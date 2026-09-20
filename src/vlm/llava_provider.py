"""
Production adapter for the frozen LLaVA-1.5-7B vision-language model.

Target Checkpoint: llava-hf/llava-1.5-7b-hf
Uses lazy imports to ensure core mathematical PGM and dataset modules stay lightweight and dependency-isolated.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Dict, Any, Optional, Union

from src.vlm.provider import (
    VLMProvider,
    VLMGenerationConfig,
    VLMResponse,
    compute_file_sha256,
)


class LLaVA15Provider:
    """
    Production adapter for llava-hf/llava-1.5-7b-hf.
    Enforces frozen weights, greedy decoding, and explicit execution provenance.
    """
    is_synthetic: bool = False
    provider_kind: str = "llava_15_hf"

    _loaded_model_id: Optional[str] = None
    _loaded_quantization: Optional[bool] = None
    _model_instance: Any = None
    _processor_instance: Any = None
    _resolved_revision: Optional[str] = None

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        model_revision: Optional[str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        local_files_only: bool = True,
        allow_download: bool = False,
        load_in_4bit: bool = False,
        quantization_type: str = "nf4",
        bnb_4bit_compute_dtype: Optional[str] = "float16",
        bnb_4bit_use_double_quant: bool = True,
        device_map: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only and (not allow_download)
        self.allow_download = allow_download
        self.provider_kind = "llava_15_hf"
        self.load_in_4bit = bool(load_in_4bit)
        self.quantization_type = str(quantization_type)
        self.bnb_4bit_compute_dtype = str(bnb_4bit_compute_dtype or "float16")
        self.bnb_4bit_use_double_quant = bool(bnb_4bit_use_double_quant)
        self.device_map = device_map or ("auto" if self.load_in_4bit else None)

    def resolve_revision(self) -> str:
        """
        Resolve the exact model commit hash / revision without eagerly loading full model tensor weights.
        """
        if self.model_revision:
            return self.model_revision
        if LLaVA15Provider._resolved_revision is not None and LLaVA15Provider._loaded_model_id == self.model_name:
            return LLaVA15Provider._resolved_revision

        # Try resolving revision lightweight from AutoConfig without tensor weight instantiation
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            if hasattr(config, "_commit_hash") and config._commit_hash:
                revision = str(config._commit_hash)
            elif hasattr(config, "to_dict"):
                revision = f"cfg_{hashlib.sha256(str(config.to_dict()).encode('utf-8')).hexdigest()[:12]}"
            else:
                revision = "unknown_revision"
            LLaVA15Provider._resolved_revision = revision
            LLaVA15Provider._loaded_model_id = self.model_name
            return revision
        except Exception:
            return LLaVA15Provider._resolved_revision or "not_loaded"

    def _ensure_loaded(self):
        """Lazy-load the model and processor once per process."""
        if (
            LLaVA15Provider._model_instance is not None
            and LLaVA15Provider._loaded_model_id == self.model_name
            and LLaVA15Provider._loaded_quantization == self.load_in_4bit
        ):
            return

        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as e:
            raise RuntimeError(
                f"Missing required VLM runtime dependencies: {e}. "
                "Install with 'uv add torch transformers pillow bitsandbytes' or supply a local virtual environment with PyTorch and Transformers."
            ) from e

        if self.load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit inference using bitsandbytes requires an active CUDA GPU runtime. "
                    "torch.cuda.is_available() returned False."
                )
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as err:
                raise RuntimeError(
                    f"BitsAndBytesConfig not available in transformers: {err}. "
                    "Ensure 'bitsandbytes' is installed via 'uv add --optional vlm bitsandbytes'."
                ) from err

            compute_dtype_torch = getattr(torch, self.bnb_4bit_compute_dtype, torch.float16)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.quantization_type,
                bnb_4bit_compute_dtype=compute_dtype_torch,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            )
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": self.device_map or "auto",
                "low_cpu_mem_usage": True,
                "local_files_only": self.local_files_only,
            }
        else:
            torch_dtype = getattr(torch, self.dtype, torch.float32)
            model_kwargs = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "local_files_only": self.local_files_only,
            }
            if self.device != "cpu" and torch.cuda.is_available():
                model_kwargs["device_map"] = self.device_map or self.device

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs,
            )
        except Exception as err:
            if self.local_files_only:
                raise RuntimeError(
                    f"LLaVA-1.5 model '{self.model_name}' not found in local cache. "
                    "In local-only mode, pretrained weights must be present locally. "
                    "To permit downloading weights, explicitly pass allow_download=True or use --allow-download."
                ) from err
            raise RuntimeError(f"Failed to load model '{self.model_name}': {err}") from err

        # Freeze model weights and set to eval mode
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if "device_map" not in model_kwargs and self.device != "cpu" and torch.cuda.is_available():
            model = model.to(self.device)

        # Inspect resolved revision or config hash
        revision = "unknown_revision"
        if hasattr(model, "config") and hasattr(model.config, "_commit_hash") and model.config._commit_hash:
            revision = str(model.config._commit_hash)
        elif hasattr(model, "config") and hasattr(model.config, "to_dict"):
            revision = f"cfg_{hashlib.sha256(str(model.config.to_dict()).encode('utf-8')).hexdigest()[:12]}"
        elif self.model_revision:
            revision = self.model_revision

        LLaVA15Provider._loaded_model_id = self.model_name
        LLaVA15Provider._loaded_quantization = self.load_in_4bit
        LLaVA15Provider._model_instance = model
        LLaVA15Provider._processor_instance = processor
        LLaVA15Provider._resolved_revision = revision

    def get_model_info(self) -> Dict[str, Any]:
        rev = self.resolve_revision()
        return {
            "model_name": self.model_name,
            "model_revision": rev,
            "resolved_revision": rev,
            "is_synthetic": False,
            "provider_kind": self.provider_kind,
            "device": self.device,
            "dtype": self.dtype,
            "local_files_only": self.local_files_only,
            "provider_type": "LLaVA15Provider",
            "quantization_enabled": self.load_in_4bit,
            "quantization_type": self.quantization_type if self.load_in_4bit else None,
            "compute_dtype": self.bnb_4bit_compute_dtype if self.load_in_4bit else self.dtype,
            "device_map": self.device_map or ("auto" if self.load_in_4bit else (self.device if self.device != "cpu" else None)),
        }

    def generate_caption(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        gen_config: Optional[VLMGenerationConfig] = None,
        image_id: Optional[str] = None,
        image_hash: Optional[str] = None,
    ) -> VLMResponse:
        path_obj = Path(image_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Image path does not exist: {path_obj}")

        config = gen_config or VLMGenerationConfig(
            model_name=self.model_name,
            device=self.device,
            dtype=self.dtype,
            local_files_only=self.local_files_only,
            load_in_4bit=self.load_in_4bit,
            quantization_type=self.quantization_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            compute_dtype=self.bnb_4bit_compute_dtype if self.load_in_4bit else self.dtype,
            device_map=self.device_map or ("auto" if self.load_in_4bit else (self.device if self.device != "cpu" else None)),
        )
        active_prompt = prompt or config.prompt
        img_id = image_id or path_obj.stem
        resolved_hash = image_hash if image_hash is not None else compute_file_sha256(path_obj)

        self._ensure_loaded()
        import torch
        from PIL import Image, ImageOps

        # Preprocess image: handle EXIF rotation and ensure RGB
        with Image.open(path_obj) as raw_img:
            img_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")

        # Format conversation prompt for LLaVA 1.5
        conversation_prompt = f"USER: <image>\n{active_prompt}\nASSISTANT:"

        processor = LLaVA15Provider._processor_instance
        model = LLaVA15Provider._model_instance
        revision = self.resolve_revision()

        inputs = processor(
            text=conversation_prompt,
            images=img_rgb,
            return_tensors="pt",
        )

        if self.load_in_4bit:
            target_device = getattr(model, "device", None) or (self.device if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        elif self.device != "cpu" and torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        start_time = time.time()
        with torch.inference_mode():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=None if not config.do_sample else config.temperature,
                num_beams=1,
            )
        elapsed = time.time() - start_time

        # Extract only newly generated tokens (slice past input prompt tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = output_tokens[0, input_len:]
        caption_text = processor.decode(generated_tokens, skip_special_tokens=True).strip()

        return VLMResponse.create(
            image_id=img_id,
            image_path=path_obj,
            image_hash=resolved_hash,
            caption=caption_text,
            model_name=self.model_name,
            model_revision=revision,
            prompt=active_prompt,
            gen_config=config,
            is_synthetic=False,
            provider_kind=self.provider_kind,
            execution_time_seconds=elapsed,
            generation_source="real_inference",
        )

