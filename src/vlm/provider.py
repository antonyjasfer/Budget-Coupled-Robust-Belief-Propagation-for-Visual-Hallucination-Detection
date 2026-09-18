"""
VLM Provider protocol, configuration, and synthetic test provider.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, Any, Optional, Union, Protocol, runtime_checkable
from datetime import datetime, timezone


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA256 checksum of raw bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class VLMGenerationConfig:
    """
    Standard generation hyperparameters and runtime settings for frozen VLMs.
    """
    model_name: str = "llava-hf/llava-1.5-7b-hf"
    prompt: str = (
        "Describe the visible physical objects in two short sentences.\n"
        "Do not speculate about objects outside the image."
    )
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.0
    seed: int = 42
    device: str = "cpu"
    dtype: str = "float32"
    local_files_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_param_hash(self) -> str:
        """Compute deterministic hash of the generation parameters."""
        data = {
            "model_name": self.model_name,
            "prompt": self.prompt,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "seed": self.seed,
            "device": self.device,
            "dtype": self.dtype,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMGenerationConfig":
        return cls(
            model_name=data.get("model_name", "llava-hf/llava-1.5-7b-hf"),
            prompt=data.get(
                "prompt",
                "Describe the visible physical objects in two short sentences.\nDo not speculate about objects outside the image."
            ),
            max_new_tokens=int(data.get("max_new_tokens", 64)),
            do_sample=bool(data.get("do_sample", False)),
            temperature=float(data.get("temperature", 0.0)),
            seed=int(data.get("seed", 42)),
            device=data.get("device", "cpu"),
            dtype=data.get("dtype", "float32"),
            local_files_only=bool(data.get("local_files_only", True)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VLMResponse:
    """
    Full provenance record for a generated VLM caption response.
    """
    response_id: str
    image_id: str
    image_path: str
    image_hash: str
    caption: str
    model_name: str
    model_revision: str
    prompt: str
    prompt_hash: str
    gen_config: Dict[str, Any]
    device: str
    dtype: str
    is_synthetic: bool
    created_at: str
    provider_kind: str = "synthetic_fixture"
    execution_time_seconds: float = 0.0

    @classmethod
    def create(
        cls,
        image_id: str,
        image_path: Union[str, Path],
        image_hash: str,
        caption: str,
        model_name: str,
        model_revision: str,
        prompt: str,
        gen_config: VLMGenerationConfig,
        is_synthetic: bool,
        provider_kind: str = "synthetic_fixture",
        execution_time_seconds: float = 0.0,
    ) -> "VLMResponse":
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        caption_hash = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16]
        
        # Build collision-resistant response_id distinguishing provider kind, synthetic flag, and response content
        id_content = (
            f"{image_id}:{image_hash}:{provider_kind}:{is_synthetic}:{model_name}:"
            f"{model_revision}:{prompt_hash}:{caption_hash}:{gen_config.get_param_hash()}"
        )
        response_id = f"resp_{hashlib.sha256(id_content.encode('utf-8')).hexdigest()[:16]}"
        
        created_at = datetime.now(timezone.utc).isoformat()
        
        return cls(
            response_id=response_id,
            image_id=image_id,
            image_path=str(image_path),
            image_hash=image_hash,
            caption=caption,
            model_name=model_name,
            model_revision=model_revision,
            prompt=prompt,
            prompt_hash=prompt_hash,
            gen_config=gen_config.to_dict(),
            device=gen_config.device,
            dtype=gen_config.dtype,
            is_synthetic=is_synthetic,
            created_at=created_at,
            provider_kind=provider_kind,
            execution_time_seconds=execution_time_seconds,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VLMResponse":
        is_synth = bool(data.get("is_synthetic", True))
        default_kind = "synthetic_fixture" if is_synth else "llava_15_hf"
        return cls(
            response_id=data["response_id"],
            image_id=data["image_id"],
            image_path=data["image_path"],
            image_hash=data["image_hash"],
            caption=data["caption"],
            model_name=data["model_name"],
            model_revision=data["model_revision"],
            prompt=data["prompt"],
            prompt_hash=data["prompt_hash"],
            gen_config=data["gen_config"],
            device=data["device"],
            dtype=data["dtype"],
            is_synthetic=is_synth,
            created_at=data["created_at"],
            provider_kind=str(data.get("provider_kind", default_kind)),
            execution_time_seconds=float(data.get("execution_time_seconds", 0.0)),
        )


@runtime_checkable
class VLMProvider(Protocol):
    """Protocol for frozen VLM caption generation providers."""
    is_synthetic: bool
    provider_kind: str

    def generate_caption(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        gen_config: Optional[VLMGenerationConfig] = None,
        image_id: Optional[str] = None,
        image_hash: Optional[str] = None,
    ) -> VLMResponse:
        ...

    def get_model_info(self) -> Dict[str, Any]:
        ...


class SyntheticVLMProvider:
    """
    Offline synthetic provider for tests and reproducible baseline demonstrations.
    Does not require network access, GPUs, or external model weight downloads.
    """
    is_synthetic: bool = True
    provider_kind: str = "synthetic_fixture"

    def __init__(
        self,
        mock_captions: Optional[Dict[str, str]] = None,
        default_caption: str = "A red car and a small dog are visible in the foreground.",
        model_name: str = "synthetic-fixture-model",
        model_revision: str = "synthetic_fixture_v1",
        provider_kind: str = "synthetic_fixture",
    ):
        self.mock_captions = mock_captions or {}
        self.default_caption = default_caption
        self.model_name = model_name
        self.model_revision = model_revision
        self.provider_kind = provider_kind

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "is_synthetic": True,
            "provider_kind": self.provider_kind,
            "provider_type": "SyntheticVLMProvider",
        }

    def generate_caption(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        gen_config: Optional[VLMGenerationConfig] = None,
        image_id: Optional[str] = None,
        image_hash: Optional[str] = None,
    ) -> VLMResponse:
        config = gen_config or VLMGenerationConfig(model_name=self.model_name)
        active_prompt = prompt or config.prompt
        
        path_obj = Path(image_path)
        img_id = image_id or path_obj.stem
        
        # Calculate image hash if not supplied
        if image_hash is not None:
            resolved_hash = image_hash
        elif path_obj.exists():
            resolved_hash = compute_file_sha256(path_obj)
        else:
            resolved_hash = hashlib.sha256(img_id.encode("utf-8")).hexdigest()

        start_time = time.time()
        # Lookup caption or generate deterministic mock
        caption = self.mock_captions.get(img_id, self.mock_captions.get(path_obj.name, self.default_caption))
        elapsed = time.time() - start_time

        return VLMResponse.create(
            image_id=img_id,
            image_path=path_obj,
            image_hash=resolved_hash,
            caption=caption,
            model_name=self.model_name,
            model_revision=self.model_revision,
            prompt=active_prompt,
            gen_config=config,
            is_synthetic=True,
            provider_kind=self.provider_kind,
            execution_time_seconds=elapsed,
        )
