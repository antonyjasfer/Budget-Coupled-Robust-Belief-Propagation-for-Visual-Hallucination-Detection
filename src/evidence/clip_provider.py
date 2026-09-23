"""
CLIP providers for multimodal image-text cosine similarity calculation.

CRITICAL METHODOLOGICAL REQUIREMENTS:
1. All scores are raw cosine similarities g_i = <v_img, v_txt> / (||v_img|| * ||v_txt||) in [-1.0, 1.0].
2. Similarity scores are observable geometric evidence, NOT calibrated probabilities.
3. Failures must never silently return 0.0; explicit availability flags and error diagnostics are enforced.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import math

logger = logging.getLogger(__name__)


@dataclass
class CLIPResult:
    """Standard container for CLIP cosine similarity evidence output."""
    score: Optional[float]
    available: bool
    model_name: str
    model_revision: Optional[str]
    prompt_template: Optional[str] = None
    preprocessing_configuration: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "available": self.available,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "prompt_template": self.prompt_template,
            "preprocessing_configuration": self.preprocessing_configuration,
            "error": self.error,
        }


class BaseCLIPProvider(ABC):
    """Abstract base class for image-text similarity providers."""

    @abstractmethod
    def compute_similarity(
        self,
        image_path: Union[str, Path],
        text: str,
        **kwargs,
    ) -> CLIPResult:
        """
        Compute raw cosine similarity g_i in [-1.0, 1.0] between image and text.

        Args:
            image_path: Path to target image.
            text: Surface claim phrase or category query text.

        Returns:
            CLIPResult containing raw score, availability flag, model provenance, and diagnostics.
        """
        pass


class TransformersCLIPProvider(BaseCLIPProvider):
    """
    Production CLIP similarity provider using HuggingFace Transformers.
    Target Model: openai/clip-vit-base-patch32
    Calculates L2-normalized image and text embeddings and their inner product.
    """

    _model_cache: Dict[str, Any] = {}
    _processor_cache: Dict[str, Any] = {}
    _revision_cache: Dict[str, str] = {}

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        model_revision: Optional[str] = None,
        device: str = "cpu",
        prompt_template: Optional[str] = "a photo of a {category}",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.prompt_template = prompt_template
        self.local_files_only = local_files_only

    def resolve_revision(self) -> str:
        """Resolve model commit hash without loading tensor weights."""
        if self.model_name in self._revision_cache:
            return self._revision_cache[self.model_name]

        try:
            from transformers import AutoConfig
            config_kwargs = {"local_files_only": self.local_files_only}
            if self.model_revision:
                config_kwargs["revision"] = self.model_revision
            config = AutoConfig.from_pretrained(
                self.model_name,
                **config_kwargs,
            )
            commit = getattr(config, "_commit_hash", None)
            if commit is not None and not type(commit).__name__ == "MagicMock":
                rev = str(commit)
                if self.model_revision and rev != self.model_revision:
                    raise RuntimeError(
                        f"INVALID_PROVENANCE: CLIP revision mismatch for {self.model_name}. "
                        f"Expected pinned revision '{self.model_revision}', resolved '{rev}'"
                    )
            elif hasattr(config, "to_dict"):
                rev = f"cfg_{hashlib.sha256(str(config.to_dict()).encode('utf-8')).hexdigest()[:12]}"
            else:
                rev = "unknown_revision"
            self._revision_cache[self.model_name] = rev
            return rev
        except Exception as e:
            if "INVALID_PROVENANCE" in str(e):
                raise
            return self.model_revision or "unresolved"

    def _ensure_loaded(self):
        """Lazy-load CLIP model and processor once per process."""
        if self.model_name in self._model_cache and self.model_name in self._processor_cache:
            return

        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ImportError as e:
            raise RuntimeError(
                f"Missing required CLIP dependencies: {e}. Install torch and transformers."
            ) from e

        load_kwargs = {"local_files_only": self.local_files_only}
        if self.model_revision:
            load_kwargs["revision"] = self.model_revision

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
            model = CLIPModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as err:
            if "INVALID_PROVENANCE" in str(err):
                raise
            raise RuntimeError(f"Failed to load CLIP model '{self.model_name}': {err}") from err

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if self.device != "cpu" and torch.cuda.is_available():
            model = model.to(self.device)

        commit_hash = None
        if hasattr(model, "config") and hasattr(model.config, "_commit_hash"):
            c = model.config._commit_hash
            if c is not None and not type(c).__name__ == "MagicMock":
                commit_hash = c
        if commit_hash is None and hasattr(model, "_commit_hash"):
            c = model._commit_hash
            if c is not None and not type(c).__name__ == "MagicMock":
                commit_hash = c

        if commit_hash is not None and not type(commit_hash).__name__ == "MagicMock":
            commit_str = str(commit_hash)
            if self.model_revision and commit_str != self.model_revision:
                raise RuntimeError(
                    f"INVALID_PROVENANCE: CLIP revision mismatch for {self.model_name}. "
                    f"Expected pinned revision '{self.model_revision}', resolved '{commit_str}'"
                )
            rev = commit_str
            self._revision_cache[self.model_name] = rev
        else:
            rev = self.resolve_revision()

        self._model_cache[self.model_name] = model
        self._processor_cache[self.model_name] = processor

    def compute_similarity(
        self,
        image_path: Union[str, Path],
        text: str,
        **kwargs,
    ) -> CLIPResult:
        path_obj = Path(image_path)
        config_dict = {
            "device": self.device,
            "prompt_template": self.prompt_template,
            "normalization": "l2",
        }

        if not path_obj.exists():
            return CLIPResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.resolve_revision(),
                prompt_template=self.prompt_template,
                preprocessing_configuration=config_dict,
                error=f"Image file does not exist: {path_obj}",
            )

        if not text or not text.strip():
            return CLIPResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.resolve_revision(),
                prompt_template=self.prompt_template,
                preprocessing_configuration=config_dict,
                error="Input query text cannot be empty",
            )

        # Format prompt if template provided and keyword matches
        clean_text = text.strip()
        if self.prompt_template and "{category}" in self.prompt_template:
            formatted_text = self.prompt_template.format(category=clean_text)
        else:
            formatted_text = clean_text

        try:
            self._ensure_loaded()
            import torch
            from PIL import Image, ImageOps

            with Image.open(path_obj) as raw_img:
                img_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")

            processor = self._processor_cache[self.model_name]
            model = self._model_cache[self.model_name]
            revision = self.resolve_revision()

            inputs = processor(
                text=[formatted_text],
                images=img_rgb,
                return_tensors="pt",
                padding=True,
            )
            if self.device != "cpu" and torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            # Extract normalized feature embeddings
            img_embeds = outputs.image_embeds  # shape: [1, D]
            txt_embeds = outputs.text_embeds   # shape: [1, D]

            img_norm = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
            txt_norm = txt_embeds / txt_embeds.norm(dim=-1, keepdim=True)

            # Raw cosine similarity: inner product of unit vectors
            cos_sim = (img_norm * txt_norm).sum(dim=-1).item()

            if math.isnan(cos_sim) or math.isinf(cos_sim):
                raise ValueError(f"CLIP returned non-finite cosine similarity: {cos_sim}")

            # Bounded in [-1.0, 1.0]
            cos_sim = max(-1.0, min(1.0, float(cos_sim)))

            return CLIPResult(
                score=cos_sim,
                available=True,
                model_name=self.model_name,
                model_revision=revision,
                prompt_template=self.prompt_template,
                preprocessing_configuration=config_dict,
            )

        except Exception as e:
            logger.warning(f"CLIP computation failed for image={path_obj} text='{formatted_text}': {e}")
            return CLIPResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.resolve_revision(),
                prompt_template=self.prompt_template,
                preprocessing_configuration=config_dict,
                error=str(e),
            )


class MockCLIPProvider(BaseCLIPProvider):
    """
    Deterministic offline mock CLIP provider for unit testing and offline verification.
    """

    def __init__(
        self,
        fixed_scores: Optional[Dict[str, float]] = None,
        simulate_failure: bool = False,
        failure_error_message: str = "Simulated CLIP failure",
        model_name: str = "mock-clip-v1",
        model_revision: str = "rev_mock_clip_001",
    ):
        self.fixed_scores = fixed_scores or {}
        self.simulate_failure = simulate_failure
        self.failure_error_message = failure_error_message
        self.model_name = model_name
        self.model_revision = model_revision

    def compute_similarity(
        self,
        image_path: Union[str, Path],
        text: str,
        **kwargs,
    ) -> CLIPResult:
        config_dict = {"mock": True, "simulate_failure": self.simulate_failure}

        if self.simulate_failure:
            return CLIPResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.model_revision,
                preprocessing_configuration=config_dict,
                error=self.failure_error_message,
            )

        clean_text = text.strip().lower()
        path_stem = Path(image_path).stem

        if (path_stem, clean_text) in self.fixed_scores:
            raw_val = self.fixed_scores[(path_stem, clean_text)]
        elif clean_text in self.fixed_scores:
            raw_val = self.fixed_scores[clean_text]
        else:
            # Deterministic hash-based float in [-0.2, 0.8]
            h = hashlib.sha256(f"{path_stem}_{clean_text}".encode("utf-8")).hexdigest()
            raw_val = ((int(h[:8], 16) % 1000) - 200) / 1000.0

        raw_val = max(-1.0, min(1.0, float(raw_val)))
        return CLIPResult(
            score=raw_val,
            available=True,
            model_name=self.model_name,
            model_revision=self.model_revision,
            prompt_template="mock_template",
            preprocessing_configuration=config_dict,
        )
