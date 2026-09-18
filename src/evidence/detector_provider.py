"""
Object detector providers for raw visual evidence extraction.

CRITICAL METHODOLOGICAL REQUIREMENTS:
1. All scores are RAW bounding-box presence max-scores d_i in [0.0, 1.0].
2. Detector scores are NOT calibrated probabilities and must not be calibrated here.
3. Primary detector is google/owlvit-base-patch32.
4. Failures must never silently return 0.0; explicit availability flags and error diagnostics are enforced.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import math

logger = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    """Standard container for object detection evidence output."""
    score: Optional[float]
    available: bool
    model_name: str
    model_revision: Optional[str]
    configuration: Dict[str, Any] = field(default_factory=dict)
    boxes: Optional[List[List[float]]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "available": self.available,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "configuration": self.configuration,
            "boxes": self.boxes,
            "error": self.error,
        }


class BaseDetectorProvider(ABC):
    """Abstract base class for object detection evidence providers."""

    @abstractmethod
    def detect_category(
        self,
        image_path: Union[str, Path],
        category: str,
        **kwargs,
    ) -> DetectorResult:
        """
        Compute raw bounding-box presence max-score d_i in [0.0, 1.0] for an object category.

        Args:
            image_path: Path to the target image.
            category: Canonical name of the object category to detect.

        Returns:
            DetectorResult containing raw score, availability flag, model provenance, and diagnostics.
        """
        pass


class HuggingFaceDetectorProvider(BaseDetectorProvider):
    """
    Production detector provider using HuggingFace models.
    Primary default detector: google/owlvit-base-patch32 (open-vocabulary zero-shot detector).
    Modularly supports secondary architectures (e.g., DETR) without mixing score distributions.
    """

    _model_cache: Dict[str, Any] = {}
    _processor_cache: Dict[str, Any] = {}
    _revision_cache: Dict[str, str] = {}

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch32",
        model_revision: Optional[str] = None,
        device: str = "cpu",
        prompt_template: str = "a photo of a {category}",
        local_files_only: bool = True,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.prompt_template = prompt_template
        self.local_files_only = local_files_only
        self._is_owlvit = "owlvit" in model_name.lower()
        self._is_detr = "detr" in model_name.lower()

    def resolve_revision(self) -> str:
        """Resolve the model snapshot commit hash without eagerly loading weights."""
        if self.model_revision:
            return self.model_revision
        if self.model_name in self._revision_cache:
            return self._revision_cache[self.model_name]

        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            if hasattr(config, "_commit_hash") and config._commit_hash:
                rev = str(config._commit_hash)
            elif hasattr(config, "to_dict"):
                rev = f"cfg_{hashlib.sha256(str(config.to_dict()).encode('utf-8')).hexdigest()[:12]}"
            else:
                rev = "unknown_revision"
            self._revision_cache[self.model_name] = rev
            return rev
        except Exception:
            return self.model_revision or "unresolved"

    def _ensure_loaded(self):
        """Lazy-load the model and processor once per process."""
        if self.model_name in self._model_cache and self.model_name in self._processor_cache:
            return

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForObjectDetection, OwlViTForObjectDetection
        except ImportError as e:
            raise RuntimeError(
                f"Missing required detection dependencies: {e}. Install torch and transformers."
            ) from e

        try:
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            if self._is_owlvit:
                model = OwlViTForObjectDetection.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
            else:
                model = AutoModelForObjectDetection.from_pretrained(
                    self.model_name,
                    local_files_only=self.local_files_only,
                )
        except Exception as err:
            raise RuntimeError(f"Failed to load detector model '{self.model_name}': {err}") from err

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if self.device != "cpu" and torch.cuda.is_available():
            model = model.to(self.device)

        rev = self.resolve_revision()
        if hasattr(model, "config") and hasattr(model.config, "_commit_hash") and model.config._commit_hash:
            rev = str(model.config._commit_hash)
            self._revision_cache[self.model_name] = rev

        self._model_cache[self.model_name] = model
        self._processor_cache[self.model_name] = processor

    def detect_category(
        self,
        image_path: Union[str, Path],
        category: str,
        **kwargs,
    ) -> DetectorResult:
        path_obj = Path(image_path)
        config_dict = {
            "prompt_template": self.prompt_template,
            "device": self.device,
            "model_type": "owlvit" if self._is_owlvit else ("detr" if self._is_detr else "unknown"),
        }

        if not path_obj.exists():
            return DetectorResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.resolve_revision(),
                configuration=config_dict,
                error=f"Image file does not exist: {path_obj}",
            )

        if not category or not category.strip():
            return DetectorResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.resolve_revision(),
                configuration=config_dict,
                error="Object category cannot be empty",
            )

        cat_clean = category.strip().lower()

        try:
            self._ensure_loaded()
            import torch
            from PIL import Image, ImageOps

            with Image.open(path_obj) as raw_img:
                img_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")

            processor = self._processor_cache[self.model_name]
            model = self._model_cache[self.model_name]
            revision = self.resolve_revision()

            if self._is_owlvit:
                query_text = self.prompt_template.format(category=cat_clean)
                inputs = processor(
                    text=[[query_text]],
                    images=img_rgb,
                    return_tensors="pt",
                )
                if self.device != "cpu" and torch.cuda.is_available():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = model(**inputs)

                # OWL-ViT outputs logits of shape [batch_size, num_boxes, num_queries]
                logits = outputs.logits[0, :, 0]  # shape: [num_boxes]
                probs = torch.sigmoid(logits)     # raw bounded score in (0, 1)
                max_score = float(probs.max().item())

                # Validate strict bounds
                if math.isnan(max_score) or math.isinf(max_score):
                    raise ValueError(f"Detector returned non-finite score: {max_score}")
                max_score = max(0.0, min(1.0, max_score))

                return DetectorResult(
                    score=max_score,
                    available=True,
                    model_name=self.model_name,
                    model_revision=revision,
                    configuration=config_dict,
                )

            elif self._is_detr:
                # DETR classification over fixed COCO category indices
                inputs = processor(images=img_rgb, return_tensors="pt")
                if self.device != "cpu" and torch.cuda.is_available():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    outputs = model(**inputs)

                logits = outputs.logits[0]  # [num_boxes, num_classes + 1]
                probs = torch.softmax(logits, dim=-1)[:, :-1]  # exclude 'no-object'

                # Match category name to id2label
                id2label = getattr(model.config, "id2label", {})
                target_cls_idx = None
                for idx, label in id2label.items():
                    if label.strip().lower() == cat_clean:
                        target_cls_idx = int(idx)
                        break

                if target_cls_idx is None:
                    return DetectorResult(
                        score=None,
                        available=False,
                        model_name=self.model_name,
                        model_revision=revision,
                        configuration=config_dict,
                        error=f"Category '{cat_clean}' not found in DETR label ontology",
                    )

                max_score = float(probs[:, target_cls_idx].max().item())
                max_score = max(0.0, min(1.0, max_score))

                return DetectorResult(
                    score=max_score,
                    available=True,
                    model_name=self.model_name,
                    model_revision=revision,
                    configuration=config_dict,
                )

            else:
                return DetectorResult(
                    score=None,
                    available=False,
                    model_name=self.model_name,
                    model_revision=revision,
                    configuration=config_dict,
                    error=f"Unsupported detector architecture for model '{self.model_name}'",
                )

        except Exception as e:
            logger.warning(f"Detection failed for image={path_obj} category={cat_clean}: {e}")
            return DetectorResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.resolve_revision(),
                configuration=config_dict,
                error=str(e),
            )


class MockDetectorProvider(BaseDetectorProvider):
    """
    Deterministic offline mock detector for unit testing and offline CI.
    Allows exact score injections, deterministic pseudo-scoring, and error simulation.
    """

    def __init__(
        self,
        fixed_scores: Optional[Dict[str, float]] = None,
        simulate_failure: bool = False,
        failure_error_message: str = "Simulated detector failure",
        model_name: str = "mock-detector-v1",
        model_revision: str = "rev_mock_001",
    ):
        self.fixed_scores = fixed_scores or {}
        self.simulate_failure = simulate_failure
        self.failure_error_message = failure_error_message
        self.model_name = model_name
        self.model_revision = model_revision

    def detect_category(
        self,
        image_path: Union[str, Path],
        category: str,
        **kwargs,
    ) -> DetectorResult:
        config_dict = {"mock": True, "simulate_failure": self.simulate_failure}

        if self.simulate_failure:
            return DetectorResult(
                score=None,
                available=False,
                model_name=self.model_name,
                model_revision=self.model_revision,
                configuration=config_dict,
                error=self.failure_error_message,
            )

        cat_clean = category.strip().lower()

        # Check explicit fixed mapping: (image_stem, category) or (category)
        path_stem = Path(image_path).stem
        if (path_stem, cat_clean) in self.fixed_scores:
            raw_val = self.fixed_scores[(path_stem, cat_clean)]
        elif cat_clean in self.fixed_scores:
            raw_val = self.fixed_scores[cat_clean]
        else:
            # Deterministic hash-based float in [0.05, 0.95]
            h = hashlib.sha256(f"{path_stem}_{cat_clean}".encode("utf-8")).hexdigest()
            raw_val = (int(h[:8], 16) % 900 + 50) / 1000.0

        raw_val = max(0.0, min(1.0, float(raw_val)))
        return DetectorResult(
            score=raw_val,
            available=True,
            model_name=self.model_name,
            model_revision=self.model_revision,
            configuration=config_dict,
        )
