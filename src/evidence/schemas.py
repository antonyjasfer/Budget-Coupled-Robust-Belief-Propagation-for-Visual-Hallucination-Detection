"""
Typed schemas for raw visual evidence records.

CRITICAL METHODOLOGICAL CONSTRAINTS:
1. Features are raw numerical evidence (detector scores d_i in [0, 1] and global cosine similarity g_i in [-1, 1]).
2. Raw features are NOT calibrated probabilities and must NOT be converted to unary fields theta_i,
   uncertainty radii epsilon_i, or Ising probabilities in this phase.
3. Missing or unexecuted evidence is explicitly represented with availability flags, NEVER silently mapped to zero.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Union
import math

EVIDENCE_SCHEMA_VERSION: str = "1.0.0"


@dataclass
class RawEvidenceRecord:
    """
    Minimal raw visual evidence for an atomic object-existence claim on an image view.

    Attributes:
        evidence_id: Globally unique identifier for this evidence record.
        image_id: Canonical image ID.
        claim_id: Atomic claim ID.
        object_category: Canonical object category.
        detector_score: Bounded object detector presence score in [0.0, 1.0] (None if unavailable).
        detector_available: Boolean flag indicating if detector executed successfully.
        similarity_score: Bounded global image-claim cosine similarity in [-1.0, 1.0] (None if unavailable).
        similarity_available: Boolean flag indicating if similarity model executed successfully.
        provider_id: Identifier of the evidence provider (e.g. 'offline_fixture_v1').
        view_id: Image view / perturbation identifier ('original' or named perturbation).
        is_synthetic: Boolean flag indicating if evidence is synthetic or fixture-derived.
        schema_version: Schema version string.
        metadata: Configuration and provenance metadata.
    """
    evidence_id: str
    image_id: str
    claim_id: str
    object_category: str
    detector_score: Optional[float] = None
    detector_available: bool = False
    similarity_score: Optional[float] = None
    similarity_available: bool = False
    provider_id: str = "default_provider"
    view_id: str = "original"
    is_synthetic: bool = True
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate value ranges, finiteness, and availability constraints."""
        if not self.evidence_id:
            raise ValueError("evidence_id cannot be empty")
        if not self.image_id:
            raise ValueError("image_id cannot be empty")
        if not self.claim_id:
            raise ValueError("claim_id cannot be empty")
        if not self.object_category:
            raise ValueError("object_category cannot be empty")

        # 1. Detector score validation
        if self.detector_available:
            if self.detector_score is None:
                raise ValueError("detector_score cannot be None when detector_available=True")
            if not isinstance(self.detector_score, (int, float)):
                raise ValueError(f"detector_score must be a float, got {type(self.detector_score)}")
            if math.isnan(self.detector_score) or math.isinf(self.detector_score):
                raise ValueError(f"detector_score must be finite, got {self.detector_score}")
            if not (0.0 <= self.detector_score <= 1.0):
                raise ValueError(f"detector_score must be in [0.0, 1.0], got {self.detector_score}")
        else:
            if self.detector_score is not None:
                raise ValueError("detector_score must be None when detector_available=False")

        # 2. Similarity score validation
        if self.similarity_available:
            if self.similarity_score is None:
                raise ValueError("similarity_score cannot be None when similarity_available=True")
            if not isinstance(self.similarity_score, (int, float)):
                raise ValueError(f"similarity_score must be a float, got {type(self.similarity_score)}")
            if math.isnan(self.similarity_score) or math.isinf(self.similarity_score):
                raise ValueError(f"similarity_score must be finite, got {self.similarity_score}")
            if not (-1.0 <= self.similarity_score <= 1.0):
                raise ValueError(f"similarity_score must be in [-1.0, 1.0], got {self.similarity_score}")
        else:
            if self.similarity_score is not None:
                raise ValueError("similarity_score must be None when similarity_available=False")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "image_id": self.image_id,
            "claim_id": self.claim_id,
            "object_category": self.object_category,
            "detector_score": self.detector_score,
            "detector_available": self.detector_available,
            "similarity_score": self.similarity_score,
            "similarity_available": self.similarity_available,
            "provider_id": self.provider_id,
            "view_id": self.view_id,
            "is_synthetic": self.is_synthetic,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawEvidenceRecord":
        return cls(
            evidence_id=str(data["evidence_id"]),
            image_id=str(data["image_id"]),
            claim_id=str(data["claim_id"]),
            object_category=str(data["object_category"]),
            detector_score=float(data["detector_score"]) if data.get("detector_score") is not None else None,
            detector_available=bool(data.get("detector_available", False)),
            similarity_score=float(data["similarity_score"]) if data.get("similarity_score") is not None else None,
            similarity_available=bool(data.get("similarity_available", False)),
            provider_id=str(data.get("provider_id", "default_provider")),
            view_id=str(data.get("view_id", "original")),
            is_synthetic=bool(data.get("is_synthetic", True)),
            schema_version=str(data.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ClaimLevelEvidenceRecord:
    """
    Claim-level multimodal evidence record representing raw observable evidence
    for an atomic object-existence claim on a real or fixture image.

    CRITICAL METHODOLOGICAL GUARANTEES:
    1. detector_score is a raw bounding-box presence max-score d_i in [0.0, 1.0].
       It is NOT a calibrated probability and must not be interpreted as one.
    2. clip_score is a raw global image-text cosine similarity g_i in [-1.0, 1.0].
    3. Failure or absence of a neural feature is explicitly tracked via availability
       flags (detector_available, similarity_available), never mapped silently to 0.0.
    4. Mathematical PGM potentials (theta_i, epsilon_i, J_ij) and robust posterior
       bounds are strictly NOT computed or stored in this raw evidence schema.
    """
    claim_id: str
    image_id: str
    object_category: str
    text_span: Optional[str] = None
    caption: str = ""
    image_hash: str = ""
    split: str = "train"
    detector_score: Optional[float] = None
    detector_available: bool = False
    detector_model: Optional[str] = None
    detector_revision: Optional[str] = None
    detector_configuration: Optional[Dict[str, Any]] = None
    clip_score: Optional[float] = None
    similarity_available: bool = False
    clip_model: Optional[str] = None
    clip_revision: Optional[str] = None
    clip_prompt_template: Optional[str] = None
    preprocessing_configuration: Optional[Dict[str, Any]] = None
    schema_version: str = EVIDENCE_SCHEMA_VERSION
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate string identifiers, numerical ranges, finiteness, and availability flags."""
        if not self.claim_id:
            raise ValueError("claim_id cannot be empty")
        if not self.image_id:
            raise ValueError("image_id cannot be empty")
        if not self.object_category:
            raise ValueError("object_category cannot be empty")

        # Normalize category
        self.object_category = self.object_category.strip().lower()

        # 1. Detector score validation (raw bounding-box presence max-score in [0, 1])
        if self.detector_available:
            if self.detector_score is None:
                raise ValueError("detector_score cannot be None when detector_available=True")
            if not isinstance(self.detector_score, (int, float)):
                raise ValueError(f"detector_score must be a float, got {type(self.detector_score)}")
            if math.isnan(self.detector_score) or math.isinf(self.detector_score):
                raise ValueError(f"detector_score must be finite, got {self.detector_score}")
            if not (0.0 <= self.detector_score <= 1.0):
                raise ValueError(f"detector_score must be in [0.0, 1.0], got {self.detector_score}")
        else:
            if self.detector_score is not None:
                raise ValueError("detector_score must be None when detector_available=False")

        # 2. CLIP cosine similarity validation (raw cosine similarity in [-1, 1])
        if self.similarity_available:
            if self.clip_score is None:
                raise ValueError("clip_score cannot be None when similarity_available=True")
            if not isinstance(self.clip_score, (int, float)):
                raise ValueError(f"clip_score must be a float, got {type(self.clip_score)}")
            if math.isnan(self.clip_score) or math.isinf(self.clip_score):
                raise ValueError(f"clip_score must be finite, got {self.clip_score}")
            if not (-1.0 <= self.clip_score <= 1.0):
                raise ValueError(f"clip_score must be in [-1.0, 1.0], got {self.clip_score}")
        else:
            if self.clip_score is not None:
                raise ValueError("clip_score must be None when similarity_available=False")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "object_category": self.object_category,
            "text_span": self.text_span,
            "caption": self.caption,
            "image_hash": self.image_hash,
            "split": self.split,
            "detector_score": self.detector_score,
            "detector_available": self.detector_available,
            "detector_model": self.detector_model,
            "detector_revision": self.detector_revision,
            "detector_configuration": self.detector_configuration,
            "clip_score": self.clip_score,
            "similarity_available": self.similarity_available,
            "clip_model": self.clip_model,
            "clip_revision": self.clip_revision,
            "clip_prompt_template": self.clip_prompt_template,
            "preprocessing_configuration": self.preprocessing_configuration,
            "schema_version": self.schema_version,
            "is_synthetic": self.is_synthetic,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimLevelEvidenceRecord":
        """Reconstruct ClaimLevelEvidenceRecord from dictionary."""
        return cls(
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            object_category=str(data["object_category"]),
            text_span=data.get("text_span"),
            caption=str(data.get("caption", "")),
            image_hash=str(data.get("image_hash", "")),
            split=str(data.get("split", "train")),
            detector_score=float(data["detector_score"]) if data.get("detector_score") is not None else None,
            detector_available=bool(data.get("detector_available", False)),
            detector_model=data.get("detector_model"),
            detector_revision=data.get("detector_revision"),
            detector_configuration=data.get("detector_configuration"),
            clip_score=float(data["clip_score"]) if data.get("clip_score") is not None else None,
            similarity_available=bool(data.get("similarity_available", False)),
            clip_model=data.get("clip_model"),
            clip_revision=data.get("clip_revision"),
            clip_prompt_template=data.get("clip_prompt_template"),
            preprocessing_configuration=data.get("preprocessing_configuration"),
            schema_version=str(data.get("schema_version", EVIDENCE_SCHEMA_VERSION)),
            is_synthetic=bool(data.get("is_synthetic", False)),
            metadata=dict(data.get("metadata", {})),
        )

