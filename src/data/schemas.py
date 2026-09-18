"""
Typed data contracts and schema definitions for visual hallucination detection.

Strictly covers:
- Atomic object-existence claims only (no attributes, spatial relations, or arbitrary graphs).
- Ground truth status (SUPPORTED, HALLUCINATED, UNKNOWN) vs model decisions (SUPPORTED, HALLUCINATED, ABSTAIN).
- Explicit PGM Ising spin mapping (SUPPORTED -> -1, HALLUCINATED -> +1, UNKNOWN -> unobserved/no spin label).
- Reproducible JSON serialization and deserialization.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json

SCHEMA_VERSION: str = "1.0.0"


class GroundTruthStatus(str, Enum):
    """
    Ground truth visual support annotation.
    SUPPORTED: The claimed object exists and is visually supported in the image.
    HALLUCINATED: The claimed object does not exist in the image / is unsupported.
    UNKNOWN: Ambiguous, occluded, or unadjudicated claim where visual support is indeterminate.
    """
    SUPPORTED = "supported"
    HALLUCINATED = "hallucinated"
    UNKNOWN = "unknown"


class DecisionStatus(str, Enum):
    """
    Model output prediction decision under robust inference.
    SUPPORTED: Model predicts the claim is visually supported (e.g. robust upper bound < threshold).
    HALLUCINATED: Model predicts the claim is hallucinated (e.g. robust lower bound > threshold).
    ABSTAIN: Robust interval spans across the decision boundary; model refuses to make a brittle claim.
    """
    SUPPORTED = "supported"
    HALLUCINATED = "hallucinated"
    ABSTAIN = "abstain"


class AnnotationSource(str, Enum):
    """Provenance and reliability class of ground truth annotations."""
    HUMAN_ADJUDICATED = "human_adjudicated"
    BENCHMARK_REFERENCE = "benchmark_reference"
    METADATA_HEURISTIC = "metadata_heuristic"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class DatasetSource(str, Enum):
    """Origin dataset for images and annotations."""
    COCO = "coco"
    POPE = "pope"
    SYNTHETIC = "synthetic"


class SplitName(str, Enum):
    """Standard project split partitions."""
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


def ground_truth_to_pgm_label(gt: Union[GroundTruthStatus, str]) -> int:
    """
    Map ground-truth visual support to binary Ising spin state h in {-1, +1}.

    Convention:
        SUPPORTED    -> h = -1 (non-hallucinated)
        HALLUCINATED -> h = +1 (hallucinated)
        UNKNOWN      -> Raises ValueError (UNKNOWN is an unobserved latent variable,
                        NOT an observed label, negative label, or third spin state).

    Args:
        gt: GroundTruthStatus enum or string value.

    Returns:
        Spin value -1 or +1.
    """
    if isinstance(gt, str):
        try:
            gt = GroundTruthStatus(gt)
        except ValueError:
            raise ValueError(f"Unknown ground truth status: '{gt}'")

    if gt == GroundTruthStatus.SUPPORTED:
        return -1
    elif gt == GroundTruthStatus.HALLUCINATED:
        return +1
    elif gt == GroundTruthStatus.UNKNOWN:
        raise ValueError(
            "UNKNOWN cannot be converted to a binary Ising spin label (+1 or -1). "
            "In the binary attractive PGM (h in {-1, +1}), UNKNOWN represents an "
            "unobserved marginal variable without a fixed observed binary clamp."
        )
    else:
        raise ValueError(f"Unrecognized GroundTruthStatus: {gt}")


@dataclass
class ImageRecord:
    """
    Canonical representation of an image in the dataset.

    Attributes:
        image_id: Globally unique canonical identifier (e.g., 'coco_2017_000000123456').
        dataset_source: Source dataset origin.
        file_name: Optional relative file path or image file name.
        file_hash: Optional SHA-256 hash of image file contents for duplicate detection.
        width: Image width in pixels.
        height: Image height in pixels.
        coco_id: Original COCO integer image ID if applicable.
        metadata: Additional provenance metadata (license, url, split tags).
    """
    image_id: str
    dataset_source: DatasetSource
    file_name: Optional[str] = None
    file_hash: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    coco_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        ds_val = self.dataset_source.value if hasattr(self.dataset_source, "value") else str(self.dataset_source)
        return {
            "image_id": self.image_id,
            "dataset_source": ds_val,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "width": self.width,
            "height": self.height,
            "coco_id": self.coco_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=str(data["image_id"]),
            dataset_source=DatasetSource(data["dataset_source"]),
            file_name=data.get("file_name"),
            file_hash=data.get("file_hash"),
            width=data.get("width"),
            height=data.get("height"),
            coco_id=data.get("coco_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GeneratedResponseRecord:
    """
    VLM-generated text response associated with an image.

    Attributes:
        response_id: Unique identifier for this response.
        image_id: Foreign key linking to the image.
        model_name: Identifier of the generating VLM.
        response_text: Full generated caption or description text.
        prompt: Optional prompt text provided to the model.
        metadata: Generation parameters (temperature, max_tokens, etc.).
    """
    response_id: str
    image_id: str
    model_name: str
    response_text: str
    prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id": self.response_id,
            "image_id": self.image_id,
            "model_name": self.model_name,
            "response_text": self.response_text,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedResponseRecord":
        return cls(
            response_id=str(data["response_id"]),
            image_id=str(data["image_id"]),
            model_name=str(data["model_name"]),
            response_text=str(data["response_text"]),
            prompt=data.get("prompt"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AtomicObjectExistenceClaim:
    """
    Atomic claim stating the existence or presence of a specific object category in an image.

    Attributes:
        claim_id: Unique identifier for this atomic claim.
        image_id: Foreign key linking to the target image.
        object_category: Canonical object category name (e.g. 'dog', 'person').
        response_id: Optional reference to the generating response.
        text_span: Optional substring or character offset range where the claim appears.
        raw_claim_text: Extracted surface phrase (e.g. 'a golden retriever sitting on the rug').
        provenance: Extraction metadata (extractor version, rule ID, confidence).
    """
    claim_id: str
    image_id: str
    object_category: str
    response_id: Optional[str] = None
    text_span: Optional[str] = None
    raw_claim_text: Optional[str] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Normalize category
        self.object_category = self.object_category.strip().lower()
        if not self.object_category:
            raise ValueError("object_category must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "object_category": self.object_category,
            "response_id": self.response_id,
            "text_span": self.text_span,
            "raw_claim_text": self.raw_claim_text,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicObjectExistenceClaim":
        return cls(
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            object_category=str(data["object_category"]),
            response_id=data.get("response_id"),
            text_span=data.get("text_span"),
            raw_claim_text=data.get("raw_claim_text"),
            provenance=data.get("provenance", {}),
        )


@dataclass
class AnnotationRecord:
    """
    Ground truth adjudication for an atomic object-existence claim.

    Attributes:
        annotation_id: Unique identifier for this annotation.
        claim_id: Foreign key linking to the evaluated atomic claim.
        ground_truth: SUPPORTED, HALLUCINATED, or UNKNOWN.
        source: Provenance class (human, benchmark, heuristic, synthetic).
        annotator_notes: Optional clarifying notes.
        metadata: Detailed annotation metadata (timestamp, annotator_id, verification flag).
    """
    annotation_id: str
    claim_id: str
    ground_truth: GroundTruthStatus
    source: AnnotationSource
    annotator_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "claim_id": self.claim_id,
            "ground_truth": self.ground_truth.value,
            "source": self.source.value,
            "annotator_notes": self.annotator_notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRecord":
        return cls(
            annotation_id=str(data["annotation_id"]),
            claim_id=str(data["claim_id"]),
            ground_truth=GroundTruthStatus(data["ground_truth"]),
            source=AnnotationSource(data["source"]),
            annotator_notes=data.get("annotator_notes"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PredictionDecisionRecord:
    """
    Model output prediction decision for an atomic claim under robust inference.

    Attributes:
        decision_id: Unique identifier for this decision.
        claim_id: Foreign key linking to the evaluated claim.
        decision: Model verdict: SUPPORTED, HALLUCINATED, or ABSTAIN.
        lower_bound: Robust lower bound probability P_min(h = +1).
        upper_bound: Robust upper bound probability P_max(h = +1).
        nominal_marginal: Nominal marginal probability P_nom(h = +1).
        threshold_low: Lower decision threshold (e.g. 0.40).
        threshold_high: Upper decision threshold (e.g. 0.60).
        metadata: Detailed solver diagnostics (budget, certification gap, witness).
    """
    decision_id: str
    claim_id: str
    decision: DecisionStatus
    lower_bound: float
    upper_bound: float
    nominal_marginal: float
    threshold_low: float = 0.40
    threshold_high: float = 0.60
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "claim_id": self.claim_id,
            "decision": self.decision.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "nominal_marginal": self.nominal_marginal,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionDecisionRecord":
        return cls(
            decision_id=str(data["decision_id"]),
            claim_id=str(data["claim_id"]),
            decision=DecisionStatus(data["decision"]),
            lower_bound=float(data["lower_bound"]),
            upper_bound=float(data["upper_bound"]),
            nominal_marginal=float(data["nominal_marginal"]),
            threshold_low=float(data.get("threshold_low", 0.40)),
            threshold_high=float(data.get("threshold_high", 0.60)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DatasetManifestEntry:
    """
    Aggregated dataset entry grouping an image and all its associated artifacts.

    Attributes:
        image: Canonical image record.
        claims: List of atomic object-existence claims for this image.
        annotations: List of ground-truth annotations corresponding to the claims.
        responses: Optional list of model responses generated for this image.
        split: Assigned split partition (train, validation, calibration, test).
    """
    image: ImageRecord
    claims: List[AtomicObjectExistenceClaim] = field(default_factory=list)
    annotations: List[AnnotationRecord] = field(default_factory=list)
    responses: List[GeneratedResponseRecord] = field(default_factory=list)
    split: Optional[SplitName] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image.to_dict(),
            "claims": [c.to_dict() for c in self.claims],
            "annotations": [a.to_dict() for a in self.annotations],
            "responses": [r.to_dict() for r in self.responses],
            "split": self.split.value if self.split is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifestEntry":
        return cls(
            image=ImageRecord.from_dict(data["image"]),
            claims=[AtomicObjectExistenceClaim.from_dict(c) for c in data.get("claims", [])],
            annotations=[AnnotationRecord.from_dict(a) for a in data.get("annotations", [])],
            responses=[GeneratedResponseRecord.from_dict(r) for r in data.get("responses", [])],
            split=SplitName(data["split"]) if data.get("split") is not None else None,
        )


@dataclass
class DatasetManifest:
    """
    Top-level reproducible dataset manifest.

    Attributes:
        schema_version: Semantic schema version (e.g. '1.0.0').
        manifest_id: Unique manifest identifier.
        created_at: ISO timestamp of creation.
        description: Human-readable description of dataset contents.
        entries: List of grouped image entries.
        metadata: Additional provenance and configuration dictionary.
    """
    schema_version: str = SCHEMA_VERSION
    manifest_id: str = "default_manifest"
    created_at: str = ""
    description: str = ""
    entries: List[DatasetManifestEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "description": self.description,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            manifest_id=str(data.get("manifest_id", "manifest")),
            created_at=str(data.get("created_at", "")),
            description=str(data.get("description", "")),
            entries=[DatasetManifestEntry.from_dict(e) for e in data.get("entries", [])],
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "DatasetManifest":
        return cls.from_dict(json.loads(json_str))
