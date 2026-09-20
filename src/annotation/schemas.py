"""
M7 Annotation and Ground-Truth Schemas.

Preserves and reuses:
- src.data.schemas.GroundTruthStatus (SUPPORTED, HALLUCINATED, UNKNOWN)
- src.data.schemas.SplitName (TRAIN, VALIDATION, CALIBRATION, TEST)
- src.evidence.schemas.ClaimLevelEvidenceRecord

Strictly ensures:
1. No parallel GroundTruthStatus enum is defined.
2. Case-insensitive parsing helper `parse_ground_truth_status` handles raw input.
3. UNKNOWN is preserved as indeterminate visual support, never conflated with ABSTAIN or Ising spins.
4. Raw M6 ClaimLevelEvidenceRecord is encapsulated without losing fields or reinterpreting scores as probabilities.
5. All records are standard dataclasses with strict serialization/deserialization.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import json

from src.data.schemas import GroundTruthStatus, SplitName
from src.evidence.schemas import ClaimLevelEvidenceRecord, EVIDENCE_SCHEMA_VERSION

M7_ANNOTATION_SCHEMA_VERSION: str = "1.0.0"


def parse_ground_truth_status(value: Union[str, GroundTruthStatus]) -> GroundTruthStatus:
    """
    Safely parse user or annotator input into canonical GroundTruthStatus enum.

    Behavior:
    - Accepts GroundTruthStatus directly.
    - Accepts strings case-insensitively.
    - Strips whitespace.
    - Strictly accepts 'supported', 'hallucinated', 'unknown'.
    - Explicitly rejects non-standard tokens such as 'abstain', 'uncertain',
      'true', 'false', 'yes', 'no'.

    Args:
        value: Input string or GroundTruthStatus enum.

    Returns:
        Canonical GroundTruthStatus enum.

    Raises:
        ValueError: If value is unrecognized or forbidden.
    """
    if isinstance(value, GroundTruthStatus):
        return value

    if not isinstance(value, str):
        raise ValueError(f"Ground truth status must be a string or GroundTruthStatus, got {type(value)}")

    cleaned = value.strip().lower()
    for status in GroundTruthStatus:
        if status.value == cleaned:
            return status

    raise ValueError(
        f"Invalid ground truth status '{value}'. Allowed canonical labels: "
        f"{[s.value for s in GroundTruthStatus]}. Non-standard labels (abstain, "
        "uncertain, true, false, yes, no) are strictly forbidden."
    )


@dataclass
class HumanAnnotationTask:
    """
    Masked record presented to a human annotator.

    STRICT METHODOLOGICAL CONSTRAINTS:
    Contains ONLY information necessary to judge if the claimed object is visually present.
    Strictly forbids exposing:
      - detector_score, detector_available
      - clip_score, similarity_available
      - model confidence, posterior
      - theta, epsilon, J
      - prediction, decision, abstain
      - split
      - full generated caption by default (avoids linguistic confirmation bias)
    """
    task_id: str
    claim_id: str
    image_id: str
    image_path: str
    object_category: str
    surface_form: str
    minimal_context: Optional[str] = None
    schema_version: str = M7_ANNOTATION_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "object_category": self.object_category,
            "surface_form": self.surface_form,
            "minimal_context": self.minimal_context,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanAnnotationTask":
        return cls(
            task_id=str(data["task_id"]),
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            image_path=str(data["image_path"]),
            object_category=str(data["object_category"]),
            surface_form=str(data["surface_form"]),
            minimal_context=data.get("minimal_context"),
            schema_version=str(data.get("schema_version", M7_ANNOTATION_SCHEMA_VERSION)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AnnotationRecord:
    """
    Independent human annotation record.

    Attributes:
        claim_id: Target claim ID.
        image_id: Target image ID.
        annotator_id: Unique identifier of the annotator (e.g., 'annotator_A').
        ground_truth_status: Canonical status (SUPPORTED, HALLUCINATED, UNKNOWN).
        rationale: Optional notes or visual evidence justification.
        timestamp: ISO-format UTC timestamp.
        schema_version: Schema version string.
    """
    claim_id: str
    image_id: str
    annotator_id: str
    ground_truth_status: GroundTruthStatus
    rationale: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = M7_ANNOTATION_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "annotator_id": self.annotator_id,
            "ground_truth_status": self.ground_truth_status.value,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRecord":
        return cls(
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            annotator_id=str(data["annotator_id"]),
            ground_truth_status=parse_ground_truth_status(data["ground_truth_status"]),
            rationale=data.get("rationale"),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            schema_version=str(data.get("schema_version", M7_ANNOTATION_SCHEMA_VERSION)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AdjudicationRecord:
    """
    Adjudication record resolving annotator disagreements.

    Attributes:
        claim_id: Target claim ID.
        image_id: Target image ID.
        adjudicator_id: ID of the expert adjudicator.
        adjudicated_status: Final resolved status.
        notes: Reason for resolving disagreement.
        timestamp: ISO-format UTC timestamp.
    """
    claim_id: str
    image_id: str
    adjudicator_id: str
    adjudicated_status: GroundTruthStatus
    notes: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = M7_ANNOTATION_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "adjudicator_id": self.adjudicator_id,
            "adjudicated_status": self.adjudicated_status.value,
            "notes": self.notes,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdjudicationRecord":
        return cls(
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            adjudicator_id=str(data["adjudicator_id"]),
            adjudicated_status=parse_ground_truth_status(data["adjudicated_status"]),
            notes=data.get("notes"),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            schema_version=str(data.get("schema_version", M7_ANNOTATION_SCHEMA_VERSION)),
        )


@dataclass
class M7ClaimRecord:
    """
    M7 Master Claim record wrapping M6 ClaimLevelEvidenceRecord with split assignment.

    Preserves all original M6 evidence, provenance, and raw scores without
    treating them as calibrated probabilities.
    """
    claim_id: str
    image_id: str
    object_category: str
    text_span: Optional[str]
    caption: str
    image_hash: str
    split: str
    evidence: ClaimLevelEvidenceRecord
    is_synthetic: bool = False
    schema_version: str = M7_ANNOTATION_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "object_category": self.object_category,
            "text_span": self.text_span,
            "caption": self.caption,
            "image_hash": self.image_hash,
            "split": self.split,
            "evidence": self.evidence.to_dict(),
            "is_synthetic": self.is_synthetic,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "M7ClaimRecord":
        ev_data = data["evidence"]
        evidence = ClaimLevelEvidenceRecord.from_dict(ev_data)
        return cls(
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            object_category=str(data["object_category"]),
            text_span=data.get("text_span"),
            caption=str(data.get("caption", "")),
            image_hash=str(data.get("image_hash", "")),
            split=str(data.get("split", "train")),
            evidence=evidence,
            is_synthetic=bool(data.get("is_synthetic", False)),
            schema_version=str(data.get("schema_version", M7_ANNOTATION_SCHEMA_VERSION)),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_m6_evidence(
        cls,
        evidence: ClaimLevelEvidenceRecord,
        assigned_split: Optional[Union[str, SplitName]] = None,
    ) -> "M7ClaimRecord":
        """Transform an M6 ClaimLevelEvidenceRecord into an M7ClaimRecord."""
        split_val = (
            assigned_split.value
            if isinstance(assigned_split, SplitName)
            else (assigned_split or evidence.split)
        )
        return cls(
            claim_id=evidence.claim_id,
            image_id=evidence.image_id,
            object_category=evidence.object_category,
            text_span=evidence.text_span,
            caption=evidence.caption,
            image_hash=evidence.image_hash,
            split=split_val,
            evidence=evidence,
            is_synthetic=evidence.is_synthetic,
            metadata=dict(evidence.metadata),
        )


@dataclass
class FinalGroundTruthRecord:
    """
    Consolidated Ground-Truth benchmark record.

    Permanently retains:
    - Raw independent Annotator A label & rationale
    - Raw independent Annotator B label & rationale
    - Disagreement flag
    - Adjudication record if disagreement occurred
    - Final resolved ground truth status (or None if unannotated / pending adjudication)
    - Full M7 claim and evidence record for internal benchmarking
    """
    claim_id: str
    image_id: str
    object_category: str
    text_span: Optional[str]
    split: str
    claim_record: M7ClaimRecord
    annotation_a: Optional[AnnotationRecord] = None
    annotation_b: Optional[AnnotationRecord] = None
    has_disagreement: bool = False
    adjudication: Optional[AdjudicationRecord] = None
    final_ground_truth: Optional[GroundTruthStatus] = None
    is_synthetic: bool = False
    schema_version: str = M7_ANNOTATION_SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "object_category": self.object_category,
            "text_span": self.text_span,
            "split": self.split,
            "claim_record": self.claim_record.to_dict(),
            "annotation_a": self.annotation_a.to_dict() if self.annotation_a else None,
            "annotation_b": self.annotation_b.to_dict() if self.annotation_b else None,
            "has_disagreement": self.has_disagreement,
            "adjudication": self.adjudication.to_dict() if self.adjudication else None,
            "final_ground_truth": self.final_ground_truth.value if self.final_ground_truth else None,
            "is_synthetic": self.is_synthetic,
            "schema_version": self.schema_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FinalGroundTruthRecord":
        claim_record = M7ClaimRecord.from_dict(data["claim_record"])
        ann_a = AnnotationRecord.from_dict(data["annotation_a"]) if data.get("annotation_a") else None
        ann_b = AnnotationRecord.from_dict(data["annotation_b"]) if data.get("annotation_b") else None
        adj = AdjudicationRecord.from_dict(data["adjudication"]) if data.get("adjudication") else None
        f_gt = parse_ground_truth_status(data["final_ground_truth"]) if data.get("final_ground_truth") else None

        return cls(
            claim_id=str(data["claim_id"]),
            image_id=str(data["image_id"]),
            object_category=str(data["object_category"]),
            text_span=data.get("text_span"),
            split=str(data.get("split", claim_record.split)),
            claim_record=claim_record,
            annotation_a=ann_a,
            annotation_b=ann_b,
            has_disagreement=bool(data.get("has_disagreement", False)),
            adjudication=adj,
            final_ground_truth=f_gt,
            is_synthetic=bool(data.get("is_synthetic", claim_record.is_synthetic)),
            schema_version=str(data.get("schema_version", M7_ANNOTATION_SCHEMA_VERSION)),
            metadata=data.get("metadata", {}),
        )
