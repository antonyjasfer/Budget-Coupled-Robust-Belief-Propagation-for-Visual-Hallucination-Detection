"""
Unit tests for M7 annotation schemas and GroundTruthStatus parsing.
"""

import pytest
from src.data.schemas import GroundTruthStatus, ground_truth_to_pgm_label, SplitName
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import (
    parse_ground_truth_status,
    HumanAnnotationTask,
    AnnotationRecord,
    AdjudicationRecord,
    M7ClaimRecord,
    FinalGroundTruthRecord,
)


def test_parse_ground_truth_status_valid_variants():
    """Verify parse_ground_truth_status handles case-insensitivity and whitespace."""
    assert parse_ground_truth_status("supported") == GroundTruthStatus.SUPPORTED
    assert parse_ground_truth_status("SUPPORTED") == GroundTruthStatus.SUPPORTED
    assert parse_ground_truth_status("  Supported  ") == GroundTruthStatus.SUPPORTED

    assert parse_ground_truth_status("hallucinated") == GroundTruthStatus.HALLUCINATED
    assert parse_ground_truth_status("HALLUCINATED") == GroundTruthStatus.HALLUCINATED
    assert parse_ground_truth_status("  hallucinated  ") == GroundTruthStatus.HALLUCINATED

    assert parse_ground_truth_status("unknown") == GroundTruthStatus.UNKNOWN
    assert parse_ground_truth_status("UNKNOWN") == GroundTruthStatus.UNKNOWN
    assert parse_ground_truth_status(" Unknown ") == GroundTruthStatus.UNKNOWN

    # Direct enum pass-through
    assert parse_ground_truth_status(GroundTruthStatus.SUPPORTED) == GroundTruthStatus.SUPPORTED


def test_parse_ground_truth_status_rejects_non_standard_labels():
    """Explicitly test that non-standard labels (abstain, uncertain, true/false, yes/no) are rejected."""
    forbidden_labels = ["abstain", "uncertain", "true", "false", "yes", "no", "maybe", "invalid"]
    for label in forbidden_labels:
        with pytest.raises(ValueError, match="Invalid ground truth status"):
            parse_ground_truth_status(label)


def test_ground_truth_to_pgm_label_semantics_preserved():
    """Verify that canonical Ising mapping is strictly preserved."""
    assert ground_truth_to_pgm_label(GroundTruthStatus.SUPPORTED) == -1
    assert ground_truth_to_pgm_label(GroundTruthStatus.HALLUCINATED) == +1
    with pytest.raises(ValueError, match="UNKNOWN cannot be converted to a binary Ising spin label"):
        ground_truth_to_pgm_label(GroundTruthStatus.UNKNOWN)


def test_human_annotation_task_roundtrip():
    """Test serialization/deserialization of HumanAnnotationTask."""
    task = HumanAnnotationTask(
        task_id="task_001",
        claim_id="claim_001",
        image_id="img_001",
        image_path="images/img_001.jpg",
        object_category="dog",
        surface_form="a fluffy dog",
        minimal_context="A fluffy dog sits on the grass.",
    )
    d = task.to_dict()
    recovered = HumanAnnotationTask.from_dict(d)
    assert recovered.task_id == task.task_id
    assert recovered.claim_id == task.claim_id
    assert recovered.image_id == task.image_id
    assert recovered.object_category == task.object_category
    assert recovered.surface_form == task.surface_form
    assert recovered.minimal_context == task.minimal_context


def test_annotation_record_roundtrip():
    """Test serialization/deserialization of AnnotationRecord."""
    rec = AnnotationRecord(
        claim_id="claim_002",
        image_id="img_002",
        annotator_id="annotator_A",
        ground_truth_status=GroundTruthStatus.SUPPORTED,
        rationale="Clear visible golden retriever",
    )
    d = rec.to_dict()
    assert d["ground_truth_status"] == "supported"
    recovered = AnnotationRecord.from_dict(d)
    assert recovered.ground_truth_status == GroundTruthStatus.SUPPORTED
    assert recovered.rationale == rec.rationale


def test_m7_claim_record_wrapping_m6_evidence():
    """Test wrapping M6 ClaimLevelEvidenceRecord into M7ClaimRecord."""
    ev = ClaimLevelEvidenceRecord(
        claim_id="claim_test_1",
        image_id="coco_test_1",
        object_category="bicycle",
        text_span="bicycle",
        caption="A bicycle parked next to a tree.",
        image_hash="hash123",
        split="train",
        detector_score=0.85,
        detector_available=True,
        clip_score=0.32,
        similarity_available=True,
        is_synthetic=True,
    )
    m7_rec = M7ClaimRecord.from_m6_evidence(ev, assigned_split=SplitName.VALIDATION)
    assert m7_rec.claim_id == "claim_test_1"
    assert m7_rec.split == "validation"
    assert m7_rec.evidence.detector_score == 0.85

    d = m7_rec.to_dict()
    recovered = M7ClaimRecord.from_dict(d)
    assert recovered.split == "validation"
    assert recovered.evidence.clip_score == 0.32
