"""
Unit tests for independent human double annotation and annotator privacy (Corrections 9, 10).
"""

import pytest
from src.annotation.schemas import (
    AnnotationRecord,
    parse_ground_truth_status,
    GroundTruthStatus,
)


def test_independent_double_annotation_pseudonymous_ids():
    """
    Verify double annotations use pseudonymous IDs (ANNOTATOR_A, ANNOTATOR_B)
    with zero personally identifiable information (PII).
    """
    rec_a = AnnotationRecord(
        claim_id="claim_001",
        image_id="coco_001",
        annotator_id="ANNOTATOR_A",
        ground_truth_status=GroundTruthStatus.SUPPORTED,
    )
    rec_b = AnnotationRecord(
        claim_id="claim_001",
        image_id="coco_001",
        annotator_id="ANNOTATOR_B",
        ground_truth_status=GroundTruthStatus.HALLUCINATED,
    )

    d_a = rec_a.to_dict()
    d_b = rec_b.to_dict()

    assert d_a["annotator_id"] == "ANNOTATOR_A"
    assert d_b["annotator_id"] == "ANNOTATOR_B"
    assert d_a["ground_truth_status"] == "supported"
    assert d_b["ground_truth_status"] == "hallucinated"

    # Zero PII in metadata
    assert "email" not in d_a.get("metadata", {})
    assert "name" not in d_a.get("metadata", {})


def test_annotation_record_from_dict_safe_parsing():
    """Verify safe parsing of raw annotator input strings."""
    raw = {
        "claim_id": "c_123",
        "image_id": "img_456",
        "annotator_id": "ANNOTATOR_A",
        "ground_truth_status": "  HaLLuCiNaTeD  ",
    }
    rec = AnnotationRecord.from_dict(raw)
    assert rec.ground_truth_status == GroundTruthStatus.HALLUCINATED
