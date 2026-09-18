"""
Unit tests for RawEvidenceRecord and validation constraints.
"""

import pytest

from src.evidence.schemas import RawEvidenceRecord, EVIDENCE_SCHEMA_VERSION


def test_raw_evidence_valid_record():
    """Verify creation and serialization of valid RawEvidenceRecord."""
    rec = RawEvidenceRecord(
        evidence_id="ev_001",
        image_id="coco_101",
        claim_id="claim_001_dog",
        object_category="dog",
        detector_score=0.85,
        detector_available=True,
        similarity_score=0.32,
        similarity_available=True,
        provider_id="fixture_provider_v1",
        view_id="original",
        is_synthetic=True,
    )

    d = rec.to_dict()
    assert d["detector_score"] == 0.85
    assert d["similarity_score"] == 0.32
    assert d["view_id"] == "original"
    assert d["is_synthetic"] is True

    reconstructed = RawEvidenceRecord.from_dict(d)
    assert reconstructed.evidence_id == "ev_001"
    assert reconstructed.detector_score == 0.85


def test_raw_evidence_unavailable_scores():
    """Verify handling when detector or similarity is unavailable."""
    rec = RawEvidenceRecord(
        evidence_id="ev_002",
        image_id="coco_101",
        claim_id="claim_001_cat",
        object_category="cat",
        detector_score=None,
        detector_available=False,
        similarity_score=0.15,
        similarity_available=True,
    )
    assert rec.detector_score is None
    assert not rec.detector_available


def test_raw_evidence_rejects_out_of_bounds_and_nan():
    """Verify strict validation against invalid ranges and NaN/inf."""
    # Detector score > 1.0
    with pytest.raises(ValueError, match="detector_score must be in"):
        RawEvidenceRecord(
            evidence_id="ev_bad",
            image_id="img_1",
            claim_id="c_1",
            object_category="dog",
            detector_score=1.5,
            detector_available=True,
        )

    # Similarity score < -1.0
    with pytest.raises(ValueError, match="similarity_score must be in"):
        RawEvidenceRecord(
            evidence_id="ev_bad",
            image_id="img_1",
            claim_id="c_1",
            object_category="dog",
            similarity_score=-1.2,
            similarity_available=True,
        )

    # NaN detector score
    with pytest.raises(ValueError, match="detector_score must be finite"):
        RawEvidenceRecord(
            evidence_id="ev_bad",
            image_id="img_1",
            claim_id="c_1",
            object_category="dog",
            detector_score=float("nan"),
            detector_available=True,
        )

    # Score provided when available=False
    with pytest.raises(ValueError, match="detector_score must be None when detector_available=False"):
        RawEvidenceRecord(
            evidence_id="ev_bad",
            image_id="img_1",
            claim_id="c_1",
            object_category="dog",
            detector_score=0.5,
            detector_available=False,
        )
