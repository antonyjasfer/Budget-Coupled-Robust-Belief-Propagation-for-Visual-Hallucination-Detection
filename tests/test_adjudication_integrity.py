"""
Unit tests for adjudication integrity and resolution of disagreements.
"""

import pytest
from src.data.schemas import GroundTruthStatus
from src.annotation.schemas import (
    AnnotationRecord,
    AdjudicationRecord,
    M7ClaimRecord,
    FinalGroundTruthRecord,
)
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.agreement import merge_and_adjudicate_annotations


def test_adjudication_resolves_disagreement():
    """
    Verify that an adjudication record properly resolves an A/B disagreement
    and produces a valid final ground-truth record.
    """
    ann_a = AnnotationRecord(
        claim_id="c_disputed",
        image_id="img_100",
        annotator_id="ANNOTATOR_A",
        ground_truth_status=GroundTruthStatus.SUPPORTED,
    )
    ann_b = AnnotationRecord(
        claim_id="c_disputed",
        image_id="img_100",
        annotator_id="ANNOTATOR_B",
        ground_truth_status=GroundTruthStatus.HALLUCINATED,
    )
    adj = AdjudicationRecord(
        claim_id="c_disputed",
        image_id="img_100",
        adjudicator_id="ADJUDICATOR_1",
        adjudicated_status=GroundTruthStatus.HALLUCINATED,
        notes="High-resolution crop shows object is completely absent.",
    )

    ev = ClaimLevelEvidenceRecord(
        claim_id="c_disputed",
        image_id="img_100",
        object_category="dog",
        caption="A dog in the park",
        image_hash="hash123",
        split="test",
    )
    claim_rec = M7ClaimRecord(
        claim_id="c_disputed",
        image_id="img_100",
        object_category="dog",
        text_span="dog",
        caption="A dog in the park",
        image_hash="hash123",
        split="test",
        evidence=ev,
    )

    final_rec = merge_and_adjudicate_annotations(
        claim_record=claim_rec,
        annotation_a=ann_a,
        annotation_b=ann_b,
        adjudication=adj,
    )

    assert final_rec.has_disagreement is True
    assert final_rec.final_ground_truth == GroundTruthStatus.HALLUCINATED
    assert final_rec.adjudication is not None
    assert final_rec.adjudication.adjudicator_id == "ADJUDICATOR_1"

    d = final_rec.to_dict()
    assert d["final_ground_truth"] == "hallucinated"
    assert d["adjudication"]["adjudicated_status"] == "hallucinated"

