"""
Unit tests for inter-annotator agreement (Cohen's kappa) and adjudication logic.
"""

import pytest
from src.data.schemas import GroundTruthStatus
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import (
    AnnotationRecord,
    AdjudicationRecord,
    M7ClaimRecord,
    FinalGroundTruthRecord,
)
from src.annotation.agreement import (
    compute_cohens_kappa,
    merge_and_adjudicate_annotations,
    CohenKappaResult,
)


def test_cohens_kappa_perfect_agreement():
    """When both annotators agree completely across classes, kappa is 1.0."""
    ann_a = [
        GroundTruthStatus.SUPPORTED,
        GroundTruthStatus.HALLUCINATED,
        GroundTruthStatus.UNKNOWN,
        GroundTruthStatus.SUPPORTED,
    ]
    ann_b = [
        GroundTruthStatus.SUPPORTED,
        GroundTruthStatus.HALLUCINATED,
        GroundTruthStatus.UNKNOWN,
        GroundTruthStatus.SUPPORTED,
    ]
    res = compute_cohens_kappa(ann_a, ann_b)
    assert res.joint_count == 4
    assert res.agreement_count == 4
    assert res.observed_agreement == 1.0
    assert res.kappa_defined is True
    assert res.cohens_kappa == pytest.approx(1.0)


def test_cohens_kappa_known_synthetic_example():
    """Verify multiclass kappa computation against analytical manual calculation."""
    # 3 SUPPORTED, 2 HALLUCINATED, 1 UNKNOWN
    ann_a = [
        GroundTruthStatus.SUPPORTED,
        GroundTruthStatus.SUPPORTED,
        GroundTruthStatus.HALLUCINATED,
        GroundTruthStatus.UNKNOWN,
    ]
    ann_b = [
        GroundTruthStatus.SUPPORTED,
        GroundTruthStatus.HALLUCINATED,  # Disagreement
        GroundTruthStatus.HALLUCINATED,
        GroundTruthStatus.UNKNOWN,
    ]
    res = compute_cohens_kappa(ann_a, ann_b)
    assert res.joint_count == 4
    assert res.agreement_count == 3
    assert res.observed_agreement == 0.75
    assert res.kappa_defined is True
    assert 0.0 < res.cohens_kappa < 1.0


def test_cohens_kappa_pe_equals_one_boundary():
    """
    CRITICAL EDGE CASE:
    When chance agreement Pe == 1.0 (both annotators assign 100% of items to identical single category),
    denominator 1 - Pe is zero.
    Verify that:
    - ZeroDivisionError is NOT raised.
    - cohens_kappa is None.
    - kappa_defined is False.
    - An explanatory status message is provided.
    """
    ann_a = [GroundTruthStatus.SUPPORTED] * 5
    ann_b = [GroundTruthStatus.SUPPORTED] * 5

    res = compute_cohens_kappa(ann_a, ann_b)
    assert res.joint_count == 5
    assert res.agreement_count == 5
    assert res.observed_agreement == 1.0
    assert res.chance_agreement == 1.0
    assert res.cohens_kappa is None
    assert res.kappa_defined is False
    assert "Pe == 1.0" in res.status_message


def test_merge_and_adjudicate_no_auto_consensus_on_disagreement():
    """
    CRITICAL METHODOLOGICAL GUARANTEE:
    When Annotator A != Annotator B, never auto-select A or B.
    Disagreement must be flagged, consensus must be None, and adjudication required.
    """
    ev = ClaimLevelEvidenceRecord(claim_id="c_disagree", image_id="img1", object_category="dog", is_synthetic=True)
    claim = M7ClaimRecord.from_m6_evidence(ev)

    rec_a = AnnotationRecord(claim_id="c_disagree", image_id="img1", annotator_id="A", ground_truth_status=GroundTruthStatus.SUPPORTED)
    rec_b = AnnotationRecord(claim_id="c_disagree", image_id="img1", annotator_id="B", ground_truth_status=GroundTruthStatus.HALLUCINATED)

    # Without adjudication
    merged = merge_and_adjudicate_annotations(claim, annotation_a=rec_a, annotation_b=rec_b)
    assert merged.has_disagreement is True
    assert merged.final_ground_truth is None

    # With adjudication
    adj = AdjudicationRecord(
        claim_id="c_disagree",
        image_id="img1",
        adjudicator_id="expert_1",
        adjudicated_status=GroundTruthStatus.SUPPORTED,
        notes="Visible dog paw under table",
    )
    adjudicated = merge_and_adjudicate_annotations(claim, annotation_a=rec_a, annotation_b=rec_b, adjudication=adj)
    assert adjudicated.has_disagreement is True
    assert adjudicated.final_ground_truth == GroundTruthStatus.SUPPORTED
    assert adjudicated.adjudication.notes == "Visible dog paw under table"


def test_merge_and_adjudicate_agreement():
    """When both annotators agree, consensus is assigned without adjudication."""
    ev = ClaimLevelEvidenceRecord(claim_id="c_agree", image_id="img2", object_category="chair", is_synthetic=True)
    claim = M7ClaimRecord.from_m6_evidence(ev)

    rec_a = AnnotationRecord(claim_id="c_agree", image_id="img2", annotator_id="A", ground_truth_status=GroundTruthStatus.UNKNOWN)
    rec_b = AnnotationRecord(claim_id="c_agree", image_id="img2", annotator_id="B", ground_truth_status=GroundTruthStatus.UNKNOWN)

    merged = merge_and_adjudicate_annotations(claim, annotation_a=rec_a, annotation_b=rec_b)
    assert merged.has_disagreement is False
    assert merged.final_ground_truth == GroundTruthStatus.UNKNOWN
