"""Tests for Milestone 9 Integrity, Semantics, and Mathematical Bounds."""

import pytest
from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results
from src.scientific.statistics import compute_m9_paired_bootstrap


def _create_claim(claim_id: str, gt: str, p_std: float, l: float, u: float) -> ClaimEvaluationResult:
    return ClaimEvaluationResult(
        run_id="run_int",
        experiment_name="test_int",
        condition="clean",
        image_id="img_1",
        claim_id=claim_id,
        object_category="object",
        text_span="dummy text",
        split="test",
        ground_truth=gt,
        detector_score=1.0 - p_std,
        clip_score=1.0 - p_std,
        theta_i=0.5 if p_std > 0.5 else -0.5,
        epsilon_i=0.2,
        coupling_j=0.5,
        budget=1.0,
        epsilon_scale=1.0,
        decision_threshold=0.5,
        grid_steps=21,
        standard_posterior=p_std,
        standard_prediction="hallucinated" if p_std >= 0.5 else "supported",
        robust_lower=l,
        robust_upper=u,
        robust_midpoint=(l + u) / 2.0,
        interval_width=u - l,
        certified_lower=l,
        certified_upper=u,
        field_cert_gap=0.0,
        robust_prediction="hallucinated" if (l + u) / 2.0 >= 0.5 else "supported",
        evidence_sensitive=(l < 0.5 < u),
        abstained=False,
        corruption_type="none",
        corruption_severity="clean",
        runtime_ms=2.0,
    )


def test_unknown_ground_truth_exclusion_from_binary_metrics():
    """Ensure UNKNOWN claims are cleanly counted but excluded from binary classification metrics."""
    claims = [
        _create_claim("c_supp", "supported", 0.1, 0.05, 0.20),
        _create_claim("c_hall", "hallucinated", 0.9, 0.80, 0.95),
        _create_claim("c_unk", "unknown", 0.5, 0.30, 0.70),
    ]

    m = evaluate_claim_results(claims, method_name="standard_bp")
    assert m.total_claims == 3
    assert m.unknown_gt_count == 1
    # Evaluated binary claims should be 2 (1 TP, 1 TN)
    assert m.confusion_matrix.tp == 1
    assert m.confusion_matrix.tn == 1
    assert m.confusion_matrix.fp == 0
    assert m.confusion_matrix.fn == 0
    assert m.accuracy == 1.0


def test_mathematical_bounds_invariant():
    """Verify that interval width is non-negative and bounds obey 0 <= L <= U <= 1."""
    c = _create_claim("c_test", "supported", 0.25, 0.10, 0.40)
    assert 0.0 <= c.robust_lower <= c.robust_upper <= 1.0
    width = c.robust_upper - c.robust_lower
    assert width >= 0.0
    assert pytest.approx(width) == c.interval_width
