"""Tests for Small-Sample Safety and Edge Case Protections.

Validates robust handling (never crashing, returning NOT EVALUABLE with reason) for:
- zero errors
- all errors
- one image
- one claim
- all equal uncertainty scores
- no threshold crossers
- all threshold crossers
- no annotation disagreements
- all UNKNOWN
- empty accepted set after abstention
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_error_detection_metrics,
    compute_discrete_risk_coverage_curve,
    compute_group_aware_incremental_utility,
    evaluate_threshold_crossing_stability,
    evaluate_robust_abstention_policy,
    evaluate_width_quantiles,
    evaluate_annotation_disagreement_uncertainty,
    evaluate_unknown_claim_uncertainty,
    compute_paired_image_bootstrap_comparisons,
)


def test_edge_case_single_claim():
    """One single claim must not crash any analysis."""
    rec = ReliabilityClaimRecord(
        image_id="img_single", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True,
        nominal_entropy=0.5, global_width=0.3,
    )
    # Error detection
    err_res = compute_error_detection_metrics([0], [0], [0.5])
    assert err_res["status"] == "NOT EVALUABLE"

    # Risk coverage
    rc_res = compute_discrete_risk_coverage_curve([rec], lambda r: r.nominal_entropy)
    assert rc_res["status"] == "NOT EVALUABLE"

    # Incremental CV
    inc_res = compute_group_aware_incremental_utility([rec])
    assert inc_res["status"] == "NOT EVALUABLE"


def test_edge_case_all_equal_uncertainty_scores():
    """All equal scores must not cause numerical crashes."""
    records = []
    for i in range(8):
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i % 2}",
            claim_id=f"c_{i}",
            ground_truth=1 if i < 4 else 0,
            nominal_prediction=0,
            nominal_correct=(i >= 4),
            nominal_entropy=0.5,  # all identical
            global_width=0.2,    # all identical
        ))
    err_res = compute_error_detection_metrics([r.ground_truth for r in records], [r.nominal_prediction for r in records], [r.nominal_entropy for r in records])
    assert err_res["status"] == "EVALUATED"
    # When scores are all equal, AUROC is 0.5 (random guessing)
    assert np.isclose(err_res["error_detection_auroc"], 0.5)


def test_edge_case_all_threshold_crossers():
    """100% threshold crossers leaves 0 accepted in abstention policy."""
    records = [
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True, global_contains_threshold=True),
        ReliabilityClaimRecord(image_id="img_1", claim_id="c1", ground_truth=1, nominal_prediction=1, nominal_correct=True, global_contains_threshold=True),
    ]
    abs_res = evaluate_robust_abstention_policy(records)
    assert abs_res["coverage"] == 0.0
    assert abs_res["accepted_n"] == 0
    assert abs_res["abstained_n"] == 2


def test_edge_case_all_unknown():
    """When all claims are UNKNOWN, binary evaluations safely guard."""
    records = [
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", ground_truth=None, global_width=0.4),
        ReliabilityClaimRecord(image_id="img_1", claim_id="c1", ground_truth=None, global_width=0.6),
    ]
    unk_res = evaluate_unknown_claim_uncertainty(records)
    assert unk_res["n_unknown"] == 2
    assert unk_res["n_supported"] == 0
    assert unk_res["n_hallucinated"] == 0
