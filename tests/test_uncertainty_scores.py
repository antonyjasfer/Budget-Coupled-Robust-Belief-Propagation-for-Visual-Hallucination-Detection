"""Tests for Nominal and Robust Uncertainty Scores.

Validates:
- U1 margin uncertainty: 1 - |2p - 1|
- U2 binary entropy: -p log2 p - (1-p) log2(1-p)
- U3 threshold distance uncertainty: -|p - tau| and normalized version
- U4 detector uncertainty
- U5 logistic uncertainty
- Robust uncertainty scores: W_global, W_local, strict crossing, contains_threshold, robust_margin
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    compute_margin_uncertainty,
    compute_binary_entropy,
    compute_threshold_distance_uncertainty,
    compute_normalized_threshold_uncertainty,
    compute_detector_uncertainty,
    compute_logistic_uncertainty,
    compute_robust_uncertainty_scores,
    classify_robust_decision,
    RobustDecisionCategory,
)


def test_margin_uncertainty():
    # p=0.5 -> max uncertainty 1.0
    assert np.isclose(compute_margin_uncertainty(0.5), 1.0)
    # p=0.0 -> min uncertainty 0.0
    assert np.isclose(compute_margin_uncertainty(0.0), 0.0)
    # p=1.0 -> min uncertainty 0.0
    assert np.isclose(compute_margin_uncertainty(1.0), 0.0)
    # p=0.8 and p=0.2 have identical margin uncertainty
    assert np.isclose(compute_margin_uncertainty(0.8), compute_margin_uncertainty(0.2))
    assert np.isclose(compute_margin_uncertainty(0.8), 0.4)


def test_binary_entropy():
    # p=0.5 -> exactly 1.0 bit
    assert np.isclose(compute_binary_entropy(0.5), 1.0)
    # p near 0 or 1 -> near 0
    assert compute_binary_entropy(0.001) < 0.02
    assert compute_binary_entropy(0.999) < 0.02
    # symmetric
    assert np.isclose(compute_binary_entropy(0.3), compute_binary_entropy(0.7))


def test_threshold_distance_uncertainty():
    tau = 0.5
    # p=0.5 -> distance is 0, score is 0.0 (maximum)
    assert np.isclose(compute_threshold_distance_uncertainty(0.5, tau=tau), 0.0)
    # p=0.1 -> distance 0.4, score is -0.4
    assert np.isclose(compute_threshold_distance_uncertainty(0.1, tau=tau), -0.4)
    # Closer to tau must be strictly greater (less negative)
    assert compute_threshold_distance_uncertainty(0.48, tau=tau) > compute_threshold_distance_uncertainty(0.2, tau=tau)

    # Normalized version
    assert np.isclose(compute_normalized_threshold_uncertainty(0.5, tau=tau), 1.0)
    assert np.isclose(compute_normalized_threshold_uncertainty(0.0, tau=tau), 0.0)
    assert np.isclose(compute_normalized_threshold_uncertainty(1.0, tau=tau), 0.0)


def test_robust_uncertainty_scores_strict_vs_contains():
    tau = 0.5
    # Interval strictly containing tau: [0.3, 0.7]
    res1 = compute_robust_uncertainty_scores(0.3, 0.7, tau=tau)
    assert np.isclose(res1["width"], 0.4)
    assert res1["strict_crossing"] is True
    assert res1["contains_threshold"] is True
    assert res1["robust_margin"] == 0.0
    assert res1["robust_decision"] == RobustDecisionCategory.EVIDENCE_SENSITIVE.value

    # Interval with boundary at tau: [0.5, 0.8]
    res2 = compute_robust_uncertainty_scores(0.5, 0.8, tau=tau)
    assert res2["strict_crossing"] is False
    assert res2["contains_threshold"] is True
    assert res2["robust_margin"] == 0.0
    assert res2["robust_decision"] == RobustDecisionCategory.EVIDENCE_SENSITIVE.value

    # Interval entirely above tau: [0.6, 0.9] -> ROBUST_HALLUCINATED
    res3 = compute_robust_uncertainty_scores(0.6, 0.9, tau=tau)
    assert res3["strict_crossing"] is False
    assert res3["contains_threshold"] is False
    assert np.isclose(res3["robust_margin"], 0.1)
    assert np.isclose(res3["u_robust_margin"], -0.1)
    assert res3["robust_decision"] == RobustDecisionCategory.ROBUST_HALLUCINATED.value

    # Interval entirely below tau: [0.1, 0.4] -> ROBUST_SUPPORTED
    res4 = compute_robust_uncertainty_scores(0.1, 0.4, tau=tau)
    assert res4["strict_crossing"] is False
    assert res4["contains_threshold"] is False
    assert np.isclose(res4["robust_margin"], 0.1)
    assert res4["robust_decision"] == RobustDecisionCategory.ROBUST_SUPPORTED.value
