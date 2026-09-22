"""Tests for Selective Prediction and Risk-Coverage Curves.

Validates:
- Sorting by lowest uncertainty to highest
- Deterministic tie-breaking
- Prefix evaluation: coverage, risk, selective accuracy, selective F1
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_discrete_risk_coverage_curve,
)


def test_risk_coverage_curve_perfect_ranking():
    # 5 claims: 4 correct (score 0.1, 0.2, 0.3, 0.4), 1 error (score 0.9)
    records = []
    for i in range(4):
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"claim_{i}",
            ground_truth=0,
            nominal_prediction=0,
            nominal_correct=True,
            nominal_entropy=0.1 * (i + 1),
        ))
    # Error claim
    records.append(ReliabilityClaimRecord(
        image_id="img_err",
        claim_id="claim_err",
        ground_truth=1,
        nominal_prediction=0,
        nominal_correct=False,
        nominal_entropy=0.9,
    ))

    rc = compute_discrete_risk_coverage_curve(records, lambda r: r.nominal_entropy)
    assert rc["status"] == "EVALUATED"
    assert len(rc["points"]) == 5

    # Check first 4 prefixes have risk = 0.0
    for k in range(4):
        assert rc["points"][k].risk == 0.0
        assert np.isclose(rc["points"][k].coverage, (k + 1) / 5)
        assert rc["points"][k].selective_accuracy == 1.0

    # 5th prefix has 1 error out of 5 -> risk = 0.2
    assert np.isclose(rc["points"][4].risk, 0.2)
    assert np.isclose(rc["points"][4].coverage, 1.0)
    assert np.isclose(rc["points"][4].selective_accuracy, 0.8)


def test_deterministic_tie_breaking():
    # Same score, tie-broken by claim_id alphabetically
    records = [
        ReliabilityClaimRecord(image_id="img_0", claim_id="claim_b", ground_truth=0, nominal_prediction=0, nominal_correct=True, nominal_entropy=0.5),
        ReliabilityClaimRecord(image_id="img_1", claim_id="claim_a", ground_truth=1, nominal_prediction=0, nominal_correct=False, nominal_entropy=0.5),
    ]
    rc = compute_discrete_risk_coverage_curve(records, lambda r: r.nominal_entropy, min_samples=2)
    # claim_a comes first -> error at k=1, risk = 1.0
    assert rc["points"][0].risk == 1.0
