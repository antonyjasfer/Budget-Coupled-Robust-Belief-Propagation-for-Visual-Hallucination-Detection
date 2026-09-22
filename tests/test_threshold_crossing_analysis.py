"""Tests for Threshold-Crossing Stability Analysis.

Validates:
- Crossers (contains_threshold: L <= tau <= U) vs non-crossers
- Error rate comparison and relative risk
- Image-level bootstrap confidence intervals
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_threshold_crossing_stability,
)


def test_threshold_crossing_stability():
    # 4 images, 8 claims:
    # 4 crossers (contains_threshold=True): 3 errors, 1 correct -> error rate = 0.75
    # 4 non-crossers (contains_threshold=False): 1 error, 3 correct -> error rate = 0.25
    records = []
    for i in range(4):
        # Crosser
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"cr_{i}",
            ground_truth=1 if i < 3 else 0,
            nominal_prediction=0,
            nominal_correct=(i >= 3),
            global_contains_threshold=True,
            global_lower=0.4,
            global_upper=0.6,
        ))
        # Non-crosser
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"nocr_{i}",
            ground_truth=1 if i == 0 else 0,
            nominal_prediction=0,
            nominal_correct=(i != 0),
            global_contains_threshold=False,
            global_lower=0.1,
            global_upper=0.3,
        ))

    res = evaluate_threshold_crossing_stability(records, tau=0.5, n_bootstraps=50)
    assert res["status"] == "EVALUATED"
    assert res["n_claims"] == 8
    assert res["n_crossers"] == 4
    assert res["n_non_crossers"] == 4
    assert np.isclose(res["error_rate_crossers"], 0.75)
    assert np.isclose(res["error_rate_non_crossers"], 0.25)
    assert np.isclose(res["relative_risk"], 3.0)
    assert np.isclose(res["risk_difference"], 0.50)
    assert res["risk_diff_bootstrap_ci"] is not None


def test_threshold_crossing_no_crossers():
    records = [
        ReliabilityClaimRecord(
            image_id="img_0", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True,
            global_contains_threshold=False, global_lower=0.1, global_upper=0.3,
        ),
        ReliabilityClaimRecord(
            image_id="img_1", claim_id="c1", ground_truth=1, nominal_prediction=1, nominal_correct=True,
            global_contains_threshold=False, global_lower=0.7, global_upper=0.9,
        ),
    ]
    res = evaluate_threshold_crossing_stability(records, tau=0.5)
    assert res["status"] == "EVALUATED"
    assert res["n_crossers"] == 0
    assert res["n_non_crossers"] == 2
    assert res["error_rate_crossers"] == 0.0
    assert res["error_rate_non_crossers"] == 0.0
