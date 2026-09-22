"""Tests for Budget Ratio Utility Sweep.

Validates:
- Evaluating utility metrics across budget ratio rho = B / sum_i epsilon_i
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_budget_ratio_utility_sweep,
)


def test_budget_ratio_utility_sweep():
    records_by_rho = {}
    rhos = [0.0, 0.5, 1.0]

    for rho in rhos:
        recs = []
        for i in range(6):
            is_err = (i >= 4)
            recs.append(ReliabilityClaimRecord(
                image_id=f"img_{i % 2}",
                claim_id=f"c_{i}",
                ground_truth=1 if is_err else 0,
                nominal_prediction=0,
                nominal_correct=not is_err,
                global_width=float(rho * 0.4 + 0.1),
                global_contains_threshold=(rho > 0.4 and is_err),
            ))
        records_by_rho[rho] = recs

    res = evaluate_budget_ratio_utility_sweep(records_by_rho)
    assert len(res) == 3
    assert 0.0 in res
    assert 0.5 in res
    assert 1.0 in res

    # Mean width should increase with rho
    assert res[0.0]["mean_width"] < res[0.5]["mean_width"] < res[1.0]["mean_width"]
    assert res[1.0]["threshold_crossing_rate"] > res[0.0]["threshold_crossing_rate"]
