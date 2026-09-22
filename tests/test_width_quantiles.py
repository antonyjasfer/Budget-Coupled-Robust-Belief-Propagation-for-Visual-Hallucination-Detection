"""Tests for Robust Width Quantile Analysis.

Validates:
- Quantile partition of claims by robust width W_global
- Small-sample guard returning NOT EVALUABLE
- Metrics per quantile bin
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_width_quantiles,
)


def test_width_quantiles_evaluation():
    records = []
    # 16 claims with varying widths and error indicators
    for i in range(16):
        w = 0.05 + 0.05 * i  # widths from 0.05 to 0.80
        is_err = (i >= 8)    # errors in upper half
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i % 4}",
            claim_id=f"c_{i}",
            ground_truth=1 if is_err else 0,
            nominal_prediction=0,
            nominal_correct=not is_err,
            global_width=w,
            nominal_posterior=0.7 if is_err else 0.3,
            global_contains_threshold=(w > 0.4),
        ))

    res = evaluate_width_quantiles(records, n_quantiles=4, min_samples_per_bin=3)
    assert res["status"] == "EVALUATED"
    assert len(res["bins"]) == 4
    # Q1 and Q2 should have low error rate, Q3 and Q4 high error rate
    assert res["bins"][0]["error_rate"] == 0.0
    assert res["bins"][3]["error_rate"] == 1.0


def test_width_quantiles_small_sample_returns_not_evaluable():
    records = [
        ReliabilityClaimRecord(
            image_id="img_0", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True,
            global_width=0.2,
        ),
        ReliabilityClaimRecord(
            image_id="img_1", claim_id="c1", ground_truth=1, nominal_prediction=0, nominal_correct=False,
            global_width=0.4,
        ),
    ]
    res = evaluate_width_quantiles(records, n_quantiles=4, min_samples_per_bin=3)
    assert res["status"] == "NOT EVALUABLE"
    assert "minimum required" in res["reason"]
