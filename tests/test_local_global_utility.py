"""Tests for Local vs Global Interval Utility Comparison.

Validates:
- Comparative evaluation of W_local vs W_global on error detection and selective prediction
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compare_local_vs_global_utility,
)


def test_local_vs_global_utility_comparison():
    records = []
    # 6 claims with both local and global widths
    for i in range(6):
        is_err = (i >= 4)
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i % 2}",
            claim_id=f"c_{i}",
            ground_truth=1 if is_err else 0,
            nominal_prediction=0,
            nominal_correct=not is_err,
            global_width=0.45 if is_err else 0.15,
            local_width=0.60 if is_err else 0.30,
        ))

    res = compare_local_vs_global_utility(records)
    assert res["status"] == "EVALUATED"
    assert res["n_claims"] == 6
    # Global width should be narrower than local width
    assert res["mean_width_global"] < res["mean_width_local"]
    assert res["global_error_auroc"] is not None
    assert res["local_error_auroc"] is not None
    assert res["global_discrete_aurc"] is not None
