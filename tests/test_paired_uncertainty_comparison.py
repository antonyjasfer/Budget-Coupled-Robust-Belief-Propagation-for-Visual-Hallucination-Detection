"""Tests for Paired Uncertainty Comparison.

Validates:
- Paired comparison of uncertainty metrics on identical claims
- Bootstrap difference distributions: Delta AUROC, Delta AURC
- No-effect interval detection
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_paired_image_bootstrap_comparisons,
)


def test_paired_comparison_computes_differences():
    records = []
    for img_idx in range(6):
        for c_idx in range(2):
            is_err = (img_idx % 2 == 1 and c_idx == 0)
            records.append(ReliabilityClaimRecord(
                image_id=f"img_{img_idx}",
                claim_id=f"claim_{img_idx}_{c_idx}",
                ground_truth=1 if is_err else 0,
                nominal_prediction=0,
                nominal_correct=not is_err,
                nominal_entropy=0.7 if is_err else 0.2,
                global_width=0.8 if is_err else 0.1,
                local_width=0.9 if is_err else 0.2,
            ))

    res = compute_paired_image_bootstrap_comparisons(records, n_bootstraps=50, min_images=4)
    assert res["status"] == "EVALUATED"
    assert res["delta_auroc_mean_ci"] is not None
    mean_diff, ci_low, ci_high = res["delta_auroc_mean_ci"]
    assert ci_low <= mean_diff <= ci_high
    assert res["n_images"] == 6
    assert res["n_claims"] == 12
