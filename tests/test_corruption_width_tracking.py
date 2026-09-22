"""Tests for Repeated-Measures Corruption Width Tracking.

Validates:
- Within-claim tracking of Delta W_{i,c} = W_{i,c} - W_{i,clean}
- Repeated-measures structure (same image, same claim across severities)
- Spearman severity vs width correlation
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_corruption_width_tracking,
)


def test_corruption_width_tracking():
    records = []
    # 3 claims, each evaluated across clean, light, medium, heavy
    for i in range(3):
        severities = [("clean", 0.20), ("light", 0.30), ("medium", 0.45), ("heavy", 0.65)]
        for sev_name, base_w in severities:
            records.append(ReliabilityClaimRecord(
                image_id=f"img_{i}",
                claim_id=f"c_{i}",
                corruption_type="gaussian_blur",
                corruption_severity=sev_name,
                global_width=base_w + 0.02 * i,
            ))

    res = evaluate_corruption_width_tracking(records, min_claims=3)
    assert res["status"] == "EVALUATED"
    assert res["n_paired_claims"] == 3
    # delta for light = 0.30 - 0.20 = 0.10
    assert np.isclose(res["mean_delta_light"], 0.10)
    # delta for medium = 0.45 - 0.20 = 0.25
    assert np.isclose(res["mean_delta_medium"], 0.25)
    # delta for heavy = 0.65 - 0.20 = 0.45
    assert np.isclose(res["mean_delta_heavy"], 0.45)
    # Spearman rank correlation should be exactly 1.0 (monotonic increase)
    assert np.isclose(res["mean_spearman_severity_width"], 1.0)


def test_corruption_insufficient_claims_returns_not_evaluable():
    records = [
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", corruption_severity="clean", global_width=0.2),
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", corruption_severity="heavy", global_width=0.4),
    ]
    res = evaluate_corruption_width_tracking(records, min_claims=3)
    assert res["status"] == "NOT EVALUABLE"
