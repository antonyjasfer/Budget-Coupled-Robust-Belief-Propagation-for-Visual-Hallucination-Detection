"""Tests for Operating Points: Coverage at Fixed Risk & Risk at Fixed Coverage.

Validates:
- Coverage@Risk<=r: maximum observed achievable coverage without linear interpolation
- Risk@FixedCoverage: closest discrete prefix coverage >= target
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_discrete_risk_coverage_curve,
)


def test_fixed_risk_coverage_operating_points():
    # 10 claims:
    # 8 correct (score 0.1 .. 0.8)
    # 2 errors (score 0.85, 0.95)
    records = []
    for i in range(8):
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"c_{i}",
            ground_truth=0,
            nominal_prediction=0,
            nominal_correct=True,
            nominal_entropy=0.1 * (i + 1),
        ))
    for i in range(2):
        records.append(ReliabilityClaimRecord(
            image_id=f"img_err_{i}",
            claim_id=f"c_err_{i}",
            ground_truth=1,
            nominal_prediction=0,
            nominal_correct=False,
            nominal_entropy=0.85 + 0.1 * i,
        ))

    rc = compute_discrete_risk_coverage_curve(
        records,
        lambda r: r.nominal_entropy,
        risk_targets=(0.0, 0.05, 0.10, 0.20),
        coverage_targets=(0.50, 0.80, 0.90, 1.00),
    )

    cov_at_risk = rc["coverage_at_fixed_risk"]
    # At risk <= 0.0, prefixes 1..8 have risk 0.0 -> max coverage is 0.8
    assert np.isclose(cov_at_risk[0.0], 0.8)
    assert np.isclose(cov_at_risk[0.05], 0.8)
    # Prefix 9: 1 error in 9 claims -> risk = 1/9 = 0.111 (> 0.10, <= 0.20)
    # At risk <= 0.10, prefix 9 fails, so max coverage is still 0.8
    assert np.isclose(cov_at_risk[0.10], 0.8)
    # At risk <= 0.20, prefix 9 has risk 0.111 <= 0.20, prefix 10 has risk 2/10 = 0.20 <= 0.20 -> max coverage is 1.0
    assert np.isclose(cov_at_risk[0.20], 1.0)

    risk_at_cov = rc["risk_at_fixed_coverage"]
    # Coverage 0.50 -> prefix 5 (5 claims, 0 errors) -> risk = 0.0
    assert np.isclose(risk_at_cov[0.50], 0.0)
    # Coverage 0.80 -> prefix 8 (8 claims, 0 errors) -> risk = 0.0
    assert np.isclose(risk_at_cov[0.80], 0.0)
    # Coverage 0.90 -> prefix 9 (9 claims, 1 error) -> risk = 1/9
    assert np.isclose(risk_at_cov[0.90], 1.0 / 9.0)
    # Coverage 1.00 -> prefix 10 (10 claims, 2 errors) -> risk = 2/10 = 0.2
    assert np.isclose(risk_at_cov[1.00], 0.20)
