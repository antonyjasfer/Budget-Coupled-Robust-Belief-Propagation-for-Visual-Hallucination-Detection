"""Tests for Area Under Risk-Coverage Curve (AURC) and Excess AURC (E-AURC).

Validates:
- Standard discrete AURC formulation: (1/N) * sum_{k=1}^N r_k
- Oracle AURC* formulation
- Excess AURC = AURC - AURC* >= 0
- E-AURC == 0 for perfect ranking
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_discrete_risk_coverage_curve,
)


def test_aurc_perfect_ranking_excess_zero():
    # 5 claims: 4 correct (score 0.1-0.4), 1 error (score 0.9)
    records = []
    for i in range(4):
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"c_{i}",
            ground_truth=0,
            nominal_prediction=0,
            nominal_correct=True,
            nominal_entropy=0.1 * (i + 1),
        ))
    records.append(ReliabilityClaimRecord(
        image_id="img_err",
        claim_id="c_err",
        ground_truth=1,
        nominal_prediction=0,
        nominal_correct=False,
        nominal_entropy=0.9,
    ))

    rc = compute_discrete_risk_coverage_curve(records, lambda r: r.nominal_entropy)
    # Risks: [0, 0, 0, 0, 0.2] -> discrete AURC = (0 + 0 + 0 + 0 + 0.2) / 5 = 0.04
    assert np.isclose(rc["discrete_aurc"], 0.04)
    # Oracle risk is identical -> oracle AURC* = 0.04
    assert np.isclose(rc["oracle_aurc"], 0.04)
    # Excess AURC must be exactly 0.0
    assert np.isclose(rc["excess_aurc"], 0.0)


def test_aurc_worst_ranking_excess_positive():
    # 5 claims: 1 error (score 0.1), 4 correct (score 0.6-0.9) - worst possible ranking
    records = [
        ReliabilityClaimRecord(
            image_id="img_err",
            claim_id="c_err",
            ground_truth=1,
            nominal_prediction=0,
            nominal_correct=False,
            nominal_entropy=0.1,
        )
    ]
    for i in range(4):
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"c_{i}",
            ground_truth=0,
            nominal_prediction=0,
            nominal_correct=True,
            nominal_entropy=0.6 + 0.1 * i,
        ))

    rc = compute_discrete_risk_coverage_curve(records, lambda r: r.nominal_entropy)
    # Risks: k=1: 1/1=1.0, k=2: 1/2=0.5, k=3: 1/3=0.333, k=4: 1/4=0.25, k=5: 1/5=0.20
    # Discrete AURC is mean of [1.0, 0.5, 0.333, 0.25, 0.20]
    expected_aurc = np.mean([1.0, 0.5, 1.0 / 3.0, 0.25, 0.20])
    assert np.isclose(rc["discrete_aurc"], expected_aurc)
    # Oracle AURC* is 0.04
    assert np.isclose(rc["oracle_aurc"], 0.04)
    # Excess AURC must be strictly positive
    assert rc["excess_aurc"] > 0.3
