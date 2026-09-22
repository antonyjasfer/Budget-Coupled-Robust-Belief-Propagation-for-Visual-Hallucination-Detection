"""Unit tests for Phase 9C budget saturation equivalence and B=0 collapse.

Verifies:
1. Zero budget (B = 0):
   Perturbation set collapses to {0}, yielding:
       L_i = P_nominal_i = U_i, width = 0.
2. Saturated budget (B >= sum_i eps_i):
   Budget constraint is non-binding, so:
       L_global ~= L_local
       U_global ~= U_local
       W_global ~= W_local.
"""

import pytest
import numpy as np
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
)
from src.calibration.ladder import BaselineLadderRunner, BaselineMethod


def create_fitted_runner(epsilon: float, budget: float):
    model = LogisticEvidenceModel(feature_type="combined")
    recs = [
        {"detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"detector_score": 0.2, "clip_score": 0.1, "label": 1},
        {"detector_score": 0.7, "clip_score": 0.25, "label": 0},
        {"detector_score": 0.15, "clip_score": 0.05, "label": 1},
    ]
    model.fit(recs)
    p_raw = model.predict_proba(recs)
    y_cal = np.array([r["label"] for r in recs])
    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(p_raw, y_cal)
    return BaselineLadderRunner(
        logistic_model=model,
        prob_calibrator=calibrator,
        epsilon_val=epsilon,
        budget_val=budget,
        coupling_lambda=0.4,
    )


def test_b_zero_collapses_uncertainty():
    """When B = 0, robust interval collapses to nominal marginal: L = P_nom = U."""
    runner = create_fitted_runner(epsilon=0.3, budget=0.0)
    records = [
        {"claim_id": "c1", "image_id": "img1", "detector_score": 0.7, "clip_score": 0.2, "label": 1},
        {"claim_id": "c2", "image_id": "img1", "detector_score": 0.3, "clip_score": 0.8, "label": 0},
    ]

    res = runner.run_image(records, method=BaselineMethod.M6_GLOBAL_ROBUST)
    for r in res:
        assert r.robust_lower is not None
        assert r.robust_upper is not None
        # With B=0, width is 0 up to discrete DP step size
        assert pytest.approx(r.robust_lower, abs=0.05) == r.nominal_posterior
        assert pytest.approx(r.robust_upper, abs=0.05) == r.nominal_posterior
        assert pytest.approx(r.robust_width, abs=0.05) == 0.0


def test_budget_saturation_matches_local_box():
    """When B >= sum_i eps_i, global budget results match local box."""
    n = 3
    eps = 0.2
    saturated_b = n * eps + 0.1  # Exceeds total sum of epsilons
    runner = create_fitted_runner(epsilon=eps, budget=saturated_b)
    records = [
        {"claim_id": f"c_{i}", "image_id": "img1", "detector_score": 0.2 + 0.3 * i, "clip_score": 0.6 - 0.2 * i, "label": i % 2}
        for i in range(n)
    ]

    res_global = runner.run_image(records, method=BaselineMethod.M6_GLOBAL_ROBUST)
    res_local = runner.run_image(records, method=BaselineMethod.M5_LOCAL_ROBUST)

    for rg, rl in zip(res_global, res_local):
        # Global bounds should match local bounds within discrete DP tolerance
        assert pytest.approx(rg.robust_lower, abs=0.02) == rl.robust_lower
        assert pytest.approx(rg.robust_upper, abs=0.02) == rl.robust_upper
        assert pytest.approx(rg.robust_width, abs=0.02) == rl.robust_width
