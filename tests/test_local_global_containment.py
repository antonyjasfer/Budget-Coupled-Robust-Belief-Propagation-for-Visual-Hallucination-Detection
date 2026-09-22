"""Unit tests for Phase 9C local box vs global budget uncertainty containment.

Theoretical Guarantee:
Because the global-budget feasible perturbation set
    U_global = {delta : |delta_i| <= eps_i, sum |delta_i| <= B}
is a subset of the local box feasible set
    U_local = {delta : |delta_i| <= eps_i},
we MUST have:
    L_local <= L_global + 1e-4
    U_global <= U_local + 1e-4
and consequently:
    W_global <= W_local + 1e-4
across all nodes in any tree topology.
"""

import pytest
import numpy as np
from src.calibration.ablation_study import run_local_vs_global_budget_ablation
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
)
from src.calibration.ladder import BaselineLadderRunner


def create_mock_runner(epsilon: float = 0.3, budget: float = 0.4):
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
        coupling_lambda=0.35,
    )


def test_local_global_containment_invariants():
    """Verify L_local <= L_global, U_global <= U_local, and W_global <= W_local."""
    runner = create_mock_runner(epsilon=0.3, budget=0.35)
    # 3 images of sizes 1, 3, 4
    records = []
    for c in range(1):
        records.append({"claim_id": f"c_1_{c}", "image_id": "img1", "detector_score": 0.5, "clip_score": 0.4, "label": 0})
    for c in range(3):
        records.append({"claim_id": f"c_2_{c}", "image_id": "img2", "detector_score": 0.2 + 0.3 * c, "clip_score": 0.7 - 0.2 * c, "label": c % 2})
    for c in range(4):
        records.append({"claim_id": f"c_3_{c}", "image_id": "img3", "detector_score": 0.8 - 0.1 * c, "clip_score": 0.3 + 0.1 * c, "label": 1})

    res = run_local_vs_global_budget_ablation(runner=runner, records=records)

    assert res["num_claims_evaluated"] == 8
    assert res["containment_violations"] == 0, f"Found containment violations: {res['containment_violations']}"
    assert res["mean_width_reduction"] >= -1e-6
    assert res["mean_global_width"] <= res["mean_local_width"] + 1e-6
