"""Unit tests for Phase 9C Graph x Uncertainty Factorial Matrix (2x3).

Conditions:
A: J=0, no uncertainty
B: J>0, no uncertainty
C: J=0, local uncertainty
D: J>0, local uncertainty
E: J=0, global uncertainty
F: J>0, global uncertainty
"""

import pytest
import numpy as np
from src.calibration.ablation_study import run_graph_uncertainty_factorial_ablation
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
)
from src.calibration.ladder import BaselineLadderRunner


def create_mock_runner():
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
        epsilon_val=0.25,
        budget_val=0.4,
        coupling_lambda=0.4,
    )


def test_factorial_matrix_all_conditions():
    """Verify all 6 factorial conditions A-F are evaluated with valid metrics."""
    runner = create_mock_runner()
    records = [
        {"claim_id": f"c_{i}", "image_id": f"img_{i // 2}", "detector_score": 0.2 + 0.1 * i, "clip_score": 0.8 - 0.1 * i, "label": i % 2, "split": "test"}
        for i in range(6)
    ]

    results = run_graph_uncertainty_factorial_ablation(runner=runner, records=records)

    assert len(results) == 6
    ids = [r.condition_id for r in results]
    assert ids == ["A", "B", "C", "D", "E", "F"]

    # Check that conditions A and B have no uncertainty (mean_width is None)
    assert results[0].mean_width is None
    assert results[1].mean_width is None

    # Check that conditions C, D, E, F have valid numeric widths
    for i in range(2, 6):
        assert results[i].mean_width is not None
        assert results[i].mean_width >= 0.0

    # Under identical unaries, condition A has J=0 and condition B has J>0
    assert results[0].has_graph is False
    assert results[1].has_graph is True
