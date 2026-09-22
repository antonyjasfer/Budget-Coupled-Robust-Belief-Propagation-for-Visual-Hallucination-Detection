"""Unit tests for the controlled baseline ladder (M0 to M6)."""

import numpy as np
import pytest

from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
)
from src.calibration.ladder import (
    BaselineLadderRunner,
    BaselineMethod,
    build_baseline_comparison_table,
    build_component_contribution_table,
)


def create_fixture_ladder_runner():
    """Create a calibrated ladder runner with trained model and calibrator."""
    model = LogisticEvidenceModel(feature_type="combined")
    train_records = [
        {"detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"detector_score": 0.2, "clip_score": 0.1, "label": 1},
        {"detector_score": 0.7, "clip_score": 0.25, "label": 0},
        {"detector_score": 0.15, "clip_score": 0.05, "label": 1},
    ]
    model.fit(train_records)

    p_raw = model.predict_proba(train_records)
    y_cal = np.array([r["label"] for r in train_records])
    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(p_raw, y_cal)

    runner = BaselineLadderRunner(
        logistic_model=model,
        prob_calibrator=calibrator,
        epsilon_val=0.25,
        budget_val=0.40,
        coupling_lambda=0.35,
    )
    return runner


def test_baseline_ladder_executes_all_methods():
    """Verify that all 7 methods (M0 to M6) run successfully on an image tree."""
    runner = create_fixture_ladder_runner()

    test_image_records = [
        {"claim_id": "c1", "image_id": "img1", "detector_score": 0.75, "clip_score": 0.28, "label": 0},
        {"claim_id": "c2", "image_id": "img1", "detector_score": 0.20, "clip_score": 0.08, "label": 1},
    ]
    edges = [(0, 1, 0.8)]

    results_by_method = runner.run_all_methods_on_records(
        records=test_image_records,
        tree_edges_by_image={"img1": edges},
    )

    assert len(results_by_method) == 7
    for m in BaselineMethod:
        res = results_by_method[m.value]
        assert len(res) == 2  # Both claims evaluated
        assert res[0].claim_id == "c1"
        assert res[1].claim_id == "c2"
        assert 0.0 <= res[0].nominal_posterior <= 1.0
        assert 0.0 <= res[1].nominal_posterior <= 1.0


def test_m3_isolated_unary_recovers_m2_probabilities():
    """Verify that M3 (theta = 0.5 * logit(p), J=0) reproduces calibrated M2 probability."""
    runner = create_fixture_ladder_runner()
    recs = [{"claim_id": "c1", "image_id": "img1", "detector_score": 0.6, "clip_score": 0.2, "label": 0}]

    res_m2 = runner.run_image(recs, method=BaselineMethod.M2_FUSION)[0]
    res_m3 = runner.run_image(recs, method=BaselineMethod.M3_UNARY_ISOLATED)[0]

    assert res_m2.nominal_posterior == pytest.approx(res_m3.nominal_posterior, abs=1e-5)


def test_baseline_summary_and_tables():
    """Verify generation of comparison and component contribution tables."""
    runner = create_fixture_ladder_runner()
    records = [
        {"claim_id": "c1", "image_id": "img1", "detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"claim_id": "c2", "image_id": "img1", "detector_score": 0.2, "clip_score": 0.1, "label": 1},
    ]
    results_by_method = runner.run_all_methods_on_records(records)
    summaries = [runner.evaluate_method_summary(r) for r in results_by_method.values()]

    table_comp = build_baseline_comparison_table(summaries, mode="DEVELOPMENT")
    assert "M0_detector_only" in table_comp
    assert "M6_budget_coupled_robust" in table_comp

    table_contrib = build_component_contribution_table({s.method_name: s for s in summaries})
    assert "M0 -> M2" in table_contrib
    assert "M5 -> M6" in table_contrib
