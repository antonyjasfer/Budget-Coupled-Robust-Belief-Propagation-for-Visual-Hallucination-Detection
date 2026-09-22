"""Unit tests for graph necessity test (M3 vs M4)."""

import pytest

from src.calibration.ablation_study import run_graph_necessity_test
from src.calibration.evidence_models import LogisticEvidenceModel, ProbabilityCalibrator
from src.calibration.ladder import BaselineLadderRunner


def test_graph_necessity_comparison():
    """Verify that graph necessity test correctly compares M3 (J=0) and M4 (J>0)."""
    model = LogisticEvidenceModel(feature_type="combined")
    train_records = [
        {"detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"detector_score": 0.2, "clip_score": 0.1, "label": 1},
        {"detector_score": 0.7, "clip_score": 0.25, "label": 0},
        {"detector_score": 0.3, "clip_score": 0.15, "label": 1},
    ]
    model.fit(train_records)
    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(model.predict_proba(train_records), [r["label"] for r in train_records])

    runner = BaselineLadderRunner(
        logistic_model=model,
        prob_calibrator=calibrator,
        coupling_lambda=0.40,
    )

    test_records = [
        {"claim_id": "c1", "image_id": "img1", "detector_score": 0.75, "clip_score": 0.25, "label": 0},
        {"claim_id": "c2", "image_id": "img1", "detector_score": 0.70, "clip_score": 0.22, "label": 0},
    ]
    edges = {"img1": [(0, 1, 0.9)]}

    out = run_graph_necessity_test(test_records, runner, edges)

    assert "m3_summary" in out
    assert "m4_summary" in out
    assert "delta_brier" in out
    assert "delta_log_loss" in out
    assert isinstance(out["graph_improved_probabilistically"], bool)


def test_graph_necessity_deterministic():
    """Verify that rerun produces identical quantitative deltas."""
    model = LogisticEvidenceModel(feature_type="combined")
    train_records = [
        {"detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"detector_score": 0.2, "clip_score": 0.1, "label": 1},
    ]
    model.fit(train_records)
    calibrator = ProbabilityCalibrator(method="none")

    runner = BaselineLadderRunner(logistic_model=model, prob_calibrator=calibrator)
    records = [
        {"claim_id": "c1", "image_id": "img1", "detector_score": 0.8, "clip_score": 0.3, "label": 0},
        {"claim_id": "c2", "image_id": "img1", "detector_score": 0.2, "clip_score": 0.1, "label": 1},
    ]
    edges = {"img1": [(0, 1, 0.5)]}

    res1 = run_graph_necessity_test(records, runner, edges)
    res2 = run_graph_necessity_test(records, runner, edges)

    assert res1["delta_brier"] == pytest.approx(res2["delta_brier"], abs=1e-9)
    assert res1["delta_log_loss"] == pytest.approx(res2["delta_log_loss"], abs=1e-9)
