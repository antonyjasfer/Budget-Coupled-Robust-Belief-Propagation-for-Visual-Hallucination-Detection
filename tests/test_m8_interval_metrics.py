"""Tests for Milestone 8 Interval Metrics and Statistics."""

import pytest

from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.interval_metrics import compute_interval_statistics
from src.data.schemas import GroundTruthStatus


def test_interval_statistics_computation():
    results = [
        ClaimEvaluationResult(
            run_id="test",
            experiment_name="test",
            condition="test",
            image_id="img1",
            claim_id="c1",
            object_category="dog",
            text_span="dog",
            split="TEST",
            ground_truth="hallucinated",
            detector_score=0.2,
            clip_score=0.1,
            theta_i=1.0,
            epsilon_i=0.5,
            coupling_j=0.5,
            budget=1.0,
            epsilon_scale=1.0,
            decision_threshold=0.5,
            grid_steps=20,
            standard_posterior=0.8,
            standard_prediction="hallucinated",
            robust_lower=0.6,
            robust_upper=0.9,
            robust_midpoint=0.75,
            interval_width=0.3,
            certified_lower=0.58,
            certified_upper=0.92,
            field_cert_gap=0.05,
            robust_prediction="hallucinated",
            evidence_sensitive=False,
            abstained=False,
            seed=42,
        ),
        ClaimEvaluationResult(
            run_id="test",
            experiment_name="test",
            condition="test",
            image_id="img1",
            claim_id="c2",
            object_category="cat",
            text_span="cat",
            split="TEST",
            ground_truth="supported",
            detector_score=0.4,
            clip_score=0.2,
            theta_i=0.5,
            epsilon_i=0.5,
            coupling_j=0.5,
            budget=1.0,
            epsilon_scale=1.0,
            decision_threshold=0.5,
            grid_steps=20,
            standard_posterior=0.6,  # Error: predicted "hallucinated" but GT is "supported"
            standard_prediction="hallucinated",
            robust_lower=0.4,
            robust_upper=0.8,
            robust_midpoint=0.6,
            interval_width=0.4,
            certified_lower=0.38,
            certified_upper=0.82,
            field_cert_gap=0.05,
            robust_prediction="abstain",
            evidence_sensitive=True,
            abstained=True,
            seed=42,
        ),
    ]

    stats = compute_interval_statistics(results, decision_threshold=0.5)

    assert stats.total_claims == 2
    assert stats.mean_width == pytest.approx(0.35)
    assert stats.min_width == pytest.approx(0.3)
    assert stats.max_width == pytest.approx(0.4)
    assert stats.evidence_sensitive_fraction == pytest.approx(0.5)
    assert stats.mean_width_correct_bp == pytest.approx(0.3)
    assert stats.mean_width_wrong_bp == pytest.approx(0.4)
