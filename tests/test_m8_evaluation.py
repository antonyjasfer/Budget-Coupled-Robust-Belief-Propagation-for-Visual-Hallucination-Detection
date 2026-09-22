"""Tests for Milestone 8 Evaluation Metrics and Safeguards."""

import pytest
import numpy as np

from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results, compute_classification_metrics
from src.data.schemas import GroundTruthStatus, DecisionStatus


def test_classification_metrics_computation():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1])
    y_prob = np.array([0.9, 0.4, 0.1, 0.2, 0.8, 0.7])

    acc, prec, rec, f1, roc_auc, pr_auc, cm = compute_classification_metrics(y_true, y_pred, y_prob)

    assert acc == pytest.approx(4 / 6)
    assert cm.tp == 2
    assert cm.fn == 1
    assert cm.tn == 2
    assert cm.fp == 1
    assert roc_auc is not None
    assert 0.0 <= roc_auc <= 1.0


def test_single_class_roc_auc_safeguard():
    # Only positive class in y_true
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.7])

    acc, prec, rec, f1, roc_auc, pr_auc, cm = compute_classification_metrics(y_true, y_pred, y_prob)
    assert roc_auc is None  # Handled safely without crashing
    assert pr_auc is None


def test_unknown_ground_truth_exclusion():
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
            ground_truth="unknown",
            detector_score=0.5,
            clip_score=0.2,
            theta_i=0.0,
            epsilon_i=0.5,
            coupling_j=0.5,
            budget=1.0,
            epsilon_scale=1.0,
            decision_threshold=0.5,
            grid_steps=20,
            standard_posterior=0.5,
            standard_prediction="hallucinated",
            robust_lower=0.2,
            robust_upper=0.8,
            robust_midpoint=0.5,
            interval_width=0.6,
            certified_lower=0.18,
            certified_upper=0.82,
            field_cert_gap=0.05,
            robust_prediction="abstain",
            evidence_sensitive=True,
            abstained=True,
            seed=42,
        ),
        ClaimEvaluationResult(
            run_id="test",
            experiment_name="test",
            condition="test",
            image_id="img2",
            claim_id="c3",
            object_category="car",
            text_span="car",
            split="TEST",
            ground_truth="supported",
            detector_score=0.8,
            clip_score=0.3,
            theta_i=-1.0,
            epsilon_i=0.5,
            coupling_j=0.5,
            budget=1.0,
            epsilon_scale=1.0,
            decision_threshold=0.5,
            grid_steps=20,
            standard_posterior=0.2,
            standard_prediction="supported",
            robust_lower=0.1,
            robust_upper=0.3,
            robust_midpoint=0.2,
            interval_width=0.2,
            certified_lower=0.08,
            certified_upper=0.32,
            field_cert_gap=0.05,
            robust_prediction="supported",
            evidence_sensitive=False,
            abstained=False,
            seed=42,
        ),
    ]

    metrics_res = evaluate_claim_results(results, method_name="standard_bp", decision_threshold=0.5, exclude_unknown=True)

    assert metrics_res.total_claims == 3
    assert metrics_res.evaluated_claims == 2
    assert metrics_res.unknown_gt_count == 1
    assert metrics_res.accuracy == 1.0
