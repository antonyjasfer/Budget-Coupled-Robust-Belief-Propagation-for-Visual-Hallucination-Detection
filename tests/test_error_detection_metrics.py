"""Tests for Error Detection Metrics (AUROC, AUPRC, Mean/Median Uncertainty).

Validates evaluation of uncertainty scores against prediction error e_i = I(y_pred != y_true).
"""

import numpy as np
import pytest

from src.calibration.reliability import compute_error_detection_metrics


def test_error_detection_ranking():
    """When uncertainty scores perfectly rank errors, AUROC must equal 1.0."""
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
    # Errors at index 3 (0 vs 1) and index 7 (1 vs 0) -> 2 errors
    # Let uncertainty score be high on errors (0.9, 0.8) and low on correct (0.1, 0.2, 0.15, 0.05, 0.2, 0.1)
    uncertainty = [0.1, 0.2, 0.15, 0.9, 0.05, 0.2, 0.1, 0.8]

    res = compute_error_detection_metrics(y_true, y_pred, uncertainty)
    assert res["status"] == "EVALUATED"
    assert res["n_samples"] == 8
    assert res["n_errors"] == 2
    assert np.isclose(res["error_detection_auroc"], 1.0)
    assert np.isclose(res["error_detection_auprc"], 1.0)
    assert res["mean_uncertainty_incorrect"] > res["mean_uncertainty_correct"]


def test_error_detection_zero_errors_returns_not_evaluable():
    """Zero errors in dataset cannot compute binary ROC AUC -> must return NOT EVALUABLE."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    uncertainty = [0.1, 0.2, 0.3, 0.4]

    res = compute_error_detection_metrics(y_true, y_pred, uncertainty)
    assert res["status"] == "NOT EVALUABLE"
    assert "Single error class" in res["reason"]
    assert res["error_detection_auroc"] is None


def test_error_detection_all_errors_returns_not_evaluable():
    """All errors in dataset cannot compute binary ROC AUC -> must return NOT EVALUABLE."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    uncertainty = [0.9, 0.8, 0.7, 0.6]

    res = compute_error_detection_metrics(y_true, y_pred, uncertainty)
    assert res["status"] == "NOT EVALUABLE"
    assert "Single error class" in res["reason"]
    assert res["error_detection_auroc"] is None


def test_error_detection_insufficient_samples():
    y_true = [0, 1]
    y_pred = [0, 0]
    uncertainty = [0.2, 0.8]

    res = compute_error_detection_metrics(y_true, y_pred, uncertainty, min_samples=4)
    assert res["status"] == "NOT EVALUABLE"
    assert "minimum required" in res["reason"]
