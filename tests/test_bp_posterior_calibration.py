"""Tests for Standard BP Posterior Calibration Audit.

Validates:
- Evaluating calibration of standard BP posterior after graph coupling vs independent unary
- Primary metrics: Brier score, log loss
- Secondary diagnostic: ECE with documented bins
"""

import numpy as np
import pytest

from src.calibration.reliability import audit_bp_posterior_calibration


def test_bp_posterior_calibration_audit():
    y_true = [0, 0, 0, 1, 1, 1]
    # Unary probabilities (slightly miscalibrated)
    p_unary = [0.3, 0.4, 0.2, 0.6, 0.7, 0.8]
    # BP probabilities (better calibrated after smoothing)
    p_bp = [0.1, 0.15, 0.05, 0.85, 0.90, 0.95]

    res = audit_bp_posterior_calibration(p_unary, p_bp, y_true, n_bins=3, min_samples=6)
    assert res["status"] == "EVALUATED"
    assert res["n_samples"] == 6
    assert res["n_bins"] == 3
    assert res["bp_brier"] < res["unary_brier"]
    assert res["bp_log_loss"] < res["unary_log_loss"]
    assert res["bp_ece"] is not None


def test_bp_calibration_insufficient_samples_returns_not_evaluable():
    y_true = [0, 1]
    p_unary = [0.2, 0.8]
    p_bp = [0.1, 0.9]
    res = audit_bp_posterior_calibration(p_unary, p_bp, y_true, min_samples=6)
    assert res["status"] == "NOT EVALUABLE"
