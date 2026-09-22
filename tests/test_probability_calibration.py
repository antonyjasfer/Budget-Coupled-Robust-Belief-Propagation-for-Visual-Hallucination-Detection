"""Unit tests for post-hoc probability calibration and evaluation metrics."""

import numpy as np
import pytest

from src.calibration.evidence_models import (
    ProbabilityCalibrator,
    compute_calibration_metrics,
)


def test_uncalibrated_identity():
    """Verify that method='none' returns raw uncalibrated probabilities unchanged."""
    cal = ProbabilityCalibrator(method="none")
    p = np.array([0.1, 0.5, 0.9])
    y = np.array([0, 1, 1])
    cal.fit(p, y)
    out = cal.calibrate(p)
    assert np.allclose(p, out)


def test_platt_calibration_fit():
    """Verify Platt (logistic) calibration fits monotonic sigmoid remapping."""
    rng = np.random.RandomState(42)
    # Synthetic uncalibrated overconfident probabilities
    p = np.linspace(0.05, 0.95, 40)
    # True probabilities roughly match p with some noise
    y = rng.binomial(1, p)

    cal = ProbabilityCalibrator(method="platt")
    cal.fit(p, y)
    calibrated = cal.calibrate(p)

    assert len(calibrated) == len(p)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    # Platt scaling should preserve monotonicity
    assert np.all(np.diff(calibrated) >= -1e-6)


def test_isotonic_calibration_fit():
    """Verify Isotonic regression calibration on moderate sample sizes."""
    p = np.linspace(0.1, 0.9, 30)
    y = np.array([0]*15 + [1]*15)

    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(p, y)
    calibrated = cal.calibrate(p)

    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    # Non-decreasing
    assert np.all(np.diff(calibrated) >= 0.0)


def test_calibration_metrics():
    """Verify Brier score, log loss, and ECE computation."""
    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    # Perfect predictions
    p_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    m_perf = compute_calibration_metrics(y_true, p_perfect)
    assert m_perf["brier_score"] == pytest.approx(0.0, abs=1e-5)
    assert m_perf["ece"] == pytest.approx(0.0, abs=1e-5)

    # Imperfect predictions
    p_imperfect = np.array([0.2, 0.3, 0.8, 0.7])
    m_imp = compute_calibration_metrics(y_true, p_imperfect)
    assert m_imp["brier_score"] > 0.0
    assert m_imp["log_loss"] > 0.0
    assert m_imp["ece"] >= 0.0


def test_calibrator_serialization():
    """Verify calibrator to_dict and from_dict roundtrip."""
    cal = ProbabilityCalibrator(method="platt")
    p = np.array([0.2, 0.4, 0.6, 0.8])
    y = np.array([0, 0, 1, 1])
    cal.fit(p, y)

    d = cal.to_dict()
    recon = ProbabilityCalibrator.from_dict(d)
    assert recon.fitted
    assert recon.method == "platt"

    out1 = cal.calibrate(p)
    out2 = recon.calibrate(p)
    assert np.allclose(out1, out2)
