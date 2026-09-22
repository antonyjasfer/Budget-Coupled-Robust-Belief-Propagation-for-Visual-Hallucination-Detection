"""Unit tests for empirical global budget B calibration."""

import numpy as np
import pytest

from src.calibration.budget_calibrator import BudgetCalibrator


def test_budget_aggregation_per_image():
    """Verify sum of residuals per image and quantile calibration."""
    # Image 1 has 3 claims with shifts 0.1, 0.2, 0.3 -> sum = 0.6
    # Image 2 has 2 claims with shifts 0.4, 0.6 -> sum = 1.0
    # Image 3 has 1 claim with shift 0.2 -> sum = 0.2
    residuals = [
        {"image_id": "img1", "claim_id": "c1", "condition": "blur", "residual": 0.1},
        {"image_id": "img1", "claim_id": "c2", "condition": "blur", "residual": 0.2},
        {"image_id": "img1", "claim_id": "c3", "condition": "blur", "residual": 0.3},
        {"image_id": "img2", "claim_id": "c4", "condition": "blur", "residual": 0.4},
        {"image_id": "img2", "claim_id": "c5", "condition": "blur", "residual": 0.6},
        {"image_id": "img3", "claim_id": "c6", "condition": "blur", "residual": 0.2},
    ]

    cal = BudgetCalibrator(default_quantile=0.90)
    cal.fit_residuals(residuals)

    # Distinct image sums under "blur" are: [0.6, 1.0, 0.2]
    assert len(cal.observed_sums) == 3
    assert set(round(s, 2) for s in cal.observed_sums) == {0.2, 0.6, 1.0}

    # B at q=0.0 should be 0.2, at q=1.0 should be 1.0
    assert cal.get_budget(quantile=0.0) == pytest.approx(0.2, abs=1e-3)
    assert cal.get_budget(quantile=1.0) == pytest.approx(1.0, abs=1e-3)
    assert cal.get_budget(quantile=0.5) >= cal.get_budget(quantile=0.1)


def test_normalized_budget_calculation():
    """Verify rho = B / sum(epsilon_i) calculation."""
    cal = BudgetCalibrator()
    # Mock budget of 1.2
    cal.observed_sums = [1.0, 1.2, 1.5]

    epsilons = [0.5, 0.5, 0.5]  # sum = 1.5
    # For budget = 1.2, rho = 1.2 / 1.5 = 0.8
    rho = cal.get_normalized_budget(epsilons, budget=1.2)
    assert rho == pytest.approx(0.8, abs=1e-5)


def test_empty_residuals_fallback():
    """Verify fallback when no residuals are provided."""
    cal = BudgetCalibrator()
    cal.fit_residuals([])
    assert cal.get_budget() >= 0.0
