"""Unit tests for empirical perturbation-set inclusion rate diagnostics."""

import numpy as np
import pytest

from src.calibration.budget_calibrator import BudgetCalibrator


def test_inclusion_rates_logic():
    """Verify local, global, and joint empirical inclusion rate calculations."""
    cal = BudgetCalibrator()

    # 4 observed perturbation vectors on a 2-node claim tree
    observed = [
        np.array([0.1, 0.2]),  # sum = 0.3
        np.array([0.3, 0.1]),  # sum = 0.4
        np.array([0.5, 0.2]),  # sum = 0.7
        np.array([0.2, 0.6]),  # sum = 0.8
    ]

    eps = np.array([0.4, 0.4])  # node 0 max 0.4, node 1 max 0.4
    b = 0.5  # budget 0.5

    # Check vector 0: [0.1, 0.2] -> local: 0.1<=0.4 & 0.2<=0.4 (T), sum: 0.3<=0.5 (T) -> joint (T)
    # Check vector 1: [0.3, 0.1] -> local: T, sum: 0.4<=0.5 (T) -> joint (T)
    # Check vector 2: [0.5, 0.2] -> local: 0.5>0.4 (F), sum: 0.7>0.5 (F) -> joint (F)
    # Check vector 3: [0.2, 0.6] -> local: 0.6>0.4 (F), sum: 0.8>0.5 (F) -> joint (F)

    diag = cal.compute_inclusion_rate(observed, eps, budget=b)

    assert diag.n_samples == 4
    assert diag.local_bound_inclusion_rate == pytest.approx(2 / 4, abs=1e-5)
    assert diag.global_budget_inclusion_rate == pytest.approx(2 / 4, abs=1e-5)
    assert diag.joint_inclusion_rate == pytest.approx(2 / 4, abs=1e-5)


def test_inclusion_hierarchical_inequality():
    """Verify that joint inclusion is always <= min(local, global)."""
    cal = BudgetCalibrator()

    observed = [
        np.array([0.3, 0.3]),  # local T (<=0.4), sum=0.6 > 0.5 (global F)
        np.array([0.5, 0.0]),  # local F (0.5>0.4), sum=0.5 <= 0.5 (global T)
    ]
    eps = np.array([0.4, 0.4])
    b = 0.5

    diag = cal.compute_inclusion_rate(observed, eps, budget=b)
    # local: 1/2 (vec 0 passes)
    # global: 1/2 (vec 1 passes)
    # joint: 0/2 (neither passes both)
    assert diag.local_bound_inclusion_rate == 0.5
    assert diag.global_budget_inclusion_rate == 0.5
    assert diag.joint_inclusion_rate == 0.0
    assert diag.joint_inclusion_rate <= min(
        diag.local_bound_inclusion_rate, diag.global_budget_inclusion_rate
    )


def test_infinite_budget_recovers_local_bound():
    """When budget is infinite, joint inclusion rate must equal local bound inclusion rate."""
    cal = BudgetCalibrator()
    observed = [np.array([0.1, 0.2]), np.array([0.9, 0.1])]
    eps = np.array([0.5, 0.5])
    diag = cal.compute_inclusion_rate(observed, eps, budget=1e6)
    assert diag.global_budget_inclusion_rate == 1.0
    assert diag.joint_inclusion_rate == diag.local_bound_inclusion_rate
