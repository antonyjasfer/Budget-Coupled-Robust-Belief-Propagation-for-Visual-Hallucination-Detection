"""
Tests for Budget-Coupled Robust Belief Propagation core mathematical properties.
"""

import pytest
import numpy as np

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import run_standard_bp
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds


def test_robust_bp_zero_budget_gives_nominal():
    """B = 0 must give nominal BP inference exactly."""
    n = 4
    theta = np.array([0.2, -0.1, 0.4, 0.0])
    couplings = np.array([0.5, 0.3, 0.8])
    epsilon = np.array([0.2, 0.2, 0.2, 0.2])

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    nom_bp = run_standard_bp(model)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=0.0, num_grid_steps=50)
        assert np.isclose(res.nominal_marginal, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.lower_grid, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.upper_grid, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.lower_certified, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.upper_certified, nom_bp.marginals[target], atol=1e-12)


def test_robust_bp_zero_epsilon_gives_nominal():
    """epsilon = 0 must give nominal BP inference even if B > 0."""
    n = 3
    theta = np.array([0.1, 0.2, -0.3])
    couplings = np.array([0.4, 0.6])
    epsilon = np.zeros(n)

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    nom_bp = run_standard_bp(model)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=1.5, num_grid_steps=50)
        assert np.isclose(res.lower_grid, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.upper_grid, nom_bp.marginals[target], atol=1e-12)


def test_robust_bp_monotonicity_in_budget():
    """Increasing budget B must not narrow the robust interval [lower_grid, upper_grid]."""
    n = 5
    rng = np.random.default_rng(999)
    theta = rng.uniform(-0.4, 0.4, size=n)
    couplings = rng.uniform(0.2, 0.8, size=n - 1)
    epsilon = np.full(n, 0.5)

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    target = 2

    budgets = [0.0, 0.2, 0.5, 0.8, 1.2, 2.0]
    intervals = []

    for b in budgets:
        res = solve_robust_bp(model, target_node=target, budget=b, num_grid_steps=100)
        intervals.append((res.lower_grid, res.upper_grid))

    for i in range(len(budgets) - 1):
        low1, up1 = intervals[i]
        low2, up2 = intervals[i + 1]
        assert low2 <= low1 + 1e-6, f"Lower bound grew when budget increased: {low1} -> {low2}"
        assert up2 >= up1 - 1e-6, f"Upper bound shrank when budget increased: {up1} -> {up2}"


def test_robust_bp_outputs_valid_probabilities():
    """All computed probabilities must remain in [0, 1] and be properly ordered."""
    n = 4
    theta = np.array([1.5, -2.0, 0.5, -0.8])
    couplings = np.array([1.2, 0.9, 1.5])
    epsilon = np.array([0.8, 0.8, 0.8, 0.8])

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=1.5, num_grid_steps=60)
        # Probabilities bounded in [0, 1]
        assert 0.0 <= res.lower_certified <= 1.0
        assert 0.0 <= res.lower_grid <= 1.0
        assert 0.0 <= res.nominal_marginal <= 1.0
        assert 0.0 <= res.upper_grid <= 1.0
        assert 0.0 <= res.upper_certified <= 1.0

        # Strict sandwich ordering
        assert res.lower_certified <= res.lower_grid + 1e-12
        assert res.lower_grid <= res.nominal_marginal + 1e-12
        assert res.nominal_marginal <= res.upper_grid + 1e-12
        assert res.upper_grid <= res.upper_certified + 1e-12


def test_robust_bp_independent_box_nested():
    """Budget-coupled bounds must be nested inside independent box uncertainty bounds."""
    n = 3
    theta = np.zeros(n)
    couplings = np.array([np.arctanh(0.5), np.arctanh(0.5)])
    epsilon = np.full(n, np.arctanh(0.5))

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    target = 1  # Center node

    box_low, box_up, _, _ = solve_independent_box_bounds(model, target_node=target)
    budget_res = solve_robust_bp(model, target_node=target, budget=float(np.arctanh(0.5)), num_grid_steps=100)

    assert budget_res.lower_grid >= box_low - 1e-6
    assert budget_res.upper_grid <= box_up + 1e-6
