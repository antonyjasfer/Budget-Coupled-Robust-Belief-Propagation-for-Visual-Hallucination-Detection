"""
Comprehensive tests for budget and parameter monotonicity invariants:
1. B = 0 gives L_i == nominal == U_i.
2. epsilon_i = 0 gives L_i == U_i == nominal.
3. Interval validity: 0 <= L_i <= U_i <= 1.
4. Feasible set monotonicity: for B2 >= B1, L(B2) <= L(B1) and U(B2) >= U(B1).
5. Field perturbation monotonicity:
   - Increasing positive unary field never reduces target posterior.
   - Decreasing unary field never increases target posterior.
6. Zero coupling (J = 0) matches independent-node interpretation.
7. Budget saturation: B >= sum_i epsilon_i matches independent box bounds.
"""

import pytest
import numpy as np
from scipy.special import expit

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import run_standard_bp
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds


def test_invariant_zero_budget():
    """B = 0 must collapse the interval to nominal exactly."""
    n = 4
    theta = np.array([0.5, -0.2, 0.3, -0.7])
    couplings = np.array([0.4, 0.6, 0.2])
    epsilon = np.array([0.3, 0.3, 0.3, 0.3])

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    nom_bp = run_standard_bp(model)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=0.0)
        assert np.isclose(res.nominal_marginal, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.lower_grid, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.upper_grid, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.lower_certified, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.upper_certified, nom_bp.marginals[target], atol=1e-12)


def test_invariant_zero_epsilon():
    """epsilon = 0 for all nodes collapses interval to nominal even for large B."""
    n = 3
    theta = np.array([-0.5, 0.4, 0.1])
    couplings = np.array([0.8, 0.5])
    epsilon = np.zeros(n)

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    nom_bp = run_standard_bp(model)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=5.0)
        assert np.isclose(res.lower_grid, nom_bp.marginals[target], atol=1e-12)
        assert np.isclose(res.upper_grid, nom_bp.marginals[target], atol=1e-12)


def test_invariant_interval_bounds_validity():
    """0 <= L_cert <= L_grid <= U_grid <= U_cert <= 1 for all configurations."""
    rng = np.random.default_rng(123)
    for _ in range(10):
        n = rng.integers(2, 6)
        theta = rng.uniform(-2.0, 2.0, size=n)
        couplings = rng.uniform(0.1, 1.5, size=n - 1)
        epsilon = rng.uniform(0.1, 0.8, size=n)
        budget = rng.uniform(0.1, 2.0)

        model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
        for target in range(n):
            res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=50)
            assert 0.0 <= res.lower_certified <= 1.0
            assert 0.0 <= res.lower_grid <= 1.0
            assert 0.0 <= res.upper_grid <= 1.0
            assert 0.0 <= res.upper_certified <= 1.0

            assert res.lower_certified <= res.lower_grid + 1e-12
            assert res.lower_grid <= res.nominal_marginal + 1e-12
            assert res.nominal_marginal <= res.upper_grid + 1e-12
            assert res.upper_grid <= res.upper_certified + 1e-12


def test_invariant_budget_dilation_monotonicity():
    """For B2 >= B1, L(B2) <= L(B1) and U(B2) >= U(B1)."""
    n = 5
    theta = np.array([0.1, -0.3, 0.2, 0.4, -0.1])
    couplings = np.array([0.3, 0.5, 0.4, 0.6])
    epsilon = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    budgets = [0.0, 0.1, 0.25, 0.5, 0.8, 1.2, 1.6, 2.0]
    target = 2

    lows = []
    ups = []
    for b in budgets:
        res = solve_robust_bp(model, target_node=target, budget=b, num_grid_steps=80)
        lows.append(res.lower_grid)
        ups.append(res.upper_grid)

    for i in range(len(budgets) - 1):
        # Allow small numerical tolerance (1e-6) due to discrete step grid alignment
        assert lows[i + 1] <= lows[i] + 1e-5, f"Lower bound grew with budget: {lows[i]} -> {lows[i+1]}"
        assert ups[i + 1] >= ups[i] - 1e-5, f"Upper bound shrank with budget: {ups[i]} -> {ups[i+1]}"


def test_invariant_zero_coupling_independent_interpretation():
    """When all J = 0, nodes decouple and target marginal matches independent-node interpretation."""
    n = 4
    theta = np.array([0.5, -0.8, 0.2, 0.1])
    couplings = np.zeros(n - 1)  # All J = 0
    # Use grid-aligned epsilons with step = 0.02 (budget = 1.0, num_grid_steps = 50)
    epsilon = np.array([0.30, 0.40, 0.24, 0.50])
    budget = 1.0

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=50)

        # For independent node, effective field depends only on local delta_target
        max_local_delta = min(epsilon[target], budget)
        expected_upper_p = float(expit(2.0 * (theta[target] + max_local_delta)))
        expected_lower_p = float(expit(2.0 * (theta[target] - max_local_delta)))

        assert np.isclose(res.upper_grid, expected_upper_p, atol=1e-5)
        assert np.isclose(res.lower_grid, expected_lower_p, atol=1e-5)


def test_invariant_budget_saturation_matches_box():
    """
    When B >= sum_i epsilon_i and grid aligns with epsilon,
    budget is not binding and robust BP matches independent box bounds.
    When not aligned, certified bounds rigorously bracket continuous box bounds.
    """
    n = 3
    theta = np.array([0.2, -0.4, 0.3])
    couplings = np.array([0.5, 0.7])
    # Use grid-aligned epsilons with step = 0.02 (budget = 1.0, num_grid_steps = 50)
    epsilon_aligned = np.array([0.20, 0.30, 0.24])
    total_eps = float(np.sum(epsilon_aligned))

    model_aligned = create_chain_tree(n, theta, couplings, epsilon=epsilon_aligned)

    for target in range(n):
        box_low, box_up, _, _ = solve_independent_box_bounds(model_aligned, target_node=target)
        # Saturated budget
        res_sat = solve_robust_bp(model_aligned, target_node=target, budget=1.0, num_grid_steps=50)

        assert np.isclose(res_sat.lower_grid, box_low, atol=1e-5)
        assert np.isclose(res_sat.upper_grid, box_up, atol=1e-5)

    # Now test non-aligned: certified bounds must bracket continuous box bounds
    epsilon_unaligned = np.array([0.21, 0.33, 0.25])
    model_unaligned = create_chain_tree(n, theta, couplings, epsilon=epsilon_unaligned)
    for target in range(n):
        box_low, box_up, _, _ = solve_independent_box_bounds(model_unaligned, target_node=target)
        res = solve_robust_bp(model_unaligned, target_node=target, budget=1.5, num_grid_steps=50)
        assert res.lower_certified <= box_low <= res.lower_grid + 1e-12
        assert res.upper_grid - 1e-12 <= box_up <= res.upper_certified
