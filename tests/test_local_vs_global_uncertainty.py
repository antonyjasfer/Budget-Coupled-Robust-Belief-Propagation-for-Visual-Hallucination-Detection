"""Unit tests comparing local box uncertainty vs budget-coupled uncertainty."""

import numpy as np
import pytest

from src.pgm.brute_force import run_brute_force_inference
from src.pgm.tree_model import create_chain_tree, create_star_tree
from src.robust_bp.solver import solve_robust_bp


def test_local_box_vs_global_budget_containment():
    """Verify that global budget B < sum(epsilon_i) yields strictly tighter intervals than local box."""
    # 3-node chain tree with attractive couplings
    theta = np.array([0.2, -0.1, 0.3])
    epsilon = np.array([0.4, 0.4, 0.4])  # sum(epsilon) = 1.2
    couplings = np.array([0.3, 0.4])
    model = create_chain_tree(num_nodes=3, theta=theta, couplings=couplings, epsilon=epsilon)

    # 1. Local box: by Theorem 1 (coordinate monotonicity), the upper extremizer is delta = +epsilon,
    # and lower extremizer is delta = -epsilon.
    box_upper_res = run_brute_force_inference(model, delta=epsilon)
    box_lower_res = run_brute_force_inference(model, delta=-epsilon)
    p_box_upper = box_upper_res.marginals[0]
    p_box_lower = box_lower_res.marginals[0]
    box_width = p_box_upper - p_box_lower

    # 2. Global budget with B = 0.4 (< 1.2)
    b_constrained = 0.4
    res_budget = solve_robust_bp(
        model=model,
        target_node=0,
        budget=b_constrained,
        num_grid_steps=20,
    )
    p_budget_upper = res_budget.upper_grid
    p_budget_lower = res_budget.lower_grid
    budget_width = p_budget_upper - p_budget_lower

    # Global budget must be strictly inside the local box interval
    assert p_budget_upper <= p_box_upper + 1e-6
    assert p_budget_lower >= p_box_lower - 1e-6
    # Width under constrained budget must be strictly smaller than unconstrained box
    assert budget_width < box_width - 0.05


def test_unconstrained_budget_recovers_local_box():
    """Verify that when B >= sum(epsilon_i), global budget collapses to local box."""
    theta = np.array([0.1, -0.2, 0.15])
    epsilon = np.array([0.2, 0.2, 0.2])  # sum(epsilon) = 0.6
    couplings = np.array([0.25, 0.25])
    model = create_chain_tree(num_nodes=3, theta=theta, couplings=couplings, epsilon=epsilon)

    # Exact local box via monotonicity extremizers
    box_upper = run_brute_force_inference(model, delta=epsilon).marginals[0]
    box_lower = run_brute_force_inference(model, delta=-epsilon).marginals[0]

    # Solver with B = 0.60 (equal to sum epsilon)
    # Using fine grid num_grid_steps = 30 so step = 0.6 / 30 = 0.02, perfectly dividing 0.2
    res = solve_robust_bp(
        model=model,
        target_node=0,
        budget=0.60,
        num_grid_steps=30,
    )

    assert res.upper_grid == pytest.approx(box_upper, abs=1e-4)
    assert res.lower_grid == pytest.approx(box_lower, abs=1e-4)


def test_zero_budget_collapses_to_standard_bp():
    """Verify that B = 0 collapses both bounds to the nominal unperturbed marginal."""
    model = create_star_tree(
        num_leaves=3,
        center_theta=0.1,
        leaf_thetas=np.array([-0.1, 0.2, -0.2]),
        center_leaf_couplings=np.array([0.35, 0.35, 0.35]),
        epsilon=np.array([0.3, 0.3, 0.3, 0.3]),
    )
    nominal = run_brute_force_inference(model, delta=np.zeros(4)).marginals[0]

    res = solve_robust_bp(model=model, target_node=0, budget=0.0, num_grid_steps=10)
    assert res.upper_grid == pytest.approx(nominal, abs=1e-6)
    assert res.lower_grid == pytest.approx(nominal, abs=1e-6)
