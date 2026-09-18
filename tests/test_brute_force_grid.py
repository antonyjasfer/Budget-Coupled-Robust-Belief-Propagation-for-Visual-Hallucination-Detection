"""
Tests comparing budget-coupled robust BP dynamic programming solver against
exhaustive brute-force grid enumeration over all candidate perturbation vectors.
"""

import pytest
import numpy as np

from src.pgm.tree_model import create_chain_tree, create_star_tree
from src.pgm.brute_force import run_brute_force_robust_grid
from src.robust_bp.solver import solve_robust_bp


@pytest.mark.parametrize("target_node", [0, 1, 2])
def test_dp_matches_brute_force_grid_3node_chain(target_node):
    """3-node chain robust DP must match exhaustive brute-force grid search exactly."""
    n = 3
    theta = np.array([0.1, -0.2, 0.3])
    couplings = np.array([0.4, 0.5])
    epsilon = np.array([0.3, 0.4, 0.3])
    budget = 0.5
    num_grid_steps = 20

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    # Brute-force exhaustive search over all grid perturbation vectors
    bf_low, bf_up, bf_delta_low, bf_delta_up = run_brute_force_robust_grid(
        model, target_node=target_node, budget=budget, num_grid_steps=num_grid_steps
    )

    # Robust BP solver using tree max-plus/min-plus convolution
    dp_res = solve_robust_bp(
        model, target_node=target_node, budget=budget, num_grid_steps=num_grid_steps
    )

    assert np.isclose(dp_res.lower_grid, bf_low, atol=1e-6), (
        f"Target {target_node} lower bound mismatch: DP={dp_res.lower_grid}, BF={bf_low}"
    )
    assert np.isclose(dp_res.upper_grid, bf_up, atol=1e-6), (
        f"Target {target_node} upper bound mismatch: DP={dp_res.upper_grid}, BF={bf_up}"
    )


def test_dp_matches_brute_force_grid_star_tree():
    """Star tree (center + 3 leaves) robust DP matches brute-force grid search."""
    num_leaves = 3
    center_theta = -0.1
    leaf_thetas = np.array([0.2, -0.3, 0.1])
    couplings = np.array([0.3, 0.6, 0.4])
    epsilon = np.array([0.25, 0.25, 0.25, 0.25])
    budget = 0.4
    num_grid_steps = 10

    model = create_star_tree(num_leaves, center_theta, leaf_thetas, couplings, epsilon=epsilon)
    target_node = 0  # Center

    bf_low, bf_up, _, _ = run_brute_force_robust_grid(
        model, target_node=target_node, budget=budget, num_grid_steps=num_grid_steps
    )

    dp_res = solve_robust_bp(
        model, target_node=target_node, budget=budget, num_grid_steps=num_grid_steps
    )

    assert np.isclose(dp_res.lower_grid, bf_low, atol=1e-6)
    assert np.isclose(dp_res.upper_grid, bf_up, atol=1e-6)
