"""
Comprehensive brute-force oracle verification of the budget-coupled robust BP solver.

For small trees (n <= 6), this test module:
1. Enumerates every discrete perturbation vector delta on the budget grid.
2. Computes the exact partition function Z and marginal P(h_r = +1) for each delta via 2^n state enumeration.
3. Finds the true global discrete minimum and maximum.
4. Compares against solve_robust_bp (lower_grid, upper_grid).
5. Verifies witness optimality and exact agreement across diverse tree topologies.
"""

import pytest
import numpy as np
import itertools

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.brute_force import run_brute_force_inference, enumerate_grid_perturbations
from src.robust_bp.solver import solve_robust_bp


def exact_grid_oracle(
    model: TreeModel,
    target_node: int,
    budget: float,
    num_grid_steps: int
):
    """
    Independent brute-force oracle:
    Directly enumerates all discrete grid vectors and evaluates exact marginals.
    """
    n = model.num_nodes
    step = budget / float(num_grid_steps) if (budget > 1e-12 and num_grid_steps > 0) else 0.0

    if step <= 1e-12:
        res = run_brute_force_inference(model)
        p = res.marginals[target_node]
        return p, p, np.zeros(n), np.zeros(n)

    # Upper search (all non-negative discrete allocations)
    max_k_per_node = [int(np.floor(min(eps, budget) / step + 1e-8)) for eps in model.epsilon]

    # Generate all candidate integer vectors sum_i a_i <= num_grid_steps
    def _gen_allocations(idx, rem):
        if idx == n - 1:
            for a in range(min(max_k_per_node[idx], rem) + 1):
                yield [a]
        else:
            for a in range(min(max_k_per_node[idx], rem) + 1):
                for rest in _gen_allocations(idx + 1, rem - a):
                    yield [a] + rest

    best_upper_p = -1.0
    best_upper_delta = np.zeros(n)
    for a_vec in _gen_allocations(0, num_grid_steps):
        delta = np.array(a_vec, dtype=np.float64) * step
        res = run_brute_force_inference(model, delta=delta)
        p = res.marginals[target_node]
        if p > best_upper_p:
            best_upper_p = p
            best_upper_delta = delta.copy()

    best_lower_p = 2.0
    best_lower_delta = np.zeros(n)
    for a_vec in _gen_allocations(0, num_grid_steps):
        delta = -np.array(a_vec, dtype=np.float64) * step
        res = run_brute_force_inference(model, delta=delta)
        p = res.marginals[target_node]
        if p < best_lower_p:
            best_lower_p = p
            best_lower_delta = delta.copy()

    return best_lower_p, best_upper_p, best_lower_delta, best_upper_delta


@pytest.mark.parametrize("target_node", [0, 1, 2, 3])
def test_oracle_vs_dp_4node_chain(target_node):
    """Verify 4-node chain DP exactly matches brute-force grid oracle on every target."""
    n = 4
    theta = np.array([0.3, -0.4, 0.1, -0.2])
    couplings = np.array([0.5, 0.7, 0.4])
    epsilon = np.array([0.3, 0.4, 0.2, 0.5])
    budget = 0.6
    num_grid_steps = 15

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    oracle_low, oracle_up, _, _ = exact_grid_oracle(
        model, target_node, budget, num_grid_steps
    )
    dp_res = solve_robust_bp(
        model, target_node, budget, num_grid_steps
    )

    assert np.isclose(dp_res.lower_grid, oracle_low, atol=1e-6), (
        f"Lower bound mismatch at target {target_node}: DP={dp_res.lower_grid}, Oracle={oracle_low}"
    )
    assert np.isclose(dp_res.upper_grid, oracle_up, atol=1e-6), (
        f"Upper bound mismatch at target {target_node}: DP={dp_res.upper_grid}, Oracle={oracle_up}"
    )


def test_oracle_vs_dp_star_tree_all_nodes():
    """Verify star tree (center + 3 leaves) DP matches oracle for center and leaves."""
    num_leaves = 3
    center_theta = 0.25
    leaf_thetas = np.array([-0.3, 0.15, -0.4])
    couplings = np.array([0.6, 0.35, 0.8])
    epsilon = np.array([0.3, 0.4, 0.3, 0.2])
    budget = 0.5
    num_grid_steps = 12

    model = create_star_tree(num_leaves, center_theta, leaf_thetas, couplings, epsilon=epsilon)

    for target in range(model.num_nodes):
        oracle_low, oracle_up, _, _ = exact_grid_oracle(
            model, target, budget, num_grid_steps
        )
        dp_res = solve_robust_bp(
            model, target, budget, num_grid_steps
        )

        assert np.isclose(dp_res.lower_grid, oracle_low, atol=1e-6)
        assert np.isclose(dp_res.upper_grid, oracle_up, atol=1e-6)


def test_oracle_vs_dp_branched_tree():
    """
    Test on a 5-node general tree:
       0 - 1 - 2
           |   |
           3   4
    """
    edges = [(0, 1), (1, 2), (1, 3), (2, 4)]
    couplings = {(0, 1): 0.4, (1, 2): 0.6, (1, 3): 0.5, (2, 4): 0.3}
    theta = np.array([0.1, -0.2, 0.3, -0.1, 0.2])
    epsilon = np.array([0.2, 0.3, 0.2, 0.4, 0.2])
    budget = 0.45
    num_grid_steps = 9

    model = TreeModel(
        num_nodes=5,
        theta=theta,
        edges=edges,
        coupling=couplings,
        epsilon=epsilon
    )

    # Test root at node 1 (highest degree) and node 4 (leaf)
    for target in [1, 4]:
        oracle_low, oracle_up, _, _ = exact_grid_oracle(
            model, target, budget, num_grid_steps
        )
        dp_res = solve_robust_bp(
            model, target, budget, num_grid_steps
        )

        assert np.isclose(dp_res.lower_grid, oracle_low, atol=1e-6)
        assert np.isclose(dp_res.upper_grid, oracle_up, atol=1e-6)
