"""
Targeted mathematical regression tests for Milestone 1 inference core.

Covers:
- Non-grid-aligned budgets and radii
- Heterogeneous radii including zero
- One-node and two-node tree cases
- Mixed-sign unary fields
- Zero couplings (independent nodes)
- Small trees compared with exhaustive signed-grid enumeration
- Witness feasibility and probability reproduction across multiple targets
- Random continuous feasible perturbations checked against certified bounds
"""

import pytest
import numpy as np
from scipy.special import expit

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import run_standard_bp, compute_rooted_total_field
from src.pgm.brute_force import run_brute_force_inference, run_brute_force_robust_grid
from src.robust_bp.solver import solve_robust_bp
from src.robust_bp.certification import compute_one_node_continuous_bounds


def test_non_grid_aligned_budget_and_radii():
    """Verify robust DP and witness recovery with non-grid-aligned budget and radii."""
    n = 3
    theta = np.array([0.15, -0.32, 0.45])
    couplings = np.array([0.35, 0.65])
    epsilon = np.array([0.33, 0.57, 0.19])
    budget = 0.73
    num_grid_steps = 37

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)
        bf_low, bf_up, _, _ = run_brute_force_robust_grid(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)

        assert np.isclose(res.lower_grid, bf_low, atol=1e-6)
        assert np.isclose(res.upper_grid, bf_up, atol=1e-6)
        assert res.witness_upper.is_budget_valid
        assert res.witness_upper.is_box_valid
        assert res.witness_upper.field_match
        assert res.witness_lower.is_budget_valid
        assert res.witness_lower.is_box_valid
        assert res.witness_lower.field_match


def test_heterogeneous_radii_including_zero():
    """Verify that nodes with epsilon=0 receive zero perturbation in DP and witness."""
    n = 4
    theta = np.array([0.1, 0.2, -0.1, 0.3])
    couplings = np.array([0.4, 0.5, 0.3])
    epsilon = np.array([0.4, 0.0, 0.5, 0.0])  # nodes 1 and 3 are fixed
    budget = 0.6
    num_grid_steps = 25

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    target = 0

    res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)

    # Reconstructed witnesses must have delta=0 at indices 1 and 3
    assert np.isclose(res.witness_upper.delta[1], 0.0, atol=1e-9)
    assert np.isclose(res.witness_upper.delta[3], 0.0, atol=1e-9)
    assert np.isclose(res.witness_lower.delta[1], 0.0, atol=1e-9)
    assert np.isclose(res.witness_lower.delta[3], 0.0, atol=1e-9)

    assert res.witness_upper.is_box_valid
    assert res.witness_upper.is_budget_valid
    assert res.witness_lower.is_box_valid
    assert res.witness_lower.is_budget_valid


def test_two_node_tree_exact():
    """2-node graph: BP and robust DP match brute force exactly."""
    n = 2
    theta = np.array([-0.3, 0.4])
    couplings = np.array([0.7])
    epsilon = np.array([0.25, 0.35])
    budget = 0.4
    num_grid_steps = 20

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)
        bf_low, bf_up, _, _ = run_brute_force_robust_grid(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)

        assert np.isclose(res.lower_grid, bf_low, atol=1e-6)
        assert np.isclose(res.upper_grid, bf_up, atol=1e-6)


def test_mixed_sign_unary_fields():
    """Verify behavior on large mixed-sign unary fields."""
    n = 4
    theta = np.array([-2.5, 3.0, -1.2, 0.8])
    couplings = np.array([0.6, 0.8, 0.4])
    epsilon = np.array([0.5, 0.5, 0.5, 0.5])
    budget = 1.0
    num_grid_steps = 30

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)
        assert res.lower_certified <= res.lower_grid <= res.upper_grid <= res.upper_certified
        assert 0.0 <= res.lower_certified
        assert res.upper_certified <= 1.0


def test_zero_couplings_independent_nodes():
    """When all couplings J_ij = 0, robust BP on target node matches single-node formula."""
    n = 3
    theta = np.array([0.2, -0.4, 0.6])
    couplings = np.array([0.0, 0.0])  # Independent
    epsilon = np.array([0.3, 0.5, 0.4])
    budget = 0.25
    num_grid_steps = 50

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    for target in range(n):
        res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)
        expected_low, expected_up = compute_one_node_continuous_bounds(theta[target], epsilon[target], budget)
        assert np.isclose(res.lower_grid, expected_low, atol=1e-5)
        assert np.isclose(res.upper_grid, expected_up, atol=1e-5)


def test_continuous_feasible_perturbations_containment():
    """Verify that randomly sampled continuous perturbations strictly lie within certified bounds."""
    rng = np.random.default_rng(2026)
    n = 4
    theta = rng.uniform(-0.5, 0.5, size=n)
    couplings = rng.uniform(0.2, 0.8, size=n - 1)
    epsilon = rng.uniform(0.2, 0.6, size=n)
    budget = 0.8
    num_grid_steps = 50

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    target = 1

    res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=num_grid_steps)

    for _ in range(100):
        raw = rng.uniform(-1.0, 1.0, size=n)
        raw_box = np.clip(raw, -1.0, 1.0) * epsilon
        l1_norm = np.sum(np.abs(raw_box))
        if l1_norm > budget:
            continuous_delta = raw_box * (budget / l1_norm)
        else:
            continuous_delta = raw_box

        assert np.all(np.abs(continuous_delta) <= epsilon + 1e-12)
        assert np.sum(np.abs(continuous_delta)) <= budget + 1e-12

        _, p_perturbed, _ = compute_rooted_total_field(model, root=target, delta=continuous_delta)

        assert res.lower_certified - 1e-7 <= p_perturbed <= res.upper_certified + 1e-7, (
            f"Perturbed probability {p_perturbed} violated certified interval "
            f"[{res.lower_certified}, {res.upper_certified}]"
        )
