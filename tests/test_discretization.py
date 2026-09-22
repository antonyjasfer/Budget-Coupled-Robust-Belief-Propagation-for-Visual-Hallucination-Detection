"""
Tests for numerical precision, extreme parameter regimes, and grid discretization refinement:
- Extreme unary fields (large positive and large negative theta).
- Extreme coupling strengths (large J, near deterministic).
- Tiny epsilon limits (1e-8).
- Single-node graph (n = 1).
- Grid refinement: increasing K reduces the certification gap Delta = B / K to 0.
- No NaN, inf, or domain errors in any calculation.
"""

import pytest
import numpy as np

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import run_standard_bp, stable_f_J
from src.robust_bp.solver import solve_robust_bp


def test_numerical_large_positive_theta():
    """Verify stability under extreme positive fields (saturated hallucination)."""
    n = 3
    theta = np.array([25.0, 30.0, 20.0])
    couplings = np.array([1.0, 2.0])
    epsilon = np.array([1.0, 1.0, 1.0])
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    res = solve_robust_bp(model, target_node=0, budget=2.0, num_grid_steps=50)

    assert not np.isnan(res.lower_grid)
    assert not np.isnan(res.upper_grid)
    assert not np.isnan(res.field_lower_grid)
    assert not np.isnan(res.field_upper_grid)
    assert np.isclose(res.lower_grid, 1.0, atol=1e-6)
    assert np.isclose(res.upper_grid, 1.0, atol=1e-6)


def test_numerical_large_negative_theta():
    """Verify stability under extreme negative fields (saturated supported)."""
    n = 3
    theta = np.array([-25.0, -30.0, -20.0])
    couplings = np.array([1.0, 2.0])
    epsilon = np.array([1.0, 1.0, 1.0])
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    res = solve_robust_bp(model, target_node=1, budget=2.0, num_grid_steps=50)

    assert not np.isnan(res.lower_grid)
    assert not np.isnan(res.upper_grid)
    assert np.isclose(res.lower_grid, 0.0, atol=1e-6)
    assert np.isclose(res.upper_grid, 0.0, atol=1e-6)


def test_numerical_large_coupling():
    """Verify stability under very strong coupling (J = 15.0, 30.0)."""
    n = 3
    theta = np.array([0.5, -0.5, 0.2])
    couplings = np.array([15.0, 30.0])
    epsilon = np.array([0.5, 0.5, 0.5])
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    res = solve_robust_bp(model, target_node=1, budget=1.0, num_grid_steps=50)

    assert not np.isnan(res.lower_grid)
    assert not np.isnan(res.upper_grid)
    assert 0.0 <= res.lower_grid <= res.upper_grid <= 1.0
    assert 0.0 <= res.lower_certified <= res.upper_certified <= 1.0


def test_numerical_tiny_epsilon():
    """Verify stability under tiny perturbation limits (epsilon = 1e-8)."""
    n = 4
    theta = np.array([0.1, -0.2, 0.3, -0.1])
    couplings = np.array([0.5, 0.4, 0.6])
    epsilon = np.full(n, 1e-8)
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    res = solve_robust_bp(model, target_node=2, budget=1.0, num_grid_steps=50)

    assert np.isclose(res.lower_grid, res.nominal_marginal, atol=1e-6)
    assert np.isclose(res.upper_grid, res.nominal_marginal, atol=1e-6)


def test_single_node_tree():
    """Verify single-node graph (n = 1) degenerate handling."""
    model = TreeModel(
        num_nodes=1,
        theta=np.array([0.4]),
        edges=[],
        coupling={},
        epsilon=np.array([0.3])
    )

    res = solve_robust_bp(model, target_node=0, budget=0.2, num_grid_steps=20)

    assert not np.isnan(res.lower_grid)
    assert not np.isnan(res.upper_grid)
    # Target 0: delta in [-0.2, +0.2]
    # lower: theta - 0.2 = 0.2 -> sigmoid(2 * 0.2)
    # upper: theta + 0.2 = 0.6 -> sigmoid(2 * 0.6)
    from scipy.special import expit
    assert np.isclose(res.lower_grid, float(expit(2.0 * 0.2)), atol=1e-5)
    assert np.isclose(res.upper_grid, float(expit(2.0 * 0.6)), atol=1e-5)


def test_grid_refinement_shrinks_gap():
    """
    Increasing grid steps K must decrease discretization step Delta = B / K,
    causing the certification gap to shrink monotonically toward zero.
    """
    n = 4
    theta = np.array([0.2, -0.3, 0.4, -0.1])
    couplings = np.array([0.5, 0.3, 0.6])
    epsilon = np.array([0.4, 0.4, 0.4, 0.4])
    budget = 1.0
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    grid_resolutions = [10, 25, 50, 100, 200]
    gaps = []
    lower_certified = []
    upper_certified = []

    for k in grid_resolutions:
        res = solve_robust_bp(model, target_node=1, budget=budget, num_grid_steps=k)
        gaps.append(res.field_cert_gap)
        lower_certified.append(res.lower_certified)
        upper_certified.append(res.upper_certified)

        # Mathematical bounds must hold at every resolution
        assert 0.0 <= res.lower_certified <= res.lower_grid <= res.upper_grid <= res.upper_certified <= 1.0

    # Gaps must decrease strictly
    for i in range(len(grid_resolutions) - 1):
        assert gaps[i + 1] < gaps[i], f"Gap failed to decrease: {gaps[i]} -> {gaps[i+1]}"
        # Certified bounds must tighten (inner certified bounds expand outward to grid bounds)
        assert lower_certified[i + 1] >= lower_certified[i] - 1e-6
        assert upper_certified[i + 1] <= upper_certified[i] + 1e-6
