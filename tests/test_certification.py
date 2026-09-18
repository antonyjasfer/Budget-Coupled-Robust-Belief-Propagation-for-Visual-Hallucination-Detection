"""
Tests for continuous-error certification and one-node analytical bounds.
"""

import pytest
import numpy as np
from scipy.special import expit

from src.pgm.tree_model import TreeModel, create_chain_tree
from src.robust_bp.certification import compute_one_node_continuous_bounds, compute_continuous_certificate
from src.robust_bp.solver import solve_robust_bp


def test_one_node_continuous_bounds_analytical():
    """Verify analytical 1-node continuous bounds match theory."""
    theta = 0.5
    eps = 0.8
    budget = 0.3

    # Effective delta is min(0.8, 0.3) = 0.3
    low, up = compute_one_node_continuous_bounds(theta, eps, budget)
    expected_low = float(expit(2.0 * (0.5 - 0.3)))
    expected_up = float(expit(2.0 * (0.5 + 0.3)))

    assert np.isclose(low, expected_low, atol=1e-12)
    assert np.isclose(up, expected_up, atol=1e-12)


def test_one_node_solver_recovers_analytical():
    """Single node TreeModel solved via robust BP matches analytical bounds."""
    theta = np.array([-0.2])
    eps = np.array([0.6])
    budget = 0.4

    model = TreeModel(num_nodes=1, theta=theta, epsilon=eps)
    res = solve_robust_bp(model, target_node=0, budget=budget, num_grid_steps=100)

    exact_low, exact_up = compute_one_node_continuous_bounds(theta[0], eps[0], budget)

    assert np.isclose(res.lower_grid, exact_low, atol=1e-10)
    assert np.isclose(res.upper_grid, exact_up, atol=1e-10)


def test_certified_bounds_sandwich():
    """Certified bounds must strictly sandwich grid bounds."""
    n = 3
    theta = np.array([0.0, 0.0, 0.0])
    couplings = np.array([0.5, 0.5])
    epsilon = np.array([0.4, 0.4, 0.4])
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    res = solve_robust_bp(model, target_node=1, budget=0.6, num_grid_steps=50)

    assert res.lower_certified <= res.lower_grid
    assert res.lower_grid <= res.upper_grid
    assert res.upper_grid <= res.upper_certified


def test_grid_refinement_convergence():
    """As number of grid steps K -> infinity (Delta -> 0), certified bounds converge to grid bounds."""
    n = 3
    theta = np.array([0.2, -0.1, 0.3])
    couplings = np.array([0.4, 0.4])
    epsilon = np.array([0.3, 0.3, 0.3])
    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)

    res_coarse = solve_robust_bp(model, target_node=1, budget=0.5, num_grid_steps=10)
    res_fine = solve_robust_bp(model, target_node=1, budget=0.5, num_grid_steps=500)

    gap_coarse = res_coarse.upper_certified - res_coarse.lower_certified
    gap_fine = res_fine.upper_certified - res_fine.lower_certified

    # Certified gap tightens as grid is refined
    assert res_fine.field_cert_gap < res_coarse.field_cert_gap
    assert np.isclose(res_fine.lower_grid, res_fine.lower_certified, atol=0.01)
    assert np.isclose(res_fine.upper_grid, res_fine.upper_certified, atol=0.01)
