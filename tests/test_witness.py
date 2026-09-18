"""
Tests for witness / backpointer reconstruction and validation.
"""

import pytest
import numpy as np

from src.pgm.tree_model import create_chain_tree, create_star_tree
from src.pgm.standard_bp import compute_rooted_total_field
from src.robust_bp.solver import solve_robust_bp


def test_witness_reconstruction_3node_chain():
    """Witness reconstruction on 3-node chain satisfies budget and reproduces optimal fields."""
    n = 3
    theta = np.array([0.0, 0.0, 0.0])
    couplings = np.array([np.arctanh(0.5), np.arctanh(0.5)])
    epsilon = np.full(n, np.arctanh(0.5))
    budget = float(np.arctanh(0.5))

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    target = 1  # Center node

    res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=100)

    # Upper witness validation
    wit_up = res.witness_upper
    assert wit_up.is_budget_valid, f"Budget violated: {wit_up.total_budget_spent} > {budget}"
    assert wit_up.is_box_valid, f"Box bound violated: {wit_up.delta}"
    assert wit_up.field_match, f"Field mismatch: BP={wit_up.witness_field}, DP={wit_up.predicted_grid_field}"
    assert np.all(wit_up.delta >= -1e-9), "Upper witness must be non-negative"

    # Lower witness validation
    wit_low = res.witness_lower
    assert wit_low.is_budget_valid, f"Budget violated: {wit_low.total_budget_spent} > {budget}"
    assert wit_low.is_box_valid, f"Box bound violated: {wit_low.delta}"
    assert wit_low.field_match, f"Field mismatch: BP={wit_low.witness_field}, DP={wit_low.predicted_grid_field}"
    assert np.all(wit_low.delta <= 1e-9), "Lower witness must be non-positive"


def test_witness_reconstruction_star_tree():
    """Witness reconstruction on star tree with multiple children satisfies all constraints."""
    num_leaves = 4
    center_theta = 0.1
    leaf_thetas = np.array([-0.2, 0.3, -0.1, 0.2])
    couplings = np.array([0.4, 0.5, 0.3, 0.6])
    epsilon = np.array([0.3, 0.25, 0.25, 0.25, 0.25])
    budget = 0.5

    model = create_star_tree(num_leaves, center_theta, leaf_thetas, couplings, epsilon=epsilon)
    target = 0  # Center root

    res = solve_robust_bp(model, target_node=target, budget=budget, num_grid_steps=80)

    for wit, sign_name in [(res.witness_upper, "upper"), (res.witness_lower, "lower")]:
        assert wit.is_budget_valid
        assert wit.is_box_valid
        assert wit.field_match

        # Direct verification with independent standard BP call
        eta_check, _, _ = compute_rooted_total_field(model, root=target, delta=wit.delta)
        assert np.isclose(eta_check, wit.predicted_grid_field, atol=1e-5)
