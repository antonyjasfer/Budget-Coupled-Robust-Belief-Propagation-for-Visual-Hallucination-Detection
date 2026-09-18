"""
Unit tests for Standard Belief Propagation and comparison against exact brute-force inference.
"""

import pytest
import numpy as np

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import stable_f_J, run_standard_bp, compute_rooted_total_field
from src.pgm.brute_force import run_brute_force_inference


def test_stable_f_J_properties():
    """Verify numerical stability and exact identities of stable_f_J."""
    # f_J(0) == 0
    assert np.isclose(stable_f_J(0.5, 0.0), 0.0)
    # f_0(x) == 0
    assert np.isclose(stable_f_J(0.0, 10.0), 0.0)
    # Sign symmetry: f_J(-x) == -f_J(x)
    assert np.isclose(stable_f_J(0.7, -1.2), -stable_f_J(0.7, 1.2))

    # Known analytical value: atanh(tanh(atanh(0.5)) * tanh(atanh(0.5))) = atanh(0.25)
    j_val = np.arctanh(0.5)
    x_val = np.arctanh(0.5)
    expected = np.arctanh(0.25)
    assert np.isclose(stable_f_J(j_val, x_val), expected, atol=1e-12)

    # Numerical stability with extreme inputs (no NaN or inf)
    extreme_large = stable_f_J(0.8, 1e8)
    assert np.isclose(extreme_large, 0.8, atol=1e-7)

    extreme_neg = stable_f_J(0.8, -1e8)
    assert np.isclose(extreme_neg, -0.8, atol=1e-7)


def test_standard_bp_single_node():
    """Standard BP on single node matches analytical sigmoid(2*theta)."""
    model = TreeModel(num_nodes=1, theta=np.array([0.6]))
    res = run_standard_bp(model)
    expected_prob = 1.0 / (1.0 + np.exp(-2.0 * 0.6))
    assert np.isclose(res.marginals[0], expected_prob, atol=1e-12)
    assert np.isclose(res.effective_fields[0], 0.6, atol=1e-12)


@pytest.mark.parametrize("num_nodes", [2, 3, 4, 5, 6, 7])
def test_standard_bp_chain_matches_brute_force(num_nodes):
    """Standard BP on random chain trees must match exact brute force."""
    rng = np.random.default_rng(42 + num_nodes)
    theta = rng.uniform(-1.0, 1.0, size=num_nodes)
    couplings = rng.uniform(0.1, 1.5, size=num_nodes - 1)

    model = create_chain_tree(num_nodes, theta, couplings)

    bp_res = run_standard_bp(model)
    bf_res = run_brute_force_inference(model)

    assert np.allclose(bp_res.marginals, bf_res.marginals, atol=1e-10)
    assert np.allclose(bp_res.effective_fields, bf_res.effective_fields, atol=1e-10)


def test_standard_bp_star_matches_brute_force():
    """Standard BP on star tree matches exact brute force."""
    rng = np.random.default_rng(123)
    num_leaves = 5
    center_theta = 0.3
    leaf_thetas = rng.uniform(-0.8, 0.8, size=num_leaves)
    couplings = rng.uniform(0.2, 1.2, size=num_leaves)

    model = create_star_tree(num_leaves, center_theta, leaf_thetas, couplings)

    bp_res = run_standard_bp(model)
    bf_res = run_brute_force_inference(model)

    assert np.allclose(bp_res.marginals, bf_res.marginals, atol=1e-10)
    assert np.allclose(bp_res.effective_fields, bf_res.effective_fields, atol=1e-10)


def test_standard_bp_rooted_bottom_up_consistency():
    """compute_rooted_total_field should match full 2-pass BP at root."""
    rng = np.random.default_rng(789)
    n = 6
    theta = rng.uniform(-0.5, 0.5, size=n)
    couplings = rng.uniform(0.1, 1.0, size=n - 1)
    model = create_chain_tree(n, theta, couplings)

    bp_full = run_standard_bp(model)

    for root_idx in range(n):
        eta_root, prob_root, _ = compute_rooted_total_field(model, root=root_idx)
        assert np.isclose(eta_root, bp_full.effective_fields[root_idx], atol=1e-11)
        assert np.isclose(prob_root, bp_full.marginals[root_idx], atol=1e-11)


def test_standard_bp_with_perturbation():
    """Standard BP with delta perturbation matches brute-force with delta."""
    n = 4
    theta = np.array([0.1, -0.2, 0.3, -0.1])
    couplings = np.array([0.4, 0.7, 0.5])
    delta = np.array([0.2, -0.1, 0.15, -0.05])

    model = create_chain_tree(n, theta, couplings)

    bp_res = run_standard_bp(model, delta=delta)
    bf_res = run_brute_force_inference(model, delta=delta)

    assert np.allclose(bp_res.marginals, bf_res.marginals, atol=1e-10)
