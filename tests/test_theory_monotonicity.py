"""
Tests verifying the formal monotonicity properties of attractive Ising models:
- Cov(h_r, h_i) >= 0 under attractive couplings (J_ij >= 0).
- d E[h_r] / d theta_i = Cov(h_r, h_i).
- Finite differences match analytical covariances.
- Monotonicity holds for arbitrary mixed-sign base unary fields theta.
- Counterexample verification: non-attractive couplings (J < 0) can violate monotonicity.
"""

import pytest
import numpy as np

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import run_standard_bp
from src.pgm.brute_force import run_brute_force_inference


def compute_exact_covariance_matrix(model: TreeModel) -> np.ndarray:
    """Compute exact Cov(h_i, h_j) matrix via 2^N state enumeration."""
    n = model.num_nodes
    # Generate all states
    states = np.array(
        np.meshgrid(*[[-1.0, 1.0]] * n)
    ).T.reshape(-1, n)

    unary = states @ model.theta
    pairwise = np.zeros(len(states))
    for u, v in model.edges:
        pairwise += model.get_coupling(u, v) * states[:, u] * states[:, v]

    log_weights = unary + pairwise
    max_lw = np.max(log_weights)
    weights = np.exp(log_weights - max_lw)
    probs = weights / np.sum(weights)

    # E[h]
    e_h = probs @ states
    # E[h_i h_j]
    e_hh = states.T @ (states * probs[:, None])
    cov = e_hh - np.outer(e_h, e_h)
    return cov


def test_covariance_identity_and_non_negativity_chain():
    """Verify d E[h_r] / d theta_i = Cov(h_r, h_i) >= 0 on a 4-node chain."""
    n = 4
    rng = np.random.default_rng(42)
    theta = rng.uniform(-1.5, 1.5, size=n)
    couplings = rng.uniform(0.1, 1.2, size=n - 1)

    model = create_chain_tree(n, theta, couplings)
    cov = compute_exact_covariance_matrix(model)

    # In an attractive model, all pairwise covariances must be non-negative
    assert np.all(cov >= -1e-12), f"Attractive Ising model produced negative covariance: {cov.min()}"

    # Verify finite difference derivative d E[h_r] / d theta_i
    eps_diff = 1e-6
    for i in range(n):
        theta_plus = theta.copy()
        theta_plus[i] += eps_diff
        model_plus = create_chain_tree(n, theta_plus, couplings)
        res_plus = run_brute_force_inference(model_plus)
        e_plus = 2.0 * res_plus.marginals - 1.0

        theta_minus = theta.copy()
        theta_minus[i] -= eps_diff
        model_minus = create_chain_tree(n, theta_minus, couplings)
        res_minus = run_brute_force_inference(model_minus)
        e_minus = 2.0 * res_minus.marginals - 1.0

        numerical_deriv = (e_plus - e_minus) / (2.0 * eps_diff)
        analytical_cov = cov[:, i]

        assert np.allclose(numerical_deriv, analytical_cov, atol=1e-5), (
            f"Derivative mismatch at coordinate {i}: {numerical_deriv} vs {analytical_cov}"
        )


def test_monotonicity_under_arbitrary_mixed_signs():
    """Verify that increasing theta_i strictly increases (or preserves) P(H_r = +1)."""
    n = 5
    # Mixed positive and negative fields
    theta = np.array([-2.5, 1.8, -0.4, 3.2, -1.0])
    couplings = np.array([0.5, 0.8, 0.3, 0.7])
    model = create_chain_tree(n, theta, couplings)

    base_res = run_standard_bp(model)

    # Increase field at each node by various positive deltas
    for pert_node in range(n):
        for delta in [0.05, 0.2, 0.5, 1.0, 2.5]:
            theta_pert = theta.copy()
            theta_pert[pert_node] += delta
            model_pert = create_chain_tree(n, theta_pert, couplings)
            pert_res = run_standard_bp(model_pert)

            for target in range(n):
                p_base = base_res.marginals[target]
                p_pert = pert_res.marginals[target]
                assert p_pert >= p_base - 1e-12, (
                    f"Target {target} decreased from {p_base} to {p_pert} when node {pert_node} increased by {delta}"
                )


def test_monotonicity_star_tree():
    """Verify monotonicity on a star tree with 1 center and 4 leaves."""
    num_leaves = 4
    center_theta = -0.8
    leaf_thetas = np.array([1.2, -1.5, 0.5, -0.3])
    couplings = np.array([0.4, 0.9, 0.2, 0.6])

    model = create_star_tree(num_leaves, center_theta, leaf_thetas, couplings)
    cov = compute_exact_covariance_matrix(model)

    assert np.all(cov >= -1e-12)

    # Decreasing any field must never increase target marginal
    base_res = run_standard_bp(model)
    for pert_node in range(model.num_nodes):
        delta = -0.4
        theta_pert = model.theta.copy()
        theta_pert[pert_node] += delta
        model_pert = TreeModel(
            num_nodes=model.num_nodes,
            theta=theta_pert,
            edges=model.edges,
            coupling=model.coupling
        )
        pert_res = run_standard_bp(model_pert)

        for target in range(model.num_nodes):
            assert pert_res.marginals[target] <= base_res.marginals[target] + 1e-12


def test_counterexample_antiferromagnetic_violates_monotonicity():
    """Verify that when J < 0 (antiferromagnetic), covariance CAN be negative, violating monotonicity."""
    # 2-node graph with J < 0
    # h1 and h2 prefer opposite spins
    states = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    theta = np.array([0.0, 0.0])
    J_anti = -1.5  # Repulsive interaction

    pairwise = J_anti * states[:, 0] * states[:, 1]
    weights = np.exp(pairwise)
    probs = weights / np.sum(weights)

    e_h = probs @ states
    e_hh = states.T @ (states * probs[:, None])
    cov = e_hh - np.outer(e_h, e_h)

    # Cov(h_0, h_1) must be strictly negative!
    assert cov[0, 1] < -0.1, f"Expected negative covariance, got {cov[0, 1]}"

    # Therefore, increasing theta_1 DECREASES P(h_0 = +1)
    weights_pert = np.exp(pairwise + 1.0 * states[:, 1])  # delta_1 = +1
    probs_pert = weights_pert / np.sum(weights_pert)
    p0_orig = float(np.sum(probs[states[:, 0] == 1.0]))
    p0_pert = float(np.sum(probs_pert[states[:, 0] == 1.0]))

    assert p0_pert < p0_orig, "Antiferromagnetic interaction should decrease target marginal"
