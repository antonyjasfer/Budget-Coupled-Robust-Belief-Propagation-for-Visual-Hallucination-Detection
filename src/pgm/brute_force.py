"""
Brute-force exact inference for binary attractive tree (and general small) Ising models.

Provides exact ground truth for small graphs (N <= 15) by enumerating all 2^N spin configurations.
Also includes brute-force grid enumeration for robust budget-constrained perturbations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Generator
import itertools
import numpy as np
from scipy.special import logsumexp, expit

from src.pgm.tree_model import TreeModel


@dataclass
class ExactInferenceResult:
    """
    Exact ground truth results from exhaustive 2^N configuration enumeration.

    Attributes:
        log_partition_function: ln(Z)
        marginals: Array of exact marginal probabilities P(h_i = +1) for each node i.
        effective_fields: Array of exact effective fields eta_i where P(h_i = +1) = sigmoid(2 * eta_i).
    """
    log_partition_function: float
    marginals: np.ndarray
    effective_fields: np.ndarray


def run_brute_force_inference(
    model: TreeModel,
    delta: Optional[np.ndarray] = None
) -> ExactInferenceResult:
    """
    Compute exact partition function and marginals by iterating over all 2^N states.

    Args:
        model: TreeModel instance.
        delta: Optional perturbation array of shape (num_nodes,).

    Returns:
        ExactInferenceResult with exact marginals and effective fields.
    """
    n = model.num_nodes
    if n == 0:
        return ExactInferenceResult(
            log_partition_function=0.0,
            marginals=np.zeros(0),
            effective_fields=np.zeros(0)
        )

    if n > 20:
        raise ValueError(f"Tree size N={n} too large for brute force enumeration (2^{n} states)")

    delta_arr = np.zeros(n, dtype=np.float64) if delta is None else np.asarray(delta, dtype=np.float64)
    theta_eff = model.theta + delta_arr

    # Generate all 2^N binary states in {-1, +1}^N
    # Shape: (2^N, N)
    states = np.array(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=np.float64)

    # Compute unnormalized log probability (Hamiltonian score) for each state
    # Unary: states @ theta_eff -> shape (2^N,)
    unary_scores = states @ theta_eff

    # Pairwise: sum_{(u,v)} J_uv * h_u * h_v
    pairwise_scores = np.zeros(len(states), dtype=np.float64)
    for u, v in model.edges:
        j_val = model.get_coupling(u, v)
        pairwise_scores += j_val * (states[:, u] * states[:, v])

    log_weights = unary_scores + pairwise_scores
    log_z = float(logsumexp(log_weights))

    # Normalized state probabilities
    probs = np.exp(log_weights - log_z)

    # Marginals P(h_i = +1)
    marginals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask_pos = (states[:, i] == 1.0)
        marginals[i] = float(np.sum(probs[mask_pos]))

    # Compute effective fields: eta_i = 0.5 * ln(P(+1) / P(-1))
    # Note: P(+1) = sigmoid(2 * eta) => 2*eta = ln(P(+1) / (1 - P(+1)))
    effective_fields = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p = np.clip(marginals[i], 1e-15, 1.0 - 1e-15)
        effective_fields[i] = 0.5 * float(np.log(p / (1.0 - p)))

    return ExactInferenceResult(
        log_partition_function=log_z,
        marginals=marginals,
        effective_fields=effective_fields
    )


def enumerate_grid_perturbations(
    num_nodes: int,
    epsilon: np.ndarray,
    budget: float,
    num_grid_steps: int,
    sign: int = 1
) -> Generator[np.ndarray, None, None]:
    """
    Generate all valid discretized perturbation vectors on the grid.

    Args:
        num_nodes: Number of nodes.
        epsilon: Max per-node perturbations.
        budget: Total shared budget B.
        num_grid_steps: Number of divisions K of the budget (step = budget / K).
        sign: +1 for upper bound (0 <= delta_i <= eps_i), -1 for lower bound (-eps_i <= delta_i <= 0).

    Yields:
        Valid perturbation numpy array of shape (num_nodes,).
    """
    if budget <= 1e-12 or num_grid_steps == 0:
        yield np.zeros(num_nodes, dtype=np.float64)
        return

    step = budget / float(num_grid_steps)
    max_k_per_node = [int(np.floor(min(eps, budget) / step + 1e-8)) for eps in epsilon]

    # Recursive generator for integer budget allocation
    def _allocate(node_idx: int, rem_k: int) -> Generator[List[int], None, None]:
        if node_idx == num_nodes - 1:
            max_alloc = min(max_k_per_node[node_idx], rem_k)
            for k in range(max_alloc + 1):
                yield [k]
        else:
            max_alloc = min(max_k_per_node[node_idx], rem_k)
            for k in range(max_alloc + 1):
                for sub in _allocate(node_idx + 1, rem_k - k):
                    yield [k] + sub

    for k_vector in _allocate(0, num_grid_steps):
        delta = sign * np.array(k_vector, dtype=np.float64) * step
        yield delta


def run_brute_force_robust_grid(
    model: TreeModel,
    target_node: int,
    budget: float,
    num_grid_steps: int
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute exact robust bounds on a tiny tree by exhaustive grid search over all perturbation vectors.

    Args:
        model: TreeModel instance.
        target_node: Node index of interest.
        budget: Shared budget B.
        num_grid_steps: Number of grid steps K.

    Returns:
        (lower_grid_prob, upper_grid_prob, best_lower_delta, best_upper_delta)
    """
    upper_prob = -np.inf
    upper_delta = np.zeros(model.num_nodes)

    lower_prob = np.inf
    lower_delta = np.zeros(model.num_nodes)

    # Upper bound search (delta >= 0)
    for delta in enumerate_grid_perturbations(model.num_nodes, model.epsilon, budget, num_grid_steps, sign=+1):
        res = run_brute_force_inference(model, delta=delta)
        p = res.marginals[target_node]
        if p > upper_prob:
            upper_prob = p
            upper_delta = delta.copy()

    # Lower bound search (delta <= 0)
    for delta in enumerate_grid_perturbations(model.num_nodes, model.epsilon, budget, num_grid_steps, sign=-1):
        res = run_brute_force_inference(model, delta=delta)
        p = res.marginals[target_node]
        if p < lower_prob:
            lower_prob = p
            lower_delta = delta.copy()

    return float(lower_prob), float(upper_prob), lower_delta, upper_delta
