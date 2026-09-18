"""
Standard Belief Propagation for binary attractive tree models.

Uses numerically stable logcosh / logaddexp representation for the BP transfer function:
    f_J(x) = atanh(tanh(J) * tanh(x))
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.special import expit

from src.pgm.tree_model import TreeModel


def stable_f_J(J: float | np.ndarray, x: float | np.ndarray) -> float | np.ndarray:
    """
    Numerically stable computation of the BP cavity message transfer function:
        f_J(x) = atanh(tanh(J) * tanh(x))

    Mathematical Derivation:
        tanh(f_J(x)) = tanh(J) * tanh(x)
        Using the identity 2 * atanh(t) = ln((1+t)/(1-t)), and substituting t = tanh(J)*tanh(x):
            (1 + tanh(J) tanh(x)) / (1 - tanh(J) tanh(x))
            = (cosh(J)cosh(x) + sinh(J)sinh(x)) / (cosh(J)cosh(x) - sinh(J)sinh(x))
            = cosh(J + x) / cosh(J - x)

        Therefore:
            f_J(x) = 0.5 * (ln cosh(J + x) - ln cosh(J - x))

        Since ln cosh(z) = logaddexp(z, -z) - ln(2):
            f_J(x) = 0.5 * (logaddexp(J + x, -(J + x)) - logaddexp(J - x, -(J - x)))

    This formulation avoids division by zero, overflow, or NaN when |x| or J is large.
    For all J >= 0, -J <= f_J(x) <= J.

    Args:
        J: Attractive coupling strength (J >= 0).
        x: Cavity field input (real scalar or numpy array).

    Returns:
        Outgoing cavity field message.
    """
    J_arr = np.asarray(J, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)

    # Calculate 0.5 * (np.logaddexp(J + x, -(J + x)) - np.logaddexp(J - x, -(J - x)))
    term1 = np.logaddexp(J_arr + x_arr, -(J_arr + x_arr))
    term2 = np.logaddexp(J_arr - x_arr, -(J_arr - x_arr))
    result = 0.5 * (term1 - term2)

    if np.ndim(x) == 0 and np.ndim(J) == 0:
        return float(result)
    return result


@dataclass
class StandardBPResult:
    """
    Results of standard tree belief propagation.

    Attributes:
        effective_fields: Array of shape (n,) containing total field eta_i = theta_i + delta_i + sum_k nu_{k->i}.
        marginals: Array of shape (n,) containing marginal probabilities P(h_i = +1) = sigmoid(2 * eta_i).
        messages: Dict mapping directed edge (u, v) -> message nu_{u->v}.
    """
    effective_fields: np.ndarray
    marginals: np.ndarray
    messages: Dict[Tuple[int, int], float]


def run_standard_bp(
    model: TreeModel,
    delta: Optional[np.ndarray] = None
) -> StandardBPResult:
    """
    Run exact 2-pass Belief Propagation on a tree model to compute all marginals.

    Args:
        model: TreeModel instance.
        delta: Optional perturbation array of shape (num_nodes,). Defaults to 0.

    Returns:
        StandardBPResult containing effective fields, marginals P(h_i = +1), and directed messages.
    """
    n = model.num_nodes
    if n == 0:
        return StandardBPResult(
            effective_fields=np.zeros(0),
            marginals=np.zeros(0),
            messages={}
        )

    delta_arr = np.zeros(n, dtype=np.float64) if delta is None else np.asarray(delta, dtype=np.float64)
    theta_eff = model.theta + delta_arr

    if n == 1:
        eta = float(theta_eff[0])
        # P(h_0 = +1) = exp(eta) / (exp(eta) + exp(-eta)) = sigmoid(2 * eta)
        prob = float(expit(2.0 * eta))
        return StandardBPResult(
            effective_fields=np.array([eta]),
            marginals=np.array([prob]),
            messages={}
        )

    # Root arbitrarily at node 0 for 2-pass tree BP
    post_order, parent, children = model.get_rooted_tree(root=0)
    messages: Dict[Tuple[int, int], float] = {}

    # Pass 1: Leaves to root (bottom-up)
    for u in post_order:
        p = parent[u]
        if p is not None:
            # Cavity field at u excluding parent p: theta_eff[u] + sum_{c in children[u]} nu_{c -> u}
            x_u_cavity = theta_eff[u] + sum(messages[(c, u)] for c in children[u])
            J_up = model.get_coupling(u, p)
            messages[(u, p)] = float(stable_f_J(J_up, x_u_cavity))

    # Pass 2: Root to leaves (top-down)
    # Process nodes in reverse post-order (root to leaves)
    pre_order = post_order[::-1]
    for u in pre_order:
        for c in children[u]:
            # Message from u to child c requires cavity field at u excluding child c:
            # theta_eff[u] + (nu_{p -> u} if p is not None else 0) + sum_{other_c in children[u] \ {c}} nu_{other_c -> u}
            p = parent[u]
            x_u_cavity = theta_eff[u]
            if p is not None:
                x_u_cavity += messages[(p, u)]
            for other_c in children[u]:
                if other_c != c:
                    x_u_cavity += messages[(other_c, u)]
            J_uc = model.get_coupling(u, c)
            messages[(u, c)] = float(stable_f_J(J_uc, x_u_cavity))

    # Compute total belief field at each node eta_i = theta_eff[i] + sum_{k in adj[i]} nu_{k -> i}
    effective_fields = np.zeros(n, dtype=np.float64)
    marginals = np.zeros(n, dtype=np.float64)

    for i in range(n):
        eta_i = theta_eff[i] + sum(messages[(k, i)] for k in model.adj[i])
        effective_fields[i] = eta_i
        marginals[i] = float(expit(2.0 * eta_i))

    return StandardBPResult(
        effective_fields=effective_fields,
        marginals=marginals,
        messages=messages
    )


def compute_rooted_total_field(
    model: TreeModel,
    root: int,
    delta: Optional[np.ndarray] = None
) -> Tuple[float, float, Dict[Tuple[int, int], float]]:
    """
    Run 1-pass bottom-up BP to compute the total effective field and marginal at a specific root.

    Args:
        model: TreeModel instance.
        root: Root node index.
        delta: Perturbation vector.

    Returns:
        (eta_root, prob_root, messages_bottom_up)
    """
    n = model.num_nodes
    delta_arr = np.zeros(n, dtype=np.float64) if delta is None else np.asarray(delta, dtype=np.float64)
    theta_eff = model.theta + delta_arr

    if n == 1:
        eta = float(theta_eff[0])
        prob = float(expit(2.0 * eta))
        return eta, prob, {}

    post_order, parent, children = model.get_rooted_tree(root=root)
    messages: Dict[Tuple[int, int], float] = {}

    for u in post_order:
        p = parent[u]
        if p is not None:
            x_u_cavity = theta_eff[u] + sum(messages[(c, u)] for c in children[u])
            J_up = model.get_coupling(u, p)
            messages[(u, p)] = float(stable_f_J(J_up, x_u_cavity))

    eta_root = theta_eff[root] + sum(messages[(c, root)] for c in children[root])
    prob_root = float(expit(2.0 * eta_root))
    return float(eta_root), prob_root, messages


# Alias for compatibility
solve_tree_bp = run_standard_bp

