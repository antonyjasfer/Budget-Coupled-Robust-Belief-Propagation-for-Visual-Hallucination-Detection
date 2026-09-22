"""
Continuous error certification for budget-coupled robust Belief Propagation.

Calculates rigorous continuous bounds accounting for grid discretization step Delta = B / K.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.special import expit


@dataclass
class CertificateResult:
    """
    Certified continuous outer bounds (via Lipschitz discretization certificate) for the target node marginal.
    Attributes:
        lower_grid: Marginal probability on the discrete grid P_grid_min(h_target = +1).
        upper_grid: Marginal probability on the discrete grid P_grid_max(h_target = +1).
        lower_certified: Certified continuous lower bound P_cont_min(h_target = +1) <= lower_grid.
        upper_certified: Certified continuous upper bound P_cont_max(h_target = +1) >= upper_grid.
        field_lower_grid: Effective total field on grid (lower).
        field_upper_grid: Effective total field on grid (upper).
        field_cert_gap: Certified discretization field gap Delta * Lipschitz_const.
        grid_step: Discretization step Delta = B / K.
    """
    lower_grid: float
    upper_grid: float
    lower_certified: float
    upper_certified: float
    field_lower_grid: float
    field_upper_grid: float
    field_cert_gap: float
    grid_step: float


def compute_tree_discretization_gap(
    model,
    root: int,
    budget: float,
    grid_step: float
) -> float:
    """
    Compute rigorous continuous discretization error certificate gap for target root on a tree model.

    Mathematical Guarantee:
        Let delta* in U(B, eps) be any continuous perturbation.
        The floored vector delta_hat_i = floor(delta*_i / Delta) * Delta is feasible on the grid
        U_grid(B, eps, K) where Delta = B / K.
        Since the DP solver finds the global grid optimum:
            eta_r(delta_hat*) >= eta_r(delta_hat).
        By Mean Value Theorem and derivative non-negativity (M9A Theorem 1):
            eta_r(delta*) - eta_r(delta_hat*) <= sum_{i in V} gamma_i * r_i
        where r_i = delta*_i - delta_hat_i in [0, min(eps_i, Delta)], and
            gamma_i = prod_{e in path(i -> root)} tanh(J_e) <= 1
        is the exact upper bound on the partial derivative d eta_r / d delta_i.
        Furthermore, since sum_i r_i <= sum_i delta*_i <= B:
            gap = min(B, sum_{i in V} gamma_i * min(eps_i, Delta)).

    Args:
        model: TreeModel instance.
        root: Target root node index.
        budget: Total shared budget B.
        grid_step: Discretization step Delta = B / K.

    Returns:
        Rigorous continuous discretization gap for the effective field.
    """
    if budget <= 1e-12 or grid_step <= 1e-12:
        return 0.0

    n = model.num_nodes
    if n <= 1:
        return float(min(budget, min(float(model.epsilon[0]), grid_step)))

    gains = np.ones(n, dtype=np.float64)
    _, _, children = model.get_rooted_tree(root=root)
    queue = [root]
    while queue:
        u = queue.pop(0)
        for c in children[u]:
            J = model.get_coupling(u, c)
            gains[c] = gains[u] * float(np.tanh(J))
            queue.append(c)

    coord_gaps = gains * np.minimum(model.epsilon, grid_step)
    gap = float(min(budget, np.sum(coord_gaps)))
    return gap


def compute_continuous_certificate(
    field_lower_grid: float,
    field_upper_grid: float,
    grid_step: float,
    lipschitz_const: float = 1.0,
    cert_gap: Optional[float] = None
) -> CertificateResult:
    """
    Compute certified continuous bounds from grid effective fields.

    Mathematical guarantee:
        Account for continuous perturbation discretization residuals bounded by cert_gap:
        lower_field_cont >= field_lower_grid - cert_gap
        upper_field_cont <= field_upper_grid + cert_gap

    Args:
        field_lower_grid: Effective field from grid solver (lower).
        field_upper_grid: Effective field from grid solver (upper).
        grid_step: Discretization step Delta = B / K.
        lipschitz_const: Fallback Lipschitz multiplier (default 1.0).
        cert_gap: Optional rigorous continuous field gap. If None, defaults to grid_step * lipschitz_const.

    Returns:
        CertificateResult with separated grid and continuous certified bounds.
    """
    if cert_gap is None:
        cert_gap = float(grid_step * lipschitz_const)
    else:
        cert_gap = float(cert_gap)

    lower_grid_prob = float(expit(2.0 * field_lower_grid))
    upper_grid_prob = float(expit(2.0 * field_upper_grid))

    field_lower_cert = field_lower_grid - cert_gap
    field_upper_cert = field_upper_grid + cert_gap

    lower_cert_prob = float(expit(2.0 * field_lower_cert))
    upper_cert_prob = float(expit(2.0 * field_upper_cert))

    # Clamp to [0, 1] and ensure strict sandwich: lower_cert <= lower_grid <= upper_grid <= upper_cert
    lower_cert_prob = min(lower_cert_prob, lower_grid_prob)
    upper_cert_prob = max(upper_cert_prob, upper_grid_prob)

    return CertificateResult(
        lower_grid=lower_grid_prob,
        upper_grid=upper_grid_prob,
        lower_certified=lower_cert_prob,
        upper_certified=upper_cert_prob,
        field_lower_grid=field_lower_grid,
        field_upper_grid=field_upper_grid,
        field_cert_gap=cert_gap,
        grid_step=grid_step
    )


def compute_one_node_continuous_bounds(
    theta: float,
    epsilon: float,
    budget: float
) -> Tuple[float, float]:
    """
    Compute exact continuous analytical bounds for a single isolated node (N=1).

    For single node:
        delta_max = min(epsilon, budget)
        delta_min = -min(epsilon, budget)
        upper_prob = sigmoid(2 * (theta + delta_max))
        lower_prob = sigmoid(2 * (theta + delta_min))

    Returns:
        (exact_lower_prob, exact_upper_prob)
    """
    delta_eff = min(float(epsilon), float(budget))
    eta_upper = theta + delta_eff
    eta_lower = theta - delta_eff
    return float(expit(2.0 * eta_lower)), float(expit(2.0 * eta_upper))
