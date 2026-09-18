"""
Continuous error certification for budget-coupled robust Belief Propagation.

Calculates rigorous continuous bounds accounting for grid discretization step Delta = B / K.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.special import expit


@dataclass
class CertificateResult:
    """
    Certified continuous confidence bounds for the target node marginal.

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


def compute_continuous_certificate(
    field_lower_grid: float,
    field_upper_grid: float,
    grid_step: float,
    lipschitz_const: float = 1.0
) -> CertificateResult:
    """
    Compute certified continuous bounds from grid effective fields.

    Mathematical guarantee:
        Because |d f_J(x)/dx| <= tanh(J) <= 1, the Lipschitz constant of effective field
        with respect to any continuous perturbation under tree BP is bounded by 1.0.
        Therefore, the discretization error in effective field for any continuous perturbation
        rounding to the grid is at most Delta * lipschitz_const.

        lower_field_cont >= field_lower_grid - field_cert_gap
        upper_field_cont <= field_upper_grid + field_cert_gap

    Args:
        field_lower_grid: Effective field from grid solver (lower).
        field_upper_grid: Effective field from grid solver (upper).
        grid_step: Discretization step Delta = B / K.
        lipschitz_const: Lipschitz constant (default 1.0 for tree Ising model).

    Returns:
        CertificateResult with separated grid and continuous certified bounds.
    """
    cert_gap = float(grid_step * lipschitz_const)

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
