"""
Budget-Coupled Robust Belief Propagation solver for binary attractive tree models.

Implements exact budget-indexed dynamic programming (exact over the discretized
uncertainty set) via max-plus and min-plus convolutions over trees, with continuous
Lipschitz discretization certification and witness extraction.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.special import expit

from src.pgm.tree_model import TreeModel
from src.pgm.standard_bp import stable_f_J, compute_rooted_total_field
from src.robust_bp.budget_convolution import max_plus_convolve, min_plus_convolve
from src.robust_bp.certification import compute_continuous_certificate, compute_tree_discretization_gap, CertificateResult
from src.robust_bp.witness import trace_witness_perturbation, validate_witness, WitnessResult


@dataclass
class RobustBPResult:
    """
    Complete output of Budget-Coupled Robust Belief Propagation.

    Attributes:
        lower_grid: Discretized grid lower bound probability P(h_target = +1).
        upper_grid: Discretized grid upper bound probability P(h_target = +1).
        lower_certified: Rigorous continuous certified lower bound probability.
        upper_certified: Rigorous continuous certified upper bound probability.
        nominal_marginal: Nominal marginal probability (delta = 0).
        field_nominal: Nominal total effective field (delta = 0).
        field_lower_grid: Discretized grid lower effective field.
        field_upper_grid: Discretized grid upper effective field.
        field_cert_gap: Discretization error certificate gap.
        budget: Shared budget limit B.
        num_grid_steps: Number of grid discretization steps K.
        target_node: Target root node index.
        witness_upper: Witness perturbation vector and validation for upper bound.
        witness_lower: Witness perturbation vector and validation for lower bound.
        budget_grid: Array of shape (K + 1,) containing budget points [0, Delta, ..., B].
        upper_field_profile: Array of shape (K + 1,) containing max field vs budget.
        lower_field_profile: Array of shape (K + 1,) containing min field vs budget.
    """
    lower_grid: float
    upper_grid: float
    lower_certified: float
    upper_certified: float
    nominal_marginal: float
    field_nominal: float
    field_lower_grid: float
    field_upper_grid: float
    field_cert_gap: float
    budget: float
    num_grid_steps: int
    target_node: int
    witness_upper: WitnessResult
    witness_lower: WitnessResult
    budget_grid: np.ndarray
    upper_field_profile: np.ndarray
    lower_field_profile: np.ndarray


def solve_robust_bp(
    model: TreeModel,
    target_node: int,
    budget: float,
    num_grid_steps: int = 100,
    lipschitz_const: float = 1.0
) -> RobustBPResult:
    """
    Solve budget-coupled robust Belief Propagation on a tree for a specified target node.

    Args:
        model: TreeModel instance.
        target_node: Index of the target node to evaluate.
        budget: Total shared perturbation budget B >= 0.
        num_grid_steps: Number of discretization intervals K (step Delta = B / K).
        lipschitz_const: Lipschitz constant for continuous error certification (default 1.0).

    Returns:
        RobustBPResult containing grid bounds, continuous certified bounds, profiles, and witnesses.
    """
    if budget < 0:
        raise ValueError(f"Budget B must be non-negative, got {budget}")
    if target_node < 0 or target_node >= model.num_nodes:
        raise ValueError(f"Invalid target node {target_node}")

    n = model.num_nodes
    K = max(0, int(num_grid_steps)) if budget > 1e-12 else 0
    grid_step = float(budget / K) if K > 0 else 0.0
    budget_grid = np.linspace(0.0, budget, K + 1) if K > 0 else np.array([0.0])

    # Compute nominal inference
    field_nom, prob_nom, _ = compute_rooted_total_field(model, root=target_node, delta=None)

    if budget <= 1e-12 or K == 0 or np.all(model.epsilon <= 1e-12):
        # Degenerate / zero budget case -> nominal
        zero_delta = np.zeros(n)
        wit_nom = validate_witness(model, target_node, zero_delta, budget, field_nom)
        return RobustBPResult(
            lower_grid=prob_nom,
            upper_grid=prob_nom,
            lower_certified=prob_nom,
            upper_certified=prob_nom,
            nominal_marginal=prob_nom,
            field_nominal=field_nom,
            field_lower_grid=field_nom,
            field_upper_grid=field_nom,
            field_cert_gap=0.0,
            budget=budget,
            num_grid_steps=K,
            target_node=target_node,
            witness_upper=wit_nom,
            witness_lower=wit_nom,
            budget_grid=budget_grid,
            upper_field_profile=np.array([field_nom]),
            lower_field_profile=np.array([field_nom])
        )

    post_order, parent, children = model.get_rooted_tree(root=target_node)

    # -------------------------------------------------------------
    # 1. UPPER BOUND DP (Max-Plus Tree Convolution)
    # -------------------------------------------------------------
    # messages_upper[(u, p)] is array of shape (K + 1,)
    messages_upper: Dict[Tuple[int, int], np.ndarray] = {}
    local_bp_upper: Dict[int, np.ndarray] = {}
    child_bp_upper: Dict[Tuple[int, int], np.ndarray] = {}
    node_x_upper: Dict[int, np.ndarray] = {}

    for u in post_order:
        c_list = children[u]
        d = len(c_list)

        # Aggregate children messages using max-plus convolution
        if d == 0:
            v_children = np.zeros(K + 1, dtype=np.float64)
        elif d == 1:
            v_children = messages_upper[(c_list[0], u)].copy()
        else:
            v_accum = messages_upper[(c_list[0], u)].copy()
            for l in range(1, d):
                c = c_list[l]
                v_accum, bp = max_plus_convolve(v_accum, messages_upper[(c, u)])
                child_bp_upper[(u, c)] = bp
            v_children = v_accum

        # Optimize local perturbation delta_u in [0, min(eps_u, available_budget)]
        # For each total budget index m in 0..K:
        # local_k ranges in 0..min(floor(eps_u / grid_step), m)
        max_k_local = int(np.floor(model.epsilon[u] / grid_step + 1e-8))
        x_u = np.zeros(K + 1, dtype=np.float64)
        local_bp = np.zeros(K + 1, dtype=np.int64)

        for m in range(K + 1):
            limit_k = min(max_k_local, m)
            k_choices = np.arange(limit_k + 1, dtype=np.int64)
            candidates = k_choices * grid_step + v_children[m - k_choices]
            best_k = int(np.argmax(candidates))
            x_u[m] = model.theta[u] + candidates[best_k]
            local_bp[m] = best_k

        local_bp_upper[u] = local_bp
        node_x_upper[u] = x_u

        p = parent[u]
        if p is not None:
            J_up = model.get_coupling(u, p)
            messages_upper[(u, p)] = np.asarray(stable_f_J(J_up, x_u), dtype=np.float64)

    field_upper_grid = float(node_x_upper[target_node][K])
    upper_field_profile = node_x_upper[target_node].copy()

    # Trace upper witness
    delta_upper = trace_witness_perturbation(
        model, target_node, children, local_bp_upper, child_bp_upper, K, grid_step, sign=+1
    )
    wit_upper = validate_witness(model, target_node, delta_upper, budget, field_upper_grid)

    # -------------------------------------------------------------
    # 2. LOWER BOUND DP (Min-Plus Tree Convolution)
    # -------------------------------------------------------------
    messages_lower: Dict[Tuple[int, int], np.ndarray] = {}
    local_bp_lower: Dict[int, np.ndarray] = {}
    child_bp_lower: Dict[Tuple[int, int], np.ndarray] = {}
    node_x_lower: Dict[int, np.ndarray] = {}

    for u in post_order:
        c_list = children[u]
        d = len(c_list)

        # Aggregate children messages using min-plus convolution
        if d == 0:
            v_children = np.zeros(K + 1, dtype=np.float64)
        elif d == 1:
            v_children = messages_lower[(c_list[0], u)].copy()
        else:
            v_accum = messages_lower[(c_list[0], u)].copy()
            for l in range(1, d):
                c = c_list[l]
                v_accum, bp = min_plus_convolve(v_accum, messages_lower[(c, u)])
                child_bp_lower[(u, c)] = bp
            v_children = v_accum

        # Local perturbation delta_u in [-min(eps_u, available_budget), 0]
        max_k_local = int(np.floor(model.epsilon[u] / grid_step + 1e-8))
        x_u = np.zeros(K + 1, dtype=np.float64)
        local_bp = np.zeros(K + 1, dtype=np.int64)

        for m in range(K + 1):
            limit_k = min(max_k_local, m)
            k_choices = np.arange(limit_k + 1, dtype=np.int64)
            # perturbation is -k_choices * grid_step
            candidates = -k_choices * grid_step + v_children[m - k_choices]
            best_k = int(np.argmin(candidates))
            x_u[m] = model.theta[u] + candidates[best_k]
            local_bp[m] = best_k

        local_bp_lower[u] = local_bp
        node_x_lower[u] = x_u

        p = parent[u]
        if p is not None:
            J_up = model.get_coupling(u, p)
            messages_lower[(u, p)] = np.asarray(stable_f_J(J_up, x_u), dtype=np.float64)

    field_lower_grid = float(node_x_lower[target_node][K])
    lower_field_profile = node_x_lower[target_node].copy()

    # Trace lower witness
    delta_lower = trace_witness_perturbation(
        model, target_node, children, local_bp_lower, child_bp_lower, K, grid_step, sign=-1
    )
    wit_lower = validate_witness(model, target_node, delta_lower, budget, field_lower_grid)

    # -------------------------------------------------------------
    # 3. CONTINUOUS CERTIFICATION
    # -------------------------------------------------------------
    rigorous_gap = compute_tree_discretization_gap(
        model, root=target_node, budget=budget, grid_step=grid_step
    )
    cert = compute_continuous_certificate(
        field_lower_grid=field_lower_grid,
        field_upper_grid=field_upper_grid,
        grid_step=grid_step,
        lipschitz_const=lipschitz_const,
        cert_gap=rigorous_gap
    )

    return RobustBPResult(
        lower_grid=cert.lower_grid,
        upper_grid=cert.upper_grid,
        lower_certified=cert.lower_certified,
        upper_certified=cert.upper_certified,
        nominal_marginal=prob_nom,
        field_nominal=field_nom,
        field_lower_grid=field_lower_grid,
        field_upper_grid=field_upper_grid,
        field_cert_gap=cert.field_cert_gap,
        budget=budget,
        num_grid_steps=K,
        target_node=target_node,
        witness_upper=wit_upper,
        witness_lower=wit_lower,
        budget_grid=budget_grid,
        upper_field_profile=upper_field_profile,
        lower_field_profile=lower_field_profile
    )


def solve_independent_box_bounds(
    model: TreeModel,
    target_node: int
) -> Tuple[float, float, float, float]:
    """
    Compute standard independent box uncertainty bounds (no coupling / infinite budget sum_i eps_i).

    Args:
        model: TreeModel instance with epsilon limits.
        target_node: Target node index.

    Returns:
        (lower_prob, upper_prob, field_lower, field_upper)
    """
    # Upper bound: delta_i = +eps_i for all i
    delta_upper = model.epsilon.copy()
    field_upper, prob_upper, _ = compute_rooted_total_field(model, root=target_node, delta=delta_upper)

    # Lower bound: delta_i = -eps_i for all i
    delta_lower = -model.epsilon.copy()
    field_lower, prob_lower, _ = compute_rooted_total_field(model, root=target_node, delta=delta_lower)

    return float(prob_lower), float(prob_upper), float(field_lower), float(field_upper)


# Alias for explicit naming
solve_budget_coupled_robust_bp = solve_robust_bp

