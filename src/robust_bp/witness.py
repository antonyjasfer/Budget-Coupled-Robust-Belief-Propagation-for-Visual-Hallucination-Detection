"""
Witness / perturbation vector reconstruction for Budget-Coupled Robust BP.

Recovers optimal perturbation vectors delta* for upper and lower bounds using dynamic
programming backpointers, and validates budget constraints and field fidelity.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from src.pgm.tree_model import TreeModel
from src.pgm.standard_bp import compute_rooted_total_field


@dataclass
class WitnessResult:
    """
    Result of witness perturbation reconstruction.

    Attributes:
        delta: Perturbation array of shape (num_nodes,).
        total_budget_spent: sum_i |delta_i|
        budget_limit: B
        is_budget_valid: bool indicating sum_i |delta_i| <= B + 1e-9
        is_box_valid: bool indicating |delta_i| <= eps_i + 1e-9 for all i
        nominal_field: Total effective field under nominal model (delta=0)
        witness_field: Total effective field under perturbed model (delta=delta)
        predicted_grid_field: Total effective field predicted by robust DP
        field_match: bool indicating abs(witness_field - predicted_grid_field) < 1e-5
    """
    delta: np.ndarray
    total_budget_spent: float
    budget_limit: float
    is_budget_valid: bool
    is_box_valid: bool
    nominal_field: float
    witness_field: float
    predicted_grid_field: float
    field_match: bool


def trace_witness_perturbation(
    model: TreeModel,
    target_root: int,
    children: Dict[int, List[int]],
    local_bp: Dict[int, np.ndarray],
    child_bp: Dict[Tuple[int, int], np.ndarray],
    total_steps: int,
    grid_step: float,
    sign: int
) -> np.ndarray:
    """
    Trace backpointers through the tree to reconstruct the exact perturbation vector delta*.

    Args:
        model: TreeModel instance.
        target_root: Target node index.
        children: Children mapping for rooted tree.
        local_bp: Local perturbation backpointers for each node.
        child_bp: Convolution backpointers for child pairs.
        total_steps: Total budget grid steps K.
        grid_step: Delta = B / K.
        sign: +1 for upper bound (delta >= 0), -1 for lower bound (delta <= 0).

    Returns:
        delta: Reconstructed perturbation array of shape (num_nodes,).
    """
    n = model.num_nodes
    delta = np.zeros(n, dtype=np.float64)

    # Stack holds (node, budget_index_available_to_subtree)
    stack: List[Tuple[int, int]] = [(target_root, total_steps)]

    while stack:
        u, budget_idx = stack.pop()
        # Budget spent locally at node u
        local_k = int(local_bp[u][budget_idx])
        delta[u] = float(sign * local_k * grid_step)

        rem_budget_idx = budget_idx - local_k
        c_list = children[u]
        d = len(c_list)

        if d == 0:
            continue
        elif d == 1:
            # Single child receives all remaining budget for children
            stack.append((c_list[0], rem_budget_idx))
        else:
            # Multiple children: unwind convolution chain in reverse order
            curr_rem = rem_budget_idx
            for l in range(d - 1, 0, -1):
                child_node = c_list[l]
                c_alloc = int(child_bp[(u, child_node)][curr_rem])
                stack.append((child_node, c_alloc))
                curr_rem -= c_alloc
            # First child receives the final remaining budget
            stack.append((c_list[0], curr_rem))

    return delta


def validate_witness(
    model: TreeModel,
    target_root: int,
    delta: np.ndarray,
    budget_limit: float,
    predicted_grid_field: float
) -> WitnessResult:
    """
    Verify that reconstructed witness respects all budget constraints and matches predicted field.

    Args:
        model: TreeModel instance.
        target_root: Target node index.
        delta: Perturbation array.
        budget_limit: Total budget B.
        predicted_grid_field: Effective field predicted by robust DP.

    Returns:
        WitnessResult containing validation metrics.
    """
    total_spent = float(np.sum(np.abs(delta)))
    is_budget_valid = bool(total_spent <= budget_limit + 1e-7)

    box_violations = np.abs(delta) - (model.epsilon + 1e-7)
    is_box_valid = bool(np.all(box_violations <= 0.0))

    # Evaluate nominal and perturbed standard BP at root
    eta_nom, _, _ = compute_rooted_total_field(model, root=target_root, delta=None)
    eta_wit, _, _ = compute_rooted_total_field(model, root=target_root, delta=delta)

    field_match = bool(np.abs(eta_wit - predicted_grid_field) < 1e-5)

    return WitnessResult(
        delta=delta,
        total_budget_spent=total_spent,
        budget_limit=budget_limit,
        is_budget_valid=is_budget_valid,
        is_box_valid=is_box_valid,
        nominal_field=eta_nom,
        witness_field=eta_wit,
        predicted_grid_field=predicted_grid_field,
        field_match=field_match
    )
