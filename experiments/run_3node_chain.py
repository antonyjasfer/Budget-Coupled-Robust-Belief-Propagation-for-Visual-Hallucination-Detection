"""
Experiment: 3-Node Chain Benchmark Validation.

Reproduces the 3-node chain example from the research design:
    Nodes: 1 - 2 - 3 (indices 0, 1, 2)
    θ1 = θ2 = θ3 = 0
    J12 = J23 = atanh(1/2) ≈ 0.549306
    ε1 = ε2 = ε3 = atanh(1/2) ≈ 0.549306

Validation Targets (for Center Node 2 / index 1):
    A) Independent box uncertainty:
       Interval ≈ [0.1071, 0.8929] (exact fraction [3/28, 25/28])
    B) Shared budget B = atanh(1/2):
       Interval = [0.25, 0.75] (exact fraction [1/4, 3/4])
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

from src.pgm.tree_model import create_chain_tree
from src.pgm.standard_bp import run_standard_bp
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds


def run_experiment() -> bool:
    print("=" * 78)
    print("EXPERIMENT: 3-Node Chain Benchmark Validation")
    print("=" * 78)

    # 1. Setup parameters
    n = 3
    theta = np.zeros(n, dtype=np.float64)
    j_val = float(np.arctanh(0.5))
    eps_val = float(np.arctanh(0.5))
    couplings = np.array([j_val, j_val], dtype=np.float64)
    epsilon = np.array([eps_val, eps_val, eps_val], dtype=np.float64)
    budget = eps_val  # B = atanh(1/2)

    model = create_chain_tree(n, theta, couplings, epsilon=epsilon)
    target_node = 1  # Center node (Node 2 in 1-based indexing)

    print(f"Graph topology: 3-node chain (0 - 1 - 2)")
    print(f"Unary fields theta: {theta}")
    print(f"Couplings J: J01 = J12 = atanh(1/2) = {j_val:.6f}")
    print(f"Local bounds epsilon: eps0 = eps1 = eps2 = atanh(1/2) = {eps_val:.6f}")
    print(f"Shared budget B: atanh(1/2) = {budget:.6f}")
    print(f"Target evaluation node: Node 2 (index {target_node})")
    print("-" * 78)

    # 2. Nominal Inference
    nominal_res = run_standard_bp(model)
    p_nom = nominal_res.marginals[target_node]
    eta_nom = nominal_res.effective_fields[target_node]
    print(f"Nominal Marginal P(h2 = +1 | delta = 0): {p_nom:.6f} (effective field: {eta_nom:.6f})")
    print("-" * 78)

    # 3. Independent Box Uncertainty (Unbudgeted / sum |delta| <= 3*eps)
    box_low_p, box_up_p, box_low_eta, box_up_eta = solve_independent_box_bounds(model, target_node=target_node)

    expected_box_low = 3.0 / 28.0   # ≈ 0.1071428...
    expected_box_up = 25.0 / 28.0   # ≈ 0.8928571...

    print("CASE A: Independent Box Uncertainty (|delta_i| <= eps_i)")
    print(f"  Calculated Interval: [{box_low_p:.4f}, {box_up_p:.4f}]")
    print(f"  Expected Target:     [{expected_box_low:.4f}, {expected_box_up:.4f}]")
    print(f"  Effective Fields:    [{box_low_eta:.6f}, {box_up_eta:.6f}]")

    box_pass = np.isclose(box_low_p, expected_box_low, atol=1e-4) and np.isclose(box_up_p, expected_box_up, atol=1e-4)
    print(f"  Validation Status:   {'PASSED' if box_pass else 'FAILED'}")
    print("-" * 78)

    # 4. Budget-Coupled Uncertainty (Shared budget B = atanh(1/2))
    num_grid_steps = 200
    robust_res = solve_robust_bp(
        model, target_node=target_node, budget=budget, num_grid_steps=num_grid_steps
    )

    expected_budget_low = 0.25  # 1/4
    expected_budget_up = 0.75   # 3/4

    print(f"CASE B: Shared Budget Uncertainty (sum |delta_i| <= B = {budget:.4f})")
    print(f"  Grid Interval:       [{robust_res.lower_grid:.4f}, {robust_res.upper_grid:.4f}]")
    print(f"  Certified Interval:  [{robust_res.lower_certified:.4f}, {robust_res.upper_certified:.4f}]")
    print(f"  Expected Target:     [{expected_budget_low:.4f}, {expected_budget_up:.4f}]")
    print(f"  Effective Fields:    [{robust_res.field_lower_grid:.6f}, {robust_res.field_upper_grid:.6f}]")
    print(f"  Cert Gap (Delta):    {robust_res.field_cert_gap:.6f}")

    budget_pass = (
        np.isclose(robust_res.lower_grid, expected_budget_low, atol=1e-4) and
        np.isclose(robust_res.upper_grid, expected_budget_up, atol=1e-4)
    )
    print(f"  Validation Status:   {'PASSED' if budget_pass else 'FAILED'}")
    print("-" * 78)

    # 5. Witness Perturbation Analysis
    print("WITNESS PERTURBATIONS:")
    up_wit = robust_res.witness_upper
    low_wit = robust_res.witness_lower

    print(f"  Upper Witness delta* : [{', '.join(f'{d:+.4f}' for d in up_wit.delta)}]")
    print(f"    Total Spent: {up_wit.total_budget_spent:.6f} / {up_wit.budget_limit:.6f}")
    print(f"    Budget Valid: {up_wit.is_budget_valid}, Box Valid: {up_wit.is_box_valid}, Field Match: {up_wit.field_match}")

    print(f"  Lower Witness delta* : [{', '.join(f'{d:+.4f}' for d in low_wit.delta)}]")
    print(f"    Total Spent: {low_wit.total_budget_spent:.6f} / {low_wit.budget_limit:.6f}")
    print(f"    Budget Valid: {low_wit.is_budget_valid}, Box Valid: {low_wit.is_box_valid}, Field Match: {low_wit.field_match}")
    print("-" * 78)

    # 6. Overall Summary Table
    print(f"{'Condition':<25} | {'Lower Bound':<15} | {'Upper Bound':<15} | {'Interval Width':<15}")
    print("-" * 78)
    print(f"{'Independent Box':<25} | {box_low_p:<15.4f} | {box_up_p:<15.4f} | {box_up_p - box_low_p:<15.4f}")
    print(f"{'Shared Budget B=0.55':<25} | {robust_res.lower_grid:<15.4f} | {robust_res.upper_grid:<15.4f} | {robust_res.upper_grid - robust_res.lower_grid:<15.4f}")
    print(f"{'Nominal (B=0)':<25} | {p_nom:<15.4f} | {p_nom:<15.4f} | {0.0:<15.4f}")
    print("=" * 78)

    all_passed = box_pass and budget_pass and up_wit.field_match and low_wit.field_match
    print(f"FINAL EXPERIMENT RESULT: {'ALL CHECKS PASSED' if all_passed else 'VALIDATION FAILED'}")
    print("=" * 78)

    return all_passed


if __name__ == "__main__":
    success = run_experiment()
    if not success:
        sys.exit(1)
