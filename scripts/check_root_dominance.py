"""Check the 'root dominance' property of budget-coupled robust BP.

Claim (provable): on an attractive Ising tree, d(eta_r)/d(delta_r) = 1 while
d(eta_r)/d(delta_i) <= prod tanh(J_e) along the path i->r  (< 1 for i != r).
Hence if B <= eps_r, the optimal adversary spends the WHOLE budget on the
target node and the robust interval has the closed form

    [L_r, U_r] = [ sigma(2(eta_r - B)), sigma(2(eta_r + B)) ]

i.e. it is a deterministic function of the point posterior: the graph, J and
the DP contribute nothing to the interval beyond eta_r itself.

Part 1: random trees vs. your own solver (src/robust_bp/solver.py).
Part 2: your M9 results file (reports/m9/final/final_claim_results.jsonl).

Run from the repository root:  python scripts/check_root_dominance.py
"""
import json
import math
import random
from pathlib import Path

import numpy as np

from src.pgm.tree_model import TreeModel
from src.robust_bp.solver import solve_robust_bp


def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def random_tree(n: int, rng: random.Random):
    return [(rng.randrange(i), i) for i in range(1, n)]


def part1(trials: int = 300, seed: int = 0) -> None:
    rng = random.Random(seed)
    worst_in, worst_out = 0.0, 0.0
    for _ in range(trials):
        n = rng.randint(2, 8)
        edges = random_tree(n, rng)
        J = {e: rng.uniform(0.05, 1.5) for e in edges}
        theta = np.array([rng.uniform(-2, 2) for _ in range(n)])
        eps = np.array([rng.uniform(0.1, 1.0) for _ in range(n)])
        model = TreeModel(num_nodes=n, theta=theta, edges=edges, coupling=J, epsilon=eps)
        r = rng.randrange(n)
        # Regime A: B <= eps_r  -> closed form must hold
        B = rng.uniform(0.0, eps[r])
        res = solve_robust_bp(model, r, B, num_grid_steps=200)
        eta = res.field_nominal
        err = max(abs(res.upper_grid - sig(2 * (eta + B))), abs(res.lower_grid - sig(2 * (eta - B))))
        worst_in = max(worst_in, err)
        # Regime B: B > eps_r -> graph matters, but extra field gain is bounded by
        # max_path_gain * (B - eps_r) <= tanh(max J) * (B - eps_r)
        B2 = eps[r] + rng.uniform(0.1, 1.5)
        res2 = solve_robust_bp(model, r, B2, num_grid_steps=200)
        extra = res2.field_upper_grid - (res2.field_nominal + eps[r])
        bound = math.tanh(max(J.values())) * (B2 - eps[r])
        worst_out = max(worst_out, extra - bound)
    print(f"[Part 1] B <= eps_r : max |solver - closed form| = {worst_in:.2e}  (should be ~grid error)")
    print(f"[Part 1] B >  eps_r : max (extra field - tanh(Jmax)(B-eps_r)) = {worst_out:.2e}  (should be <= ~0)")


def part2(path: str = "reports/m9/final/final_claim_results.jsonl") -> None:
    p = Path(path)
    if not p.exists():
        print("[Part 2] results file not found, skipped")
        return
    rows = [json.loads(line) for line in p.open()]
    by_cond = {}
    for row in rows:
        b = min(row["budget"], row["epsilon_i"])
        if row["budget"] > row["epsilon_i"] or row["condition"].startswith("ablation"):
            continue
        q = min(max(row["standard_posterior"], 1e-15), 1 - 1e-15)
        eta = 0.5 * math.log(q / (1 - q))
        err = max(abs(row["robust_upper"] - sig(2 * (eta + b))), abs(row["robust_lower"] - sig(2 * (eta - b))))
        by_cond[row["condition"]] = max(by_cond.get(row["condition"], 0.0), err)
    print("[Part 2] conditions with B <= eps: max deviation from the target-only closed form")
    for cond, err in sorted(by_cond.items()):
        print(f"   {cond:28s} {err:.2e}")
    main = [r for r in rows if r["condition"] == "main_evaluation"]
    lo, hi = sig(-2 * main[0]["budget"]), sig(2 * main[0]["budget"])
    agree = all(r["evidence_sensitive"] == (lo < r["standard_posterior"] < hi) for r in main)
    print(f"[Part 2] 'evidence_sensitive' == (point posterior in ({lo:.2f}, {hi:.2f})) for every main claim: {agree}")


if __name__ == "__main__":
    part1()
    part2()
