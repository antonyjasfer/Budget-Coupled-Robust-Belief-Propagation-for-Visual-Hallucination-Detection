"""Phase 0 regression tests.

1. Root-dominance lemma (docs/theory/root_dominance.md):
   - if B <= eps_r, the robust interval equals sigma(2(eta_r +/- B)) exactly;
   - otherwise the graph adds at most gamma_max * (B - eps_r) to the target field.
2. Report-generator integrity: with zero labelled claims, no finding is stated.
3. Ablation bookkeeping helpers in scripts/run_m8_experiments.py.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from src.pgm.tree_model import TreeModel
from src.pgm.standard_bp import compute_rooted_total_field
from src.robust_bp.solver import solve_robust_bp


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _random_model(rng: random.Random, n: int):
    edges = [(rng.randrange(i), i) for i in range(1, n)]
    coupling = {e: rng.uniform(0.05, 1.5) for e in edges}
    theta = np.array([rng.uniform(-2.0, 2.0) for _ in range(n)])
    eps = np.array([rng.uniform(0.1, 1.0) for _ in range(n)])
    return TreeModel(num_nodes=n, theta=theta, edges=edges, coupling=coupling, epsilon=eps), coupling


def _gamma_max(model: TreeModel, coupling, root: int) -> float:
    """Largest path gain prod tanh(J_e) from any non-root node to the root."""
    _, parent, _ = model.get_rooted_tree(root)
    best = 0.0
    for i in range(model.num_nodes):
        if i == root:
            continue
        g, v = 1.0, i
        while parent[v] is not None:
            p = parent[v]
            j = coupling.get((v, p), coupling.get((p, v)))
            g *= math.tanh(j)
            v = p
        best = max(best, g)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_when_budget_within_target_radius(seed):
    rng = random.Random(seed)
    for _ in range(40):
        n = rng.randint(1, 8)
        model, _ = _random_model(rng, n)
        r = rng.randrange(n)
        budget = rng.uniform(0.0, float(model.epsilon[r]))
        res = solve_robust_bp(model, r, budget, num_grid_steps=100)
        eta = res.field_nominal
        assert res.upper_grid == pytest.approx(_sig(2 * (eta + budget)), abs=1e-9)
        assert res.lower_grid == pytest.approx(_sig(2 * (eta - budget)), abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_graph_contribution_bound_when_budget_exceeds_radius(seed):
    rng = random.Random(100 + seed)
    for _ in range(40):
        n = rng.randint(2, 8)
        model, coupling = _random_model(rng, n)
        r = rng.randrange(n)
        eps_r = float(model.epsilon[r])
        budget = eps_r + rng.uniform(0.05, 1.5)
        res = solve_robust_bp(model, r, budget, num_grid_steps=200)
        gmax = _gamma_max(model, coupling, r)
        extra_up = res.field_upper_grid - (res.field_nominal + eps_r)
        extra_lo = (res.field_nominal - eps_r) - res.field_lower_grid
        bound = gmax * (budget - eps_r)
        # Lower side of the lemma: spending eps_r on the target is always feasible
        # (grid may round eps_r down by at most one step).
        step = budget / 200
        assert extra_up >= -step - 1e-9 and extra_lo >= -step - 1e-9
        assert extra_up <= bound + 1e-9
        assert extra_lo <= bound + 1e-9


def test_graph_can_matter_only_above_the_radius():
    """Sanity: with B > eps_r and strong coupling the graph adds a strictly positive amount."""
    model = TreeModel(
        num_nodes=3,
        theta=np.zeros(3),
        edges=[(0, 1), (0, 2)],
        coupling={(0, 1): 1.5, (0, 2): 1.5},
        epsilon=np.array([0.1, 0.5, 0.5]),
    )
    res = solve_robust_bp(model, 0, 1.1, num_grid_steps=110)
    assert res.field_upper_grid > res.field_nominal + 0.1 + 0.1


# ---------------------------------------------------------------------------
# Report-generator integrity
# ---------------------------------------------------------------------------

FORBIDDEN = [
    "directly leads to interval width expansion",
    "confirms consistent behavior",
    "without heuristics",
    "successfully exposes",
    "guaranteed marginal posterior intervals",
]


def _unlabelled_claims():
    from src.experiments.inference_runner import ClaimEvaluationResult

    out = []
    for i in range(4):
        out.append(
            ClaimEvaluationResult(
                run_id="r", experiment_name="e", condition="main_evaluation",
                image_id=f"img_{i // 2}", claim_id=f"c{i}", object_category="dog",
                text_span="dog", split="test", ground_truth=None,
                detector_score=0.5, clip_score=0.2, theta_i=0.0, epsilon_i=0.5,
                coupling_j=0.5, budget=0.5, epsilon_scale=1.0, decision_threshold=0.5,
                grid_steps=10, standard_posterior=0.5, standard_prediction="supported",
                robust_lower=0.27, robust_upper=0.73, robust_midpoint=0.5,
                interval_width=0.46, certified_lower=0.27, certified_upper=0.73,
                field_cert_gap=0.0, robust_prediction="abstain", evidence_sensitive=True,
                abstained=True, corruption_type="none", corruption_severity="clean",
                runtime_ms=1.0,
            )
        )
    return out


def test_reports_state_no_findings_without_labels(tmp_path):
    from src.scientific.consolidator import M9ResultsConsolidator

    out_dir = tmp_path / "final"
    M9ResultsConsolidator(output_dir=out_dir, allow_overwrite=True).consolidate(
        _unlabelled_claims(), execution_mode="DEVELOPMENT"
    )
    texts = [
        (out_dir / "research_summary.md").read_text(encoding="utf-8"),
        (out_dir / "contribution_statement.md").read_text(encoding="utf-8"),
        *[(p).read_text(encoding="utf-8") for p in (out_dir / "paper_results").glob("*.md")],
    ]
    joined = "\n".join(texts).lower()
    for phrase in FORBIDDEN:
        assert phrase not in joined, phrase
    summary = texts[0]
    assert "DEVELOPMENT ONLY" in summary
    assert "No scientific conclusion is generated automatically" in summary
    # Root-dominance regime is reported (all 4 records have B <= eps).
    assert "1.000" in summary


def test_crossing_rate_reads_legacy_key():
    from src.scientific.reporting import _crossing_rate

    assert _crossing_rate({"evidence_sensitive_fraction": 0.283}) == pytest.approx(0.283)
    assert _crossing_rate({"threshold_crossing_rate": 0.1, "evidence_sensitive_fraction": 0.9}) == pytest.approx(0.1)
    assert math.isnan(_crossing_rate({}))


# ---------------------------------------------------------------------------
# Ablation bookkeeping helpers
# ---------------------------------------------------------------------------

def test_ablation_helpers_report_nominal_not_midpoint():
    from types import SimpleNamespace
    import scripts.run_m8_experiments as m8

    out = SimpleNamespace(hallucination_score=0.45, metadata={"nominal_marginal": 0.43})
    assert m8._nominal_or_mid(out) == pytest.approx(0.43)
    assert m8._threshold_decision(0.51, 0.5) == "hallucinated"
    assert m8._threshold_decision(0.49, 0.5) == "supported"

    model = TreeModel(num_nodes=2, theta=np.array([0.3, -0.2]), edges=[(0, 1)],
                      coupling={(0, 1): 0.5}, epsilon=np.array([0.5, 0.5]))
    ctx = SimpleNamespace(model=model, claim_to_node={"a": 0, "b": 1})
    _, expected, _ = compute_rooted_total_field(model, root=0, delta=None)
    assert m8._nominal_posterior(ctx, "a") == pytest.approx(expected)
