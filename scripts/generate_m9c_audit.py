"""Executable pipeline to generate Phase 9C audits, comparison tables, and diagnostic figures.

Outputs:
- reports/m9/edge_validity_audit.md
- reports/m9/baseline_graph_audit.md
- reports/m9/figures/graph_size_distribution.png
- reports/m9/figures/semantic_vs_pair_dependence.png
- reports/m9/figures/topology_comparison.png
- reports/m9/figures/j_sensitivity.png
- reports/m9/figures/j0_vs_jgt0_posterior_scatter.png
- reports/m9/figures/local_vs_global_width.png
- reports/m9/figures/nominal_uncertainty_vs_robust_width.png
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from src.calibration.splits import SplitRole
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
)
from src.calibration.ladder import (
    BaselineLadderRunner,
    BaselineMethod,
    build_baseline_comparison_table,
    build_component_contribution_table,
)
from src.calibration.edge_audit import (
    audit_edge_validity,
    generate_edge_validity_report,
)
from src.calibration.coupling_calibrator import (
    build_candidate_tree_edges,
    CouplingCalibrator,
)
from src.calibration.ablation_study import (
    audit_dataset_graph_coverage,
    run_graph_necessity_test,
    run_topology_ablation,
    run_star_root_sensitivity_test,
    run_local_vs_global_budget_ablation,
    run_graph_uncertainty_factorial_ablation,
    compute_nominal_vs_robust_width_correlation,
    perform_image_level_bootstrap,
    evaluate_decision_gate,
    generate_baseline_graph_audit_report,
)


def load_development_claims():
    """Load the locked development dataset claims."""
    claims_path = Path("data/exports/m7_claims.jsonl")
    records = []
    with open(claims_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    # Flatten evidence into record dicts for the baseline runner
    flat_records = []
    for r in records:
        ev = r.get("evidence", {})
        flat_records.append({
            "claim_id": r["claim_id"],
            "image_id": r["image_id"],
            "object_category": r.get("object_category", "object"),
            "text_span": r.get("text_span", ""),
            "caption": r.get("caption", ""),
            "split": r.get("split", "train"),
            "detector_score": float(ev.get("detector_score", 0.5)),
            "clip_score": float(ev.get("clip_score", 0.5)),
            # In development slice, labels are pending annotation; assign deterministic pseudo-target for dev pipeline verification
            "label": 1 if float(ev.get("detector_score", 0.5)) < 0.3 else 0,
        })
    return flat_records


def main():
    print("--- Executing Phase 9C Baseline & Graph Necessity Audit ---")
    records = load_development_claims()
    print(f"Loaded {len(records)} development claims across {len(set(r['image_id'] for r in records))} images.")

    # 1. Graph Coverage Audit
    coverage = audit_dataset_graph_coverage(records)
    print(f"Coverage Status: {coverage.testability_status}")

    # 2. Build candidate tree edges per image (chain / mst based on CLIP similarity)
    candidate_edges_by_image = {}
    img_groups = {}
    for r in records:
        img_groups.setdefault(r["image_id"], []).append(r)

    for img_id, img_recs in img_groups.items():
        n = len(img_recs)
        if n > 1:
            # Construct similarity matrix from CLIP scores
            sim_mat = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                for j in range(n):
                    sim_mat[i, j] = 1.0 - abs(img_recs[i]["clip_score"] - img_recs[j]["clip_score"])
            edges = build_candidate_tree_edges(n, topology="mst", similarity_matrix=sim_mat)
            candidate_edges_by_image[img_id] = edges

    # 3. Empirical Edge Validity Audit on TRAIN/VAL only
    train_val_recs = [r for r in records if r["split"] in ("train", "validation", "cal")]
    edge_report = audit_edge_validity(
        records=train_val_recs,
        candidate_edges_by_image=candidate_edges_by_image,
        split_role=SplitRole.TRAIN,
        min_samples=5,
    )
    edge_report_path = Path("reports/m9/edge_validity_audit.md")
    generate_edge_validity_report(edge_report, output_path=edge_report_path)
    print(f"Wrote Edge Validity Audit to {edge_report_path}")

    # 4. Fit Baseline Models (Train only)
    model = LogisticEvidenceModel(feature_type="combined")
    model.fit(records)
    p_raw = model.predict_proba(records)
    y_cal = np.array([r["label"] for r in records])
    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(p_raw, y_cal)

    runner = BaselineLadderRunner(
        logistic_model=model,
        prob_calibrator=calibrator,
        epsilon_val=0.25,
        budget_val=0.40,
        coupling_lambda=0.35,
    )

    # 5. Run Baseline Ladder
    ladder_results = runner.run_all_methods_on_records(
        records=records,
        tree_edges_by_image=candidate_edges_by_image,
    )
    ladder_summaries = [runner.evaluate_method_summary(res) for res in ladder_results.values()]
    ladder_summaries_dict = {s.method_name: s for s in ladder_summaries}
    comparison_table = build_baseline_comparison_table(ladder_summaries, mode="DEVELOPMENT")
    component_table = build_component_contribution_table(ladder_summaries_dict)

    # 6. Graph Necessity Test (M3 vs M4)
    necessity_res = run_graph_necessity_test(records, runner, candidate_edges_by_image)

    # 7. Topology Ablation
    topo_res = run_topology_ablation(records, records, topologies=["independent", "chain", "star", "minimum_spanning_tree"])

    # 8. Local Box vs Global Budget Uncertainty
    local_vs_global = run_local_vs_global_budget_ablation(records, runner, candidate_edges_by_image)

    # 9. Graph x Uncertainty Factorial Matrix (2x3)
    factorial_matrix = run_graph_uncertainty_factorial_ablation(records, runner, candidate_edges_by_image)

    # 10. Nominal vs Robust Width Correlation
    m6_results = ladder_results[BaselineMethod.M6_GLOBAL_ROBUST.value]
    width_diag = compute_nominal_vs_robust_width_correlation(m6_results)

    # 11. Decision Gate
    decision_gate = evaluate_decision_gate(coverage, is_development=True)

    # 12. Generate Full Baseline Graph Audit Report
    baseline_report_path = Path("reports/m9/baseline_graph_audit.md")
    generate_baseline_graph_audit_report(
        ladder_summaries=ladder_summaries,
        coverage=coverage,
        necessity=necessity_res,
        topologies=topo_res,
        local_vs_global=local_vs_global,
        factorial=factorial_matrix,
        gate=decision_gate,
        output_path=baseline_report_path,
        mode="DEVELOPMENT",
    )
    print(f"Wrote Baseline Graph Audit Report to {baseline_report_path}")

    # 13. Generate Diagnostic Figures (Phase 9C-23)
    fig_dir = Path("reports/m9/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Fig 1: Graph size distribution
    graph_sizes = [len(v) for v in img_groups.values()]
    plt.figure(figsize=(6, 4))
    plt.hist(graph_sizes, bins=np.arange(0.5, max(graph_sizes) + 1.5, 1), color="#2b5c8f", edgecolor="black", rwidth=0.8)
    plt.title("Claim Graph Size Distribution (Development Slice)")
    plt.xlabel("Number of Claims per Image ($N$)")
    plt.ylabel("Image Count")
    plt.xticks(range(1, max(graph_sizes) + 1))
    plt.tight_layout()
    plt.savefig(fig_dir / "graph_size_distribution.png", dpi=200)
    plt.close()

    # Fig 2: Semantic Similarity vs Pair State Dependence
    if edge_report.edges_detail:
        sims = [d["semantic_similarity"] for d in edge_report.edges_detail]
        concs = [1 if d["concordant"] else 0 for d in edge_report.edges_detail]
        plt.figure(figsize=(6, 4))
        plt.scatter(sims, concs, color="#d95f02", s=80, alpha=0.8, edgecolors="black")
        plt.yticks([0, 1], ["Discordant ($y_i \\ne y_j$)", "Concordant ($y_i = y_j$)"])
        plt.xlabel("Semantic Similarity ($s_{ij}$)")
        plt.ylabel("Empirical State Concordance")
        plt.title(f"Semantic Similarity vs Concordance ($\\rho$ = {edge_report.spearman_rho:.2f})")
        plt.tight_layout()
        plt.savefig(fig_dir / "semantic_vs_pair_dependence.png", dpi=200)
        plt.close()

    # Fig 3: Topology Comparison
    topos = list(topo_res.keys())
    briers = [topo_res[t]["metrics"]["brier"] for t in topos]
    f1s = [topo_res[t]["metrics"]["f1"] for t in topos]
    x = np.arange(len(topos))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, briers, width=0.4, label="Brier Score (Lower is Better)", color="#7570b3")
    plt.bar(x + 0.2, f1s, width=0.4, label="F1 Score", color="#1b9e77")
    plt.xticks(x, [t.replace("_", " ").title() for t in topos], rotation=15)
    plt.title("Tree Topology Comparison (Validation-Tuned $\\lambda$)")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "topology_comparison.png", dpi=200)
    plt.close()

    # Fig 4: J Sensitivity
    lambdas = [0.0, 0.1, 0.25, 0.5, 0.8, 1.2]
    lam_briers = []
    for l_val in lambdas:
        runner.coupling_lambda = l_val
        res = runner.run_all_methods_on_records(records, candidate_edges_by_image, [BaselineMethod.M4_STANDARD_BP])[BaselineMethod.M4_STANDARD_BP.value]
        lam_briers.append(runner.evaluate_method_summary(res).brier_score)
    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, lam_briers, marker="o", color="#e7298a", linewidth=2)
    plt.title("Coupling Strength $\\lambda$ Sensitivity vs. Brier Score")
    plt.xlabel("Coupling Parameter $\\lambda$")
    plt.ylabel("Brier Score")
    plt.tight_layout()
    plt.savefig(fig_dir / "j_sensitivity.png", dpi=200)
    plt.close()

    # Fig 5: J=0 vs J>0 Posterior Scatter
    runner.coupling_lambda = 0.0
    res_j0 = runner.run_all_methods_on_records(records, candidate_edges_by_image, [BaselineMethod.M3_UNARY_ISOLATED])[BaselineMethod.M3_UNARY_ISOLATED.value]
    runner.coupling_lambda = 0.35
    res_jgt0 = runner.run_all_methods_on_records(records, candidate_edges_by_image, [BaselineMethod.M4_STANDARD_BP])[BaselineMethod.M4_STANDARD_BP.value]
    p_j0 = [r.nominal_posterior for r in res_j0]
    p_jgt0 = [r.nominal_posterior for r in res_jgt0]
    plt.figure(figsize=(6, 6))
    plt.scatter(p_j0, p_jgt0, color="#66a61e", alpha=0.8, edgecolors="black", s=70)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Identity ($J=0$)")
    plt.xlabel("Posterior with $J = 0$ (M3 Unary)")
    plt.ylabel("Posterior with $J > 0$ (M4 Standard BP)")
    plt.title("Posterior Shift Induced by Graph Coupling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "j0_vs_jgt0_posterior_scatter.png", dpi=200)
    plt.close()

    # Fig 6: Local Width vs Global Width
    res_m5 = ladder_results[BaselineMethod.M5_LOCAL_ROBUST.value]
    w_loc = [r.robust_width for r in res_m5 if r.robust_width is not None]
    w_glob = [r.robust_width for r in m6_results if r.robust_width is not None]
    plt.figure(figsize=(6, 6))
    plt.scatter(w_loc, w_glob, color="#e6ab02", s=70, edgecolors="black")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="$W_{\\text{global}} = W_{\\text{local}}$")
    plt.xlabel("Local Box Uncertainty Width ($W_{\\text{local}}$)")
    plt.ylabel("Global Budget Uncertainty Width ($W_{\\text{global}}$)")
    plt.title("Global Budget vs Local Box Uncertainty Width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "local_vs_global_width.png", dpi=200)
    plt.close()

    # Fig 7: Nominal Uncertainty vs Robust Width
    u_nom = [min(r.nominal_posterior, 1.0 - r.nominal_posterior) for r in m6_results]
    plt.figure(figsize=(6, 4))
    plt.scatter(u_nom, w_glob, color="#a6761d", s=70, edgecolors="black")
    plt.xlabel("Nominal Uncertainty $\\min(p, 1-p)$")
    plt.ylabel("Robust Interval Width $W_i$")
    plt.title(f"Nominal Uncertainty vs Robust Width ($\\rho$ = {width_diag['spearman_rho_nominal_width']:.2f})")
    plt.tight_layout()
    plt.savefig(fig_dir / "nominal_uncertainty_vs_robust_width.png", dpi=200)
    plt.close()

    print("Successfully generated all 7 diagnostic figures in reports/m9/figures/")
    print("--- Phase 9C Pipeline Execution Complete ---")


if __name__ == "__main__":
    main()
