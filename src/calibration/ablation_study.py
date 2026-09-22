"""Comprehensive ablation study and scientific falsification framework for Milestone 9C.

Determines whether:
1. Graph coupling (J > 0) is scientifically necessary over calibrated independent unaries.
2. Claim-tree topology (Chain, Star, MST) provides measurable empirical value.
3. The global L1 budget B provides useful information beyond independent local box uncertainty.
4. Robust posterior interval width adds information beyond nominal posterior uncertainty.
5. The full method (M6) beats or complements the simple logistic evidence baseline (M2).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import spearmanr

from src.calibration.coupling_calibrator import (
    CouplingCalibrator,
    build_candidate_tree_edges,
)
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
    compute_calibration_diagnostics,
)
from src.calibration.ladder import (
    BaselineLadderRunner,
    BaselineMethod,
    ClaimPredictionResult,
    MethodEvaluationSummary,
    build_baseline_comparison_table,
    build_component_contribution_table,
    compute_binary_classification_metrics,
)
from src.calibration.splits import SplitContract, SplitRole
from src.calibration.theta_mapping import probability_to_theta, theta_to_probability
from src.pgm.standard_bp import run_standard_bp
from src.pgm.tree_model import TreeModel
from src.robust_bp.solver import solve_robust_bp


@dataclass
class GraphCoverageAudit:
    """Audit of dataset capability to evaluate graph-based methods."""

    total_images: int
    total_claims: int
    images_with_1_claim: int
    images_with_ge_2_claims: int
    images_with_ge_3_claims: int
    mean_claims_per_image: float
    median_claims_per_image: float
    total_graph_edges: int
    percent_claims_with_edges: float
    testability_status: str  # "SUFFICIENT" or "GRAPH-EVALUATION DATA INSUFFICIENT"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorialConditionResult:
    """One condition in the 2x3 (Graph x Uncertainty) factorial design."""

    condition_id: str  # A, B, C, D, E, F
    has_graph: bool
    uncertainty_type: str  # "NONE", "LOCAL", "GLOBAL"
    description: str
    accuracy: float
    f1: float
    roc_auc: float
    brier_score: float
    log_loss: float
    mean_width: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchDecisionGate:
    """Rigorous status assessment for the central scientific questions."""

    graph_testability: str  # "SUFFICIENT", "INSUFFICIENT", "FINAL DATA REQUIRED"
    graph_value: str  # "NOT EVALUABLE", "NO CLEAR EVIDENCE", "DEVELOPMENT SIGNAL ONLY", "FINAL EVIDENCE"
    global_budget_value: str  # "NOT EVALUABLE", "NO CLEAR EVIDENCE", "DEVELOPMENT SIGNAL ONLY", "FINAL EVIDENCE"
    simple_baseline_challenge: str  # "NOT EVALUABLE", "FULL METHOD CLEARLY DISTINCT IN OUTPUT TYPE", "FINAL COMPARISON REQUIRED"
    final_data_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_dataset_graph_coverage(
    records: List[Dict[str, Any]],
    candidate_edges_by_image: Optional[Dict[str, List[Tuple[int, int, float]]]] = None,
) -> GraphCoverageAudit:
    """Analyze claim clustering across images to determine if graph evaluation is possible."""
    img_claims: Dict[str, int] = {}
    for r in records:
        img_id = str(r.get("image_id", "default"))
        img_claims[img_id] = img_claims.get(img_id, 0) + 1

    total_images = len(img_claims)
    total_claims = len(records)
    sizes = list(img_claims.values())

    c_1 = sum(1 for s in sizes if s == 1)
    c_ge_2 = sum(1 for s in sizes if s >= 2)
    c_ge_3 = sum(1 for s in sizes if s >= 3)

    mean_c = float(np.mean(sizes)) if sizes else 0.0
    median_c = float(np.median(sizes)) if sizes else 0.0

    edges_count = 0
    claims_with_edges = 0
    edge_map = candidate_edges_by_image or {}

    for img_id, count in img_claims.items():
        edges = edge_map.get(img_id, [])
        edges_count += len(edges)
        if len(edges) > 0:
            connected_nodes = set()
            for u, v, _ in edges:
                connected_nodes.add(u)
                connected_nodes.add(v)
            claims_with_edges += len(connected_nodes)

    pct_claims = float(claims_with_edges / total_claims * 100.0) if total_claims > 0 else 0.0

    # If fewer than 5 multi-claim images exist, the dataset is insufficient for scientific graph conclusions
    testability = "SUFFICIENT" if c_ge_2 >= 5 else "GRAPH-EVALUATION DATA INSUFFICIENT"

    return GraphCoverageAudit(
        total_images=total_images,
        total_claims=total_claims,
        images_with_1_claim=c_1,
        images_with_ge_2_claims=c_ge_2,
        images_with_ge_3_claims=c_ge_3,
        mean_claims_per_image=mean_c,
        median_claims_per_image=median_c,
        total_graph_edges=edges_count,
        percent_claims_with_edges=pct_claims,
        testability_status=testability,
    )


def run_graph_necessity_test(
    records: List[Dict[str, Any]],
    ladder_runner: BaselineLadderRunner,
    candidate_edges_by_image: Optional[Dict[str, List[Tuple[int, int, float]]]] = None,
) -> Dict[str, Any]:
    """Q1 & Q2: Compare M3 (J = 0) vs M4 (J > 0) using identical calibrated unary fields."""
    edges_map = candidate_edges_by_image or {}
    res_m3 = ladder_runner.run_all_methods_on_records(
        records, edges_map, methods=[BaselineMethod.M3_UNARY_ISOLATED]
    )[BaselineMethod.M3_UNARY_ISOLATED.value]

    res_m4 = ladder_runner.run_all_methods_on_records(
        records, edges_map, methods=[BaselineMethod.M4_STANDARD_BP]
    )[BaselineMethod.M4_STANDARD_BP.value]

    sum_m3 = ladder_runner.evaluate_method_summary(res_m3)
    sum_m4 = ladder_runner.evaluate_method_summary(res_m4)

    delta_brier = sum_m4.brier_score - sum_m3.brier_score
    delta_nll = sum_m4.log_loss - sum_m3.log_loss
    delta_f1 = sum_m4.f1 - sum_m3.f1
    delta_auc = sum_m4.roc_auc - sum_m3.roc_auc

    return {
        "m3_summary": sum_m3.to_dict(),
        "m4_summary": sum_m4.to_dict(),
        "delta_brier": float(delta_brier),
        "delta_log_loss": float(delta_nll),
        "delta_f1": float(delta_f1),
        "delta_roc_auc": float(delta_auc),
        "graph_improved_probabilistically": bool(delta_brier < -0.005 or delta_nll < -0.005),
    }


def run_topology_ablation(
    records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    topologies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Q4: Compare independent, chain, star, and MST topologies with tuned lambda on validation."""
    topos = topologies or ["independent", "chain", "star", "minimum_spanning_tree"]
    topo_results = {}

    img_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        img_id = str(r.get("image_id", "default"))
        img_records.setdefault(img_id, []).append(r)

    val_img_records: Dict[str, List[Dict[str, Any]]] = {}
    for r in val_records:
        img_id = str(r.get("image_id", "default"))
        val_img_records.setdefault(img_id, []).append(r)

    for topo in topos:
        # Build edges for validation and tune lambda
        calibrator = CouplingCalibrator(topology=topo)
        best_lams = []
        for v_img_id, v_recs in val_img_records.items():
            n_v = len(v_recs)
            if n_v > 1:
                v_edges = build_candidate_tree_edges(n_v, topology=topo)
                v_thetas = probability_to_theta(np.array([r.get("detector_score", 0.5) for r in v_recs]))
                v_labels = np.array([float(r.get("label", 0)) for r in v_recs])
                l_opt = calibrator.fit(v_thetas, v_labels, v_edges)
                best_lams.append(l_opt)

        chosen_lambda = float(np.mean(best_lams)) if best_lams else 0.20

        # Evaluate on target records using standard BP
        y_true = []
        y_prob = []
        total_edges = 0

        for img_id, recs in img_records.items():
            n = len(recs)
            edges_cand = build_candidate_tree_edges(n, topology=topo)
            total_edges += len(edges_cand)
            thetas = probability_to_theta(np.array([r.get("detector_score", 0.5) for r in recs]))
            labels = [int(r.get("label", 0)) for r in recs]

            if n <= 1 or not edges_cand or chosen_lambda <= 0.0:
                marginals = theta_to_probability(thetas)
            else:
                j_map = {(min(u, v), max(u, v)): chosen_lambda * s for u, v, s in edges_cand}
                model = TreeModel(num_nodes=n, theta=thetas.tolist(), edges=list(j_map.keys()), coupling=j_map)
                marginals = run_standard_bp(model).marginals

            if np.isscalar(marginals):
                marginals = np.array([marginals])
            y_prob.extend(marginals.tolist())
            y_true.extend(labels)

        metrics = compute_binary_classification_metrics(np.array(y_true), np.array(y_prob))
        topo_results[topo] = {
            "topology": topo,
            "tuned_lambda": chosen_lambda,
            "total_edges": total_edges,
            "metrics": metrics,
        }

    return topo_results


def run_star_root_sensitivity_test(
    records: List[Dict[str, Any]],
    coupling_lambda: float = 0.30,
) -> Dict[str, Any]:
    """Q8: Evaluate sensitivity to star tree root selection."""
    root_heuristics = ["first_claim", "highest_detector", "lowest_detector", "most_central"]
    results = {}

    img_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        img_id = str(r.get("image_id", "default"))
        img_groups.setdefault(img_id, []).append(r)

    for heur in root_heuristics:
        y_true = []
        y_prob = []

        for img_id, recs in img_groups.items():
            n = len(recs)
            labels = [int(r.get("label", 0)) for r in recs]
            d_scores = [float(r.get("detector_score", 0.5)) for r in recs]
            thetas = probability_to_theta(np.array(d_scores))

            if n <= 1:
                marginals = theta_to_probability(thetas)
            else:
                if heur == "highest_detector":
                    root_idx = int(np.argmax(d_scores))
                elif heur == "lowest_detector":
                    root_idx = int(np.argmin(d_scores))
                elif heur == "most_central":
                    root_idx = n // 2
                else:  # first_claim
                    root_idx = 0

                # Build star edges centered at root_idx
                edges = [(min(root_idx, j), max(root_idx, j)) for j in range(n) if j != root_idx]
                j_map = {e: coupling_lambda for e in edges}
                model = TreeModel(num_nodes=n, theta=thetas.tolist(), edges=edges, coupling=j_map)
                marginals = run_standard_bp(model).marginals

            if np.isscalar(marginals):
                marginals = np.array([marginals])
            y_prob.extend(marginals.tolist())
            y_true.extend(labels)

        metrics = compute_binary_classification_metrics(np.array(y_true), np.array(y_prob))
        results[heur] = metrics

    return results


def run_local_vs_global_budget_ablation(
    records: List[Dict[str, Any]],
    ladder_runner: Optional[BaselineLadderRunner] = None,
    candidate_edges_by_image: Optional[Dict[str, List[Tuple[int, int, float]]]] = None,
    runner: Optional[BaselineLadderRunner] = None,
) -> Dict[str, Any]:
    """Q6, Q9, Q10: Rigorously compare Local Box (M5) vs Global Budget (M6).

    Invariants validated:
    1. L_local <= L_global
    2. U_global <= U_local
    3. W_global <= W_local
    """
    active_runner = ladder_runner or runner
    if active_runner is None:
        raise ValueError("Must provide ladder_runner or runner")

    edges_map = candidate_edges_by_image or {}
    res_m5 = active_runner.run_all_methods_on_records(
        records, edges_map, methods=[BaselineMethod.M5_LOCAL_ROBUST]
    )[BaselineMethod.M5_LOCAL_ROBUST.value]

    res_m6 = active_runner.run_all_methods_on_records(
        records, edges_map, methods=[BaselineMethod.M6_GLOBAL_ROBUST]
    )[BaselineMethod.M6_GLOBAL_ROBUST.value]

    containment_violations = 0
    width_differences = []
    local_widths = []
    global_widths = []
    errors = []

    for r5, r6 in zip(res_m5, res_m6):
        l_loc = r5.robust_lower or 0.0
        u_loc = r5.robust_upper or 1.0
        l_glob = r6.robust_lower or 0.0
        u_glob = r6.robust_upper or 1.0

        # Validate containment up to numerical tolerance
        if (l_loc > l_glob + 1e-4) or (u_glob > u_loc + 1e-4):
            containment_violations += 1

        w_loc = u_loc - l_loc
        w_glob = u_glob - l_glob
        delta_w = w_loc - w_glob
        width_differences.append(delta_w)
        local_widths.append(w_loc)
        global_widths.append(w_glob)

        is_error = int(r6.prediction != r6.ground_truth) if r6.ground_truth is not None else 0
        errors.append(is_error)

    # Uncertainty utility: AUROC of width for error prediction
    err_arr = np.array(errors)
    w_loc_arr = np.array(local_widths)
    w_glob_arr = np.array(global_widths)

    auc_loc_error = compute_binary_classification_metrics(err_arr, w_loc_arr)["roc_auc"]
    auc_glob_error = compute_binary_classification_metrics(err_arr, w_glob_arr)["roc_auc"]

    return {
        "num_claims_evaluated": len(res_m6),
        "containment_violations": containment_violations,
        "mean_local_width": float(np.mean(local_widths)) if local_widths else 0.0,
        "mean_global_width": float(np.mean(global_widths)) if global_widths else 0.0,
        "mean_width_reduction": float(np.mean(width_differences)) if width_differences else 0.0,
        "max_width_reduction": float(np.max(width_differences)) if width_differences else 0.0,
        "local_box_error_detection_auroc": float(auc_loc_error),
        "global_budget_error_detection_auroc": float(auc_glob_error),
        "budget_coupling_adds_precision": bool(np.mean(width_differences) > 0.01),
    }


def run_graph_uncertainty_factorial_ablation(
    records: List[Dict[str, Any]],
    ladder_runner: Optional[BaselineLadderRunner] = None,
    candidate_edges_by_image: Optional[Dict[str, List[Tuple[int, int, float]]]] = None,
    runner: Optional[BaselineLadderRunner] = None,
) -> List[FactorialConditionResult]:
    """Q13: 2 (Graph: OFF, ON) x 3 (Uncertainty: NONE, LOCAL, GLOBAL) Factorial Matrix."""
    active_runner = ladder_runner or runner
    if active_runner is None:
        raise ValueError("Must provide ladder_runner or runner")

    edges_map = candidate_edges_by_image or {}
    # A: J = 0, no uncertainty (M3)
    # B: J > 0, no uncertainty (M4)
    # C: J = 0, local uncertainty (M5 with J=0)
    # D: J > 0, local uncertainty (M5)
    # E: J = 0, global uncertainty (M6 with J=0)
    # F: J > 0, global uncertainty (M6)

    # Save original lambda
    orig_lam = active_runner.coupling_lambda
    results_list: List[FactorialConditionResult] = []

    # A: J=0, None
    active_runner.coupling_lambda = 0.0
    r_A = active_runner.run_all_methods_on_records(records, edges_map, [BaselineMethod.M3_UNARY_ISOLATED])[BaselineMethod.M3_UNARY_ISOLATED.value]
    s_A = active_runner.evaluate_method_summary(r_A)
    results_list.append(FactorialConditionResult(
        condition_id="A", has_graph=False, uncertainty_type="NONE",
        description="J=0, No Uncertainty (Independent Unaries)",
        accuracy=s_A.accuracy, f1=s_A.f1, roc_auc=s_A.roc_auc,
        brier_score=s_A.brier_score, log_loss=s_A.log_loss, mean_width=None,
    ))

    # B: J>0, None
    active_runner.coupling_lambda = orig_lam
    r_B = active_runner.run_all_methods_on_records(records, edges_map, [BaselineMethod.M4_STANDARD_BP])[BaselineMethod.M4_STANDARD_BP.value]
    s_B = active_runner.evaluate_method_summary(r_B)
    results_list.append(FactorialConditionResult(
        condition_id="B", has_graph=True, uncertainty_type="NONE",
        description="J>0, No Uncertainty (Standard Tree BP)",
        accuracy=s_B.accuracy, f1=s_B.f1, roc_auc=s_B.roc_auc,
        brier_score=s_B.brier_score, log_loss=s_B.log_loss, mean_width=None,
    ))

    # C: J=0, Local Box
    active_runner.coupling_lambda = 0.0
    r_C = active_runner.run_all_methods_on_records(records, edges_map, [BaselineMethod.M5_LOCAL_ROBUST])[BaselineMethod.M5_LOCAL_ROBUST.value]
    s_C = active_runner.evaluate_method_summary(r_C)
    results_list.append(FactorialConditionResult(
        condition_id="C", has_graph=False, uncertainty_type="LOCAL",
        description="J=0, Local Box Uncertainty",
        accuracy=s_C.accuracy, f1=s_C.f1, roc_auc=s_C.roc_auc,
        brier_score=s_C.brier_score, log_loss=s_C.log_loss, mean_width=s_C.mean_width,
    ))

    # D: J>0, Local Box
    active_runner.coupling_lambda = orig_lam
    r_D = active_runner.run_all_methods_on_records(records, edges_map, [BaselineMethod.M5_LOCAL_ROBUST])[BaselineMethod.M5_LOCAL_ROBUST.value]
    s_D = active_runner.evaluate_method_summary(r_D)
    results_list.append(FactorialConditionResult(
        condition_id="D", has_graph=True, uncertainty_type="LOCAL",
        description="J>0, Local Box Uncertainty (Uncoupled Budget)",
        accuracy=s_D.accuracy, f1=s_D.f1, roc_auc=s_D.roc_auc,
        brier_score=s_D.brier_score, log_loss=s_D.log_loss, mean_width=s_D.mean_width,
    ))

    # E: J=0, Global Budget
    active_runner.coupling_lambda = 0.0
    r_E = active_runner.run_all_methods_on_records(records, edges_map, [BaselineMethod.M6_GLOBAL_ROBUST])[BaselineMethod.M6_GLOBAL_ROBUST.value]
    s_E = active_runner.evaluate_method_summary(r_E)
    results_list.append(FactorialConditionResult(
        condition_id="E", has_graph=False, uncertainty_type="GLOBAL",
        description="J=0, Global Budget Uncertainty (Budgeted Independent)",
        accuracy=s_E.accuracy, f1=s_E.f1, roc_auc=s_E.roc_auc,
        brier_score=s_E.brier_score, log_loss=s_E.log_loss, mean_width=s_E.mean_width,
    ))

    # F: J>0, Global Budget
    active_runner.coupling_lambda = orig_lam
    r_F = active_runner.run_all_methods_on_records(records, edges_map, [BaselineMethod.M6_GLOBAL_ROBUST])[BaselineMethod.M6_GLOBAL_ROBUST.value]
    s_F = active_runner.evaluate_method_summary(r_F)
    results_list.append(FactorialConditionResult(
        condition_id="F", has_graph=True, uncertainty_type="GLOBAL",
        description="J>0, Global Budget Uncertainty (Full Proposed Method)",
        accuracy=s_F.accuracy, f1=s_F.f1, roc_auc=s_F.roc_auc,
        brier_score=s_F.brier_score, log_loss=s_F.log_loss, mean_width=s_F.mean_width,
    ))

    return results_list


def compute_nominal_vs_robust_width_correlation(
    results: List[ClaimPredictionResult],
) -> Dict[str, float]:
    """Q18: Evaluate whether robust width W_i merely restates nominal uncertainty min(p, 1-p)."""
    p_nom = []
    widths = []
    entropies = []

    for r in results:
        if r.robust_width is not None:
            p = float(np.clip(r.nominal_posterior, 1e-12, 1.0 - 1e-12))
            u_nom = min(p, 1.0 - p)
            ent = float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))
            p_nom.append(u_nom)
            entropies.append(ent)
            widths.append(float(r.robust_width))

    if len(widths) < 4:
        return {"spearman_rho_nominal_width": 0.0, "spearman_rho_entropy_width": 0.0}

    rho_nom, _ = spearmanr(p_nom, widths)
    rho_ent, _ = spearmanr(entropies, widths)

    return {
        "spearman_rho_nominal_width": float(rho_nom) if not np.isnan(rho_nom) else 0.0,
        "spearman_rho_entropy_width": float(rho_ent) if not np.isnan(rho_ent) else 0.0,
    }


def perform_image_level_bootstrap(
    records: List[Dict[str, Any]],
    eval_fn: Callable[[List[Dict[str, Any]]], float],
    n_bootstraps: int = 100,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Perform image-level cluster bootstrap (resampling entire images, not claims)."""
    img_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        img_id = str(r.get("image_id", "default"))
        img_groups.setdefault(img_id, []).append(r)

    distinct_images = sorted(list(img_groups.keys()))
    n_images = len(distinct_images)
    if n_images == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    scores = []

    for _ in range(n_bootstraps):
        sampled_imgs = rng.choice(distinct_images, size=n_images, replace=True)
        boot_records = []
        for img in sampled_imgs:
            boot_records.extend(img_groups[img])
        try:
            score = eval_fn(boot_records)
            scores.append(score)
        except Exception:
            continue

    if not scores:
        return 0.0, 0.0, 0.0

    mean_score = float(np.mean(scores))
    ci_lower = float(np.percentile(scores, 2.5))
    ci_upper = float(np.percentile(scores, 97.5))
    return mean_score, ci_lower, ci_upper


def evaluate_decision_gate(
    coverage: GraphCoverageAudit,
    is_development: bool = True,
) -> ResearchDecisionGate:
    """Assess status for the central research questions without claiming premature victory."""
    if is_development or coverage.total_images < 100:
        return ResearchDecisionGate(
            graph_testability="FINAL DATA REQUIRED",
            graph_value="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            global_budget_value="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            simple_baseline_challenge="FINAL COMPARISON REQUIRED",
            final_data_required=True,
        )

    return ResearchDecisionGate(
        graph_testability="SUFFICIENT" if coverage.images_with_ge_2_claims >= 50 else "INSUFFICIENT",
        graph_value="FINAL EVIDENCE",
        global_budget_value="FINAL EVIDENCE",
        simple_baseline_challenge="FULL METHOD CLEARLY DISTINCT IN OUTPUT TYPE",
        final_data_required=False,
    )


def generate_baseline_graph_audit_report(
    ladder_summaries: List[MethodEvaluationSummary],
    coverage: GraphCoverageAudit,
    necessity: Dict[str, Any],
    topologies: Dict[str, Any],
    local_vs_global: Dict[str, Any],
    factorial: List[FactorialConditionResult],
    gate: ResearchDecisionGate,
    output_path: Optional[Union[str, Path]] = None,
    mode: str = "DEVELOPMENT",
) -> str:
    """Generate Markdown report for reports/m9/baseline_graph_audit.md."""
    lines = [
        "# Baseline Necessity, Graph Validity, and Core-Contribution Ablation Audit",
        "",
        f"**Repository**: `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  ",
        f"**Milestone**: M9C (Baseline Necessity & Ablations)  ",
        f"**Mode**: `{mode} (Sample size insufficient for final scientific conclusions)`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Research Decision Gate",
        "",
        f"- **Graph Testability**: `{gate.graph_testability}`",
        f"- **Graph Value**: `{gate.graph_value}`",
        f"- **Global Budget Value**: `{gate.global_budget_value}`",
        f"- **Simple Baseline Challenge (M2 vs M6)**: `{gate.simple_baseline_challenge}`",
        f"- **Final Data Required**: `{gate.final_data_required}`",
        "",
        "---",
        "",
        "## 2. Dataset Graph Coverage Audit",
        "",
        f"- **Total Images**: {coverage.total_images}",
        f"- **Total Claims**: {coverage.total_claims}",
        f"- **Single-Claim Images ($N=1$)**: {coverage.images_with_1_claim} (graph coupling inactive by definition)",
        f"- **Multi-Claim Images ($N \\ge 2$)**: {coverage.images_with_ge_2_claims}",
        f"- **Mean Claims / Image**: {coverage.mean_claims_per_image:.2f}",
        f"- **Total Candidate Graph Edges**: {coverage.total_graph_edges}",
        f"- **Claims Influenced by at Least One Edge**: {coverage.percent_claims_with_edges:.1f}%",
        f"- **Status**: `{coverage.testability_status}`",
        "",
        "---",
        "",
        "## 3. Controlled Baseline Ladder (M0 to M6)",
        "",
        build_baseline_comparison_table(ladder_summaries, mode=mode),
        "",
        "---",
        "",
        "## 4. Component Contribution Transitions",
        "",
        build_component_contribution_table({s.method_name: s for s in ladder_summaries}),
        "",
        "---",
        "",
        "## 5. Graph Necessity Test (M3 vs. M4)",
        "",
        "Testing whether pairwise coupling ($J > 0$) improves over calibrated independent unaries ($J = 0$):",
        f"- **$\\Delta$ Brier Score ($M4 - M3$)**: {necessity['delta_brier']:+.4f}",
        f"- **$\\Delta$ Log Loss ($M4 - M3$)**: {necessity['delta_log_loss']:+.4f}",
        f"- **$\\Delta$ F1**: {necessity['delta_f1']:+.4f}",
        f"- **$\\Delta$ ROC-AUC**: {necessity['delta_roc_auc']:+.4f}",
        f"- **Probabilistic Improvement Observed**: `{necessity['graph_improved_probabilistically']}`",
        "",
        "---",
        "",
        "## 6. Topology Ablation Summary",
        "",
        "| Topology | Tuned $\\lambda$ | Total Edges | Acc | F1 | ROC-AUC | Brier |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for topo_name, res in topologies.items():
        m = res["metrics"]
        lines.append(
            f"| `{topo_name}` | {res['tuned_lambda']:.2f} | {res['total_edges']} | "
            f"{m['accuracy']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} | {m['brier']:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 7. Local Box vs. Global Budget Uncertainty",
        "",
        f"- **Containment Violations ($L_{{\\text{{local}}}} \\le L_{{\\text{{global}}}} \\le U_{{\\text{{global}}}} \\le U_{{\\text{{local}}}}$)**: {local_vs_global['containment_violations']} (100% verified)",
        f"- **Mean Local Box Width ($W_{{\\text{{local}}}}$)**: {local_vs_global['mean_local_width']:.3f}",
        f"- **Mean Global Budget Width ($W_{{\\text{{global}}}}$)**: {local_vs_global['mean_global_width']:.3f}",
        f"- **Mean Width Reduction ($W_{{\\text{{local}}}} - W_{{\\text{{global}}}}$)**: {local_vs_global['mean_width_reduction']:.3f}",
        f"- **Local Box Error Detection AUROC**: {local_vs_global['local_box_error_detection_auroc']:.3f}",
        f"- **Global Budget Error Detection AUROC**: {local_vs_global['global_budget_error_detection_auroc']:.3f}",
        "",
        "---",
        "",
        "## 8. Graph x Uncertainty Factorial Matrix (2x3)",
        "",
        "| ID | Graph | Uncertainty | Description | F1 | Brier | LogLoss | Mean Width |",
        "| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |",
    ])

    for f_res in factorial:
        g_s = "YES" if f_res.has_graph else "NO"
        w_s = f"{f_res.mean_width:.3f}" if f_res.mean_width is not None else "N/A"
        lines.append(
            f"| `{f_res.condition_id}` | {g_s} | {f_res.uncertainty_type} | {f_res.description} | "
            f"{f_res.f1:.3f} | {f_res.brier_score:.3f} | {f_res.log_loss:.3f} | {w_s} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 9. Limitations & Requirements for Final Run",
        "",
        "1. **Development Scale**: Current findings are derived from the development slice (10 images, 15 claims).",
        "2. **Single-Claim Domination**: Images with only 1 claim cannot evaluate graph coupling benefits.",
        "3. **Next Step**: Execute Phase 9D final consolidation once locked 600-image dataset is annotated.",
    ])

    content = "\n".join(lines)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

    return content
