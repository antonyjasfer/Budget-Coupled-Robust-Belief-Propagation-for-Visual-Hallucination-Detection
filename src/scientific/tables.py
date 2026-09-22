"""
Milestone 9 Publication-Ready Table Generator.

Produces Tables 1 through 10 in CSV, JSON, Markdown, and LaTeX formats.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics
from src.experiments.interval_metrics import compute_interval_statistics, RobustIntervalStatistics

logger = logging.getLogger("m9_tables")


def format_num(val: Optional[float], decimals: int = 3) -> str:
    """Format float or return 'N/A'."""
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def save_m9_table(
    table_id: str,
    title: str,
    rows: List[Dict[str, Any]],
    output_dir: Path,
    default_headers: Optional[List[str]] = None,
) -> None:
    """Save a table in JSON, CSV, Markdown, and LaTeX formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else (default_headers or ["Metric", "Value"])

    # 1. JSON
    with open(output_dir / f"{table_id}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    # 2. CSV
    with open(output_dir / f"{table_id}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    # 3. Markdown
    md_lines = [f"### {title}\n", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    (output_dir / f"{table_id}.md").write_text("\n".join(md_lines), encoding="utf-8")

    # 4. LaTeX
    latex_cols = "l" + "c" * (len(headers) - 1)
    latex_lines = [
        f"% {title}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{title}}}",
        f"\\label{{tab:{table_id}}}",
        f"\\begin{{tabular}}{{{latex_cols}}}",
        "\\hline",
        " & ".join(h.replace("_", "\\_").replace("%", "\\%") for h in headers) + " \\\\",
        "\\hline",
    ]
    for r in rows:
        row_vals = [str(r.get(h, "")).replace("_", "\\_").replace("%", "\\%") for h in headers]
        latex_lines.append(" & ".join(row_vals) + " \\\\")
    latex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    (output_dir / f"{table_id}.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def generate_table_1_dataset_composition(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 1: Dataset composition."""
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed"]] or results
    splits = sorted(list({r.split for r in clean_res}))

    rows: List[Dict[str, Any]] = []
    total_imgs: Set[str] = set()
    total_claims = 0
    total_supp = 0
    total_halluc = 0
    total_unk = 0

    for s in splits:
        s_res = [r for r in clean_res if r.split == s]
        imgs = len({r.image_id for r in s_res})
        total_imgs.update({r.image_id for r in s_res})
        claims = len(s_res)
        supp = sum(1 for r in s_res if r.ground_truth == "supported")
        halluc = sum(1 for r in s_res if r.ground_truth == "hallucinated")
        unk = sum(1 for r in s_res if r.ground_truth == "unknown" or r.ground_truth is None)

        total_claims += claims
        total_supp += supp
        total_halluc += halluc
        total_unk += unk

        rows.append({
            "Split": s.upper(),
            "Images": imgs,
            "Claims": claims,
            "Supported (GT)": supp,
            "Hallucinated (GT)": halluc,
            "Unknown (GT)": unk,
        })

    # Summary row
    rows.append({
        "Split": "TOTAL",
        "Images": len(total_imgs),
        "Claims": total_claims,
        "Supported (GT)": total_supp,
        "Hallucinated (GT)": total_halluc,
        "Unknown (GT)": total_unk,
    })

    save_m9_table("table1_dataset_composition", "Table 1: Benchmark Dataset Composition", rows, out_dir)
    return rows


def generate_table_2_main_comparison(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 2: Main method comparison."""
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed", "standard_bp"]] or results

    m_point = evaluate_claim_results(clean_res, method_name="evidence_point")
    m_std = evaluate_claim_results(clean_res, method_name="standard_bp")
    m_rob_all = evaluate_claim_results(clean_res, method_name="robust_bp", exclude_abstain=False)
    m_rob_sel = evaluate_claim_results(clean_res, method_name="robust_bp", exclude_abstain=True)

    rows = [
        {
            "Method": "Evidence Point Baseline",
            "Accuracy": format_num(m_point.accuracy),
            "Precision": format_num(m_point.precision),
            "Recall": format_num(m_point.recall),
            "F1 Score": format_num(m_point.f1),
            "ROC-AUC": format_num(m_point.roc_auc),
            "Coverage": f"{m_point.coverage_rate * 100:.1f}%",
            "Abstentions": m_point.abstained_count,
        },
        {
            "Method": "Standard BP (Point Posterior)",
            "Accuracy": format_num(m_std.accuracy),
            "Precision": format_num(m_std.precision),
            "Recall": format_num(m_std.recall),
            "F1 Score": format_num(m_std.f1),
            "ROC-AUC": format_num(m_std.roc_auc),
            "Coverage": f"{m_std.coverage_rate * 100:.1f}%",
            "Abstentions": m_std.abstained_count,
        },
        {
            "Method": "Robust BP (Full / Midpoint)",
            "Accuracy": format_num(m_rob_all.accuracy),
            "Precision": format_num(m_rob_all.precision),
            "Recall": format_num(m_rob_all.recall),
            "F1 Score": format_num(m_rob_all.f1),
            "ROC-AUC": format_num(m_rob_all.roc_auc),
            "Coverage": f"{m_rob_all.coverage_rate * 100:.1f}%",
            "Abstentions": 0,
        },
        {
            "Method": "Robust BP (Selective / Non-Abstained)",
            "Accuracy": format_num(m_rob_sel.accuracy),
            "Precision": format_num(m_rob_sel.precision),
            "Recall": format_num(m_rob_sel.recall),
            "F1 Score": format_num(m_rob_sel.f1),
            "ROC-AUC": format_num(m_rob_sel.roc_auc),
            "Coverage": f"{m_rob_sel.coverage_rate * 100:.1f}%",
            "Abstentions": m_rob_sel.abstained_count,
        },
    ]

    save_m9_table("table2_main_method_comparison", "Table 2: Main Method Performance Comparison", rows, out_dir)
    return rows


def generate_table_3_interval_statistics(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 3: Robust interval statistics."""
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed"]] or results
    inv = compute_interval_statistics(clean_res)

    rows = [
        {"Statistic": "Total Evaluated Claims", "Value": str(inv.total_claims)},
        {"Statistic": "Mean Interval Width", "Value": format_num(inv.mean_width)},
        {"Statistic": "Median Interval Width", "Value": format_num(inv.median_width)},
        {"Statistic": "Std Dev Interval Width", "Value": format_num(inv.std_width)},
        {"Statistic": "Min Interval Width", "Value": format_num(inv.min_width)},
        {"Statistic": "Max Interval Width", "Value": format_num(inv.max_width)},
        {"Statistic": "25th Percentile Width", "Value": format_num(inv.quantiles.get("p25"))},
        {"Statistic": "75th Percentile Width", "Value": format_num(inv.quantiles.get("p75"))},
        {"Statistic": "90th Percentile Width", "Value": format_num(inv.quantiles.get("p90"))},
        {"Statistic": "Mean Width (Supported GT)", "Value": format_num(inv.mean_width_supported)},
        {"Statistic": "Mean Width (Hallucinated GT)", "Value": format_num(inv.mean_width_hallucinated)},
        {"Statistic": "Mean Width (BP Correct)", "Value": format_num(inv.mean_width_correct_bp)},
        {"Statistic": "Mean Width (BP Wrong)", "Value": format_num(inv.mean_width_wrong_bp)},
        {"Statistic": "Evidence-Sensitive Rate", "Value": f"{inv.evidence_sensitive_fraction * 100:.1f}%"},
        {"Statistic": "Robustly Supported Fraction", "Value": f"{inv.robustly_supported_fraction * 100:.1f}%"},
        {"Statistic": "Robustly Hallucinated Fraction", "Value": f"{inv.robustly_hallucinated_fraction * 100:.1f}%"},
    ]

    save_m9_table("table3_interval_statistics", "Table 3: Robust Posterior Interval Statistics", rows, out_dir)
    return rows


def generate_table_4_ablation_study(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 4: Ablation study."""
    conditions = sorted(list({r.condition for r in results}))
    rows = []

    for cond in conditions:
        sub = [r for r in results if r.condition == cond]
        m = evaluate_claim_results(sub, method_name="robust_bp")
        inv = compute_interval_statistics(sub)

        rows.append({
            "Condition": cond,
            "Claims": len(sub),
            "Accuracy": format_num(m.accuracy),
            "F1 Score": format_num(m.f1),
            "ROC-AUC": format_num(m.roc_auc),
            "Mean Width": format_num(inv.mean_width),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
        })

    save_m9_table(
        "table4_ablation_study",
        "Table 4: Ablation Analysis (Coupling, Budget, Uncertainty Constraints)",
        rows,
        out_dir,
        default_headers=["Condition", "Claims", "Accuracy", "F1 Score", "ROC-AUC", "Mean Width", "Evidence-Sensitive Rate"],
    )
    return rows


def generate_table_5_budget_sensitivity(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 5: Budget sensitivity."""
    budgets = sorted(list({r.budget for r in results}))
    rows = []

    for b in budgets:
        sub = [r for r in results if abs(r.budget - b) < 1e-6]
        m = evaluate_claim_results(sub, method_name="robust_bp", exclude_abstain=True)
        inv = compute_interval_statistics(sub)
        runtime = float(np.mean([r.runtime_ms for r in sub])) if sub else 0.0

        rows.append({
            "Budget B": f"{b:.4f}",
            "Claims": len(sub),
            "Mean Width": format_num(inv.mean_width),
            "Median Width": format_num(inv.median_width),
            "F1 (Selective)": format_num(m.f1),
            "ROC-AUC": format_num(m.roc_auc),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
            "Abstentions": m.abstained_count,
            "Mean Runtime (ms)": f"{runtime:.2f}",
        })

    save_m9_table(
        "table5_budget_sensitivity",
        "Table 5: Budget Sensitivity Analysis ($B \\in \\mathcal{U}(B, \\epsilon)$)",
        rows,
        out_dir,
        default_headers=["Budget B", "Claims", "Mean Width", "Median Width", "F1 (Selective)", "ROC-AUC", "Evidence-Sensitive Rate", "Abstentions", "Mean Runtime (ms)"],
    )
    return rows


def generate_table_6_epsilon_sensitivity(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 6: Epsilon sensitivity."""
    scales = sorted(list({r.epsilon_scale for r in results}))
    rows = []

    for s in scales:
        sub = [r for r in results if abs(r.epsilon_scale - s) < 1e-6]
        m = evaluate_claim_results(sub, method_name="robust_bp", exclude_abstain=True)
        inv = compute_interval_statistics(sub)
        runtime = float(np.mean([r.runtime_ms for r in sub])) if sub else 0.0

        rows.append({
            "Epsilon Scale (alpha)": f"{s:.2f}",
            "Claims": len(sub),
            "Mean Width": format_num(inv.mean_width),
            "F1 (Selective)": format_num(m.f1),
            "ROC-AUC": format_num(m.roc_auc),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
            "Mean Runtime (ms)": f"{runtime:.2f}",
        })

    save_m9_table(
        "table6_epsilon_sensitivity",
        "Table 6: Uncertainty Scaling Sensitivity ($\\epsilon'_i = \\alpha \\epsilon_i$)",
        rows,
        out_dir,
        default_headers=["Epsilon Scale (alpha)", "Claims", "Mean Width", "F1 (Selective)", "ROC-AUC", "Evidence-Sensitive Rate", "Mean Runtime (ms)"],
    )
    return rows


def generate_table_7_grid_resolution(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 7: Grid-resolution analysis."""
    resolutions = sorted(list({r.grid_steps for r in results}))
    rows = []

    for k in resolutions:
        sub = [r for r in results if r.grid_steps == k]
        inv = compute_interval_statistics(sub)
        runtimes = [r.runtime_ms for r in sub]
        mean_rt = float(np.mean(runtimes)) if runtimes else 0.0

        rows.append({
            "Grid Steps (K)": str(k),
            "Claims": len(sub),
            "Mean Width": format_num(inv.mean_width),
            "Median Width": format_num(inv.median_width),
            "Std Width": format_num(inv.std_width),
            "Mean Runtime per Claim (ms)": f"{mean_rt:.2f}",
        })

    save_m9_table(
        "table7_grid_resolution",
        "Table 7: Discrete-Budget Grid Approximation Analysis ($K$ steps)",
        rows,
        out_dir,
        default_headers=["Grid Steps (K)", "Claims", "Mean Width", "Median Width", "Std Width", "Mean Runtime per Claim (ms)"],
    )
    return rows


def generate_table_8_corruption_robustness(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 8: Corruption robustness."""
    corrupt_res = [r for r in results if r.corruption_type != "none" or r.corruption_severity == "clean"]
    conditions = sorted(list({(r.corruption_type, r.corruption_severity) for r in corrupt_res}))
    if not conditions:
        conditions = [("none", "clean")]
    rows = []

    for c_type, sev in conditions:
        sub = [r for r in corrupt_res if r.corruption_type == c_type and r.corruption_severity == sev]
        m_std = evaluate_claim_results(sub, method_name="standard_bp")
        m_rob = evaluate_claim_results(sub, method_name="robust_bp", exclude_abstain=True)
        inv = compute_interval_statistics(sub)

        rows.append({
            "Corruption Type": c_type,
            "Severity": sev,
            "Claims": len(sub),
            "Std BP Accuracy": format_num(m_std.accuracy),
            "Rob BP (Selective) Acc": format_num(m_rob.accuracy),
            "Mean Interval Width": format_num(inv.mean_width),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
        })

    save_m9_table(
        "table8_corruption_robustness",
        "Table 8: Performance and Uncertainty Under Controlled Visual Corruptions",
        rows,
        out_dir,
        default_headers=["Corruption Type", "Severity", "Claims", "Std BP Accuracy", "Rob BP (Selective) Acc", "Mean Interval Width", "Evidence-Sensitive Rate"],
    )
    return rows


def generate_table_9_computational_cost(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 9: Computational cost."""
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed"]] or results
    n_claims = len(clean_res)
    n_images = len({r.image_id for r in clean_res})
    grid_steps = clean_res[0].grid_steps if clean_res else 100

    runtimes = [r.runtime_ms for r in clean_res]
    total_time_s = float(np.sum(runtimes) / 1000.0) if runtimes else 0.0
    mean_claim_ms = float(np.mean(runtimes)) if runtimes else 0.0
    mean_img_ms = float(total_time_s * 1000.0 / n_images) if n_images > 0 else 0.0

    rows = [
        {"Parameter": "Total Images Evaluated", "Value": str(n_images)},
        {"Parameter": "Total Claims Evaluated", "Value": str(n_claims)},
        {"Parameter": "Robust Grid Resolution K", "Value": str(grid_steps)},
        {"Parameter": "Mean Runtime per Claim (ms)", "Value": f"{mean_claim_ms:.2f} ms"},
        {"Parameter": "Mean Runtime per Image (ms)", "Value": f"{mean_img_ms:.2f} ms"},
        {"Parameter": "Total Inference Time (s)", "Value": f"{total_time_s:.3f} s"},
    ]

    save_m9_table("table9_computational_cost", "Table 9: Computational Runtime and Resource Analysis", rows, out_dir)
    return rows


def generate_table_10_evidence_sensitive_analysis(results: List[ClaimEvaluationResult], out_dir: Path) -> List[Dict[str, Any]]:
    """Table 10: Evidence-sensitive decision analysis."""
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed"]] or results
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    rows = []

    for tau in thresholds:
        n_total = len(clean_res)
        n_sensitive = sum(1 for r in clean_res if r.robust_lower <= tau <= r.robust_upper)
        rate_sens = (n_sensitive / max(1, n_total)) * 100.0

        # Accuracy on non-sensitive claims
        non_sens = [r for r in clean_res if not (r.robust_lower <= tau <= r.robust_upper) and r.ground_truth in ["supported", "hallucinated"]]
        m_sel = evaluate_claim_results(non_sens, method_name="standard_bp", decision_threshold=tau)

        # Full accuracy for comparison
        all_ann = [r for r in clean_res if r.ground_truth in ["supported", "hallucinated"]]
        m_full = evaluate_claim_results(all_ann, method_name="standard_bp", decision_threshold=tau)

        acc_gain = (m_sel.accuracy - m_full.accuracy) if (m_sel.accuracy is not None and m_full.accuracy is not None) else 0.0

        rows.append({
            "Threshold (tau)": f"{tau:.2f}",
            "Evaluated Claims": n_total,
            "Evidence-Sensitive Count": n_sensitive,
            "Abstention Rate": f"{rate_sens:.1f}%",
            "Full Standard BP Acc": format_num(m_full.accuracy),
            "Selective Acc (Non-Abstaining)": format_num(m_sel.accuracy),
            "Selective Accuracy Gain": f"{acc_gain * 100:+.1f}%",
        })

    save_m9_table(
        "table10_evidence_sensitive_analysis",
        "Table 10: Risk-Coverage and Evidence-Sensitive Abstention Trade-Off",
        rows,
        out_dir,
        default_headers=["Threshold (tau)", "Evaluated Claims", "Evidence-Sensitive Count", "Abstention Rate", "Full Standard BP Acc", "Selective Acc (Non-Abstaining)", "Selective Accuracy Gain"],
    )
    return rows


def generate_all_m9_tables(results: List[ClaimEvaluationResult], output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Generate all 10 publication tables."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    t1 = generate_table_1_dataset_composition(results, tables_dir)
    t2 = generate_table_2_main_comparison(results, tables_dir)
    t3 = generate_table_3_interval_statistics(results, tables_dir)
    t4 = generate_table_4_ablation_study(results, tables_dir)
    t5 = generate_table_5_budget_sensitivity(results, tables_dir)
    t6 = generate_table_6_epsilon_sensitivity(results, tables_dir)
    t7 = generate_table_7_grid_resolution(results, tables_dir)
    t8 = generate_table_8_corruption_robustness(results, tables_dir)
    t9 = generate_table_9_computational_cost(results, tables_dir)
    t10 = generate_table_10_evidence_sensitive_analysis(results, tables_dir)

    return {
        "table1": t1,
        "table2": t2,
        "table3": t3,
        "table4": t4,
        "table5": t5,
        "table6": t6,
        "table7": t7,
        "table8": t8,
        "table9": t9,
        "table10": t10,
    }
