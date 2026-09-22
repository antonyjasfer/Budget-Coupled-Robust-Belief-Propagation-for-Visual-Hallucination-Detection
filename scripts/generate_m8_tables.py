"""
Publication-quality table generator for Milestone 8 (M8) experimental results.
Outputs tables in CSV, JSON, Markdown, and LaTeX formats.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any
import numpy as np

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics
from src.experiments.interval_metrics import compute_interval_statistics, RobustIntervalStatistics


def load_raw_results(results_dir: Path) -> List[ClaimEvaluationResult]:
    """Load raw claim results from JSONL."""
    candidates = [
        results_dir / "raw" / "claim_evaluation_results.jsonl",
        results_dir / "raw_claim_results.jsonl",
        results_dir / "claim_evaluation_results.jsonl",
    ]
    for p in candidates:
        if p.exists():
            records: List[ClaimEvaluationResult] = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(ClaimEvaluationResult.from_dict(json.loads(line)))
            return records
    raise FileNotFoundError(f"Raw results file not found in any of: {candidates}")


def format_val(val: Optional[float], decimals: int = 3) -> str:
    """Format float or return N/A."""
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def save_table_outputs(
    table_dict: Dict[str, Any],
    rows: List[Dict[str, Any]],
    table_name: str,
    tables_dir: Path,
    default_headers: Optional[List[str]] = None,
) -> None:
    """Save table in JSON, CSV, and Markdown formats."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. JSON (rows list format)
    with open(tables_dir / f"{table_name}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    headers = list(rows[0].keys()) if rows else (default_headers or ["Metric", "Value"])

    # 2. CSV
    with open(tables_dir / f"{table_name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    # 3. Markdown
    md_lines = []
    md_lines.append(f"| " + " | ".join(headers) + " |")
    md_lines.append(f"| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md_lines.append(f"| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    (tables_dir / f"{table_name}.md").write_text("\n".join(md_lines), encoding="utf-8")


def generate_table_1_dataset_summary(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 1: Dataset summary."""
    clean_res = [r for r in results if r.condition in ["clean", "robust_bp_proposed"]] or results
    unique_images = sorted(list({r.image_id for r in clean_res}))
    unique_claims = sorted(list({r.claim_id for r in clean_res}))

    splits = sorted(list({r.split for r in clean_res}))
    split_counts: Dict[str, Dict[str, int]] = {}
    for s in splits:
        s_res = [r for r in clean_res if r.split == s]
        s_imgs = len({r.image_id for r in s_res})
        s_claims = len(s_res)
        s_supp = sum(1 for r in s_res if r.ground_truth == "supported")
        s_halluc = sum(1 for r in s_res if r.ground_truth == "hallucinated")
        s_unk = sum(1 for r in s_res if r.ground_truth == "unknown")
        s_unann = sum(1 for r in s_res if r.ground_truth is None)
        split_counts[s] = {
            "images": s_imgs,
            "claims": s_claims,
            "supported": s_supp,
            "hallucinated": s_halluc,
            "unknown": s_unk,
            "unannotated": s_unann,
        }

    rows = []
    for s, d in split_counts.items():
        rows.append({
            "Split": s.upper(),
            "Images": d["images"],
            "Claims": d["claims"],
            "Supported (GT)": d["supported"],
            "Hallucinated (GT)": d["hallucinated"],
            "Unknown (GT)": d["unknown"],
            "Unannotated": d["unannotated"],
        })

    summary_dict = {
        "table_id": "table1_dataset_summary",
        "total_images": len(unique_images),
        "total_claims": len(unique_claims),
        "splits": split_counts,
    }
    save_table_outputs(summary_dict, rows, "table1_dataset_summary", tables_dir)
    return summary_dict


def generate_table_2_main_comparison(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 2: Main method comparison."""
    clean_res = [r for r in results if r.condition in ["clean", "robust_bp_proposed", "standard_bp"]] or results

    m_point = evaluate_claim_results(clean_res, method_name="evidence_point")
    m_std = evaluate_claim_results(clean_res, method_name="standard_bp")
    m_rob_all = evaluate_claim_results(clean_res, method_name="robust_bp", exclude_abstain=False)
    m_rob_sel = evaluate_claim_results(clean_res, method_name="robust_bp", exclude_abstain=True)

    rows = [
        {
            "Method": "Evidence Point Baseline",
            "Accuracy": format_val(m_point.accuracy),
            "Precision": format_val(m_point.precision),
            "Recall": format_val(m_point.recall),
            "F1": format_val(m_point.f1),
            "ROC-AUC": format_val(m_point.roc_auc),
            "Coverage": f"{m_point.coverage_rate * 100:.1f}%",
            "Abstentions": m_point.abstained_count,
        },
        {
            "Method": "Standard BP (Point Posterior)",
            "Accuracy": format_val(m_std.accuracy),
            "Precision": format_val(m_std.precision),
            "Recall": format_val(m_std.recall),
            "F1": format_val(m_std.f1),
            "ROC-AUC": format_val(m_std.roc_auc),
            "Coverage": f"{m_std.coverage_rate * 100:.1f}%",
            "Abstentions": m_std.abstained_count,
        },
        {
            "Method": "Robust BP (Full / Midpoint)",
            "Accuracy": format_val(m_rob_all.accuracy),
            "Precision": format_val(m_rob_all.precision),
            "Recall": format_val(m_rob_all.recall),
            "F1": format_val(m_rob_all.f1),
            "ROC-AUC": format_val(m_rob_all.roc_auc),
            "Coverage": f"{m_rob_all.coverage_rate * 100:.1f}%",
            "Abstentions": 0,
        },
        {
            "Method": "Robust BP (Selective / Non-Abstained)",
            "Accuracy": format_val(m_rob_sel.accuracy),
            "Precision": format_val(m_rob_sel.precision),
            "Recall": format_val(m_rob_sel.recall),
            "F1": format_val(m_rob_sel.f1),
            "ROC-AUC": format_val(m_rob_sel.roc_auc),
            "Coverage": f"{m_rob_sel.coverage_rate * 100:.1f}%",
            "Abstentions": m_rob_sel.abstained_count,
        },
    ]

    out_dict = {
        "table_id": "table2_method_comparison",
        "evidence_point": m_point.to_dict(),
        "standard_bp": m_std.to_dict(),
        "robust_bp_full": m_rob_all.to_dict(),
        "robust_bp_selective": m_rob_sel.to_dict(),
    }
    save_table_outputs(out_dict, rows, "table2_method_comparison", tables_dir)
    return out_dict


def generate_table_3_interval_statistics(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 3: Interval statistics."""
    clean_res = [r for r in results if r.condition in ["clean", "robust_bp_proposed"]] or results
    inv = compute_interval_statistics(clean_res)

    rows = [
        {"Metric": "Total Evaluated Claims", "Value": str(inv.total_claims)},
        {"Metric": "Mean Interval Width", "Value": format_val(inv.mean_width)},
        {"Metric": "Median Interval Width", "Value": format_val(inv.median_width)},
        {"Metric": "Std Dev Interval Width", "Value": format_val(inv.std_width)},
        {"Metric": "Min Interval Width", "Value": format_val(inv.min_width)},
        {"Metric": "Max Interval Width", "Value": format_val(inv.max_width)},
        {"Metric": "25th Percentile Width", "Value": format_val(inv.quantiles.get("p25"))},
        {"Metric": "75th Percentile Width", "Value": format_val(inv.quantiles.get("p75"))},
        {"Metric": "90th Percentile Width", "Value": format_val(inv.quantiles.get("p90"))},
        {"Metric": "Mean Width (Supported GT)", "Value": format_val(inv.mean_width_supported)},
        {"Metric": "Mean Width (Hallucinated GT)", "Value": format_val(inv.mean_width_hallucinated)},
        {"Metric": "Mean Width (BP Correct)", "Value": format_val(inv.mean_width_correct_bp)},
        {"Metric": "Mean Width (BP Wrong)", "Value": format_val(inv.mean_width_wrong_bp)},
        {"Metric": "Evidence-Sensitive Rate", "Value": f"{inv.evidence_sensitive_fraction * 100:.1f}%"},
        {"Metric": "Pearson r (Width vs BP Error)", "Value": format_val(inv.pearson_corr_width_error)},
        {"Metric": "Pearson p-value", "Value": format_val(inv.pearson_p_value, decimals=4)},
        {"Metric": "Spearman rho (Width vs BP Error)", "Value": format_val(inv.spearman_corr_width_error)},
        {"Metric": "Spearman p-value", "Value": format_val(inv.spearman_p_value, decimals=4)},
    ]

    out_dict = {"table_id": "table3_interval_statistics", "statistics": inv.to_dict()}
    save_table_outputs(out_dict, rows, "table3_interval_statistics", tables_dir)
    return out_dict


def generate_table_4_ablation_study(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 4: Ablation study."""
    conditions = sorted(list({r.condition for r in results}))
    rows = []
    summary: Dict[str, Any] = {}

    for cond in conditions:
        sub = [r for r in results if r.condition == cond]
        m = evaluate_claim_results(sub, method_name="robust_bp")
        inv = compute_interval_statistics(sub)

        rows.append({
            "Condition": cond,
            "Claims": len(sub),
            "Accuracy": format_val(m.accuracy),
            "F1": format_val(m.f1),
            "ROC-AUC": format_val(m.roc_auc),
            "Mean Width": format_val(inv.mean_width),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
        })
        summary[cond] = {"metrics": m.to_dict(), "intervals": inv.to_dict()}

    out_dict = {"table_id": "table4_ablation_study", "ablations": summary}
    save_table_outputs(out_dict, rows, "table4_ablation_study", tables_dir)
    return out_dict


def generate_table_5_budget_sensitivity(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 5: Budget sensitivity."""
    budgets = sorted(list({r.budget for r in results}))
    rows = []
    summary: Dict[str, Any] = {}

    for b in budgets:
        sub = [r for r in results if abs(r.budget - b) < 1e-6]
        m = evaluate_claim_results(sub, method_name="robust_bp", exclude_abstain=True)
        inv = compute_interval_statistics(sub)
        runtime = float(np.mean([r.runtime_ms for r in sub])) if sub else 0.0

        rows.append({
            "Budget B": f"{b:.4f}",
            "Claims": len(sub),
            "Mean Width": format_val(inv.mean_width),
            "Median Width": format_val(inv.median_width),
            "F1 (Selective)": format_val(m.f1),
            "ROC-AUC": format_val(m.roc_auc),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
            "Abstentions": m.abstained_count,
            "Mean Runtime (ms)": f"{runtime:.2f}",
        })
        summary[f"budget_{b:.4f}"] = {"f1": m.f1, "mean_width": inv.mean_width, "abstentions": m.abstained_count}

    out_dict = {"table_id": "table5_budget_sensitivity", "budgets": summary}
    save_table_outputs(
        out_dict,
        rows,
        "table5_budget_sensitivity",
        tables_dir,
        default_headers=["Budget B", "Claims", "Mean Width", "Median Width", "F1 (Selective)", "ROC-AUC", "Evidence-Sensitive Rate", "Abstentions", "Mean Runtime (ms)"],
    )
    return out_dict


def generate_table_6_corruption_robustness(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 6: Corruption robustness."""
    corrupt_res = [r for r in results if r.corruption_type != "none" or r.corruption_severity == "clean"]
    conditions = sorted(list({(r.corruption_type, r.corruption_severity) for r in corrupt_res}))
    if not conditions:
        conditions = [("none", "clean")]
    rows = []
    summary: Dict[str, Any] = {}

    for c_type, sev in conditions:
        sub = [r for r in corrupt_res if r.corruption_type == c_type and r.corruption_severity == sev]
        m_std = evaluate_claim_results(sub, method_name="standard_bp")
        m_rob = evaluate_claim_results(sub, method_name="robust_bp", exclude_abstain=True)
        inv = compute_interval_statistics(sub)

        rows.append({
            "Corruption Type": c_type,
            "Severity": sev,
            "Claims": len(sub),
            "Std BP Accuracy": format_val(m_std.accuracy),
            "Rob BP (Sel) Accuracy": format_val(m_rob.accuracy),
            "Mean Width": format_val(inv.mean_width),
            "Evidence-Sensitive Rate": f"{inv.evidence_sensitive_fraction * 100:.1f}%",
        })
        summary[f"{c_type}_{sev}"] = {"std_acc": m_std.accuracy, "rob_acc": m_rob.accuracy, "mean_width": inv.mean_width}

    out_dict = {"table_id": "table6_corruption_robustness", "corruptions": summary}
    save_table_outputs(
        out_dict,
        rows,
        "table6_corruption_robustness",
        tables_dir,
        default_headers=["Corruption Type", "Severity", "Claims", "Std BP Accuracy", "Rob BP (Sel) Accuracy", "Mean Width", "Evidence-Sensitive Rate"],
    )
    return out_dict


def generate_table_7_computational_cost(results: List[ClaimEvaluationResult], tables_dir: Path) -> Dict[str, Any]:
    """Table 7: Computational cost."""
    clean_res = [r for r in results if r.condition in ["clean", "robust_bp_proposed"]] or results
    n_claims = len(clean_res)
    n_images = len({r.image_id for r in clean_res})
    grid_steps = clean_res[0].grid_steps if clean_res else 100

    runtimes = [r.runtime_ms for r in clean_res]
    total_time_s = float(np.sum(runtimes) / 1000.0) if runtimes else 0.0
    mean_claim_ms = float(np.mean(runtimes)) if runtimes else 0.0
    mean_img_ms = float(total_time_s * 1000.0 / n_images) if n_images > 0 else 0.0

    rows = [
        {"Parameter": "Total Evaluated Images", "Value": str(n_images)},
        {"Parameter": "Total Evaluated Claims", "Value": str(n_claims)},
        {"Parameter": "Robust Grid Resolution K", "Value": str(grid_steps)},
        {"Parameter": "Mean Runtime per Claim (ms)", "Value": f"{mean_claim_ms:.2f} ms"},
        {"Parameter": "Mean Runtime per Image (ms)", "Value": f"{mean_img_ms:.2f} ms"},
        {"Parameter": "Total Inference Time (s)", "Value": f"{total_time_s:.3f} s"},
    ]

    out_dict = {
        "table_id": "table7_computational_cost",
        "total_images": n_images,
        "total_claims": n_claims,
        "grid_steps": grid_steps,
        "mean_runtime_ms_per_claim": mean_claim_ms,
        "mean_runtime_ms_per_image": mean_img_ms,
        "total_time_seconds": total_time_s,
    }
    save_table_outputs(out_dict, rows, "table7_computational_cost", tables_dir)
    return out_dict


def generate_all_tables(results_dir: Path) -> Dict[str, Any]:
    """Generate all 7 publication-ready tables."""
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    results = load_raw_results(results_dir)

    t1 = generate_table_1_dataset_summary(results, tables_dir)
    t2 = generate_table_2_main_comparison(results, tables_dir)
    t3 = generate_table_3_interval_statistics(results, tables_dir)
    t4 = generate_table_4_ablation_study(results, tables_dir)
    t5 = generate_table_5_budget_sensitivity(results, tables_dir)
    t6 = generate_table_6_corruption_robustness(results, tables_dir)
    t7 = generate_table_7_computational_cost(results, tables_dir)

    return {
        "table_1": t1,
        "table_2": t2,
        "table_3": t3,
        "table_4": t4,
        "table_5": t5,
        "table_6": t6,
        "table_7": t7,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready M8 tables.")
    parser.add_argument("--results-dir", type=str, default="reports/m8", help="Path to M8 experiment output directory")
    args = parser.parse_args()

    r_dir = Path(args.results_dir)
    tables = generate_all_tables(r_dir)
    print(f"Generated 7 tables in JSON, CSV, and Markdown in: {r_dir / 'tables'}")


if __name__ == "__main__":
    main()
