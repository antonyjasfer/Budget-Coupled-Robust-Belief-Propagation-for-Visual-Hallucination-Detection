"""
Offline evaluation script consuming raw claim results and generating summary reports.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics
from src.experiments.interval_metrics import compute_interval_statistics, RobustIntervalStatistics
from src.experiments.aggregation import run_image_cluster_bootstrap, select_case_studies
from scripts.generate_m8_tables import generate_all_tables
from scripts.generate_m8_plots import generate_all_plots


def load_raw_results(results_dir: Path) -> List[ClaimEvaluationResult]:
    """Load raw results from JSONL."""
    candidates = [
        results_dir / "raw" / "claim_evaluation_results.jsonl",
        results_dir / "raw_claim_results.jsonl",
        results_dir / "claim_evaluation_results.jsonl",
    ]
    for p in candidates:
        if p.exists():
            results: List[ClaimEvaluationResult] = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(ClaimEvaluationResult.from_dict(json.loads(line)))
            return results
    raise FileNotFoundError(f"Results file not found in any of: {candidates}")


def build_summary_report(
    results_dir: Path,
    manifest_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute full evaluation metrics, bootstrap CIs, case studies, and generate summary.md.
    """
    results = load_raw_results(results_dir)
    clean_res = [r for r in results if r.condition in ["clean", "robust_bp_proposed"]] or results

    # 1. Classification metrics
    m_std = evaluate_claim_results(clean_res, method_name="standard_bp")
    m_rob_sel = evaluate_claim_results(clean_res, method_name="robust_bp", exclude_abstain=True)
    m_rob_all = evaluate_claim_results(clean_res, method_name="robust_bp", exclude_abstain=False)
    m_point = evaluate_claim_results(clean_res, method_name="evidence_point")

    # 2. Interval statistics
    inv = compute_interval_statistics(clean_res)

    # 3. Cluster Bootstrap Analysis
    boot_std = run_image_cluster_bootstrap(clean_res, method_name="standard_bp")
    boot_rob = run_image_cluster_bootstrap(clean_res, method_name="robust_bp")

    # 4. Case Studies
    case_studies = select_case_studies(clean_res)

    # 5. Manifest / Provenance metadata
    manifest_file = results_dir / "experiment_manifest.json"
    manifest = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    exec_mode = manifest.get("execution_mode", "DEVELOPMENT")

    # Format Summary JSON
    summary_data = {
        "execution_mode": exec_mode,
        "run_id": manifest.get("run_id", "unknown"),
        "git_commit": manifest.get("git_commit", "unknown"),
        "total_claims": len(clean_res),
        "total_images": len({r.image_id for r in clean_res}),
        "metrics": {
            "standard_bp": m_std.to_dict(),
            "robust_bp_selective": m_rob_sel.to_dict(),
            "robust_bp_full": m_rob_all.to_dict(),
            "evidence_point": m_point.to_dict(),
        },
        "interval_statistics": inv.to_dict(),
        "bootstrap": {
            "standard_bp": {k: v.to_dict() for k, v in boot_std.items()},
            "robust_bp": {k: v.to_dict() for k, v in boot_rob.items()},
        },
        "case_studies": [c.to_dict() for c in case_studies],
    }

    # Save summary.json
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Build and save summary.md
    acc_std_str = f"{m_std.accuracy:.3f}" if m_std.accuracy is not None else "N/A"
    f1_std_str = f"{m_std.f1:.3f}" if m_std.f1 is not None else "N/A"
    auc_std_str = f"{m_std.roc_auc:.3f}" if m_std.roc_auc is not None else "N/A"

    acc_rob_str = f"{m_rob_sel.accuracy:.3f}" if m_rob_sel.accuracy is not None else "N/A"
    f1_rob_str = f"{m_rob_sel.f1:.3f}" if m_rob_sel.f1 is not None else "N/A"
    auc_rob_str = f"{m_rob_sel.roc_auc:.3f}" if m_rob_sel.roc_auc is not None else "N/A"

    corr_r_str = f"{inv.pearson_corr_width_error:.3f}" if inv.pearson_corr_width_error is not None else "N/A"
    corr_p_str = f"{inv.pearson_p_value:.4f}" if inv.pearson_p_value is not None else "N/A"

    md_content = f"""# Milestone 8 Evaluation Summary

## 1. Execution Status
- **Execution Mode**: `{exec_mode}`
- **Run ID**: `{manifest.get('run_id', 'unknown')}`
- **Git Commit**: `{manifest.get('git_commit', 'unknown')}` (Dirty: {manifest.get('git_dirty', False)})
- **Total Images Evaluated**: {len({r.image_id for r in clean_res})}
- **Total Claims Evaluated**: {len(clean_res)}
- **Dataset Locked**: {manifest.get('is_dataset_locked', False)}

## 2. Core Method Comparison

| Method | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Coverage | Abstentions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Evidence Point Baseline | {m_point.accuracy or 'N/A'} | {m_point.precision or 'N/A'} | {m_point.recall or 'N/A'} | {m_point.f1 or 'N/A'} | {m_point.roc_auc or 'N/A'} | {m_point.coverage_rate * 100:.1f}% | {m_point.abstained_count} |
| Standard BP (Point Posterior) | {acc_std_str} | {m_std.precision or 'N/A'} | {m_std.recall or 'N/A'} | {f1_std_str} | {auc_std_str} | {m_std.coverage_rate * 100:.1f}% | {m_std.abstained_count} |
| Robust BP (Selective) | {acc_rob_str} | {m_rob_sel.precision or 'N/A'} | {m_rob_sel.recall or 'N/A'} | {f1_rob_str} | {auc_rob_str} | {m_rob_sel.coverage_rate * 100:.1f}% | {m_rob_sel.abstained_count} |

## 3. Robust Interval Properties
- **Mean Interval Width**: {inv.mean_width:.4f}
- **Median Interval Width**: {inv.median_width:.4f}
- **Standard Deviation**: {inv.std_width:.4f}
- **Evidence-Sensitive Fraction ($L_i \le \\tau \le U_i$)**: {inv.evidence_sensitive_fraction * 100:.1f}%
- **Robustly Supported Fraction ($U_i < \\tau$)**: {inv.robustly_supported_fraction * 100:.1f}%
- **Robustly Hallucinated Fraction ($L_i > \\tau$)**: {inv.robustly_hallucinated_fraction * 100:.1f}%

## 4. Key Scientific Hypothesis Test
- **Pearson correlation $r(\\text{{width}}, \\text{{BP error}})$**: {corr_r_str} (p = {corr_p_str})
- **Spearman rank correlation $\\rho(\\text{{width}}, \\text{{BP error}})$**: {inv.spearman_corr_width_error or 'N/A'}

## 5. Case Studies
"""
    for cs in case_studies:
        md_content += f"""
### Example: `{cs.category_tag}` (Claim `{cs.claim_id}`)
- **Image ID**: `{cs.image_id}` | **Category**: `{cs.object_category}`
- **Ground Truth**: `{cs.ground_truth}`
- **Detector Score**: {cs.detector_score} | **CLIP Score**: {cs.clip_score}
- **Standard Posterior**: {cs.standard_posterior:.4f} (Prediction: `{cs.standard_prediction}`)
- **Robust Interval**: [{cs.robust_lower:.4f}, {cs.robust_upper:.4f}] (Width: {cs.interval_width:.4f})
- **Robust Decision**: `{cs.robust_prediction}` (Evidence-sensitive: {cs.evidence_sensitive})
- **Observation**: {cs.description}
"""

    (results_dir / "summary.md").write_text(md_content, encoding="utf-8")
    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate raw M8 results and build report.")
    parser.add_argument("--results-dir", type=str, default="reports/m8", help="Path to M8 experiment output directory")
    args = parser.parse_args()

    r_dir = Path(args.results_dir)
    build_summary_report(r_dir)
    generate_all_tables(r_dir)
    generate_all_plots(r_dir)
    print(f"Evaluation complete. Reports and summaries updated in: {r_dir}")


if __name__ == "__main__":
    main()
