"""
Publication-quality plotting script for Milestone 8 (M8) experimental results.
Generates all Figures 1-12 in PNG and PDF formats.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.inference_runner import ClaimEvaluationResult


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


def save_fig_all_formats(fig: plt.Figure, base_path: Path, formats: List[str]) -> None:
    """Save matplotlib figure in multiple formats (e.g. PNG, PDF)."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out_file = base_path.with_suffix(f".{fmt.lower()}")
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_placeholder_fig(title: str, message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="gray", wrap=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_01_confusion_matrix_standard_bp(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 1: Confusion matrix — Standard BP."""
    annotated = [r for r in results if r.ground_truth in ["supported", "hallucinated"]]
    if not annotated:
        fig = make_placeholder_fig("Confusion Matrix: Standard BP", "No annotated ground-truth claims available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    y_true = [1 if r.ground_truth == "hallucinated" else 0 for r in annotated]
    y_pred = [1 if r.standard_prediction == "hallucinated" else 0 for r in annotated]

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Supported", "Hallucinated"],
        yticklabels=["Supported", "Hallucinated"],
        title="Confusion Matrix: Standard BP",
        ylabel="Ground Truth",
        xlabel="Predicted Label",
    )
    for i in range(2):
        for j in range(2):
            val = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
            ax.text(j, i, format(val, "d"), ha="center", va="center", color="black" if val < np.max(cm)/2 else "white")
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_02_confusion_matrix_robust_bp(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 2: Confusion matrix — Robust BP."""
    annotated = [r for r in results if r.ground_truth in ["supported", "hallucinated"] and not r.abstained]
    if not annotated:
        fig = make_placeholder_fig("Confusion Matrix: Robust BP", "No non-abstained annotated claims available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    y_true = [1 if r.ground_truth == "hallucinated" else 0 for r in annotated]
    y_pred = [1 if r.robust_prediction == "hallucinated" else 0 for r in annotated]

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Supported", "Hallucinated"],
        yticklabels=["Supported", "Hallucinated"],
        title="Confusion Matrix: Robust BP (Non-Abstained)",
        ylabel="Ground Truth",
        xlabel="Predicted Label",
    )
    for i in range(2):
        for j in range(2):
            val = cm[i, j] if i < cm.shape[0] and j < cm.shape[1] else 0
            ax.text(j, i, format(val, "d"), ha="center", va="center", color="black" if val < np.max(cm)/2 else "white")
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_03_roc_curves_comparison(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 3: ROC curves — baseline comparison."""
    annotated = [r for r in results if r.ground_truth in ["supported", "hallucinated"]]
    y_true = np.array([1 if r.ground_truth == "hallucinated" else 0 for r in annotated])

    if len(annotated) < 2 or len(np.unique(y_true)) < 2:
        fig = make_placeholder_fig("ROC Curves: Method Comparison", "Requires at least 2 classes in ground truth.")
        save_fig_all_formats(fig, out_base, formats)
        return

    scores_std = np.array([r.standard_posterior for r in annotated])
    scores_rob = np.array([r.robust_midpoint for r in annotated])
    scores_det = np.array([1.0 - (r.detector_score if r.detector_score is not None else 0.5) for r in annotated])

    fpr_std, tpr_std, _ = metrics.roc_curve(y_true, scores_std)
    auc_std = metrics.auc(fpr_std, tpr_std)

    fpr_rob, tpr_rob, _ = metrics.roc_curve(y_true, scores_rob)
    auc_rob = metrics.auc(fpr_rob, tpr_rob)

    fpr_det, tpr_det, _ = metrics.roc_curve(y_true, scores_det)
    auc_det = metrics.auc(fpr_det, tpr_det)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_std, tpr_std, label=f"Standard BP (AUC = {auc_std:.3f})", color="tab:blue", lw=2)
    ax.plot(fpr_rob, tpr_rob, label=f"Robust BP Midpoint (AUC = {auc_rob:.3f})", color="tab:green", lw=2)
    ax.plot(fpr_det, tpr_det, label=f"Evidence Baseline (AUC = {auc_det:.3f})", color="tab:gray", ls="--", lw=1.5)
    ax.plot([0, 1], [0, 1], "k:", label="Chance")

    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves: Method Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_04_interval_width_distribution(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 4: Robust interval width distribution."""
    clean_results = [r for r in results if r.corruption_severity == "clean"] or results
    widths = [r.interval_width for r in clean_results]
    if not widths:
        fig = make_placeholder_fig("Distribution of Robust Interval Widths", "No interval results available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(widths, bins=10, color="tab:purple", edgecolor="black", alpha=0.7, density=False)
    ax.set(xlabel="Interval Width (U_i - L_i)", ylabel="Count", title="Distribution of Robust Interval Widths")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_05_posterior_vs_midpoint(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 5: Standard posterior vs robust midpoint."""
    clean_results = [r for r in results if r.corruption_severity == "clean"] or results
    if not clean_results:
        fig = make_placeholder_fig("Standard Posterior vs Robust Midpoint", "No results available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    x = [r.standard_posterior for r in clean_results]
    y = [r.robust_midpoint for r in clean_results]
    colors = ["tab:red" if r.evidence_sensitive else "tab:blue" for r in clean_results]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, c=colors, alpha=0.8, edgecolors="k", s=45)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y = x")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.axvline(0.5, color="gray", ls=":", alpha=0.5)

    ax.set(
        xlabel="Standard BP Posterior P(H_i = +1 | E)",
        ylabel="Robust Interval Midpoint (L_i + U_i) / 2",
        title="Standard Posterior vs Robust Midpoint",
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_06_representative_intervals(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 6: L_i/U_i interval visualization for representative claims."""
    clean_results = [r for r in results if r.corruption_severity == "clean"] or results
    if not clean_results:
        fig = make_placeholder_fig("Representative Robust Intervals", "No results available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    sample = clean_results[:min(12, len(clean_results))]
    indices = np.arange(len(sample))
    lowers = np.array([r.robust_lower for r in sample])
    uppers = np.array([r.robust_upper for r in sample])
    nominals = np.array([r.standard_posterior for r in sample])
    midpoints = (lowers + uppers) / 2.0
    err_low = np.maximum(0.0, midpoints - lowers)
    err_high = np.maximum(0.0, uppers - midpoints)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(indices, midpoints, yerr=[err_low, err_high], fmt="o", color="tab:blue", ecolor="tab:orange", elinewidth=2, capsize=4, label="Robust Interval [L_i, U_i]")
    ax.scatter(indices, nominals, color="black", marker="x", s=40, zorder=5, label="Standard BP Posterior")
    ax.axhline(0.5, color="red", ls="--", alpha=0.7, label="Threshold tau = 0.5")

    labels = [f"{r.object_category}\n({r.claim_id[-6:]})" for r in sample]
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set(ylabel="Probability of Hallucination", title="Representative Robust Intervals [L_i, U_i]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_07_width_vs_corruption(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 7: Interval width vs corruption severity."""
    severities = ["clean", "light", "medium", "heavy"]
    mean_widths = []
    available_sevs = []

    for sev in severities:
        sub = [r.interval_width for r in results if r.corruption_severity == sev]
        if sub:
            available_sevs.append(sev)
            mean_widths.append(float(np.mean(sub)))

    if not available_sevs:
        fig = make_placeholder_fig("Interval Width vs Visual Degradation", "No corruption conditions evaluated.")
        save_fig_all_formats(fig, out_base, formats)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(available_sevs, mean_widths, marker="o", lw=2, color="tab:red")
    ax.set(xlabel="Corruption Severity", ylabel="Mean Robust Interval Width", title="Interval Width vs Visual Degradation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_08_f1_vs_budget(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 8: F1 vs budget B."""
    budgets = sorted(list({r.budget for r in results}))
    f1_vals = []
    valid_budgets = []

    for b in budgets:
        sub = [r for r in results if abs(r.budget - b) < 1e-6 and r.ground_truth in ["supported", "hallucinated"] and not r.abstained]
        if len(sub) >= 1:
            y_true = [1 if r.ground_truth == "hallucinated" else 0 for r in sub]
            y_pred = [1 if r.robust_prediction == "hallucinated" else 0 for r in sub]
            f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
            valid_budgets.append(b)
            f1_vals.append(f1)

    if not valid_budgets:
        fig = make_placeholder_fig("F1 Score vs Perturbation Budget B", "No annotated budget evaluation claims.")
        save_fig_all_formats(fig, out_base, formats)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(valid_budgets, f1_vals, marker="s", lw=2, color="tab:green")
    ax.set(xlabel="Shared Budget B", ylabel="F1 Score (Non-Abstained)", title="F1 Score vs Perturbation Budget B")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_09_width_vs_budget(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 9: Interval width vs budget B."""
    budgets = sorted(list({r.budget for r in results}))
    if not budgets:
        fig = make_placeholder_fig("Interval Width vs Budget B", "No budget sweep conditions evaluated.")
        save_fig_all_formats(fig, out_base, formats)
        return

    mean_widths = []
    for b in budgets:
        sub = [r.interval_width for r in results if abs(r.budget - b) < 1e-6]
        mean_widths.append(float(np.mean(sub)) if sub else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(budgets, mean_widths, marker="^", lw=2, color="tab:blue")
    ax.set(xlabel="Shared Budget B", ylabel="Mean Interval Width", title="Interval Width vs Budget B")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_10_performance_vs_corruption(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 10: Performance (Accuracy/F1) vs corruption severity."""
    severities = ["clean", "light", "medium", "heavy"]
    acc_std_vals = []
    acc_rob_vals = []
    valid_sevs = []

    for sev in severities:
        sub = [r for r in results if r.corruption_severity == sev and r.ground_truth in ["supported", "hallucinated"]]
        if len(sub) >= 1:
            y_true = [1 if r.ground_truth == "hallucinated" else 0 for r in sub]
            y_pred_std = [1 if r.standard_prediction == "hallucinated" else 0 for r in sub]
            y_pred_rob = [1 if (r.robust_midpoint >= 0.5) else 0 for r in sub]

            acc_std = metrics.accuracy_score(y_true, y_pred_std)
            acc_rob = metrics.accuracy_score(y_true, y_pred_rob)

            valid_sevs.append(sev)
            acc_std_vals.append(acc_std)
            acc_rob_vals.append(acc_rob)

    if not valid_sevs:
        fig = make_placeholder_fig("Classification Performance under Visual Corruption", "No annotated corruption claims available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(valid_sevs, acc_std_vals, marker="o", label="Standard BP Accuracy", color="tab:blue")
    ax.plot(valid_sevs, acc_rob_vals, marker="s", label="Robust BP Midpoint Accuracy", color="tab:green")
    ax.set(xlabel="Corruption Severity", ylabel="Accuracy", title="Classification Performance under Visual Corruption")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_11_error_rate_vs_width_bins(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 11: Error rate vs interval-width bins."""
    annotated = [r for r in results if r.ground_truth in ["supported", "hallucinated"]]
    if not annotated:
        fig = make_placeholder_fig("Standard BP Error Rate by Uncertainty Width Bin", "No annotated claims available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    widths = np.array([r.interval_width for r in annotated])
    errors = np.array([1 if ((r.standard_posterior >= 0.5) != (r.ground_truth == "hallucinated")) else 0 for r in annotated])

    bins = [0.0, 0.33, 0.66, 1.0]
    bin_labels = ["[0, 0.33)", "[0.33, 0.66)", "[0.66, 1.0]"]
    bin_errs = []
    bin_counts = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        mask = (widths >= low) & (widths <= high if i == len(bins)-2 else widths < high)
        if np.any(mask):
            bin_errs.append(float(np.mean(errors[mask])))
            bin_counts.append(int(np.sum(mask)))
        else:
            bin_errs.append(0.0)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(bin_labels, bin_errs, color="tab:orange", edgecolor="black", alpha=0.7)
    for idx, rect in enumerate(bars):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.02, f"n={bin_counts[idx]}", ha='center', va='bottom', fontsize=8)

    ax.set(xlabel="Interval Width Bin", ylabel="Standard BP Error Rate", title="Standard BP Error Rate by Uncertainty Width Bin")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def plot_12_evidence_sensitive_vs_corruption(results: List[ClaimEvaluationResult], out_base: Path, formats: List[str]) -> None:
    """Plot 12: Evidence-sensitive fraction vs corruption severity."""
    severities = ["clean", "light", "medium", "heavy"]
    sens_rates = []
    valid_sevs = []

    for sev in severities:
        sub = [r for r in results if r.corruption_severity == sev]
        if sub:
            frac = float(sum(1 for r in sub if r.evidence_sensitive) / len(sub))
            valid_sevs.append(sev)
            sens_rates.append(frac)

    if not valid_sevs:
        fig = make_placeholder_fig("Evidence Sensitivity vs Degradation", "No corruption data available.")
        save_fig_all_formats(fig, out_base, formats)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(valid_sevs, sens_rates, marker="D", lw=2, color="tab:red")
    ax.set(xlabel="Corruption Severity", ylabel="Evidence-Sensitive Fraction (L_i <= tau <= U_i)", title="Evidence Sensitivity vs Degradation")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig_all_formats(fig, out_base, formats)


def generate_all_plots(results_dir: Path, formats: Optional[List[str]] = None) -> List[Path]:
    """Generate all 12 publication-quality plots."""
    fmts = formats or ["png", "pdf"]
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results = load_raw_results(results_dir)
    generated: List[Path] = []

    plot_funcs = [
        ("fig1_confusion_standard_bp", plot_01_confusion_matrix_standard_bp),
        ("fig2_confusion_robust_bp", plot_02_confusion_matrix_robust_bp),
        ("fig3_roc_curves", plot_03_roc_curves_comparison),
        ("fig4_interval_width_distribution", plot_04_interval_width_distribution),
        ("fig5_posterior_vs_midpoint", plot_05_posterior_vs_midpoint),
        ("fig6_representative_intervals", plot_06_representative_intervals),
        ("fig7_width_vs_corruption", plot_07_width_vs_corruption),
        ("fig8_f1_vs_budget", plot_08_f1_vs_budget),
        ("fig9_width_vs_budget", plot_09_width_vs_budget),
        ("fig10_performance_vs_corruption", plot_10_performance_vs_corruption),
        ("fig11_error_rate_vs_width_bins", plot_11_error_rate_vs_width_bins),
        ("fig12_sensitive_fraction_vs_corruption", plot_12_evidence_sensitive_vs_corruption),
    ]

    for name, func in plot_funcs:
        out_base = fig_dir / name
        func(results, out_base, fmts)
        generated.append(out_base.with_suffix(".png"))

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality M8 plots.")
    parser.add_argument("--results-dir", type=str, default="reports/m8", help="Path to M8 experiment output directory")
    parser.add_argument("--formats", type=str, default="png,pdf", help="Comma-separated plot formats")
    args = parser.parse_args()

    r_dir = Path(args.results_dir)
    fmts = [f.strip() for f in args.formats.split(",") if f.strip()]
    plots = generate_all_plots(r_dir, formats=fmts)
    print(f"Generated {len(plots)} figures in: {r_dir / 'figures'}")


if __name__ == "__main__":
    main()
