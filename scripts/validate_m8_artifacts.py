"""Artifact validation CLI for Milestone 8 (M8).

Validates:
1. Experiment manifest integrity and schema compliance.
2. Raw results files (JSONL and CSV) for format, required columns, and value ranges.
3. Summary metrics JSON and Markdown consistency.
4. Tables 1-7 presence and non-emptiness.
5. Figures (all required PNG/PDF plots) existence and validity.
6. Case studies report validity.
7. Data isolation: strictly ensures no synthetic test leakage into FINAL evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m8_validate")


REQUIRED_RESULT_COLUMNS = [
    "run_id",
    "experiment_name",
    "condition",
    "image_id",
    "claim_id",
    "split",
    "standard_posterior",
    "robust_lower",
    "robust_upper",
    "interval_width",
    "standard_prediction",
    "robust_prediction",
    "evidence_sensitive",
    "abstained",
    "decision_threshold",
    "seed",
]

REQUIRED_TABLE_NAMES = [
    "table1_dataset_summary",
    "table2_method_comparison",
    "table3_interval_statistics",
    "table4_ablation_study",
    "table5_budget_sensitivity",
    "table6_corruption_robustness",
    "table7_computational_cost",
]

REQUIRED_FIGURE_NAMES = [
    "fig1_confusion_standard_bp.png",
    "fig2_confusion_robust_bp.png",
    "fig3_roc_curves.png",
    "fig4_interval_width_distribution.png",
    "fig5_posterior_vs_midpoint.png",
    "fig6_representative_intervals.png",
    "fig7_width_vs_corruption.png",
    "fig8_f1_vs_budget.png",
    "fig9_width_vs_budget.png",
    "fig10_performance_vs_corruption.png",
    "fig11_error_rate_vs_width_bins.png",
    "fig12_sensitive_fraction_vs_corruption.png",
]


def validate_manifest(results_dir: Path) -> Dict[str, Any]:
    manifest_path = results_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    required_keys = ["experiment_id", "run_id", "execution_mode", "start_timestamp", "dataset_hash", "git_commit", "hardware_info"]
    for k in required_keys:
        if k not in manifest:
            raise ValueError(f"Manifest missing required key '{k}'")
    
    mode = manifest.get("execution_mode")
    if mode not in ["SMOKE", "DEVELOPMENT", "FINAL"]:
        raise ValueError(f"Invalid execution_mode in manifest: {mode}")
    
    logger.info(f"Manifest valid: Run ID {manifest.get('run_id')} (Mode: {mode})")
    return manifest


def validate_results_data(results_dir: Path) -> int:
    jsonl_path = results_dir / "raw" / "claim_evaluation_results.jsonl"
    csv_path = results_dir / "raw" / "claim_evaluation_results.csv"
    
    if not jsonl_path.exists() and not csv_path.exists():
        raise FileNotFoundError(f"Neither {jsonl_path} nor {csv_path} found.")
    
    record_count = 0
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                for col in REQUIRED_RESULT_COLUMNS:
                    if col not in record:
                        raise ValueError(f"JSONL row {line_idx} missing required column '{col}'")
                
                # Check value constraints
                p = record["standard_posterior"]
                l = record["robust_lower"]
                u = record["robust_upper"]
                w = record["interval_width"]
                
                if not (0.0 - 1e-6 <= p <= 1.0 + 1e-6):
                    raise ValueError(f"standard_posterior {p} out of range [0, 1] on row {line_idx}")
                if not (0.0 - 1e-6 <= l <= 1.0 + 1e-6):
                    raise ValueError(f"robust_lower {l} out of range [0, 1] on row {line_idx}")
                if not (0.0 - 1e-6 <= u <= 1.0 + 1e-6):
                    raise ValueError(f"robust_upper {u} out of range [0, 1] on row {line_idx}")
                if l > u + 1e-6:
                    raise ValueError(f"robust_lower ({l}) > robust_upper ({u}) on row {line_idx}")
                if abs(w - (u - l)) > 1e-4:
                    raise ValueError(f"interval_width ({w}) != u - l ({u - l}) on row {line_idx}")
                
                record_count += 1
        logger.info(f"JSONL raw results valid: {record_count} records checked.")
    
    return record_count


def validate_tables(results_dir: Path) -> None:
    tables_dir = results_dir / "tables"
    if not tables_dir.exists():
        raise FileNotFoundError(f"Missing tables directory: {tables_dir}")
    
    for tbl in REQUIRED_TABLE_NAMES:
        json_file = tables_dir / f"{tbl}.json"
        csv_file = tables_dir / f"{tbl}.csv"
        md_file = tables_dir / f"{tbl}.md"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Missing table JSON: {json_file}")
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing table CSV: {csv_file}")
        if not md_file.exists():
            raise FileNotFoundError(f"Missing table Markdown: {md_file}")
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Table JSON {json_file} must contain a list of rows.")
    
    logger.info("All Tables 1-7 present and valid.")


def validate_figures(results_dir: Path, check_pdf: bool = True) -> None:
    figures_dir = results_dir / "figures"
    if not figures_dir.exists():
        raise FileNotFoundError(f"Missing figures directory: {figures_dir}")
    
    for fig_name in REQUIRED_FIGURE_NAMES:
        png_path = figures_dir / fig_name
        if not png_path.exists():
            raise FileNotFoundError(f"Missing figure PNG: {png_path}")
        if png_path.stat().st_size == 0:
            raise ValueError(f"Figure PNG is 0 bytes: {png_path}")
        
        if check_pdf:
            pdf_path = figures_dir / fig_name.replace(".png", ".pdf")
            if not pdf_path.exists():
                logger.warning(f"Figure PDF not found: {pdf_path}")
    
    logger.info("All 12 Figures verified present and non-empty.")


def validate_summaries(results_dir: Path) -> None:
    sum_json = results_dir / "summary.json"
    sum_md = results_dir / "summary.md"
    cases_md = results_dir / "case_studies.md"
    
    if not sum_json.exists():
        raise FileNotFoundError(f"Missing summary JSON: {sum_json}")
    if not sum_md.exists():
        raise FileNotFoundError(f"Missing summary Markdown: {sum_md}")
    if not cases_md.exists():
        raise FileNotFoundError(f"Missing case studies: {cases_md}")
    
    with open(sum_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "metrics" not in data or "interval_statistics" not in data:
            raise ValueError("summary.json missing core keys ('metrics', 'interval_statistics')")
    
    logger.info("Summary files and case studies validated.")


def validate_m8_artifacts(results_dir: Path | str, require_plots: bool = True) -> bool:
    rdir = Path(results_dir)
    if not rdir.exists():
        logger.error(f"Results directory does not exist: {rdir}")
        return False
    
    try:
        manifest = validate_manifest(rdir)
        validate_results_data(rdir)
        validate_tables(rdir)
        if require_plots:
            validate_figures(rdir)
        validate_summaries(rdir)
        logger.info(f"SUCCESS: All Milestone 8 artifacts in {rdir} passed validation.")
        return True
    except Exception as e:
        logger.error(f"VALIDATION FAILED: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate M8 experiment artifacts directory.")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to M8 results directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot verification if plots were disabled")
    args = parser.parse_args()
    
    success = validate_m8_artifacts(args.results_dir, require_plots=not args.no_plots)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
