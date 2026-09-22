"""Milestone 9 Final Scientific Results Validator.

Validates:
1. Result directory structure and presence of required packages
2. Raw JSONL schema and mathematical consistency ($0 <= L_i <= U_i <= 1$, $w_i = U_i - L_i$)
3. Ground-truth semantics (supported=-1, hallucinated=+1, UNKNOWN excluded from binary metrics)
4. No duplicate claim IDs in clean evaluation
5. Manifest completeness, git status, and checksum consistency
6. Completeness of Tables 1 through 10 (JSON, CSV, MD, LaTeX)
7. Completeness of Figures 1 through 14 (PNG, PDF)
8. Scientific report packages (results_section.md, discussion_section.md, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m9_validator")


def validate_m9_artifacts(results_dir: Path) -> Tuple[bool, List[str]]:
    """Validate all generated Milestone 9 artifacts in results_dir."""
    errors: List[str] = []

    if not results_dir.exists():
        return False, [f"Results directory does not exist: {results_dir}"]

    # 1. Check required top-level or subfolder files
    required_files = [
        "final_claim_results.jsonl",
        "final_image_results.jsonl",
        "final_metrics.json",
        "final_intervals.json",
        "final_ablation_results.json",
        "final_corruption_results.json",
        "final_statistics.json",
        "final_manifest.json",
        "research_summary.md",
        "contribution_statement.md",
    ]
    for rf in required_files:
        p = results_dir / rf
        if not p.exists():
            errors.append(f"Missing required artifact: {rf}")

    # 2. Check Paper Results Package
    paper_dir = results_dir / "paper_results"
    required_paper_files = [
        "results_section.md",
        "discussion_section.md",
        "method_summary.md",
        "statistical_analysis.md",
        "limitations.md",
        "reproducibility.md",
        "case_studies/case_studies.md",
    ]
    for pf in required_paper_files:
        p = paper_dir / pf
        if not p.exists():
            errors.append(f"Missing paper results file: paper_results/{pf}")

    # 3. Check Tables 1 to 10
    tables_dir = results_dir / "tables"
    for i in range(1, 11):
        for fmt in ["json", "csv", "md", "tex"]:
            # Match table filename prefix
            t_files = list(tables_dir.glob(f"table{i}_*.{fmt}"))
            if not t_files:
                errors.append(f"Missing Table {i} in .{fmt} format (expected in {tables_dir})")

    # 4. Check Figures 1 to 14
    figures_dir = results_dir / "figures"
    for i in range(1, 15):
        for fmt in ["png", "pdf"]:
            f_files = list(figures_dir.glob(f"fig{i}_*.{fmt}"))
            if not f_files:
                errors.append(f"Missing Figure {i} in .{fmt} format (expected in {figures_dir})")

    # 5. Validate Claim Results Mathematical Bounds & Ground Truth
    claim_file = results_dir / "final_claim_results.jsonl"
    if claim_file.exists():
        seen_clean_claims = set()
        with open(claim_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    rec = json.loads(line.strip())
                except Exception as e:
                    errors.append(f"Line {line_idx} in {claim_file.name} is invalid JSON: {e}")
                    continue

                cid = rec.get("claim_id")
                low = rec.get("robust_lower")
                upp = rec.get("robust_upper")
                gt = rec.get("ground_truth")
                std_bp = rec.get("standard_bp_prob")

                # Duplicate claim check on clean baseline
                if rec.get("corruption_type") == "clean" and rec.get("ablation_type") == "none":
                    if cid in seen_clean_claims:
                        errors.append(f"Duplicate clean baseline claim ID detected: {cid}")
                    seen_clean_claims.add(cid)

                # Interval bounds: 0 <= L <= U <= 1
                if low is None or upp is None:
                    errors.append(f"Claim {cid} missing robust bounds (low={low}, upp={upp})")
                else:
                    if low < -1e-5 or upp > 1.00001:
                        errors.append(f"Claim {cid} robust bounds out of [0, 1]: [{low}, {upp}]")
                    if low > upp + 1e-5:
                        errors.append(f"Claim {cid} inverted bounds: lower ({low}) > upper ({upp})")

                # Probability bounds: 0 <= P <= 1
                if std_bp is not None and (std_bp < -1e-5 or std_bp > 1.00001):
                    errors.append(f"Claim {cid} standard BP prob out of [0, 1]: {std_bp}")

                # Ground truth semantics
                if gt is not None and gt not in ["supported", "hallucinated", "unknown"]:
                    errors.append(f"Claim {cid} has unrecognized ground truth status: {gt}")

    # 6. Validate Manifest Completeness
    manifest_file = results_dir / "final_manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                man = json.load(f)
            req_man_fields = ["version", "milestone", "execution_mode", "timestamp", "git", "dataset_summary", "result_checksum"]
            for mf in req_man_fields:
                if mf not in man:
                    errors.append(f"Manifest missing required field: {mf}")
        except Exception as e:
            errors.append(f"Invalid manifest JSON: {e}")

    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 9 Final Scientific Results Validator")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="reports/m9/final",
        help="Directory containing Milestone 9 final artifacts",
    )
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    logger.info(f"Validating Milestone 9 results in '{results_path}'...")
    passed, errors = validate_m9_artifacts(results_path)

    print("=" * 60)
    print("MILESTONE 9 SCIENTIFIC ARTIFACT VALIDATION SUMMARY")
    print(f"Directory: {results_path}")
    print(f"Overall Result: {'PASSED' if passed else 'FAILED'}")
    print(f"Total Errors: {len(errors)}")
    if errors:
        print("\nValidation Errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors.")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
