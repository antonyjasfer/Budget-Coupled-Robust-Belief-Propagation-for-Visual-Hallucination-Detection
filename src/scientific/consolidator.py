"""
Milestone 9 Master Results Consolidator.

Coordinates raw data ingestion, metric calculation, interval statistics,
ablation and corruption analysis, statistical testing, table/figure generation,
and immutable manifest production into `reports/m9/final/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Dict, List, Optional, Any

import numpy as np

from src.experiments.inference_runner import ClaimEvaluationResult, InferenceRunner
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics
from src.experiments.interval_metrics import compute_interval_statistics, RobustIntervalStatistics
from src.experiments.configs import M8ExperimentConfig, ExecutionMode, get_default_m8_config
from src.experiments.loaders import load_m8_dataset, M8DatasetBundle
from src.experiments.ablations import AblationMatrixRunner
from src.experiments.corruption import evaluate_corruption_pipeline
from src.scientific.gate import FinalDatasetGate, GateCheckResult, GateStatus
from src.scientific.statistics import (
    compute_m9_paired_bootstrap,
    compute_correlation_hypotheses,
    categorize_errors,
    select_m9_case_studies,
)
from src.scientific.tables import generate_all_m9_tables
from src.scientific.figures import generate_all_m9_figures
from src.scientific.reporting import generate_m9_research_package

logger = logging.getLogger("m9_consolidator")


@dataclass
class M9ConsolidatedData:
    """Holds all consolidated Milestone 9 evaluation outputs."""
    claim_results: List[ClaimEvaluationResult]
    metrics: Dict[str, Any]
    interval_stats: Dict[str, Any]
    bootstrap_comparison: Dict[str, Any]
    correlations: Dict[str, Any]
    error_analysis: Dict[str, Any]
    case_studies: List[Dict[str, Any]]
    tables: Dict[str, Any]
    figures: Dict[str, Any]
    manifest: Dict[str, Any]
    output_dir: Path


class M9ResultsConsolidator:
    """Consolidates and validates final scientific outputs for Milestone 9."""

    def __init__(
        self,
        config: Optional[M8ExperimentConfig] = None,
        output_dir: Optional[Path] = None,
        allow_overwrite: bool = False,
    ):
        self.config = config or get_default_m8_config()
        self.output_dir = output_dir or Path("reports/m9/final")
        self.allow_overwrite = allow_overwrite

        # Subdirectories layout
        self.dirs = {
            "raw": self.output_dir / "raw",
            "processed": self.output_dir / "processed",
            "tables": self.output_dir / "tables",
            "figures": self.output_dir / "figures",
            "cases": self.output_dir / "cases",
            "statistics": self.output_dir / "statistics",
            "manifest": self.output_dir / "manifest",
            "logs": self.output_dir / "logs",
            "paper_results": self.output_dir / "paper_results",
        }

    def initialize_directory(self) -> None:
        """Create output directories, protecting previous final packages unless explicit."""
        if self.output_dir.exists() and not self.allow_overwrite:
            if (self.output_dir / "final_manifest.json").exists():
                raise FileExistsError(
                    f"Final result package already exists at '{self.output_dir}'. "
                    "Use allow_overwrite=True or specify a different output directory."
                )

        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def load_raw_results(file_path: Path) -> List[ClaimEvaluationResult]:
        """Load claim evaluation results from a JSONL file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Results file not found: {file_path}")

        results: List[ClaimEvaluationResult] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(ClaimEvaluationResult.from_dict(json.loads(line)))
        return results

    def run_development_pipeline(self) -> List[ClaimEvaluationResult]:
        """Execute real inference pipeline on the canonical real M6 subset."""
        logger.info("Executing M9 Development Pipeline on canonical real M6 dataset...")
        bundle = load_m8_dataset(mode=ExecutionMode.DEVELOPMENT)

        runner = InferenceRunner(self.config)
        claim_results: List[ClaimEvaluationResult] = []

        # 1. Main clean evaluation
        clean_claims, _ = runner.run_batch(bundle.image_records)
        claim_results.extend(clean_claims)

        # 2. Ablations
        ablation_runner = AblationMatrixRunner(self.config)
        ablation_results = ablation_runner.run_ablation_suite(bundle.image_records)
        for _, ab_claims in ablation_results.items():
            claim_results.extend(ab_claims)

        # 3. Corruptions
        corr_results = evaluate_corruption_pipeline(bundle.image_records, self.config)
        for _, c_claims in corr_results.items():
            claim_results.extend(c_claims)

        return claim_results

    def run_final_pipeline(self, lock_path: Optional[Path] = None) -> List[ClaimEvaluationResult]:
        """Execute complete scientific evaluation IF AND ONLY IF final dataset gate passes."""
        gate = FinalDatasetGate(dataset_lock_path=lock_path or self.config.dataset_lock_path)
        gate_result = gate.verify()

        if not gate_result.passed:
            msg = (
                f"FINAL DATASET GATE FAILED. Cannot generate FINAL scientific results.\n"
                f"Status: {gate_result.status.value}\n"
                f"Diagnostics:\n" + "\n".join(f"- {d}" for d in gate_result.diagnostics)
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("FINAL DATASET GATE PASSED. Running complete locked dataset evaluation...")
        bundle = load_m8_dataset(mode=ExecutionMode.FINAL)

        runner = InferenceRunner(self.config)
        claim_results: List[ClaimEvaluationResult] = []

        # 1. Clean run
        clean_claims, _ = runner.run_batch(bundle.image_records)
        claim_results.extend(clean_claims)

        # 2. Ablation suite
        ablation_runner = AblationMatrixRunner(self.config)
        ablation_results = ablation_runner.run_ablation_suite(bundle.image_records)
        for _, ab_claims in ablation_results.items():
            claim_results.extend(ab_claims)

        # 3. Corruption suite
        corr_results = evaluate_corruption_pipeline(bundle.image_records, self.config)
        for _, c_claims in corr_results.items():
            claim_results.extend(c_claims)

        return claim_results

    def consolidate(
        self,
        results: List[ClaimEvaluationResult],
        execution_mode: str = "DEVELOPMENT",
        provenance_info: Optional[Dict[str, Any]] = None,
    ) -> M9ConsolidatedData:
        """Consolidate, analyze, tabulate, and generate reports from claim results."""
        self.initialize_directory()

        logger.info(f"Consolidating {len(results)} claim evaluation records (mode={execution_mode})...")

        # 1. Separate clean and unablated records for core metrics
        clean_claims = [
            r for r in results
            if getattr(r, "corruption_type", "clean") in ["clean", "none"]
            and getattr(r, "ablation_type", getattr(r, "condition", "none")) in ["none", "clean", "standard_bp_proposed", "main"]
        ]
        if not clean_claims:
            clean_claims = [
                r for r in results
                if getattr(r, "ablation_type", getattr(r, "condition", "none")) in ["none", "clean", "standard_bp_proposed", "main"]
            ]
        if not clean_claims:
            clean_claims = results

        # 2. Compute Method Metrics
        methods = ["evidence_point", "standard_bp", "robust_bp"]
        metrics_dict: Dict[str, Any] = {}
        for m_name in methods:
            eval_metrics = evaluate_claim_results(clean_claims, method_name=m_name)
            metrics_dict[m_name] = eval_metrics.to_dict()

        # 3. Compute Robust Interval Statistics
        interval_stats = compute_interval_statistics(clean_claims)
        interval_dict = interval_stats.to_dict()

        # 4. Statistical Testing & Paired Bootstrap
        bootstrap_comp = compute_m9_paired_bootstrap(
            clean_claims,
            n_bootstrap=1000,
            seed=getattr(self.config, "seed", 42),
        )

        # 5. Correlation & Hypothesis Testing
        corr_list = compute_correlation_hypotheses(results)
        correlations = {c.signal_name: c.to_dict() for c in corr_list}

        # 6. Error Categorization
        error_recs = categorize_errors(clean_claims)
        error_summary: Dict[str, Any] = {}
        for er in error_recs:
            error_summary.setdefault(er.category_id, {
                "category_name": er.category_name,
                "count": 0,
                "sample_claim_ids": [],
            })
            error_summary[er.category_id]["count"] += 1
            if len(error_summary[er.category_id]["sample_claim_ids"]) < 5:
                error_summary[er.category_id]["sample_claim_ids"].append(er.claim_id)

        # 7. Case Study Selection
        case_studies_recs = select_m9_case_studies(clean_claims)
        case_studies = [cs.to_dict() for cs in case_studies_recs]

        # 8. Ablations & Corruption Summary
        ablations_summary: Dict[str, Any] = {}
        ab_types = sorted(list(set(getattr(r, "ablation_type", getattr(r, "condition", "none")) for r in results)))
        for ab in ab_types:
            ab_res = [
                r for r in results
                if getattr(r, "ablation_type", getattr(r, "condition", "none")) == ab
                and getattr(r, "corruption_type", "clean") in ["clean", "none"]
            ]
            ablations_summary[ab] = evaluate_claim_results(ab_res, method_name="robust_bp").to_dict()

        corruptions_summary: Dict[str, Any] = {}
        c_types = sorted(list(set(getattr(r, "corruption_type", getattr(r, "condition", "clean")) for r in results)))
        for ct in c_types:
            ct_res = [
                r for r in results
                if getattr(r, "corruption_type", getattr(r, "condition", "clean")) == ct
                and getattr(r, "ablation_type", getattr(r, "condition", "none")) in ["none", "clean", "standard_bp_proposed", "main"]
            ]
            corruptions_summary[ct] = evaluate_claim_results(ct_res, method_name="robust_bp").to_dict()

        # 9. Save Structured JSON and JSONL Data
        claim_jsonl_path = self.dirs["raw"] / "final_claim_results.jsonl"
        with open(claim_jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict()) + "\n")

        image_groups: Dict[str, List[ClaimEvaluationResult]] = {}
        for r in clean_claims:
            image_groups.setdefault(r.image_id, []).append(r)

        image_jsonl_path = self.dirs["raw"] / "final_image_results.jsonl"
        with open(image_jsonl_path, "w", encoding="utf-8") as f:
            for img_id, img_recs in image_groups.items():
                m_std = evaluate_claim_results(img_recs, method_name="standard_bp")
                m_rob = evaluate_claim_results(img_recs, method_name="robust_bp")
                row = {
                    "image_id": img_id,
                    "num_claims": len(img_recs),
                    "standard_bp_acc": m_std.accuracy,
                    "robust_bp_acc": m_rob.accuracy,
                    "mean_interval_width": float(np.mean([r.robust_upper - r.robust_lower for r in img_recs])),
                }
                f.write(json.dumps(row) + "\n")

        metrics_json_path = self.dirs["processed"] / "final_metrics.json"
        with open(metrics_json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        intervals_json_path = self.dirs["processed"] / "final_intervals.json"
        with open(intervals_json_path, "w", encoding="utf-8") as f:
            json.dump(interval_dict, f, indent=2)

        ablation_json_path = self.dirs["processed"] / "final_ablation_results.json"
        with open(ablation_json_path, "w", encoding="utf-8") as f:
            json.dump(ablations_summary, f, indent=2)

        corruption_json_path = self.dirs["processed"] / "final_corruption_results.json"
        with open(corruption_json_path, "w", encoding="utf-8") as f:
            json.dump(corruptions_summary, f, indent=2)

        stats_json_path = self.dirs["statistics"] / "final_statistics.json"
        with open(stats_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "paired_bootstrap": bootstrap_comp,
                "correlations": correlations,
                "error_categories": error_summary,
            }, f, indent=2)

        cases_json_path = self.dirs["cases"] / "case_studies.json"
        with open(cases_json_path, "w", encoding="utf-8") as f:
            json.dump(case_studies, f, indent=2)

        # 10. Generate All Tables 1-10
        tables_res = generate_all_m9_tables(results, self.output_dir)

        # 11. Generate All Figures 1-14
        figures_res = generate_all_m9_figures(results, self.output_dir)

        # 12. Generate Final Immutable Manifest
        manifest_data = self._generate_final_manifest(
            results=results,
            execution_mode=execution_mode,
            provenance_info=provenance_info,
        )
        manifest_path = self.dirs["manifest"] / "final_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 13. Copy Top-Level Conveniences to output root
        top_level_files = [
            (claim_jsonl_path, self.output_dir / "final_claim_results.jsonl"),
            (image_jsonl_path, self.output_dir / "final_image_results.jsonl"),
            (metrics_json_path, self.output_dir / "final_metrics.json"),
            (intervals_json_path, self.output_dir / "final_intervals.json"),
            (ablation_json_path, self.output_dir / "final_ablation_results.json"),
            (corruption_json_path, self.output_dir / "final_corruption_results.json"),
            (stats_json_path, self.output_dir / "final_statistics.json"),
            (manifest_path, self.output_dir / "final_manifest.json"),
        ]
        for src_f, dst_f in top_level_files:
            if src_f != dst_f:
                shutil.copy2(src_f, dst_f)

        # 14. Generate Paper-Ready Results Package
        consolidated = M9ConsolidatedData(
            claim_results=results,
            metrics=metrics_dict,
            interval_stats=interval_dict,
            bootstrap_comparison=bootstrap_comp,
            correlations=correlations,
            error_analysis=error_summary,
            case_studies=case_studies,
            tables=tables_res,
            figures=figures_res,
            manifest=manifest_data,
            output_dir=self.output_dir,
        )

        generate_m9_research_package(consolidated, self.dirs["paper_results"])

        logger.info(f"Consolidation complete. Outputs written to: {self.output_dir}")
        return consolidated

    def _generate_final_manifest(
        self,
        results: List[ClaimEvaluationResult],
        execution_mode: str,
        provenance_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct deterministic final reproducibility manifest."""
        h = hashlib.sha256()
        for r in sorted(results, key=lambda x: (x.claim_id, getattr(x, "ablation_type", getattr(x, "condition", "none")), getattr(x, "corruption_type", "clean"))):
            h.update(f"{r.claim_id}_{r.standard_posterior:.4f}_{r.robust_lower:.4f}_{r.robust_upper:.4f}".encode("utf-8"))
        result_checksum = h.hexdigest()

        git_sha = "unknown"
        git_clean = False
        try:
            import subprocess
            res_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            if res_sha.returncode == 0:
                git_sha = res_sha.stdout.strip()
            res_diff = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
            git_clean = (len(res_diff.stdout.strip()) == 0)
        except Exception:
            pass

        unique_images = list(set(r.image_id for r in results))
        unique_claims = list(set(r.claim_id for r in results))

        manifest = {
            "version": "1.0.0",
            "milestone": "M9",
            "execution_mode": execution_mode,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "git": {
                "commit_sha": git_sha,
                "is_clean": git_clean,
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
            },
            "dataset_summary": {
                "total_records": len(results),
                "unique_images": len(unique_images),
                "unique_claims": len(unique_claims),
                "splits": list(getattr(self.config, "eval_splits", ["test"])),
            },
            "config": {
                "random_seed": getattr(self.config, "seed", 42),
                "budget_sweep": getattr(self.config.ablations, "budget_sweep_multipliers", [0.0, 0.5, 1.0]),
                "decision_threshold": getattr(self.config.pgm, "decision_threshold", 0.5),
            },
            "provenance": provenance_info or {},
            "result_checksum": result_checksum,
        }
        return manifest
