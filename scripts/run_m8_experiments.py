"""Master Experiment Runner CLI for Milestone 8 (M8).

Executes the end-to-end evaluation and experiment pipeline:
1. Validates configuration and execution mode (SMOKE / DEVELOPMENT / FINAL).
2. Loads dataset bundle (real M7 or synthetic smoke fixture) with split isolation.
3. Runs Standard BP, Robust BP, Baselines, Ablations, and Corruption experiments.
4. Computes core classification metrics and robust interval statistics.
5. Computes image-level cluster bootstrap confidence intervals.
6. Generates machine-readable Tables 1-7 (JSON, CSV, Markdown).
7. Generates publication-quality Figures 1-12 (PNG, PDF).
8. Generates representative case studies report.
9. Writes complete experiment manifest with execution provenance.
10. Validates all generated artifacts.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.configs import (
    M8ExperimentConfig,
    ExecutionMode,
    PGMParameterConfig,
    TreeTopology,
    load_m8_config,
    save_m8_config,
)
from src.experiments.loaders import (
    load_m8_dataset,
    create_synthetic_smoke_bundle,
    validate_m8_dataset_integrity,
    M8DatasetBundle,
)
from src.experiments.inference_runner import InferenceRunner, ClaimEvaluationResult
from src.experiments.ablations import AblationMatrixRunner
from src.experiments.corruption import evaluate_corruption_pipeline
from src.experiments.evaluation import evaluate_claim_results
from src.experiments.interval_metrics import compute_interval_statistics
from src.experiments.aggregation import run_image_cluster_bootstrap, select_case_studies
from src.experiments.provenance import create_experiment_manifest, save_experiment_manifest
from scripts.generate_m8_tables import generate_all_tables
from scripts.generate_m8_plots import generate_all_plots
from scripts.validate_m8_artifacts import validate_m8_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m8_runner")


def run_experiments(
    config_path: Optional[str] = None,
    smoke: bool = False,
    subset: Optional[int] = None,
    split: Optional[str] = None,
    experiments: Optional[List[str]] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    resume: bool = False,
    no_plots: bool = False,
    no_corruption: bool = False,
    force_unsafe: bool = False,
) -> Path:
    start_time = time.time()
    start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # 1. Load configuration
    if config_path and Path(config_path).exists():
        cfg = load_m8_config(config_path)
    elif smoke:
        cfg = M8ExperimentConfig(
            experiment_id="m8_smoke_test",
            mode=ExecutionMode.SMOKE,
            eval_splits=["all"],
            subset_size=5,
        )
    else:
        cfg = load_m8_config("configs/m8_default.yaml") if Path("configs/m8_default.yaml").exists() else M8ExperimentConfig()

    # CLI Overrides
    if smoke:
        cfg.mode = ExecutionMode.SMOKE
        cfg.output.output_dir = output_dir or "reports/m8_smoke"
        cfg.output.cache_dir = "data/cache/m8_smoke"
        cfg.bootstrap.n_resamples = 50
    elif output_dir:
        cfg.output.output_dir = output_dir

    if seed is not None:
        cfg.seed = seed
    if split:
        cfg.eval_splits = ["all"] if split.lower() == "all" else [split.lower()]
    if resume:
        cfg.resume = True
    if force_unsafe:
        cfg.force = True
    if no_plots:
        cfg.output.save_plots = False
    if no_corruption:
        cfg.corruption.enabled = False

    out_dir = Path(cfg.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"STARTING MILESTONE 8 EXPERIMENT RUN (Mode: {cfg.mode.value})")
    logger.info(f"Output Directory: {out_dir.resolve()}")
    logger.info(f"Target Splits:    {cfg.eval_splits}")
    logger.info("=" * 60)

    # 2. Data Loading & Split Isolation
    if cfg.mode == ExecutionMode.SMOKE:
        logger.info("Generating synthetic smoke fixture bundle...")
        bundle = create_synthetic_smoke_bundle(num_images=5, seed=cfg.seed)
    else:
        logger.info(f"Loading real dataset from {cfg.evidence_path}...")
        bundle = load_m8_dataset(cfg)

    # Validate integrity
    validate_m8_dataset_integrity(bundle)
    logger.info(f"Loaded {len(bundle.images)} images, {len(bundle.claims)} claims.")

    # 3. Main Standard & Robust BP Evaluation
    runner = InferenceRunner(cfg, bundle=bundle, cache_dir=raw_dir)
    
    t0_inf = time.time()
    logger.info("Executing Standard and Robust BP inference...")
    main_results = runner.run_all(splits=cfg.eval_splits, condition="main_evaluation")
    t_inf = time.time() - t0_inf

    # Also compute comparison baselines
    from src.experiments.baselines import run_evidence_point_baseline, run_robust_bp_no_coupling, run_robust_bp_independent_box
    from src.experiments.parameterization import build_image_pgm_context
    
    point_results: List[ClaimEvaluationResult] = []
    no_coup_results: List[ClaimEvaluationResult] = []
    indep_results: List[ClaimEvaluationResult] = []
    
    for claim in bundle.claims:
        p_pt = run_evidence_point_baseline(claim, threshold=cfg.pgm.decision_threshold)
        point_results.append(
            ClaimEvaluationResult(
                run_id=cfg.experiment_id,
                experiment_name="evidence_point_baseline",
                condition="baseline",
                image_id=claim.image_id,
                claim_id=claim.claim_id,
                object_category=claim.object_category,
                text_span=claim.text_span,
                split=claim.split,
                ground_truth=bundle.get_ground_truth_for_claim(claim.claim_id).value if bundle.get_ground_truth_for_claim(claim.claim_id) else None,
                detector_score=claim.evidence.detector_score if claim.evidence.detector_available else None,
                clip_score=claim.evidence.clip_score if claim.evidence.similarity_available else None,
                theta_i=0.0,
                epsilon_i=0.0,
                coupling_j=0.0,
                budget=0.0,
                epsilon_scale=1.0,
                decision_threshold=cfg.pgm.decision_threshold,
                grid_steps=cfg.pgm.grid_steps,
                standard_posterior=p_pt.hallucination_score,
                standard_prediction=p_pt.predicted_decision.value,
                robust_lower=p_pt.hallucination_score,
                robust_upper=p_pt.hallucination_score,
                robust_midpoint=p_pt.hallucination_score,
                interval_width=0.0,
                certified_lower=p_pt.hallucination_score,
                certified_upper=p_pt.hallucination_score,
                field_cert_gap=0.0,
                robust_prediction=p_pt.predicted_decision.value,
                evidence_sensitive=False,
                abstained=False,
                seed=cfg.seed,
            )
        )
        
        ctx = build_image_pgm_context(
            image_id=claim.image_id,
            claims=bundle.get_claims_for_image(claim.image_id),
            config=cfg.pgm,
        )
        
        nc_map = run_robust_bp_no_coupling(ctx)
        nc = nc_map.get(claim.claim_id)
        if nc:
            no_coup_results.append(
                ClaimEvaluationResult(
                    run_id=cfg.experiment_id,
                    experiment_name="robust_bp_no_coupling",
                    condition="ablation_j0",
                    image_id=claim.image_id,
                    claim_id=claim.claim_id,
                    object_category=claim.object_category,
                    text_span=claim.text_span,
                    split=claim.split,
                    ground_truth=bundle.get_ground_truth_for_claim(claim.claim_id).value if bundle.get_ground_truth_for_claim(claim.claim_id) else None,
                    detector_score=claim.evidence.detector_score if claim.evidence.detector_available else None,
                    clip_score=claim.evidence.clip_score if claim.evidence.similarity_available else None,
                    theta_i=float(ctx.model.theta[ctx.claim_to_node[claim.claim_id]]),
                    epsilon_i=float(ctx.model.epsilon[ctx.claim_to_node[claim.claim_id]]),
                    coupling_j=0.0,
                    budget=ctx.budget,
                    epsilon_scale=1.0,
                    decision_threshold=cfg.pgm.decision_threshold,
                    grid_steps=cfg.pgm.grid_steps,
                    standard_posterior=nc.hallucination_score,
                    standard_prediction=nc.predicted_decision.value,
                    robust_lower=nc.interval_lower or nc.hallucination_score,
                    robust_upper=nc.interval_upper or nc.hallucination_score,
                    robust_midpoint=nc.hallucination_score,
                    interval_width=nc.interval_width or 0.0,
                    certified_lower=nc.certified_lower or 0.0,
                    certified_upper=nc.certified_upper or 1.0,
                    field_cert_gap=0.0,
                    robust_prediction=nc.predicted_decision.value,
                    evidence_sensitive=nc.is_evidence_sensitive,
                    abstained=(nc.predicted_decision.value == "abstain"),
                    seed=cfg.seed,
                )
            )

        ib_map = run_robust_bp_independent_box(ctx)
        ib = ib_map.get(claim.claim_id)
        if ib:
            indep_results.append(
                ClaimEvaluationResult(
                    run_id=cfg.experiment_id,
                    experiment_name="robust_bp_independent_box",
                    condition="ablation_independent_box",
                    image_id=claim.image_id,
                    claim_id=claim.claim_id,
                    object_category=claim.object_category,
                    text_span=claim.text_span,
                    split=claim.split,
                    ground_truth=bundle.get_ground_truth_for_claim(claim.claim_id).value if bundle.get_ground_truth_for_claim(claim.claim_id) else None,
                    detector_score=claim.evidence.detector_score if claim.evidence.detector_available else None,
                    clip_score=claim.evidence.clip_score if claim.evidence.similarity_available else None,
                    theta_i=float(ctx.model.theta[ctx.claim_to_node[claim.claim_id]]),
                    epsilon_i=float(ctx.model.epsilon[ctx.claim_to_node[claim.claim_id]]),
                    coupling_j=float(cfg.pgm.base_coupling_j),
                    budget=0.0,
                    epsilon_scale=1.0,
                    decision_threshold=cfg.pgm.decision_threshold,
                    grid_steps=cfg.pgm.grid_steps,
                    standard_posterior=ib.hallucination_score,
                    standard_prediction=ib.predicted_decision.value,
                    robust_lower=ib.interval_lower or ib.hallucination_score,
                    robust_upper=ib.interval_upper or ib.hallucination_score,
                    robust_midpoint=ib.hallucination_score,
                    interval_width=ib.interval_width or 0.0,
                    certified_lower=ib.certified_lower or 0.0,
                    certified_upper=ib.certified_upper or 1.0,
                    field_cert_gap=0.0,
                    robust_prediction=ib.predicted_decision.value,
                    evidence_sensitive=ib.is_evidence_sensitive,
                    abstained=(ib.predicted_decision.value == "abstain"),
                    seed=cfg.seed,
                )
            )

    # 4. Core Evaluation & Interval Statistics
    eval_metrics = evaluate_claim_results(main_results, method_name="standard_bp", decision_threshold=cfg.pgm.decision_threshold)
    interval_stats = compute_interval_statistics(main_results, decision_threshold=cfg.pgm.decision_threshold)

    # 5. Ablation Studies
    logger.info("Executing Ablation Matrix...")
    ablation_runner = AblationMatrixRunner(runner, config=cfg, bundle=bundle)
    ablation_results = ablation_runner.run_all_ablations(bundle)

    # 6. Corruption Robustness Experiment
    corruption_results = {}
    if cfg.corruption.enabled:
        logger.info("Executing Visual Corruption Experiment...")
        corruption_results = evaluate_corruption_pipeline(bundle, cfg)

    # 7. Collect All Evaluated Claim Results
    all_evaluated_results: List[ClaimEvaluationResult] = list(main_results) + list(point_results) + list(no_coup_results) + list(indep_results)
    for ablation_rows in ablation_results.values():
        all_evaluated_results.extend(ablation_rows)
    for corrupt_rows in corruption_results.values():
        all_evaluated_results.extend(corrupt_rows)

    # Save All Raw Results
    jsonl_out = raw_dir / "claim_evaluation_results.jsonl"
    csv_out = raw_dir / "claim_evaluation_results.csv"
    
    with open(jsonl_out, "w", encoding="utf-8") as f_json, open(csv_out, "w", encoding="utf-8", newline="") as f_csv:
        import csv
        writer = None
        for r in all_evaluated_results:
            d = r.to_dict()
            f_json.write(json.dumps(d) + "\n")
            if writer is None:
                writer = csv.DictWriter(f_csv, fieldnames=list(d.keys()))
                writer.writeheader()
            writer.writerow(d)

    # 8. Bootstrap Confidence Intervals
    logger.info("Computing Image-Level Cluster Bootstrap Statistics...")
    bootstrap_stats = run_image_cluster_bootstrap(
        main_results,
        method_name="standard_bp",
        n_resamples=cfg.bootstrap.n_resamples,
        confidence_level=cfg.bootstrap.confidence_level,
        seed=cfg.bootstrap.seed,
    )

    # 9. Case Studies Selection
    logger.info("Selecting Representative Case Studies...")
    case_studies = select_case_studies(main_results, tau=cfg.pgm.decision_threshold)
    with open(out_dir / "case_studies.md", "w", encoding="utf-8") as f:
        f.write("# M8 Representative Case Studies\n\n")
        f.write(f"Generated at: {start_iso}\n\n")
        for c in case_studies:
            f.write(f"### Category: {c.category_tag}\n")
            f.write(f"- **Image ID**: `{c.image_id}`\n")
            f.write(f"- **Claim ID**: `{c.claim_id}`\n")
            f.write(f"- **Category**: {c.object_category}\n")
            f.write(f"- **Ground Truth**: `{c.ground_truth}`\n")
            f.write(f"- **Standard BP Posterior**: `{c.standard_posterior:.4f}` (Prediction: `{c.standard_prediction}`)\n")
            f.write(f"- **Robust BP Interval**: `[{c.robust_lower:.4f}, {c.robust_upper:.4f}]` (Width: `{c.interval_width:.4f}`)\n")
            f.write(f"- **Robust Decision**: `{c.robust_prediction}` | Evidence-Sensitive: `{c.evidence_sensitive}`\n")
            f.write(f"- **Rationale**: {c.description}\n\n---\n\n")

    # 10. Generate Tables 1-7
    logger.info("Generating Publication Tables 1-7...")
    generate_all_tables(out_dir)

    # 11. Generate Figures 1-12
    if cfg.output.save_plots:
        logger.info("Generating Publication Figures 1-12...")
        generate_all_plots(out_dir)

    # 12. Save Summary JSON and Markdown
    end_time = time.time()
    total_duration = end_time - start_time
    
    summary_data = {
        "run_id": cfg.experiment_id,
        "execution_mode": cfg.mode.value,
        "dataset": {
            "total_images": len(bundle.images),
            "total_claims": len(bundle.claims),
            "splits": cfg.eval_splits,
        },
        "timing": {
            "inference_duration_sec": t_inf,
            "total_duration_sec": total_duration,
            "time_per_claim_sec": t_inf / max(1, len(bundle.claims)),
            "time_per_image_sec": t_inf / max(1, len(bundle.images)),
        },
        "metrics": eval_metrics.to_dict(),
        "interval_statistics": interval_stats.to_dict(),
        "bootstrap": {k: v.to_dict() for k, v in bootstrap_stats.items()},
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# M8 Evaluation Summary ({cfg.mode.value})\n\n")
        f.write(f"**Run ID**: `{cfg.experiment_id}` | **Splits**: `{cfg.eval_splits}` | **Timestamp**: `{start_iso}`\n\n")
        f.write("## 1. Classification Metrics\n\n")
        f.write(f"- **Total Evaluated Claims**: {eval_metrics.total_claims} (Excluded UNKNOWN: {eval_metrics.unknown_gt_count})\n")
        acc_str = f"{eval_metrics.accuracy:.4f}" if eval_metrics.accuracy is not None else "N/A"
        f1_str = f"{eval_metrics.f1:.4f}" if eval_metrics.f1 is not None else "N/A"
        roc_str = f"{eval_metrics.roc_auc:.4f}" if eval_metrics.roc_auc is not None else "N/A"
        f.write(f"- **Standard BP**: Accuracy = {acc_str} | F1 = {f1_str} | ROC-AUC = {roc_str}\n")
        f.write(f"- **Robust BP Abstentions**: {eval_metrics.abstained_count}\n\n")
        f.write("## 2. Robust Interval Statistics\n\n")
        f.write(f"- **Mean Width**: {interval_stats.mean_width:.4f} (Median: {interval_stats.median_width:.4f}, Std: {interval_stats.std_width:.4f})\n")
        f.write(f"- **Evidence-Sensitive Fraction**: {interval_stats.evidence_sensitive_fraction * 100:.2f}%\n\n")
        f.write("## 3. Computational Timing\n\n")
        f.write(f"- **Total Runtime**: {total_duration:.2f} s ({summary_data['timing']['time_per_claim_sec']*1000:.2f} ms/claim)\n")

    # 13. Create & Save Manifest
    manifest = create_experiment_manifest(
        config=cfg,
        bundle=bundle,
        run_id=cfg.experiment_id,
        start_time=start_time,
        end_time=end_time,
        splits_evaluated=cfg.eval_splits,
        total_images=len(bundle.images),
        total_claims=len(bundle.claims),
        annotated_claims=eval_metrics.evaluated_claims,
    )
    save_experiment_manifest(manifest, out_dir / "experiment_manifest.json")

    # 14. Run validation on output artifacts
    logger.info("Validating generated artifacts...")
    validate_m8_artifacts(out_dir, require_plots=cfg.output.save_plots)

    logger.info("=" * 60)
    logger.info(f"MILESTONE 8 EXPERIMENTS COMPLETED SUCCESSFULLY in {total_duration:.2f}s")
    logger.info(f"Results available in: {out_dir.resolve()}")
    logger.info("=" * 60)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Master runner for Milestone 8 experiments.")
    parser.add_argument("--config", type=str, help="Path to M8 config YAML")
    parser.add_argument("--smoke", action="store_true", help="Run in fast synthetic smoke test mode")
    parser.add_argument("--subset", type=int, help="Limit number of images to run")
    parser.add_argument("--split", type=str, help="Evaluate specific split (TRAIN/VALIDATION/CALIBRATION/TEST/ALL)")
    parser.add_argument("--experiments", type=str, help="Comma-separated experiment list")
    parser.add_argument("--seed", type=int, help="Random seed override")
    parser.add_argument("--output-dir", type=str, help="Output directory path")
    parser.add_argument("--resume", action="store_true", help="Resume cached runs")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    parser.add_argument("--no-corruption", action="store_true", help="Disable corruption experiment")
    parser.add_argument("--force-unsafe", action="store_true", help="Bypass strict safety checks")
    args = parser.parse_args()

    exp_list = [x.strip() for x in args.experiments.split(",")] if args.experiments else None

    run_experiments(
        config_path=args.config,
        smoke=args.smoke,
        subset=args.subset,
        split=args.split,
        experiments=exp_list,
        seed=args.seed,
        output_dir=args.output_dir,
        resume=args.resume,
        no_plots=args.no_plots,
        no_corruption=args.no_corruption,
        force_unsafe=args.force_unsafe,
    )


if __name__ == "__main__":
    main()
