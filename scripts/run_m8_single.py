"""CLI for running single-image or single-claim inference with Standard & Robust BP.

Useful for deep-dive diagnostics, case study analysis, and interactive debugging.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.configs import M8ExperimentConfig
from src.experiments.loaders import load_m8_dataset
from src.experiments.parameterization import build_image_pgm_context
from src.experiments.baselines import (
    run_evidence_point_baseline,
    run_standard_bp_baseline,
    run_robust_bp_proposed,
    run_robust_bp_no_coupling,
    run_robust_bp_independent_box,
)
from src.experiments.robust_runner import generate_claim_robust_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m8_single")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect single image/claim inference under Standard and Robust BP.")
    parser.add_argument("--image-id", type=str, help="Target image ID (e.g. 'coco_000000000064')")
    parser.add_argument("--claim-id", type=str, help="Target claim ID")
    parser.add_argument("--dataset-path", type=str, default="data/m7_evidence/m7_verified_evidence.jsonl")
    parser.add_argument("--ground-truth-path", type=str, default="data/annotations/claim_annotations.jsonl")
    parser.add_argument("--manifest-path", type=str, default="data/manifests/m7_image_manifest.json")
    parser.add_argument("--budget", type=float, default=None, help="Global budget B override")
    parser.add_argument("--coupling", type=float, default=None, help="Coupling J override")
    parser.add_argument("--detailed-profile", action="store_true", help="Generate profile sweep across budgets")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic smoke fixture for quick testing")
    args = parser.parse_args()

    cfg = M8ExperimentConfig()
    if args.budget is not None:
        cfg.pgm.budget_ratio = args.budget
    if args.coupling is not None:
        cfg.pgm.base_coupling_j = args.coupling

    # Load dataset
    if args.smoke:
        from src.experiments.loaders import create_synthetic_smoke_bundle
        bundle = create_synthetic_smoke_bundle(num_images=1, seed=42)
    else:
        bundle = load_m8_dataset(cfg)

    image_ids = list(bundle.images.keys())
    if not image_ids:
        logger.error("No images found in dataset.")
        sys.exit(1)

    target_image_id = args.image_id or image_ids[0]
    if target_image_id not in bundle.images:
        logger.error(f"Image ID '{target_image_id}' not found in loaded images: {image_ids}")
        sys.exit(1)

    claims = bundle.get_claims_for_image(target_image_id)
    if not claims:
        logger.error(f"No claims found for image '{target_image_id}'.")
        sys.exit(1)

    target_claim = None
    if args.claim_id:
        for c in claims:
            if c.claim_id == args.claim_id:
                target_claim = c
                break
        if target_claim is None:
            logger.error(f"Claim ID '{args.claim_id}' not found in image '{target_image_id}'. Available: {[c.claim_id for c in claims]}")
            sys.exit(1)
    else:
        target_claim = claims[0]

    logger.info(f"Analyzing Image '{target_image_id}' | Claim '{target_claim.claim_id}'")
    logger.info(f"Claim Text: {target_claim.text_span}")
    gt = bundle.get_ground_truth_for_claim(target_claim.claim_id)
    logger.info(f"Ground Truth: {gt.value if gt else 'None'}")

    ctx = build_image_pgm_context(
        image_id=target_image_id,
        claim_records=claims,
        config=cfg.pgm,
    )

    cid = target_claim.claim_id

    # Baseline calculations
    p_point = run_evidence_point_baseline(target_claim, threshold=cfg.pgm.decision_threshold)
    std_map = run_standard_bp_baseline(ctx)
    rob_map = run_robust_bp_proposed(ctx)
    nc_map = run_robust_bp_no_coupling(ctx)
    box_map = run_robust_bp_independent_box(ctx)

    p_std = std_map[cid]
    rob_res = rob_map[cid]
    no_coup_res = nc_map[cid]
    indep_res = box_map[cid]

    logger.info("\n" + "="*50)
    logger.info("SINGLE CLAIM INFERENCE RESULTS")
    logger.info("="*50)
    logger.info(f"Evidence Point Posterior:     {p_point.hallucination_score:.4f} (Decision: {p_point.predicted_decision.value})")
    logger.info(f"Standard BP Posterior:        {p_std.hallucination_score:.4f} (Decision: {p_std.predicted_decision.value})")
    logger.info(f"Robust BP Interval [L, U]:    [{rob_res.interval_lower:.4f}, {rob_res.interval_upper:.4f}] (Width: {rob_res.interval_width:.4f})")
    logger.info(f"  Decision at tau={cfg.pgm.decision_threshold}: {rob_res.predicted_decision.value}")
    logger.info(f"  Evidence-Sensitive:           {rob_res.is_evidence_sensitive}")
    logger.info(f"Ablation (J=0) Interval:        [{no_coup_res.interval_lower:.4f}, {no_coup_res.interval_upper:.4f}] (Width: {no_coup_res.interval_width:.4f})")
    logger.info(f"Independent Box Interval:       [{indep_res.interval_lower:.4f}, {indep_res.interval_upper:.4f}] (Width: {indep_res.interval_width:.4f})")
    logger.info("="*50)

    if args.detailed_profile:
        logger.info("\nGenerating Budget Sensitivity Profile...")
        profile = generate_claim_robust_profile(ctx, target_claim_id=cid, ground_truth=gt)
        logger.info(json.dumps(profile.to_dict(), indent=2))


if __name__ == "__main__":
    main()
