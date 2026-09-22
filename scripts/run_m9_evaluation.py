"""Master Evaluation and Results Consolidation CLI for Milestone 9 (M9).

Usage:
    # 1. Check readiness for final execution (gate check)
    python scripts/run_m9_evaluation.py --mode ready

    # 2. Run real development consolidation (canonical M6 subset)
    python scripts/run_m9_evaluation.py --mode dev --output-dir reports/m9/final --allow-overwrite

    # 3. Consolidate from pre-existing claim evaluation results JSONL
    python scripts/run_m9_evaluation.py --mode dev --source-results reports/m8/raw_claim_results.jsonl --output-dir reports/m9/final --allow-overwrite

    # 4. Execute final locked experiment (strictly gates on full annotated dataset)
    python scripts/run_m9_evaluation.py --mode final --output-dir reports/m9/final
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experiments.configs import get_default_m8_config, ExecutionMode
from src.experiments.inference_runner import ClaimEvaluationResult, InferenceRunner
from src.experiments.loaders import create_synthetic_smoke_bundle
from src.scientific.gate import FinalDatasetGate, GateStatus
from src.scientific.consolidator import M9ResultsConsolidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m9_runner")


def main() -> int:
    parser = argparse.ArgumentParser(description="Milestone 9 Scientific Evaluation & Consolidation CLI")
    parser.add_argument(
        "--mode",
        choices=["ready", "dev", "development", "final", "smoke"],
        default="ready",
        help="Execution mode (ready: gate check; dev: canonical subset; final: locked dataset)",
    )
    parser.add_argument(
        "--source-results",
        type=str,
        default=None,
        help="Path to pre-existing claim evaluation results JSONL to consolidate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/m9/final",
        help="Output directory for final publication package",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow overwriting an existing final results package",
    )
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    config = get_default_m8_config()

    # Mode 1: READY / GATE CHECK
    if args.mode == "ready":
        logger.info("Checking Milestone 9 Final Dataset Gate...")
        gate = FinalDatasetGate()
        gate_res = gate.verify()
        print("=" * 60)
        print("MILESTONE 9 FINAL DATASET GATE STATUS:")
        print(f"Status: {gate_res.status.value}")
        print(f"Passed: {gate_res.passed}")
        print("Diagnostics:")
        for diag in gate_res.diagnostics:
            print(f"  - {diag}")
        print("=" * 60)

        if gate_res.passed:
            print("STATE: READY_FOR_FINAL_EXECUTION (Locked dataset verified)")
            return 0
        else:
            print("STATE: READY (Software complete; full 600-image dataset pending lock)")
            return 0

    # Mode 2: SMOKE
    consolidator = M9ResultsConsolidator(
        config=config,
        output_dir=out_path,
        allow_overwrite=args.allow_overwrite,
    )

    if args.mode == "smoke":
        logger.info("Executing M9 pipeline in SMOKE mode with synthetic fixture...")
        bundle = create_synthetic_smoke_bundle()
        runner = InferenceRunner(config)
        claim_results, _ = runner.run_batch(bundle.image_records)
        consolidator.consolidate(claim_results, execution_mode="SMOKE")
        print(f"SUCCESS: M9 Smoke consolidation completed at {out_path}")
        return 0

    # Mode 3: SOURCE-RESULTS or DEV
    if args.source_results:
        src_file = Path(args.source_results)
        logger.info(f"Loading pre-computed results from {src_file}...")
        claim_results = consolidator.load_raw_results(src_file)
        consolidator.consolidate(claim_results, execution_mode="DEVELOPMENT")
        print(f"SUCCESS: M9 Consolidation completed at {out_path}")
        return 0

    if args.mode in ["dev", "development"]:
        logger.info("Executing M9 Development Pipeline on canonical real M6 subset...")
        claim_results = consolidator.run_development_pipeline()
        consolidator.consolidate(claim_results, execution_mode="DEVELOPMENT")
        print(f"SUCCESS: M9 Development evaluation and consolidation completed at {out_path}")
        return 0

    # Mode 4: FINAL
    if args.mode == "final":
        logger.info("Attempting FINAL scientific execution...")
        gate = FinalDatasetGate()
        gate_res = gate.verify()
        if not gate_res.passed:
            logger.error("CANNOT RUN FINAL SCIENTIFIC EXPERIMENT: Final dataset gate failed.")
            for diag in gate_res.diagnostics:
                logger.error(f"  - {diag}")
            print("\nFINAL SCIENTIFIC EXPERIMENT: NOT RUN (Full M7 locked dataset required)")
            return 1

        claim_results = consolidator.run_final_pipeline()
        consolidator.consolidate(claim_results, execution_mode="FINAL")
        print(f"SUCCESS: FINAL SCIENTIFIC RESULTS GENERATED at {out_path}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
