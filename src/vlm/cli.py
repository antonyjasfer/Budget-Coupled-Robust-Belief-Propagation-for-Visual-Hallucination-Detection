"""
Command-line interface for frozen-VLM caption generation, caching, preflight validation, and pilot execution.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional, List

from src.data.schemas import DatasetManifest
from src.data.manifests import load_manifest
from src.vlm.provider import VLMGenerationConfig, SyntheticVLMProvider
from src.vlm.llava_provider import LLaVA15Provider
from src.vlm.cache import VLMCache
from src.vlm.pipeline import (
    run_vlm_pilot,
    export_annotation_bundle,
)
from src.vlm.preflight import run_preflight, PreflightReport
from src.vlm.pilot_bundle import create_portable_pilot_bundle, validate_portable_pilot_bundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vlm_cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-VLM Caption Acquisition & Pilot Execution CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: preflight
    preflight_parser = subparsers.add_parser("preflight", help="Run comprehensive preflight checks before launching real pilot")
    preflight_parser.add_argument("--manifest", type=str, required=True, help="Path to input dataset manifest JSON")
    preflight_parser.add_argument("--config", type=str, default="configs/vlm_generation.json", help="Path to VLM generation config JSON")
    preflight_parser.add_argument("--split-registry", type=str, default=None, help="Optional path to split registry JSON")
    preflight_parser.add_argument("--image-base-dir", type=str, default=None, help="Base directory for image files")
    preflight_parser.add_argument("--cache-dir", type=str, default="data/cache/vlm", help="Directory for response disk cache")
    preflight_parser.add_argument("--output-bundle", type=str, default="data/exports/annotation_pilot_bundle.json", help="Path for exported annotation bundle")
    preflight_parser.add_argument("--sample-size", type=int, default=10, help="Maximum number of training images to sample (max 10)")
    preflight_parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    preflight_parser.add_argument("--allow-download", action="store_true", help="Allow downloading model weights from HuggingFace Hub")
    preflight_parser.add_argument("--json", action="store_true", help="Output raw structured JSON report")

    # Subcommand: export-bundle
    export_parser = subparsers.add_parser("export-bundle", help="Export a portable self-contained real-image pilot bundle")
    export_parser.add_argument("--manifest", type=str, required=True, help="Path to input dataset manifest JSON")
    export_parser.add_argument("--output-dir", type=str, required=True, help="Directory to create portable bundle in")
    export_parser.add_argument("--config", type=str, default="configs/vlm_generation.json", help="Path to VLM generation config JSON")
    export_parser.add_argument("--split-registry", type=str, default=None, help="Optional path to split registry JSON")
    export_parser.add_argument("--image-base-dir", type=str, default=None, help="Base directory for image files")
    export_parser.add_argument("--sample-size", type=int, default=10, help="Maximum number of training images to sample (max 10)")
    export_parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    export_parser.add_argument("--no-copy-images", action="store_true", help="Do not copy image files, reference them relatively")

    # Subcommand: validate-bundle
    val_parser = subparsers.add_parser("validate-bundle", help="Validate a portable pilot bundle structure and hashes")
    val_parser.add_argument("--bundle-dir", type=str, required=True, help="Path to exported bundle directory")

    # Subcommand: run-pilot
    pilot_parser = subparsers.add_parser("run-pilot", help="Run the development-only real image pilot")
    pilot_parser.add_argument("--manifest", type=str, required=True, help="Path to input dataset manifest JSON")
    pilot_parser.add_argument("--split-registry", type=str, default=None, help="Optional path to split registry JSON")
    pilot_parser.add_argument("--cache-dir", type=str, default="data/cache/vlm", help="Directory for response disk cache")
    pilot_parser.add_argument("--output-bundle", type=str, default="data/exports/annotation_pilot_bundle.json", help="Path to save exported annotation bundle")
    pilot_parser.add_argument("--config", type=str, default="configs/vlm_generation.json", help="Path to VLM generation config JSON")
    pilot_parser.add_argument("--synthetic", action="store_true", help="Use offline synthetic mock provider for testing")
    pilot_parser.add_argument("--allow-download", action="store_true", help="Allow downloading model weights from HuggingFace Hub if missing locally")
    pilot_parser.add_argument("--sample-size", type=int, default=10, help="Maximum number of training images to sample (max 10)")
    pilot_parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    pilot_parser.add_argument("--image-base-dir", type=str, default=None, help="Base directory for image files")

    # Subcommand: generate-caption
    gen_parser = subparsers.add_parser("generate-caption", help="Generate a caption for a single image")
    gen_parser.add_argument("--image-path", type=str, required=True, help="Path to input image file")
    gen_parser.add_argument("--prompt", type=str, default=None, help="Custom prompt text")
    gen_parser.add_argument("--cache-dir", type=str, default="data/cache/vlm", help="Directory for response disk cache")
    gen_parser.add_argument("--synthetic", action="store_true", help="Use synthetic mock provider")
    gen_parser.add_argument("--allow-download", action="store_true", help="Allow downloading model weights")
    gen_parser.add_argument("--config", type=str, default="configs/vlm_generation.json", help="Path to VLM generation config JSON")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.command == "preflight":
        report = run_preflight(
            manifest_path=parsed.manifest,
            config_path=parsed.config,
            split_registry_path=parsed.split_registry,
            image_base_dir=parsed.image_base_dir,
            cache_dir=parsed.cache_dir,
            output_bundle_path=parsed.output_bundle,
            sample_size=parsed.sample_size,
            seed=parsed.seed,
            allow_download=parsed.allow_download,
        )

        if parsed.json:
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if report.overall_passed else 1

        print("=" * 78)
        print("REAL-PILOT PREFLIGHT VERIFICATION REPORT")
        print("=" * 78)
        print(f"Overall Status   : {'READY' if report.overall_passed else 'BLOCKED'}")
        print(f"Gates Passed     : {report.passed_gates} / {report.total_gates}")
        print(f"Blocked Gates    : {report.blocked_gates}")
        print(f"Warnings         : {report.warning_gates}")
        print("-" * 78)
        for gate in report.gates:
            status_tag = f"[{gate.status}]"
            print(f"{status_tag:<10} {gate.gate_name:<30} : {gate.message}")
        print("-" * 78)
        if report.blockers:
            print("EXPLICIT BLOCKERS:")
            for b in report.blockers:
                print(f"  - {b}")
        if report.warnings:
            print("WARNINGS:")
            for w in report.warnings:
                print(f"  - {w}")
        print("=" * 78)
        return 0 if report.overall_passed else 1

    elif parsed.command == "export-bundle":
        out_dir = create_portable_pilot_bundle(
            manifest_path=parsed.manifest,
            output_dir=parsed.output_dir,
            config_path=parsed.config,
            split_registry_path=parsed.split_registry,
            image_base_dir=parsed.image_base_dir,
            sample_size=parsed.sample_size,
            seed=parsed.seed,
            copy_images=not parsed.no_copy_images,
        )
        logger.info(f"Successfully created portable pilot bundle at: {out_dir}")
        return 0

    elif parsed.command == "validate-bundle":
        res = validate_portable_pilot_bundle(parsed.bundle_dir)
        if res["valid"]:
            print(f"Portable bundle validation PASSED: {res['bundle_id']} ({res['image_count']} images)")
            return 0
        else:
            print(f"Portable bundle validation FAILED:")
            for err in res["errors"]:
                print(f"  - {err}")
            return 1

    elif parsed.command == "run-pilot":
        # Load config
        gen_config = VLMGenerationConfig()
        if parsed.config and Path(parsed.config).exists():
            with open(parsed.config, "r", encoding="utf-8") as f:
                gen_config = VLMGenerationConfig.from_dict(json.load(f))

        # Enforce pilot max sample size <= 10
        sample_size = min(parsed.sample_size, 10)

        # Load manifest
        manifest = load_manifest(parsed.manifest)

        split_registry = None
        if parsed.split_registry and Path(parsed.split_registry).exists():
            with open(parsed.split_registry, "r", encoding="utf-8") as f:
                split_registry = json.load(f)

        # Setup cache
        cache = VLMCache(cache_dir=parsed.cache_dir)

        # Setup provider
        if parsed.synthetic:
            provider = SyntheticVLMProvider()
            logger.info("Using SyntheticVLMProvider (offline mode)")
        else:
            provider = LLaVA15Provider(
                model_name=gen_config.model_name,
                device=gen_config.device,
                dtype=gen_config.dtype,
                local_files_only=not parsed.allow_download,
                allow_download=parsed.allow_download,
            )
            logger.info(f"Using LLaVA15Provider ({gen_config.model_name})")

        try:
            bundle, stats = run_vlm_pilot(
                manifest=manifest,
                provider=provider,
                cache=cache,
                gen_config=gen_config,
                split_registry=split_registry,
                sample_size=sample_size,
                seed=parsed.seed,
                image_base_dir=parsed.image_base_dir,
            )
        except RuntimeError as e:
            logger.error(f"Pilot execution blocked or failed: {e}")
            return 1

        out_path = export_annotation_bundle(bundle, parsed.output_bundle)
        logger.info(f"Exported annotation-ready bundle with {bundle.total_responses} entries to: {out_path}")
        logger.info(f"Execution stats: {stats.to_dict()}")
        return 0

    elif parsed.command == "generate-caption":
        gen_config = VLMGenerationConfig()
        if parsed.config and Path(parsed.config).exists():
            with open(parsed.config, "r", encoding="utf-8") as f:
                gen_config = VLMGenerationConfig.from_dict(json.load(f))

        if parsed.prompt:
            gen_config.prompt = parsed.prompt

        cache = VLMCache(cache_dir=parsed.cache_dir)

        if parsed.synthetic:
            provider = SyntheticVLMProvider()
        else:
            provider = LLaVA15Provider(
                model_name=gen_config.model_name,
                device=gen_config.device,
                dtype=gen_config.dtype,
                local_files_only=not parsed.allow_download,
                allow_download=parsed.allow_download,
            )

        resp = provider.generate_caption(
            image_path=parsed.image_path,
            prompt=gen_config.prompt,
            gen_config=gen_config,
        )
        cache.put(resp, gen_config=gen_config)
        print(f"Generated caption:\n{resp.caption}")
        print(f"Response ID: {resp.response_id}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
