"""
Command-Line Interface (CLI) for data pipeline operations:
- Manifest construction
- Manifest validation
- Deterministic split generation
- Overlap auditing
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.data.schemas import DatasetManifest, SplitName
from src.data.manifests import load_manifest, save_manifest, validate_manifest, create_manifest
from src.data.splits import split_manifest_by_image_groups
from src.data.image_registry import ImageRegistry


def cmd_validate_manifest(args: argparse.Namespace) -> int:
    """Validate manifest file for referential integrity."""
    path = Path(args.manifest_path)
    print(f"Validating manifest: {path}")
    try:
        manifest = load_manifest(path, validate=False)
        errors = validate_manifest(manifest)
        if errors:
            print(f"VALIDATION FAILED with {len(errors)} error(s):")
            for err in errors:
                print(f"  - {err}")
            return 1
        else:
            print(f"VALIDATION PASSED: Manifest is valid ({len(manifest.entries)} image entries).")
            return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


def cmd_generate_splits(args: argparse.Namespace) -> int:
    """Split manifest deterministically by image identity groups."""
    in_path = Path(args.manifest_path)
    out_dir = Path(args.output_dir) if args.output_dir else in_path.parent
    seed = int(args.seed)

    print(f"Loading manifest from: {in_path}")
    manifest = load_manifest(in_path, validate=True)

    reserved_ids: List[str] = []
    if args.reserved_file:
        res_path = Path(args.reserved_file)
        if res_path.exists():
            with open(res_path, "r", encoding="utf-8") as f:
                reserved_ids = json.load(f)
            print(f"Loaded {len(reserved_ids)} reserved external image IDs from {res_path}")

    props = {
        SplitName.TRAIN: args.train_prop,
        SplitName.VALIDATION: args.val_prop,
        SplitName.CALIBRATION: args.calib_prop,
        SplitName.TEST: args.test_prop,
    }

    print(f"Generating splits with seed={seed}, proportions={props}...")
    split_res = split_manifest_by_image_groups(
        manifest,
        seed=seed,
        proportions=props,
        reserved_external_ids=reserved_ids,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save unified manifest with split assignments
    unified_path = out_dir / f"{in_path.stem}_split_unified.json"
    save_manifest(split_res.unified_manifest, unified_path)
    print(f"Saved unified split manifest to: {unified_path}")

    # Save per-split manifests
    for s_name, s_manifest in split_res.manifests_by_split.items():
        s_path = out_dir / f"{in_path.stem}_{s_name.value}.json"
        save_manifest(s_manifest, s_path)
        print(f"  - Saved {s_name.value:<12} ({len(s_manifest.entries)} images) -> {s_path}")

    # Save split metadata
    meta_path = out_dir / f"{in_path.stem}_split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": split_res.metadata.seed,
                "input_manifest_hash": split_res.metadata.input_manifest_hash,
                "achieved_counts": split_res.metadata.achieved_counts,
                "achieved_proportions": split_res.metadata.achieved_proportions,
                "reserved_external_count": split_res.metadata.reserved_external_count,
            },
            f,
            indent=2,
        )
    print(f"Saved split metadata to: {meta_path}")
    print("SPLIT GENERATION COMPLETE.")
    return 0


def cmd_audit_overlap(args: argparse.Namespace) -> int:
    """Audit overlap between two manifests or split files."""
    path_a = Path(args.manifest_a)
    path_b = Path(args.manifest_b)

    print(f"Auditing overlap between:")
    print(f"  A: {path_a}")
    print(f"  B: {path_b}")

    manifest_a = load_manifest(path_a, validate=True)
    manifest_b = load_manifest(path_b, validate=True)

    registry = ImageRegistry()
    registry.register_manifest(manifest_a)
    registry.register_manifest(manifest_b)

    ids_a = {e.image.image_id for e in manifest_a.entries}
    ids_b = {e.image.image_id for e in manifest_b.entries}

    report = registry.audit_overlap(ids_a, ids_b, label_a=path_a.name, label_b=path_b.name)
    print("=" * 70)
    print("OVERLAP AUDIT REPORT")
    print("=" * 70)
    print(report.summary)
    print("=" * 70)

    return 1 if report.has_overlap else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.data.cli",
        description="CLI utilities for hallucination detection dataset manifests and splits.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # validate-manifest
    p_val = subparsers.add_parser("validate-manifest", help="Validate a dataset manifest file.")
    p_val.add_argument("manifest_path", help="Path to manifest JSON file.")

    # generate-splits
    p_split = subparsers.add_parser("generate-splits", help="Partition a manifest deterministically.")
    p_split.add_argument("manifest_path", help="Path to input manifest JSON file.")
    p_split.add_argument("--output-dir", default="", help="Output directory for split manifests.")
    p_split.add_argument("--seed", type=int, default=42, help="Random seed (default 42).")
    p_split.add_argument("--train-prop", type=float, default=0.50, help="Train proportion (default 0.50).")
    p_split.add_argument("--val-prop", type=float, default=0.15, help="Validation proportion (default 0.15).")
    p_split.add_argument("--calib-prop", type=float, default=0.15, help="Calibration proportion (default 0.15).")
    p_split.add_argument("--test-prop", type=float, default=0.20, help="Test proportion (default 0.20).")
    p_split.add_argument("--reserved-file", default="", help="Optional JSON file listing reserved image IDs.")

    # audit-overlap
    p_audit = subparsers.add_parser("audit-overlap", help="Audit overlap between two manifests.")
    p_audit.add_argument("manifest_a", help="Path to first manifest JSON.")
    p_audit.add_argument("manifest_b", help="Path to second manifest JSON.")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "validate-manifest":
        return cmd_validate_manifest(args)
    elif args.command == "generate-splits":
        return cmd_generate_splits(args)
    elif args.command == "audit-overlap":
        return cmd_audit_overlap(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
