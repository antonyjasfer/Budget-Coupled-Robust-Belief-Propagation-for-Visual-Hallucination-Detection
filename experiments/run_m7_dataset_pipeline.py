"""
Milestone 7 Pipeline Runner and Quality Audit CLI.

Usage:
    python -m experiments.run_m7_dataset_pipeline manifest [options]
    python -m experiments.run_m7_dataset_pipeline export-templates [options]
    python -m experiments.run_m7_dataset_pipeline merge [options]
    python -m experiments.run_m7_dataset_pipeline audit [options]
    python -m experiments.run_m7_dataset_pipeline all [options]
"""

import argparse
from pathlib import Path
import json
import sys

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    DatasetSource,
    SplitName,
)
from src.data.manifests import create_manifest
from src.data.splits import split_manifest_by_image_groups, DEFAULT_PROPORTIONS
from src.annotation.workflow import (
    load_m6_evidence_file,
    ingest_m6_evidence_to_m7_claims,
    export_masked_templates,
    load_annotations_file,
    load_adjudications_file,
    build_final_ground_truth_dataset,
    generate_quality_report,
)
from src.annotation.agreement import compute_cohens_kappa


def cmd_manifest(args):
    """Build or audit the master M7 image manifest."""
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Ingest available real M6 evidence images to form the initial manifest base
    evidence_path = Path(args.evidence_file)
    if not evidence_path.exists():
        print(f"Error: Evidence file {evidence_path} not found.")
        sys.exit(1)

    evidence_records = load_m6_evidence_file(evidence_path)
    seen_images = {}
    for ev in evidence_records:
        if ev.image_id not in seen_images:
            seen_images[ev.image_id] = ev

    entries = []
    for idx, (img_id, ev) in enumerate(seen_images.items()):
        img_rec = ImageRecord(
            image_id=img_id,
            dataset_source=DatasetSource.COCO,
            file_hash=ev.image_hash,
            metadata={"m6_response_id": ev.metadata.get("response_id")},
        )
        entries.append(
            DatasetManifestEntry(
                image=img_rec,
                split=SplitName.TRAIN,
            )
        )

    manifest = create_manifest(
        manifest_id="m7_coco_master_manifest",
        description="M7 master image manifest with deterministic group splitting",
        entries=entries,
        metadata={
            "target_benchmark_size": args.target_size,
            "seed": args.seed,
        },
    )

    # Deterministic split via src.data.splits
    split_res = split_manifest_by_image_groups(manifest, seed=args.seed)
    final_manifest = split_res.unified_manifest

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest.to_dict(), f, indent=2)

    print(f"Manifest written to: {manifest_path}")
    print(f"Images in manifest: {len(final_manifest.entries)}")
    print(f"Target benchmark size: {args.target_size}")


def cmd_export_templates(args):
    """Export masked annotation templates from M6 evidence."""
    evidence_path = Path(args.evidence_file)
    if not evidence_path.exists():
        print(f"Error: Evidence file {evidence_path} not found.")
        sys.exit(1)

    evidence_records = load_m6_evidence_file(evidence_path)
    claims = ingest_m6_evidence_to_m7_claims(evidence_records, manifest=None, allow_missing_images=True)

    out_dir = Path(args.output_dir)
    tpl_a, tpl_b = export_masked_templates(claims, output_dir=out_dir)

    print(f"Exported {len(claims)} masked tasks to:")
    print(f"  Annotator A template: {tpl_a}")
    print(f"  Annotator B template: {tpl_b}")


def cmd_merge(args):
    """Merge dual annotations, apply adjudications, and write final ground-truth dataset."""
    evidence_path = Path(args.evidence_file)
    evidence_records = load_m6_evidence_file(evidence_path)
    claims = ingest_m6_evidence_to_m7_claims(evidence_records, manifest=None, allow_missing_images=True)

    anns_a = load_annotations_file(args.annotator_a) if args.annotator_a else {}
    anns_b = load_annotations_file(args.annotator_b) if args.annotator_b else {}
    adjudications = load_adjudications_file(args.adjudicated) if args.adjudicated else {}

    final_records = build_final_ground_truth_dataset(
        claims=claims,
        annotations_a=anns_a,
        annotations_b=anns_b,
        adjudications=adjudications,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r.to_dict()) + "\n")

    print(f"Final ground truth dataset written to: {out_path}")
    print(f"Total claims: {len(final_records)}")


def cmd_audit(args):
    """Audit the ground truth dataset and generate quality report."""
    evidence_path = Path(args.evidence_file)
    evidence_records = load_m6_evidence_file(evidence_path)
    claims = ingest_m6_evidence_to_m7_claims(evidence_records, manifest=None, allow_missing_images=True)

    anns_a = load_annotations_file(args.annotator_a) if args.annotator_a else {}
    anns_b = load_annotations_file(args.annotator_b) if args.annotator_b else {}
    adjudications = load_adjudications_file(args.adjudicated) if args.adjudicated else {}

    final_records = build_final_ground_truth_dataset(
        claims=claims,
        annotations_a=anns_a,
        annotations_b=anns_b,
        adjudications=adjudications,
    )

    manifest = None
    if args.manifest and Path(args.manifest).exists():
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = DatasetManifest.from_dict(json.load(f))

    report = generate_quality_report(
        manifest=manifest,
        claims=claims,
        final_records=final_records,
        target_reserved_count=args.target_size,
    )

    report_json_path = Path(args.report_json)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    report_md_path = Path(args.report_md)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    print("=" * 70)
    print("M7 DATASET QUALITY AUDIT REPORT")
    print("=" * 70)
    print(report.to_markdown())
    print("=" * 70)
    print(f"JSON report saved: {report_json_path}")
    print(f"Markdown report saved: {report_md_path}")


def cmd_all(args):
    """Execute complete end-to-end M7 pipeline on available real data."""
    print("=== Step 1: Generating M7 Master Image Manifest ===")
    cmd_manifest(args)

    print("\n=== Step 2: Exporting Masked Annotation Templates ===")
    cmd_export_templates(args)

    print("\n=== Step 3: Merging Dataset & Annotations ===")
    cmd_merge(args)

    print("\n=== Step 4: Auditing Quality & Generating Reports ===")
    cmd_audit(args)


def main():
    parser = argparse.ArgumentParser(description="Milestone 7 Ground-Truth & Annotation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common options
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--evidence-file", default="data/exports/claim_level_evidence.jsonl", help="M6 evidence export")
    common_parser.add_argument("--manifest-out", default="data/manifests/m7_image_manifest.json", help="Manifest output path")
    common_parser.add_argument("--manifest", default="data/manifests/m7_image_manifest.json", help="Manifest path for audit")
    common_parser.add_argument("--output-dir", default="data/annotations", help="Annotation templates output directory")
    common_parser.add_argument("--annotator-a", default="data/annotations/m7_annotator_A.jsonl", help="Annotator A completed file")
    common_parser.add_argument("--annotator-b", default="data/annotations/m7_annotator_B.jsonl", help="Annotator B completed file")
    common_parser.add_argument("--adjudicated", default="data/annotations/m7_adjudications.jsonl", help="Adjudications file")
    common_parser.add_argument("--output", default="data/exports/m7_ground_truth.jsonl", help="Final ground truth output")
    common_parser.add_argument("--report-json", default="reports/m7_dataset_quality_report.json", help="Quality report JSON")
    common_parser.add_argument("--report-md", default="reports/m7_dataset_quality_report.md", help="Quality report Markdown")
    common_parser.add_argument("--target-size", type=int, default=600, help="Target benchmark image count")
    common_parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")

    # Subcommands
    subparsers.add_parser("manifest", parents=[common_parser])
    subparsers.add_parser("export-templates", parents=[common_parser])
    subparsers.add_parser("merge", parents=[common_parser])
    subparsers.add_parser("audit", parents=[common_parser])
    subparsers.add_parser("all", parents=[common_parser])

    args = parser.parse_args()

    if args.command == "manifest":
        cmd_manifest(args)
    elif args.command == "export-templates":
        cmd_export_templates(args)
    elif args.command == "merge":
        cmd_merge(args)
    elif args.command == "audit":
        cmd_audit(args)
    elif args.command == "all":
        cmd_all(args)


if __name__ == "__main__":
    main()
