"""
CLI utilities for object-existence claim extraction and raw evidence attachment.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    GeneratedResponseRecord,
    AtomicObjectExistenceClaim,
)
from src.data.manifests import load_manifest, save_manifest, validate_manifest
from src.claims.vocabulary import create_coco_category_registry, CategoryRegistry
from src.claims.extraction import ConservativeClaimExtractor, ExtractionReport
from src.evidence.fixture_provider import FixtureEvidenceProvider
from src.evidence.schemas import RawEvidenceRecord


def cmd_extract_claims(args: argparse.Namespace) -> int:
    """Extract object-existence claims from responses in a manifest."""
    manifest_path = Path(args.manifest_path)
    out_path = Path(args.output_manifest) if args.output_manifest else manifest_path
    diag_path = Path(args.diagnostics_out) if args.diagnostics_out else manifest_path.parent / f"{manifest_path.stem}_extraction_diagnostics.json"

    print(f"Loading manifest from: {manifest_path}")
    manifest = load_manifest(manifest_path, validate=True)

    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)

    total_responses = 0
    total_accepted = 0
    total_rejected = 0
    diagnostics: List[Dict[str, Any]] = []

    updated_entries: List[DatasetManifestEntry] = []

    for entry in manifest.entries:
        existing_claims = list(entry.claims)
        new_atomic_claims: List[AtomicObjectExistenceClaim] = []

        for resp in entry.responses:
            total_responses += 1
            report = extractor.extract_from_response(resp)
            diagnostics.append(report.to_dict())

            total_accepted += report.num_accepted_claims
            total_rejected += report.num_rejected_mentions

            for ext_claim in report.accepted_claims:
                new_atomic_claims.append(ext_claim.to_atomic_claim())

        # Combine claims avoiding duplicate claim IDs
        combined_claims = existing_claims + [
            c for c in new_atomic_claims if c.claim_id not in {ec.claim_id for ec in existing_claims}
        ]

        updated_entries.append(
            DatasetManifestEntry(
                image=entry.image,
                claims=combined_claims,
                annotations=entry.annotations,
                responses=entry.responses,
                split=entry.split,
            )
        )

    updated_manifest = DatasetManifest(
        schema_version=manifest.schema_version,
        manifest_id=manifest.manifest_id,
        created_at=manifest.created_at,
        description=manifest.description,
        entries=updated_entries,
        metadata={
            **manifest.metadata,
            "extraction_stats": {
                "total_responses_processed": total_responses,
                "total_accepted_claims": total_accepted,
                "total_rejected_mentions": total_rejected,
            },
        },
    )

    save_manifest(updated_manifest, out_path, validate=True)
    print(f"Saved updated manifest with {total_accepted} extracted claims to: {out_path}")

    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved extraction diagnostics ({total_rejected} rejected mentions) to: {diag_path}")

    return 0


def cmd_attach_evidence(args: argparse.Namespace) -> int:
    """Attach stored fixture evidence records to claims in a manifest."""
    manifest_path = Path(args.manifest_path)
    fixture_path = Path(args.evidence_fixture)
    out_path = Path(args.output_manifest) if args.output_manifest else manifest_path
    view_id = args.view_id

    print(f"Loading manifest from: {manifest_path}")
    manifest = load_manifest(manifest_path, validate=True)

    print(f"Loading fixture evidence from: {fixture_path}")
    provider = FixtureEvidenceProvider(fixture_path=fixture_path)

    total_claims = 0
    attached_evidence_count = 0
    evidence_by_image: Dict[str, List[Dict[str, Any]]] = {}

    for entry in manifest.entries:
        img_id = entry.image.image_id
        entry_ev: List[Dict[str, Any]] = []

        for claim in entry.claims:
            total_claims += 1
            try:
                ev_rec = provider.get_evidence(img_id, claim, view_id=view_id)
                entry_ev.append(ev_rec.to_dict())
                attached_evidence_count += 1
            except KeyError as e:
                print(f"  [WARNING] {e}")

        evidence_by_image[img_id] = entry_ev
        entry.image.metadata[f"evidence_{view_id}"] = entry_ev

    save_manifest(manifest, out_path, validate=True)
    print(f"Attached {attached_evidence_count}/{total_claims} raw evidence records for view '{view_id}' -> {out_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.claims.cli",
        description="CLI utilities for object-existence claim extraction and raw evidence contracts.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # extract-claims
    p_ext = subparsers.add_parser("extract-claims", help="Extract claims from responses in manifest.")
    p_ext.add_argument("manifest_path", help="Path to manifest JSON file.")
    p_ext.add_argument("--output-manifest", default="", help="Path for output manifest (defaults to in-place).")
    p_ext.add_argument("--diagnostics-out", default="", help="Path for extraction diagnostics JSON.")

    # attach-evidence
    p_ev = subparsers.add_parser("attach-evidence", help="Attach raw fixture evidence to manifest claims.")
    p_ev.add_argument("manifest_path", help="Path to manifest JSON file.")
    p_ev.add_argument("--evidence-fixture", required=True, help="Path to fixture evidence JSON file.")
    p_ev.add_argument("--output-manifest", default="", help="Path for output manifest (defaults to in-place).")
    p_ev.add_argument("--view-id", default="original", help="View identifier (default 'original').")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "extract-claims":
        return cmd_extract_claims(args)
    elif args.command == "attach-evidence":
        return cmd_attach_evidence(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
