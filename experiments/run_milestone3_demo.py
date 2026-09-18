"""
End-to-end Milestone 3 fixture demonstration.

Executes:
1. Loading synthetic VLM generated responses from manifest.
2. Conservative object-existence claim extraction and contextual filtering.
3. Diagnostic reporting of accepted claims and rejected mentions with reason codes.
4. Loading and attaching raw visual evidence fixtures (detector scores and cosine similarities).
5. Referential integrity validation.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    GeneratedResponseRecord,
    DatasetSource,
    SplitName,
)
from src.data.manifests import create_manifest, validate_manifest
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.evidence.fixture_provider import FixtureEvidenceProvider


def run_demo() -> bool:
    print("=" * 78)
    print("MILESTONE 3 FIXTURE DEMONSTRATION")
    print("=" * 78)

    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    m3_fixtures_dir = fixtures_dir / "milestone3"
    evidence_path = m3_fixtures_dir / "fixture_evidence.json"

    # 1. Prepare sample synthetic responses
    sample_responses = [
        GeneratedResponseRecord(
            response_id="resp_101",
            image_id="coco_101",
            model_name="llava-1.5-7b",
            response_text="The photo displays a golden retriever dog and a red car parked nearby. There is no cat in the driveway.",
        ),
        GeneratedResponseRecord(
            response_id="resp_102",
            image_id="coco_102",
            model_name="instructblip-7b",
            response_text="A cute cat is resting on the couch. There might be a dog hiding in the shadows.",
        ),
    ]

    entries = [
        DatasetManifestEntry(
            image=ImageRecord(image_id="coco_101", dataset_source=DatasetSource.COCO, file_name="000000000101.jpg"),
            responses=[sample_responses[0]],
            split=SplitName.TRAIN,
        ),
        DatasetManifestEntry(
            image=ImageRecord(image_id="coco_102", dataset_source=DatasetSource.COCO, file_name="000000000102.jpg"),
            responses=[sample_responses[1]],
            split=SplitName.VALIDATION,
        ),
    ]

    manifest = create_manifest(manifest_id="milestone3_demo_manifest", entries=entries)

    # 2. Extract object-existence claims
    print("1. Initializing CategoryRegistry and ConservativeClaimExtractor...")
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)

    total_accepted = 0
    total_rejected = 0

    print("\n2. Processing VLM responses:")
    print("-" * 78)
    for entry in manifest.entries:
        for resp in entry.responses:
            print(f"Response [{resp.response_id}] for Image [{resp.image_id}]:")
            print(f"  Text: \"{resp.response_text}\"")

            report = extractor.extract_from_response(resp)
            print(f"  Accepted Claims ({report.num_accepted_claims}):")
            for c in report.accepted_claims:
                atomic = c.to_atomic_claim()
                entry.claims.append(atomic)
                total_accepted += 1
                print(f"    - Object: '{c.object_category}' | Spans: {len(c.spans)} | ID: {c.claim_id}")

            print(f"  Rejected Mentions ({report.num_rejected_mentions}):")
            for r in report.rejected_mentions:
                total_rejected += 1
                print(f"    - Candidate: '{r.candidate_category}' | Reason: {r.reason.value:<15} | Context: \"{r.mention_span.matched_text}\"")
            print("-" * 78)

    # 3. Attach Raw Evidence Fixtures
    print("\n3. Attaching Raw Evidence Fixtures from:", evidence_path.name)
    provider = FixtureEvidenceProvider(fixture_path=evidence_path)

    for entry in manifest.entries:
        img_id = entry.image.image_id
        print(f"\nImage [{img_id}] Evidence Summary:")
        for claim in entry.claims:
            ev_orig = provider.get_evidence(img_id, claim, view_id="original")
            det_str = f"{ev_orig.detector_score:.4f}" if ev_orig.detector_available else "UNAVAILABLE (None)"
            sim_str = f"{ev_orig.similarity_score:.4f}" if ev_orig.similarity_available else "UNAVAILABLE (None)"
            print(f"  Claim [{claim.claim_id}] (Category: '{claim.object_category}'):")
            print(f"    Detector Score  d_i : {det_str} (available: {ev_orig.detector_available})")
            print(f"    Cosine Sim      g_i : {sim_str} (available: {ev_orig.similarity_available})")
            print(f"    View ID             : {ev_orig.view_id} | Synthetic: {ev_orig.is_synthetic}")

    # 4. Referential Integrity Validation
    errors = validate_manifest(manifest)
    print("\n4. Final Manifest Referential Integrity Check:", "PASSED" if not errors else "FAILED")
    print("=" * 78)
    print(f"DEMO RESULT: {'SUCCESS' if not errors else 'FAILURE'}")
    print(f"Summary: {total_accepted} claims accepted, {total_rejected} mentions rejected across {len(manifest.entries)} images.")
    print("=" * 78)

    return not bool(errors)


if __name__ == "__main__":
    success = run_demo()
    if not success:
        sys.exit(1)
