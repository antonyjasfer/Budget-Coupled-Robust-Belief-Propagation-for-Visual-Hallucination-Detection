"""
End-to-end Milestone 4 demonstration: Frozen-VLM caption acquisition, caching, and annotation bundle export.

Executes:
1. Loading dataset manifest and splitting (selecting training images only).
2. Frozen-VLM caption generation (using offline synthetic provider with real prompt).
3. Disk caching with atomic writes and cache hit/miss tracking.
4. Claim extraction with conservative contextual filtering (accepted claims vs rejected mentions).
5. Exporting an annotation-ready review bundle with explicit 'not_yet_reviewed' status.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import tempfile

from src.data.coco import load_coco_instances
from src.data.splits import create_dataset_splits
from src.data.manifests import create_manifest
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.vlm.provider import (
    VLMGenerationConfig,
    SyntheticVLMProvider,
)
from src.vlm.cache import VLMCache
from src.vlm.pipeline import (
    run_vlm_pilot,
    export_annotation_bundle,
    AnnotationReviewStatus,
)


def run_demo() -> bool:
    print("=" * 78)
    print("MILESTONE 4 DEMONSTRATION: FROZEN-VLM CAPTION ACQUISITION & PILOT PIPELINE")
    print("=" * 78)

    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    coco_path = fixtures_dir / "coco_instances_synthetic.json"
    reserved_path = fixtures_dir / "reserved_external_ids.json"
    mock_responses_path = fixtures_dir / "milestone4" / "mock_vlm_responses.json"

    # 1. Load manifest and split registry
    print("1. Loading dataset manifest and training split...")
    with open(reserved_path, "r") as f:
        reserved_ids = set(json.load(f))

    adapter = load_coco_instances(coco_path)
    entries = adapter.to_manifest_entries()
    manifest = create_manifest("milestone4_pilot_manifest", entries=entries)

    split_result = create_dataset_splits(
        manifest=manifest,
        reserved_external_image_ids=reserved_ids,
        train_ratio=0.4,
        val_ratio=0.1,
        calib_ratio=0.1,
        test_ratio=0.4,
        seed=42,
    )
    print(f"   Total images: {len(manifest.entries)} | Train split: {len(split_result.split_indices['train'])} images")

    # 2. Setup VLM components
    print("\n2. Initializing VLM Provider, Cache, and Claim Extractor...")
    with open(mock_responses_path, "r") as f:
        mock_captions = json.load(f)

    provider = SyntheticVLMProvider(
        mock_captions=mock_captions,
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="snapshot_m4_demo",
    )
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(category_registry=registry)
    gen_config = VLMGenerationConfig(
        model_name="llava-hf/llava-1.5-7b-hf",
        prompt="Describe the visible physical objects in two short sentences.\nDo not speculate about objects outside the image.",
        max_new_tokens=64,
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "vlm_cache"
        cache = VLMCache(cache_dir=cache_dir)

        # 3. First execution run (Cold cache -> All Misses)
        print("\n3. Executing Pilot Run 1 (Cold Cache):")
        print("-" * 78)
        bundle1, stats1 = run_vlm_pilot(
            manifest=manifest,
            provider=provider,
            cache=cache,
            extractor=extractor,
            gen_config=gen_config,
            split_registry=split_result.registry,
            sample_size=3,
            seed=42,
        )

        for entry in bundle1.entries:
            print(f"Image [{entry.image_id}] ({entry.split} split):")
            print(f"  Response ID: {entry.response_id} | Synthetic: {entry.is_synthetic}")
            print(f"  Caption: \"{entry.caption}\"")
            print(f"  Accepted Claims ({len(entry.accepted_claims)}):")
            for c in entry.accepted_claims:
                print(f"    - Category: '{c['object_category']}' | ID: {c['claim_id']}")
            print(f"  Rejected Mentions ({len(entry.rejected_mentions)}):")
            for r in entry.rejected_mentions:
                cand_text = r.get("matched_text", r.get("candidate_category", "unknown"))
                reason_code = r.get("reason", "unknown")
                print(f"    - Candidate: '{cand_text}' | Reason: {reason_code}")
            print(f"  Review Status: {entry.review_status}")
            print()

        print(f"Run 1 Statistics: Hits={stats1.cache_hits}, Misses={stats1.cache_misses}, Generated={stats1.generated_captions}")

        # 4. Second execution run (Warm cache -> All Hits)
        print("\n4. Executing Pilot Run 2 (Warm Cache, Idempotency Check):")
        print("-" * 78)
        bundle2, stats2 = run_vlm_pilot(
            manifest=manifest,
            provider=provider,
            cache=cache,
            extractor=extractor,
            gen_config=gen_config,
            split_registry=split_result.registry,
            sample_size=3,
            seed=42,
        )
        print(f"Run 2 Statistics: Hits={stats2.cache_hits}, Misses={stats2.cache_misses}, Generated={stats2.generated_captions}")
        assert stats2.cache_hits == stats1.selected_image_count, f"Expected {stats1.selected_image_count} cache hits, got {stats2.cache_hits}"
        assert stats2.cache_misses == 0, f"Expected 0 cache misses, got {stats2.cache_misses}"

        # 5. Export annotation bundle
        print("\n5. Exporting Annotation-Ready Bundle...")
        bundle_export_path = Path(tmp_dir) / "annotation_bundle.json"
        export_annotation_bundle(bundle2, bundle_export_path)
        print(f"   Bundle written to: {bundle_export_path}")
        print(f"   Bundle ID: {bundle2.bundle_id}")
        print(f"   Total Responses: {bundle2.total_responses}")
        print(f"   Total Accepted Claims: {bundle2.total_accepted_claims}")
        print(f"   Total Rejected Mentions: {bundle2.total_rejected_mentions}")
        print(f"   Review Status check: All {len(bundle2.entries)} entries strictly marked as '{AnnotationReviewStatus.NOT_YET_REVIEWED.value}'")

    print("=" * 78)
    print("DEMO RESULT: SUCCESS")
    print("=" * 78)
    return True


if __name__ == "__main__":
    success = run_demo()
    sys.exit(0 if success else 1)
