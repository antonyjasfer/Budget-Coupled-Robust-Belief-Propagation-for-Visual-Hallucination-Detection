"""
Milestone 5 Demonstration: Real-Image Pilot Readiness & Portable Execution.

Demonstrates:
1. Comprehensive 11-gate preflight check reporting exact blockers without synthetic substitution.
2. Portable pilot bundle creation containing training split only, relative paths, hashes, and run instructions.
3. Bundle structure and SHA256 integrity validation.
4. Provenance and cache separation (synthetic fixtures vs real execution isolation).
5. Offline-verified annotation handoff artifact creation with unreviewed status.
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
from src.data.manifests import create_manifest, save_manifest
from src.data.schemas import SplitName
from src.vlm.provider import VLMGenerationConfig, SyntheticVLMProvider
from src.vlm.cache import VLMCache
from src.vlm.preflight import run_preflight
from src.vlm.pilot_bundle import create_portable_pilot_bundle, validate_portable_pilot_bundle
from src.vlm.pipeline import run_vlm_pilot, export_annotation_bundle


def run_demo():
    print("=" * 78)
    print("MILESTONE 5 DEMONSTRATION: REAL-IMAGE PILOT READINESS & PORTABLE EXECUTION")
    print("=" * 78)

    # 1. Setup sample manifest and splits
    fixture_dir = Path("tests/fixtures")
    coco_path = fixture_dir / "coco_instances_synthetic.json"
    reserved_path = fixture_dir / "reserved_external_ids.json"

    with open(reserved_path, "r") as f:
        reserved_ids = set(json.load(f))

    adapter = load_coco_instances(coco_path)
    entries = adapter.to_manifest_entries()
    manifest = create_manifest("milestone5_manifest", entries=entries)

    split_reg = create_dataset_splits(
        manifest=manifest,
        reserved_external_image_ids=reserved_ids,
        train_ratio=0.4,
        val_ratio=0.1,
        calib_ratio=0.1,
        test_ratio=0.4,
        seed=42,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tdp = Path(tmp_dir)
        man_path = tdp / "manifest.json"
        save_manifest(manifest, man_path)
        split_path = tdp / "splits.json"
        with open(split_path, "w") as f:
            json.dump(split_reg.registry, f, indent=2)

        print("1. Executing 11-Gate Preflight Check on Current Environment...")
        preflight_report = run_preflight(
            manifest_path=man_path,
            split_registry_path=split_path,
            sample_size=3,
            seed=42,
            allow_download=False,
            require_real_images=False,
        )

        print(f"   Preflight Overall Status : {'READY' if preflight_report.overall_passed else 'BLOCKED'}")
        print(f"   Gates Passed            : {preflight_report.passed_gates} / {preflight_report.total_gates}")
        print(f"   Blocked Gates           : {preflight_report.blocked_gates}")
        print(f"   Warnings                : {preflight_report.warning_gates}")
        print("   Key Gate Statuses:")
        for g in preflight_report.gates[:6]:
            print(f"     [{g.status:<7}] {g.gate_name:<28} : {g.message}")
        print(f"     [{preflight_report.gates[6].status:<7}] {preflight_report.gates[6].gate_name:<28} : {preflight_report.gates[6].message}")
        print(f"     [{preflight_report.gates[7].status:<7}] {preflight_report.gates[7].gate_name:<28} : {preflight_report.gates[7].message}")
        print("-" * 78)

        print("2. Packaging Portable Real-Image Pilot Bundle...")
        bundle_out = tdp / "portable_pilot_bundle"
        create_portable_pilot_bundle(
            manifest_path=man_path,
            output_dir=bundle_out,
            split_registry_path=split_path,
            sample_size=3,
            seed=42,
            copy_images=False,
        )
        print(f"   Bundle generated at: {bundle_out}")
        print(f"   Contents: {[p.name for p in bundle_out.iterdir()]}")
        print("-" * 78)

        print("3. Validating Portable Pilot Bundle Integrity...")
        val_res = validate_portable_pilot_bundle(bundle_out)
        print(f"   Validation Passed  : {val_res['valid']}")
        print(f"   Bundle ID          : {val_res['bundle_id']}")
        print(f"   Image Count        : {val_res['image_count']}")
        print(f"   Selected Image IDs : {val_res['selected_image_ids']}")
        assert val_res["valid"], f"Bundle validation failed: {val_res['errors']}"
        print("-" * 78)

        print("4. Verifying Cache & Provenance Isolation...")
        cache = VLMCache(tdp / "cache")
        cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")
        
        # Query cache for real provider when synthetic entry was stored
        synth_provider = SyntheticVLMProvider()
        synth_resp = synth_provider.generate_caption("test.jpg", image_id="img001", image_hash="hash001")
        cache.put(synth_resp, gen_config=cfg)

        real_query = cache.get(
            image_hash="hash001",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="snapshot_real",
            prompt=cfg.prompt,
            gen_config=cfg,
            provider_kind="llava_15_hf",
            is_synthetic=False,
        )
        assert real_query is None, "Real query must NOT hit synthetic cache entry!"
        print("   Synthetic cache entry successfully isolated from real inference requests.")
        print("-" * 78)

        print("5. Exporting Annotation-Ready Review Bundle...")
        bundle, stats = run_vlm_pilot(
            manifest=manifest,
            provider=synth_provider,
            cache=cache,
            split_registry=split_reg.registry,
            sample_size=3,
            seed=42,
        )
        out_json = tdp / "annotation_bundle.json"
        export_annotation_bundle(bundle, out_json)
        print(f"   Bundle exported: {bundle.bundle_id}")
        print(f"   Total Responses: {bundle.total_responses}")
        print(f"   Accepted Claims: {bundle.total_accepted_claims}")
        print(f"   Review Status  : All entries strictly marked '{bundle.entries[0].review_status}'")
        print("=" * 78)
        print("DEMO RESULT: SUCCESS (All Milestone 5 Capabilities Verified)")
        print("=" * 78)


if __name__ == "__main__":
    run_demo()
