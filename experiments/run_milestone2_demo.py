"""
End-to-end Milestone 2 fixture demonstration.

Executes:
1. Manifest construction from synthetic COCO instances and POPE benchmark queries.
2. Referential integrity validation.
3. Deterministic 50/15/15/20 split generation with external evaluation reservation.
4. Cross-split overlap and leakage auditing.
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json

from src.data.coco import COCOMetadataAdapter
from src.data.pope import POPEAdapter
from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    AtomicObjectExistenceClaim,
    AnnotationRecord,
    GroundTruthStatus,
    AnnotationSource,
    SplitName,
)
from src.data.manifests import create_manifest, validate_manifest, save_manifest, load_manifest
from src.data.splits import split_manifest_by_image_groups
from src.data.image_registry import ImageRegistry


def run_demo() -> bool:
    print("=" * 78)
    print("MILESTONE 2 FIXTURE DEMONSTRATION")
    print("=" * 78)

    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    coco_path = fixtures_dir / "coco_instances_synthetic.json"
    pope_path = fixtures_dir / "pope_synthetic.jsonl"
    reserved_path = fixtures_dir / "reserved_external_ids.json"

    # 1. Load COCO and POPE adapters
    print(f"1. Loading COCO adapter from: {coco_path.name}")
    coco_adapter = COCOMetadataAdapter(coco_path)
    print(f"   Loaded {len(coco_adapter.images)} images, {len(coco_adapter.categories)} categories.")

    print(f"2. Loading POPE adapter from: {pope_path.name}")
    pope_adapter = POPEAdapter(pope_path)
    print(f"   Loaded {len(pope_adapter.records)} probe questions across {len(pope_adapter.records_by_image)} images.")

    with open(reserved_path, "r", encoding="utf-8") as f:
        reserved_ids = json.load(f)
    print(f"3. Loaded {len(reserved_ids)} reserved external image IDs: {reserved_ids}")

    # 2. Construct manifest entries
    entries = []
    for image_id, img_rec in sorted(coco_adapter.images.items()):
        claims = []
        annotations = []

        # Convert POPE questions into benchmark reference claims for evaluation
        pope_records = pope_adapter.get_records_for_image(image_id)
        for p_idx, p_rec in enumerate(pope_records):
            c_id = f"claim_{image_id}_{p_rec.category_name}_{p_idx}"
            claim = AtomicObjectExistenceClaim(
                claim_id=c_id,
                image_id=image_id,
                object_category=p_rec.category_name,
                raw_claim_text=p_rec.question_text,
                provenance={"pope_question_id": p_rec.question_id, "variant": p_rec.variant},
            )
            claims.append(claim)

            # Map POPE benchmark label to benchmark reference annotation
            gt = GroundTruthStatus.SUPPORTED if p_rec.benchmark_label == "yes" else GroundTruthStatus.HALLUCINATED
            ann = AnnotationRecord(
                annotation_id=f"ann_{c_id}",
                claim_id=c_id,
                ground_truth=gt,
                source=AnnotationSource.BENCHMARK_REFERENCE,
                annotator_notes=f"POPE reference label: {p_rec.benchmark_label}",
            )
            annotations.append(ann)

        entries.append(DatasetManifestEntry(image=img_rec, claims=claims, annotations=annotations))

    manifest = create_manifest(
        manifest_id="synthetic_demo_manifest",
        description="Synthetic offline demonstration manifest for visual hallucination detection.",
        entries=entries,
    )

    # 3. Validate manifest
    errors = validate_manifest(manifest)
    print(f"4. Manifest validation: {'PASSED (0 errors)' if not errors else f'FAILED ({len(errors)} errors)'}")
    if errors:
        for e in errors:
            print(f"   - {e}")
        return False

    # 4. Generate splits
    split_res = split_manifest_by_image_groups(
        manifest,
        seed=42,
        proportions={SplitName.TRAIN: 0.50, SplitName.VALIDATION: 0.15, SplitName.CALIBRATION: 0.15, SplitName.TEST: 0.20},
        reserved_external_ids=reserved_ids,
    )

    print("-" * 78)
    print("SPLIT GENERATION SUMMARY:")
    print(f"  Total Images:        {split_res.metadata.total_images}")
    print(f"  Total Groups:        {split_res.metadata.total_groups}")
    print(f"  Reserved External:   {split_res.metadata.reserved_external_count} images")
    print(f"  Input Manifest Hash: {split_res.metadata.input_manifest_hash[:16]}...")
    print("\n  Achieved Counts & Proportions:")
    for s_name in [SplitName.TRAIN, SplitName.VALIDATION, SplitName.CALIBRATION, SplitName.TEST]:
        cnt = split_res.metadata.achieved_counts[s_name.value]
        prop = split_res.metadata.achieved_proportions[s_name.value]
        print(f"    - {s_name.value:<12} : {cnt:>2} images ({prop*100:>5.1f}%)")

    # 5. Overlap Audit
    print("-" * 78)
    print("CROSS-SPLIT OVERLAP AUDIT:")
    registry = ImageRegistry()
    registry.register_manifest(manifest)

    train_ids = {e.image.image_id for e in split_res.manifests_by_split[SplitName.TRAIN].entries}
    val_ids = {e.image.image_id for e in split_res.manifests_by_split[SplitName.VALIDATION].entries}
    calib_ids = {e.image.image_id for e in split_res.manifests_by_split[SplitName.CALIBRATION].entries}
    test_ids = {e.image.image_id for e in split_res.manifests_by_split[SplitName.TEST].entries}

    r_tv = registry.audit_overlap(train_ids, val_ids, "Train", "Val")
    r_tc = registry.audit_overlap(train_ids, calib_ids, "Train", "Calib")
    r_tt = registry.audit_overlap(train_ids, test_ids, "Train", "Test")
    r_vt = registry.audit_overlap(val_ids, test_ids, "Val", "Test")

    all_disjoint = not (r_tv.has_overlap or r_tc.has_overlap or r_tt.has_overlap or r_vt.has_overlap)
    print(f"  Train vs Val   : {'DISJOINT (0 overlap)' if not r_tv.has_overlap else 'OVERLAP DETECTED'}")
    print(f"  Train vs Calib : {'DISJOINT (0 overlap)' if not r_tc.has_overlap else 'OVERLAP DETECTED'}")
    print(f"  Train vs Test  : {'DISJOINT (0 overlap)' if not r_tt.has_overlap else 'OVERLAP DETECTED'}")
    print(f"  Val vs Test    : {'DISJOINT (0 overlap)' if not r_vt.has_overlap else 'OVERLAP DETECTED'}")

    # Check reserved external containment
    for r_id in reserved_ids:
        assert r_id in test_ids and r_id not in train_ids, f"Reserved image {r_id} leaked!"

    print(f"  Reserved ID Check: ALL {len(reserved_ids)} RESERVED IMAGES EXCLUSIVELY IN TEST SPLIT")
    print("=" * 78)
    print(f"DEMO RESULT: {'SUCCESS' if all_disjoint else 'FAILURE'}")
    print("=" * 78)
    return all_disjoint


if __name__ == "__main__":
    success = run_demo()
    if not success:
        sys.exit(1)
