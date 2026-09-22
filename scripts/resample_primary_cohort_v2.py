"""
Phase 10A-R2-3 & 10A-R2-4:
Resample 600-Image Primary Representative Cohort and Recreate 300/90/90/120 Splits from Candidate Universe V2.

Strict Compliance Rules (Phase 10A-R2 Approval Corrections 3 & 4):
1. Samples uniformly without replacement from the union of official MS COCO 2017 train and validation records.
2. Distinctly separates:
   - coco_source_split: "train2017" / "val2017"
   - research_split: "TRAIN" / "VALIDATION" / "CALIBRATION" / "TEST"
3. Records realized source-split counts: sample_train2017, sample_val2017.
4. Generates:
   - data/manifests/final_sampling_manifest_v2.json
   - data/manifests/final_corruption_manifest_v2.json
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sampling import (
    sample_primary_representative_cohort,
    assign_frozen_splits,
    create_predeclared_corruption_manifest,
    DEFAULT_SPLIT_COUNTS,
)

def main():
    universe_p = PROJECT_ROOT / "data" / "manifests" / "coco_candidate_universe_v2.json"
    out_dir = PROJECT_ROOT / "data" / "manifests"
    out_sampling_p = out_dir / "final_sampling_manifest_v2.json"
    out_corruption_p = out_dir / "final_corruption_manifest_v2.json"

    print("Phase 10A-R2: Loading Candidate Universe V2...")
    with open(universe_p, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    universe_hash = universe_data.get("universe_hash")
    all_images = universe_data.get("images", [])
    n_universe = len(all_images)
    print(f"Candidate Universe Size: {n_universe} images, Hash: {universe_hash}")
    assert n_universe == 123287, f"Expected 123,287 images in universe, got {n_universe}"

    # Sample exactly 600 images using seed 42
    seed = 42
    cohort_size = 600
    print(f"Sampling exactly {cohort_size} primary representative images (seed={seed})...")
    sampled_images = sample_primary_representative_cohort(all_images, cohort_size=cohort_size, seed=seed)
    assert len(sampled_images) == cohort_size, f"Expected {cohort_size} sampled images, got {len(sampled_images)}"

    # Assign frozen splits: 300 Train, 90 Val, 90 Cal, 120 Test
    print("Assigning frozen splits: 300 Train, 90 Val, 90 Cal, 120 Test (seed=42)...")
    split_assignments, achieved_counts = assign_frozen_splits(
        sampled_images,
        split_counts=DEFAULT_SPLIT_COUNTS,
        seed=seed,
    )

    # Validate split counts
    assert achieved_counts["train"] == 300
    assert achieved_counts["validation"] == 90
    assert achieved_counts["calibration"] == 90
    assert achieved_counts["test"] == 120

    # Build separated split dictionaries
    research_splits = {}
    coco_source_splits = {}
    source_counts = Counter()

    # Create image lookup from sampled images
    img_meta_map = {img["image_id"]: img for img in sampled_images}
    selected_ids = sorted(list(split_assignments.keys()))

    for img_id in selected_ids:
        r_split = split_assignments[img_id]
        research_splits[img_id] = r_split.upper()  # TRAIN, VALIDATION, CALIBRATION, TEST
        c_split = img_meta_map[img_id]["coco_source_split"]
        coco_source_splits[img_id] = c_split
        source_counts[c_split] += 1

    sample_train2017 = source_counts["train2017"]
    sample_val2017 = source_counts["val2017"]

    print(f"Realized Source Split Distribution: train2017={sample_train2017}, val2017={sample_val2017}")

    # Build full image records for the manifest
    detailed_images = []
    for img_id in selected_ids:
        meta = dict(img_meta_map[img_id])
        meta["research_split"] = research_splits[img_id]
        meta["split"] = split_assignments[img_id]  # legacy field
        detailed_images.append(meta)

    # Build SamplingManifest v2
    sampling_manifest = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "acquisition_version": "10A-R2",
        "manifest_id": "coco_600_primary_sampling_manifest_v2",
        "cohort_type": "primary_representative",
        "candidate_population_size": n_universe,
        "candidate_universe_hash": universe_hash,
        "selected_image_ids": selected_ids,
        "sampling_rule": (
            "600 images were sampled uniformly without replacement from the union of the "
            "official COCO 2017 train and validation image records."
        ),
        "seed": seed,
        "code_sha": "6510402d106eb1b8e0716d7d648a3a6fa1b03896",
        "split_assignments": split_assignments,  # legacy compatibility
        "split_counts": achieved_counts,
        "research_splits": research_splits,
        "coco_source_splits": coco_source_splits,
        "sample_train2017": sample_train2017,
        "sample_val2017": sample_val2017,
        "image_identity_groups": {img_id: [img_id] for img_id in selected_ids},
        "images": detailed_images,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Compute manifest hash
    canonical_sampling_json = json.dumps(sampling_manifest, sort_keys=True, separators=(",", ":"))
    sampling_manifest_hash = hashlib.sha256(canonical_sampling_json.encode("utf-8")).hexdigest()
    sampling_manifest["manifest_hash"] = sampling_manifest_hash

    with open(out_sampling_p, "w", encoding="utf-8") as f:
        json.dump(sampling_manifest, f, indent=2)

    print(f"Generated: {out_sampling_p}")
    print(f"Sampling Manifest V2 Hash: {sampling_manifest_hash}")

    # Build Corruption Manifest V2
    print("Generating Corruption Manifest V2 for 600 Primary Cohort Images...")
    corruption_manifest = create_predeclared_corruption_manifest(
        source_image_ids=selected_ids,
        corruption_families=["gaussian_noise", "gaussian_blur", "jpeg_compression", "contrast_reduction"],
        severities=[1, 2, 3, 4, 5],
        seed=seed,
    )
    corruption_manifest["schema_version"] = "2.0.0"
    corruption_manifest["dataset_version"] = "v2"
    corruption_manifest["sampling_manifest_v2_hash"] = sampling_manifest_hash
    # recompute hash with updated metadata
    corruption_manifest.pop("manifest_hash", None)
    c_hash = hashlib.sha256(
        json.dumps(corruption_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    corruption_manifest["manifest_hash"] = c_hash

    with open(out_corruption_p, "w", encoding="utf-8") as f:
        json.dump(corruption_manifest, f, indent=2)

    print(f"Generated: {out_corruption_p}")
    print(f"Corruption Manifest V2 Hash: {c_hash}")

if __name__ == "__main__":
    main()
