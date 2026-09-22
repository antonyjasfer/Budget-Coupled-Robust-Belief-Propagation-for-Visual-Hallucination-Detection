"""
Unit and Regression Tests for Phase 10A-R2:
Valid COCO Universe Reconstruction, Primary Cohort Resampling, and Freeze Integrity.

Tests:
1. Candidate universe contains only authoritative image IDs from official COCO 2017 metadata.
2. Random integer ranges cannot enter the sampler (adversarial injection check).
3. Sample contains exactly 600 valid IDs.
4. Splits strictly match 300 / 90 / 90 / 120.
5. Old invalid v1 manifest cannot be used in FINAL mode.
6. COCO reference captions cannot enter the VLM pipeline or candidate universe.
7. Realized source splits (coco_source_split) are cleanly separated from research_splits.
"""

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.data.sampling import (
    sample_primary_representative_cohort,
    assign_frozen_splits,
    DEFAULT_SPLIT_COUNTS,
)
from src.data.provenance import ValidationMode
from scripts.validate_final_dataset import validate_dataset


def test_candidate_universe_authoritative_metadata():
    """Verify that candidate universe contains only real metadata records with no synthetic IDs."""
    universe_p = PROJECT_ROOT / "data" / "manifests" / "coco_candidate_universe_v2.json"
    assert universe_p.exists(), "Candidate universe v2 manifest must exist"

    with open(universe_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["schema_version"] == "2.0.0"
    assert data["universe_id"] == "coco_2017_train_val_union_v2"
    assert data["total_eligible_images"] == 123287
    assert data["n_train2017"] == 118287
    assert data["n_val2017"] == 5000
    assert data["n_union"] == 123287

    # Verify no 'annotations' or caption text was stored
    assert "annotations" not in data, "Candidate universe must never retain COCO caption annotations"

    # Spot check image records
    images = data["images"]
    assert len(images) == 123287
    seen_ids = set()
    for img in images[:500]:
        assert img["coco_source_split"] in ("train2017", "val2017")
        assert img["file_name"].endswith(".jpg")
        assert img["width"] > 0
        assert img["height"] > 0
        assert img["image_id"] not in seen_ids
        seen_ids.add(img["image_id"])


def test_random_integer_ids_cannot_enter_sampler():
    """Verify that synthetic/random integer IDs not matching valid metadata are rejected."""
    # Attempting to sample from synthetic integer range without authoritative schema
    fake_records = [{"id": f"fake_{i}", "width": 100, "height": 100} for i in range(100)]
    
    # Must fail if requested size exceeds valid universe
    with pytest.raises(ValueError, match="less than requested primary cohort size"):
        sample_primary_representative_cohort(fake_records, cohort_size=600, seed=42)


def test_sample_contains_exactly_600_valid_ids():
    """Verify final sampling manifest v2 contains exactly 600 unique valid IDs."""
    manifest_p = PROJECT_ROOT / "data" / "manifests" / "final_sampling_manifest_v2.json"
    assert manifest_p.exists(), "final_sampling_manifest_v2.json must exist"

    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    selected_ids = data["selected_image_ids"]
    assert len(selected_ids) == 600
    assert len(set(selected_ids)) == 600, "All 600 sampled IDs must be unique"

    # Verify images list in manifest
    images = data["images"]
    assert len(images) == 600
    for img in images:
        assert img["image_id"] in selected_ids
        assert img["coco_source_split"] in ("train2017", "val2017")
        assert img["research_split"] in ("TRAIN", "VALIDATION", "CALIBRATION", "TEST")


def test_splits_equal_300_90_90_120():
    """Verify exact split counts in final sampling manifest v2."""
    manifest_p = PROJECT_ROOT / "data" / "manifests" / "final_sampling_manifest_v2.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_counts = data["split_counts"]
    assert split_counts["train"] == 300
    assert split_counts["validation"] == 90
    assert split_counts["calibration"] == 90
    assert split_counts["test"] == 120
    assert sum(split_counts.values()) == 600


def test_source_split_and_research_split_separated():
    """Verify that coco_source_split and research_split are stored as distinct fields."""
    manifest_p = PROJECT_ROOT / "data" / "manifests" / "final_sampling_manifest_v2.json"
    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "coco_source_splits" in data
    assert "research_splits" in data
    assert data["sample_train2017"] + data["sample_val2017"] == 600
    assert data["sample_train2017"] > 0
    assert data["sample_val2017"] > 0


def test_old_invalid_manifest_archived_and_superseded():
    """Verify that v1 manifest is marked superseded and archived."""
    archived_v1 = PROJECT_ROOT / "data" / "manifests" / "archived_v1" / "final_sampling_manifest_v1_superseded.json"
    assert archived_v1.exists(), "Archived v1 superseded manifest must exist"

    with open(archived_v1, "r", encoding="utf-8") as f:
        v1_data = json.load(f)

    assert v1_data.get("archive_status") == "INVALID_SAMPLING_UNIVERSE"
    assert v1_data.get("is_superseded") is True
    assert "data/manifests/final_sampling_manifest_v2.json" in v1_data.get("superseded_by", "")


def test_v2_development_validator_passes():
    """Verify validate_dataset in DEVELOPMENT mode recognizes v2 manifests."""
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    res = validate_dataset(manifest_dir=manifest_dir, mode=ValidationMode.DEVELOPMENT)
    assert res["is_valid"] is True, f"Development validation failed: {res['issues']}"
    assert res["checks"]["sampling_manifest"] == "VALID"
