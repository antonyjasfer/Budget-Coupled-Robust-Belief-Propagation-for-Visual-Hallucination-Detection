"""Tests for Milestone 8 Data Loaders and Split Isolation."""

import pytest
from pathlib import Path

from src.experiments.loaders import (
    load_m8_dataset,
    create_synthetic_smoke_bundle,
    validate_m8_dataset_integrity,
    M8DatasetBundle,
)
from src.annotation.schemas import M7ClaimRecord
from src.data.schemas import GroundTruthStatus, SplitName


def test_synthetic_smoke_bundle_creation():
    bundle = create_synthetic_smoke_bundle(num_images=4, seed=42)
    assert len(bundle.images) == 4
    assert len(bundle.claims) >= 4
    assert bundle.is_synthetic is True

    # Check that UNKNOWN status exists and is handled
    unknowns = [
        c for c in bundle.claims
        if bundle.get_ground_truth_for_claim(c.claim_id) == GroundTruthStatus.UNKNOWN
    ]
    assert len(unknowns) > 0

    # Validate integrity passes
    validate_m8_dataset_integrity(bundle)


def test_split_leakage_detection():
    bundle = create_synthetic_smoke_bundle(num_images=3, seed=42)
    
    # Intentionally corrupt one claim split to create mismatch with manifest
    claim_corrupt = bundle.claims[0]
    claim_corrupt.split = "test"
    # Find matching manifest entry and set to train
    for entry in bundle.manifest.entries:
        if entry.image.image_id == claim_corrupt.image_id:
            entry.split = SplitName.TRAIN
            break

    with pytest.raises(ValueError, match="split mismatch"):
        validate_m8_dataset_integrity(bundle)


def test_duplicate_claim_id_detection():
    bundle = create_synthetic_smoke_bundle(num_images=2, seed=42)
    
    # Duplicate a claim ID
    bundle.claims[1].claim_id = bundle.claims[0].claim_id

    with pytest.raises(ValueError, match="Duplicate claim_id"):
        validate_m8_dataset_integrity(bundle)


def test_load_real_m7_dataset_subset():
    evidence_path = Path("data/m7_evidence/m7_verified_evidence.jsonl")
    gt_path = Path("data/annotations/claim_annotations.jsonl")
    manifest_path = Path("data/manifests/m7_image_manifest.json")

    if evidence_path.exists() and manifest_path.exists():
        bundle = load_m8_dataset(
            evidence_path=evidence_path,
            ground_truth_path=gt_path,
            manifest_path=manifest_path,
            check_integrity=True,
        )
        assert len(bundle.images) == 10
        assert len(bundle.claims) == 15
        assert bundle.is_synthetic is False
