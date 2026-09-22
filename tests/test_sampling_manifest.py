"""
Unit tests for SamplingManifest creation, serialization, and cryptographic hashing.
"""

import copy
import json
import pytest
from src.data.sampling import (
    create_sampling_manifest,
    SamplingManifest,
    CohortType,
)


def test_sampling_manifest_hash_integrity():
    """Verify SHA-256 hash calculation and sensitivity to changes."""
    universe = [{"id": f"img_{i:03d}"} for i in range(10)]
    sampled = universe[:5]
    splits = {f"img_{i:03d}": "train" for i in range(5)}
    counts = {"train": 5}

    manifest = create_sampling_manifest(
        eligible_universe=universe,
        sampled_images=sampled,
        split_assignments=splits,
        split_counts=counts,
        seed=42,
    )

    h1 = manifest.manifest_hash
    assert len(h1) == 64  # SHA-256 hex string

    # Recomputing from identical content yields identical hash
    h2 = manifest.compute_hash()
    assert h1 == h2

    # Tampering with split assignment changes hash
    tampered_data = manifest.to_dict()
    tampered_data["split_assignments"]["img_000"] = "test"
    tampered = SamplingManifest.from_dict(tampered_data)
    assert tampered.compute_hash() != h1


def test_sampling_manifest_json_roundtrip():
    """Verify dictionary / JSON serialization preserves all fields."""
    universe = [{"id": f"img_{i:03d}"} for i in range(6)]
    sampled = universe[:4]
    splits = {"img_000": "train", "img_001": "validation", "img_002": "calibration", "img_003": "test"}
    counts = {"train": 1, "validation": 1, "calibration": 1, "test": 1}

    manifest = create_sampling_manifest(
        eligible_universe=universe,
        sampled_images=sampled,
        split_assignments=splits,
        split_counts=counts,
        seed=42,
        cohort_type=CohortType.PRIMARY_REPRESENTATIVE,
    )

    d = manifest.to_dict()
    recovered = SamplingManifest.from_dict(d)

    assert recovered.manifest_id == manifest.manifest_id
    assert recovered.cohort_type == CohortType.PRIMARY_REPRESENTATIVE
    assert recovered.selected_image_ids == manifest.selected_image_ids
    assert recovered.split_assignments == manifest.split_assignments
    assert recovered.split_counts == manifest.split_counts
    assert recovered.manifest_hash == manifest.manifest_hash
