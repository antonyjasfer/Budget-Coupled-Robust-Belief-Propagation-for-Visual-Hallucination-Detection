"""
Unit tests for pre-declared corruption cohort and identity group inheritance (Correction 14).
"""

import pytest
from src.data.sampling import create_predeclared_corruption_manifest


def test_corruption_variants_inherit_source_identity():
    """
    Verify that all synthesized corruption variants inherit source image identity,
    and are pre-declared without conditioning on interval width or model error.
    """
    source_ids = ["coco_001", "coco_002"]
    families = ["gaussian_noise", "jpeg_compression"]
    severities = [1, 2, 3]

    manifest = create_predeclared_corruption_manifest(
        source_image_ids=source_ids,
        corruption_families=families,
        severities=severities,
        seed=42,
    )

    # Total variants = 2 source * 2 families * 3 severities = 12
    assert manifest["total_corrupted_variants"] == 12
    assert manifest["source_images_count"] == 2
    assert len(manifest["entries"]) == 12

    # Verify source inheritance
    for entry in manifest["entries"]:
        assert entry["source_image_id"] in source_ids
        assert entry["corruption_family"] in families
        assert entry["severity"] in severities
        assert entry["derived_image_id"].startswith(entry["source_image_id"])

    # Verify cryptographic hash present
    assert "manifest_hash" in manifest
    assert len(manifest["manifest_hash"]) == 64
