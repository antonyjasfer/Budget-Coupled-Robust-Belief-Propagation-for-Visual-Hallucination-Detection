"""
Unit tests for eligible universe filtering, representative sampling, and frozen splits.
"""

import pytest
import numpy as np
from src.data.sampling import (
    filter_eligible_coco_universe,
    sample_primary_representative_cohort,
    assign_frozen_splits,
    DEFAULT_SPLIT_COUNTS,
)


def test_filter_eligible_coco_universe():
    """Verify pre-label image-level metadata filtering."""
    raw_images = [
        {"id": "img1", "width": 640, "height": 480},
        {"id": "img2", "width": 50, "height": 480},   # too small width
        {"id": "img3", "width": 640, "height": 60},    # too small height
        {"id": "img1", "width": 640, "height": 480},   # duplicate
        {"id": "", "width": 640, "height": 480},       # missing ID
        {"id": "img4", "width": 300, "height": 300},
    ]
    eligible = filter_eligible_coco_universe(raw_images, min_width=100, min_height=100)
    assert len(eligible) == 2
    assert [x["id"] for x in eligible] == ["img1", "img4"]


def test_sample_primary_representative_cohort_reproducibility():
    """Verify reproducible sampling with frozen seed."""
    universe = [{"id": f"coco_{i:06d}", "width": 640, "height": 480} for i in range(1000)]
    
    sample1 = sample_primary_representative_cohort(universe, cohort_size=600, seed=42)
    sample2 = sample_primary_representative_cohort(universe, cohort_size=600, seed=42)
    sample3 = sample_primary_representative_cohort(universe, cohort_size=600, seed=999)

    assert len(sample1) == 600
    assert len(sample2) == 600
    # Exactly identical across runs with seed=42
    assert [x["id"] for x in sample1] == [x["id"] for x in sample2]
    # Different selection with different seed
    assert [x["id"] for x in sample1] != [x["id"] for x in sample3]


def test_sample_primary_insufficient_population_raises():
    """Verify ValueError when eligible universe is smaller than requested cohort."""
    small_universe = [{"id": f"coco_{i:04d}"} for i in range(100)]
    with pytest.raises(ValueError, match="Eligible universe has only 100 images"):
        sample_primary_representative_cohort(small_universe, cohort_size=600, seed=42)


def test_assign_frozen_splits_counts():
    """Verify exact 300 / 90 / 90 / 120 split allocation."""
    sampled = [{"id": f"coco_{i:06d}"} for i in range(600)]
    assignments, counts = assign_frozen_splits(sampled, seed=42)

    assert counts["train"] == 300
    assert counts["validation"] == 90
    assert counts["calibration"] == 90
    assert counts["test"] == 120
    assert len(assignments) == 600

    # Ensure every image has exactly one valid split assignment
    assigned_splits = set(assignments.values())
    assert assigned_splits == {"train", "validation", "calibration", "test"}
