"""
Unit tests for duplicate and image-identity group split preservation (Correction 2).
"""

import pytest
from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    DatasetSource,
    SplitName,
)
from src.data.splits import split_manifest_by_image_groups


def test_duplicate_images_assigned_same_split():
    """
    Verify that identical or aliased images (sharing content hash)
    are placed strictly into the same split partition.
    """
    # Create entries with duplicate file hashes
    entries = [
        DatasetManifestEntry(
            image=ImageRecord(image_id="img_orig", dataset_source=DatasetSource.COCO, file_hash="hash_alpha"),
            claims=[],
            annotations=[],
            responses=[],
        ),
        DatasetManifestEntry(
            image=ImageRecord(image_id="img_dup_1", dataset_source=DatasetSource.COCO, file_hash="hash_alpha"),  # duplicate of img_orig
            claims=[],
            annotations=[],
            responses=[],
        ),
        DatasetManifestEntry(
            image=ImageRecord(image_id="img_other_1", dataset_source=DatasetSource.COCO, file_hash="hash_beta"),
            claims=[],
            annotations=[],
            responses=[],
        ),
        DatasetManifestEntry(
            image=ImageRecord(image_id="img_other_2", dataset_source=DatasetSource.COCO, file_hash="hash_gamma"),
            claims=[],
            annotations=[],
            responses=[],
        ),
    ]

    manifest = DatasetManifest(manifest_id="test_manifest", entries=entries)

    split_res = split_manifest_by_image_groups(manifest, seed=42)
    meta = split_res.metadata

    # Both img_orig and img_dup_1 must receive identical split assignment
    assert meta.assignments["img_orig"] == meta.assignments["img_dup_1"]
