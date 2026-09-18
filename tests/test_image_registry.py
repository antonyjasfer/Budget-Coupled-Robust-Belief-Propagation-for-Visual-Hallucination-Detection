"""
Unit tests for ImageRegistry and overlap detection.
"""

from pathlib import Path
import pytest

from src.data.image_registry import ImageRegistry, compute_file_sha256
from src.data.schemas import ImageRecord, DatasetSource, DatasetManifest, DatasetManifestEntry


def test_image_registry_grouping_by_hash():
    """Verify that images with identical content hashes are grouped together."""
    registry = ImageRegistry()

    img1 = ImageRecord(
        image_id="coco_101",
        dataset_source=DatasetSource.COCO,
        file_hash="hash_alpha_12345",
    )
    img2 = ImageRecord(
        image_id="pope_alias_101",
        dataset_source=DatasetSource.POPE,
        file_hash="hash_alpha_12345",  # Duplicate content
    )
    img3 = ImageRecord(
        image_id="coco_102",
        dataset_source=DatasetSource.COCO,
        file_hash="hash_beta_67890",
    )

    g1 = registry.register_image(img1)
    g2 = registry.register_image(img2)
    g3 = registry.register_image(img3)

    assert g1 == g2, "Images with identical file_hash must be merged into the same group"
    assert g1 != g3

    group = registry.groups[g1]
    assert group.canonical_image_ids == {"coco_101", "pope_alias_101"}
    assert group.file_hashes == {"hash_alpha_12345"}


def test_image_registry_overlap_audit():
    """Verify overlap audit between sets with and without overlap."""
    registry = ImageRegistry()

    img1 = ImageRecord(image_id="img_1", dataset_source=DatasetSource.COCO, file_hash="hash_1")
    img2 = ImageRecord(image_id="img_2", dataset_source=DatasetSource.COCO, file_hash="hash_2")
    img3 = ImageRecord(image_id="img_3", dataset_source=DatasetSource.COCO, file_hash="hash_3")
    img4 = ImageRecord(image_id="img_4_alias", dataset_source=DatasetSource.POPE, file_hash="hash_1")  # alias of img_1

    for img in [img1, img2, img3, img4]:
        registry.register_image(img)

    # Disjoint sets
    report_disjoint = registry.audit_overlap({"img_2"}, {"img_3"}, label_a="Train", label_b="Val")
    assert not report_disjoint.has_overlap

    # Overlapping via alias hash
    report_overlap = registry.audit_overlap({"img_1"}, {"img_4_alias"}, label_a="Train", label_b="Test")
    assert report_overlap.has_overlap
    assert "OVERLAP DETECTED" in report_overlap.summary
