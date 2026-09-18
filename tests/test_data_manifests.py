"""
Unit tests for manifest validation and referential integrity.
"""

from pathlib import Path
import pytest

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    AtomicObjectExistenceClaim,
    AnnotationRecord,
    GroundTruthStatus,
    AnnotationSource,
    DatasetSource,
)
from src.data.manifests import create_manifest, validate_manifest, save_manifest, load_manifest


def test_manifest_validation_catches_integrity_errors():
    """Verify validation detects duplicate IDs, missing references, and orphaned annotations."""
    # 1. Valid manifest
    img1 = ImageRecord(image_id="img_1", dataset_source=DatasetSource.COCO)
    claim1 = AtomicObjectExistenceClaim(claim_id="c_1", image_id="img_1", object_category="cat")
    ann1 = AnnotationRecord(annotation_id="a_1", claim_id="c_1", ground_truth=GroundTruthStatus.SUPPORTED, source=AnnotationSource.HUMAN_ADJUDICATED)

    entry1 = DatasetManifestEntry(image=img1, claims=[claim1], annotations=[ann1])
    manifest = create_manifest(manifest_id="m_valid", entries=[entry1])

    errors = validate_manifest(manifest)
    assert len(errors) == 0

    # 2. Duplicate image_id
    img2 = ImageRecord(image_id="img_1", dataset_source=DatasetSource.COCO)  # Duplicate
    entry2 = DatasetManifestEntry(image=img2, claims=[], annotations=[])
    manifest_dup = create_manifest(manifest_id="m_dup", entries=[entry1, entry2])
    errs_dup = validate_manifest(manifest_dup)
    assert any("Duplicate image_id" in e for e in errs_dup)

    # 3. Claim references different image_id
    claim_bad_img = AtomicObjectExistenceClaim(claim_id="c_2", image_id="other_img", object_category="dog")
    entry_bad_claim = DatasetManifestEntry(image=img1, claims=[claim_bad_img])
    manifest_bad_claim = create_manifest(manifest_id="m_bad", entries=[entry_bad_claim])
    errs_bad_claim = validate_manifest(manifest_bad_claim)
    assert any("Referential integrity failure: claim" in e for e in errs_bad_claim)

    # 4. Annotation references non-existent claim
    ann_orphaned = AnnotationRecord(annotation_id="a_2", claim_id="non_existent_c", ground_truth=GroundTruthStatus.HALLUCINATED, source=AnnotationSource.HUMAN_ADJUDICATED)
    entry_bad_ann = DatasetManifestEntry(image=img1, claims=[claim1], annotations=[ann_orphaned])
    manifest_bad_ann = create_manifest(manifest_id="m_ann", entries=[entry_bad_ann])
    errs_bad_ann = validate_manifest(manifest_bad_ann)
    assert any("Referential integrity failure: annotation" in e for e in errs_bad_ann)


def test_manifest_save_load_file(tmp_path):
    """Verify save and load round-trip to disk with validation."""
    img = ImageRecord(image_id="img_disk", dataset_source=DatasetSource.COCO)
    claim = AtomicObjectExistenceClaim(claim_id="c_disk", image_id="img_disk", object_category="table")
    entry = DatasetManifestEntry(image=img, claims=[claim])
    manifest = create_manifest(manifest_id="m_disk", entries=[entry])

    file_path = tmp_path / "test_manifest.json"
    save_manifest(manifest, file_path)

    loaded = load_manifest(file_path, validate=True)
    assert loaded.manifest_id == "m_disk"
    assert len(loaded.entries) == 1
    assert loaded.entries[0].claims[0].object_category == "table"
