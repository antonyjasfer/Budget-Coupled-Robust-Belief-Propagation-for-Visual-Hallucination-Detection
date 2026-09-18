"""
Unit tests for deterministic, leakage-safe dataset splitting.
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
    SplitName,
)
from src.data.splits import split_manifest_by_image_groups


def _make_dummy_manifest(num_images: int = 20) -> DatasetManifest:
    entries = []
    for i in range(1, num_images + 1):
        img_id = f"coco_{100 + i}"
        img = ImageRecord(
            image_id=img_id,
            dataset_source=DatasetSource.COCO,
            file_name=f"{100 + i}.jpg",
            file_hash=f"hash_{100 + i}",
        )
        claim = AtomicObjectExistenceClaim(
            claim_id=f"claim_{img_id}_dog",
            image_id=img_id,
            object_category="dog",
        )
        ann = AnnotationRecord(
            annotation_id=f"ann_{img_id}_dog",
            claim_id=f"claim_{img_id}_dog",
            ground_truth=GroundTruthStatus.SUPPORTED if i % 2 == 0 else GroundTruthStatus.HALLUCINATED,
            source=AnnotationSource.HUMAN_ADJUDICATED,
        )
        entries.append(DatasetManifestEntry(image=img, claims=[claim], annotations=[ann]))

    return DatasetManifest(
        manifest_id="dummy_manifest",
        entries=entries,
    )


def test_split_determinism():
    """Identical seed produces identical split assignments."""
    manifest = _make_dummy_manifest(20)

    res1 = split_manifest_by_image_groups(manifest, seed=42)
    res2 = split_manifest_by_image_groups(manifest, seed=42)
    res3 = split_manifest_by_image_groups(manifest, seed=99)

    assert res1.metadata.assignments == res2.metadata.assignments
    assert res1.metadata.input_manifest_hash == res2.metadata.input_manifest_hash
    # Different seed gives different assignments
    assert res1.metadata.assignments != res3.metadata.assignments


def test_split_disjointness_no_leakage():
    """Verify that all splits are mutually exclusive and collectively exhaustive."""
    manifest = _make_dummy_manifest(20)
    res = split_manifest_by_image_groups(manifest, seed=123)

    train_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.TRAIN].entries}
    val_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.VALIDATION].entries}
    calib_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.CALIBRATION].entries}
    test_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.TEST].entries}

    # Mutual exclusivity
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(calib_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(calib_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0
    assert len(calib_ids.intersection(test_ids)) == 0

    # Collective exhaustiveness
    all_assigned = train_ids.union(val_ids, calib_ids, test_ids)
    assert len(all_assigned) == 20


def test_split_respects_reserved_external_ids():
    """Reserved external images must NEVER enter train, validation, or calibration splits."""
    manifest = _make_dummy_manifest(20)
    reserved = ["coco_101", "coco_102", "coco_103"]

    res = split_manifest_by_image_groups(manifest, seed=42, reserved_external_ids=reserved)

    train_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.TRAIN].entries}
    val_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.VALIDATION].entries}
    calib_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.CALIBRATION].entries}
    test_ids = {e.image.image_id for e in res.manifests_by_split[SplitName.TEST].entries}

    for r_id in reserved:
        assert r_id not in train_ids, f"Reserved image {r_id} leaked into TRAIN"
        assert r_id not in val_ids, f"Reserved image {r_id} leaked into VALIDATION"
        assert r_id not in calib_ids, f"Reserved image {r_id} leaked into CALIBRATION"
        assert r_id in test_ids, f"Reserved image {r_id} must be in TEST"


def test_split_rejects_conflicting_reservations():
    """Forced assignment into train conflicting with reserved external evaluation must raise ValueError."""
    manifest = _make_dummy_manifest(10)
    with pytest.raises(ValueError, match="Conflicting reservation"):
        split_manifest_by_image_groups(
            manifest,
            reserved_external_ids=["coco_101"],
            forced_assignments={"coco_101": SplitName.TRAIN},
        )
