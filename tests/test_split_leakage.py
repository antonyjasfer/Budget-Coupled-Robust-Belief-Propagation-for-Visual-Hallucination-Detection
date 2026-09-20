"""
Unit tests for split leakage detection and duplicate assignment auditing.
"""

import pytest
from src.data.schemas import (
    DatasetManifestEntry,
    ImageRecord,
    DatasetSource,
    SplitName,
)
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import M7ClaimRecord
from src.annotation.leakage import (
    validate_no_image_split_overlap,
    validate_hash_split_disjointness,
    validate_claim_image_split_consistency,
    SplitLeakageError,
)


def test_validate_no_image_split_overlap_clean_sequence():
    """Valid assignments produce clean audit report."""
    assignments = [
        ("img_1", SplitName.TRAIN),
        ("img_2", SplitName.VALIDATION),
        ("img_3", SplitName.CALIBRATION),
        ("img_4", SplitName.TEST),
    ]
    report = validate_no_image_split_overlap(assignments)
    assert report.is_valid is True
    assert report.unique_images == 4
    assert len(report.duplicate_assignment_attempts) == 0


def test_validate_no_image_split_overlap_detects_duplicate_attempts():
    """A sequence with duplicate assignments (which would be silently overwritten by dict) is caught."""
    assignments = [
        ("img_1", "train"),
        ("img_2", "validation"),
        ("img_1", "train"),  # Duplicate assignment attempt
    ]
    with pytest.raises(SplitLeakageError, match="duplicate assignment attempt"):
        validate_no_image_split_overlap(assignments)


def test_validate_no_image_split_overlap_detects_cross_split_overlap():
    """Image assigned to two conflicting splits is caught."""
    assignments = [
        ("img_1", "train"),
        ("img_1", "test"),  # Cross-split leakage!
        ("img_2", "train"),
    ]
    with pytest.raises(SplitLeakageError, match="assigned to multiple splits"):
        validate_no_image_split_overlap(assignments)


def test_validate_hash_split_disjointness_catches_cross_split_hash():
    """Identical file hash in two splits is detected."""
    img1 = ImageRecord(image_id="img_1", dataset_source=DatasetSource.SYNTHETIC, file_hash="same_hash_abc")
    img2 = ImageRecord(image_id="img_2", dataset_source=DatasetSource.SYNTHETIC, file_hash="same_hash_abc")

    entries = [
        DatasetManifestEntry(image=img1, split=SplitName.TRAIN),
        DatasetManifestEntry(image=img2, split=SplitName.TEST),
    ]

    with pytest.raises(SplitLeakageError, match="Hash cross-split leakage detected"):
        validate_hash_split_disjointness(entries)


def test_validate_claim_image_split_consistency():
    """Claims whose split does not match parent image split are rejected."""
    ev = ClaimLevelEvidenceRecord(
        claim_id="c1",
        image_id="img_1",
        object_category="cat",
        is_synthetic=True,
    )
    claim = M7ClaimRecord.from_m6_evidence(ev, assigned_split="train")
    image_splits = {"img_1": "test"}

    with pytest.raises(SplitLeakageError, match="Claim/image split consistency failure"):
        validate_claim_image_split_consistency([claim], image_splits)
