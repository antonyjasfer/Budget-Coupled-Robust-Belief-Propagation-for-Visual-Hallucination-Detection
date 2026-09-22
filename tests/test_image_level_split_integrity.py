"""Regression tests verifying image-level split integrity and zero cross-split image leakage."""

import pytest

from src.calibration.splits import (
    SplitContract,
    SplitLeakageError,
    SplitRole,
    create_image_level_deterministic_splits,
)


def test_cross_split_image_leakage_rejected():
    """Verify that assigning two different claims from the same image to different splits is REJECTED."""
    # Claim c1 from img_001 is in TRAIN
    # Claim c2 from img_001 is in TEST
    claim_to_image = {
        "claim_001": "img_001",
        "claim_002": "img_001",  # Same image!
        "claim_003": "img_002",
        "claim_004": "img_003",
    }

    # Although claim IDs are disjoint, image_id img_001 crosses splits!
    with pytest.raises(SplitLeakageError, match="Image-level leakage detected"):
        SplitContract(
            train_ids=["claim_001"],
            val_ids=["claim_003"],
            cal_ids=["claim_004"],
            test_ids=["claim_002"],  # Belongs to img_001!
            claim_to_image=claim_to_image,
        )


def test_cross_split_image_leakage_train_val_rejected():
    """Verify that same image crossing TRAIN and VALIDATION is REJECTED."""
    claim_to_image = {
        "c_tr": "img_shared",
        "c_val": "img_shared",
    }
    with pytest.raises(SplitLeakageError, match="Image-level leakage detected"):
        SplitContract(
            train_ids=["c_tr"],
            val_ids=["c_val"],
            claim_to_image=claim_to_image,
        )


def test_assert_no_leakage_catches_test_image_id():
    """Verify that assert_no_leakage catches records bearing TEST image IDs."""
    contract = SplitContract(
        train_ids=["c1"],
        val_ids=["c2"],
        cal_ids=["c3"],
        test_ids=["c4"],
        claim_to_image={"c1": "img_train", "c2": "img_val", "c3": "img_cal", "c4": "img_test"},
    )

    # Candidate train record has a new claim ID "c999", but its image is "img_test"!
    bad_train_records = [{"claim_id": "c999", "image_id": "img_test"}]
    with pytest.raises(SplitLeakageError, match="TEST image IDs found in TRAIN records"):
        contract.assert_no_leakage(bad_train_records, role=SplitRole.TRAIN)


def test_create_image_level_deterministic_splits_keeps_images_atomic():
    """Verify that create_image_level_deterministic_splits assigns all claims of each image to one split."""
    # 5 images with varying number of claims
    records = [
        {"claim_id": "c1_a", "image_id": "img1"},
        {"claim_id": "c1_b", "image_id": "img1"},
        {"claim_id": "c2_a", "image_id": "img2"},
        {"claim_id": "c3_a", "image_id": "img3"},
        {"claim_id": "c3_b", "image_id": "img3"},
        {"claim_id": "c3_c", "image_id": "img3"},
        {"claim_id": "c4_a", "image_id": "img4"},
        {"claim_id": "c5_a", "image_id": "img5"},
    ]

    contract = create_image_level_deterministic_splits(
        records,
        train_ratio=0.40,
        val_ratio=0.20,
        cal_ratio=0.20,
    )

    # Validate image isolation
    contract.validate_isolation()

    # Verify every image has all its claims in the same split
    img_splits = {
        "TRAIN": contract.train_ids,
        "VAL": contract.val_ids,
        "CAL": contract.cal_ids,
        "TEST": contract.test_ids,
    }

    for img_id in ["img1", "img2", "img3", "img4", "img5"]:
        claims = [r["claim_id"] for r in records if r["image_id"] == img_id]
        # Find which split contains the first claim
        split_found = None
        for s_name, s_claims in img_splits.items():
            if claims[0] in s_claims:
                split_found = s_name
                break
        assert split_found is not None
        # All other claims of this image must be in the exact same split
        for c in claims:
            assert c in img_splits[split_found]
