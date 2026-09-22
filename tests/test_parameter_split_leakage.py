"""Unit tests for split isolation and test leakage prevention."""

import pytest

from src.calibration.splits import (
    SplitContract,
    SplitLeakageError,
    SplitRole,
    create_deterministic_splits,
)


def test_clean_disjoint_split():
    """Verify that properly disjoint splits pass validation."""
    train_ids = ["c1", "c2", "c3"]
    val_ids = ["c4", "c5"]
    cal_ids = ["c6", "c7"]
    test_ids = ["c8", "c9"]

    contract = SplitContract(
        train_ids=train_ids,
        val_ids=val_ids,
        cal_ids=cal_ids,
        test_ids=test_ids,
    )
    assert len(contract.train_ids) == 3
    assert len(contract.test_ids) == 2
    assert contract.train_hash != contract.test_hash


def test_overlap_raises_leakage_error():
    """Verify that any overlap between splits raises SplitLeakageError."""
    # Test overlaps with train
    with pytest.raises(SplitLeakageError, match="Leakage detected"):
        SplitContract(
            train_ids=["c1", "c2", "c3"],
            val_ids=["c4"],
            cal_ids=["c5"],
            test_ids=["c3", "c6"],  # c3 overlaps
        )

    # Train overlaps with val
    with pytest.raises(SplitLeakageError, match="Leakage detected"):
        SplitContract(
            train_ids=["c1", "c2"],
            val_ids=["c2", "c3"],  # c2 overlaps
            cal_ids=["c4"],
            test_ids=["c5"],
        )

    # Cal overlaps with test
    with pytest.raises(SplitLeakageError, match="Leakage detected"):
        SplitContract(
            train_ids=["c1"],
            val_ids=["c2"],
            cal_ids=["c3"],
            test_ids=["c3"],  # c3 overlaps
        )


def test_assert_no_leakage_guard():
    """Verify runtime assertions against leaked record lists."""
    contract = SplitContract(
        train_ids=["c1", "c2"],
        val_ids=["c3"],
        cal_ids=["c4"],
        test_ids=["test_001", "test_002"],
    )

    clean_records = [{"claim_id": "c1"}, {"claim_id": "c2"}]
    contract.assert_no_leakage(clean_records, role=SplitRole.TRAIN)

    leaked_records = [{"claim_id": "c1"}, {"claim_id": "test_001"}]
    with pytest.raises(SplitLeakageError, match="TEST IDs found in TRAIN records"):
        contract.assert_no_leakage(leaked_records, role=SplitRole.TRAIN)


def test_deterministic_hash_reproducibility():
    """Verify that split hashes are order-independent and cryptographically deterministic."""
    c1 = SplitContract(
        train_ids=["c3", "c1", "c2"],
        val_ids=["v1"],
        cal_ids=["ca1"],
        test_ids=["t1"],
    )
    c2 = SplitContract(
        train_ids=["c1", "c2", "c3"],
        val_ids=["v1"],
        cal_ids=["ca1"],
        test_ids=["t1"],
    )
    assert c1.train_hash == c2.train_hash

    # Changing a single ID must change the hash
    c3 = SplitContract(
        train_ids=["c1", "c2", "c4"],
        val_ids=["v1"],
        cal_ids=["ca1"],
        test_ids=["t1"],
    )
    assert c1.train_hash != c3.train_hash


def test_create_deterministic_splits():
    """Verify automatic split partitioning preserves disjointness."""
    ids = [f"claim_{i:03d}" for i in range(20)]
    contract = create_deterministic_splits(ids, train_ratio=0.4, val_ratio=0.2, cal_ratio=0.2)

    assert len(contract.train_ids) == 8
    assert len(contract.val_ids) == 4
    assert len(contract.cal_ids) == 4
    assert len(contract.test_ids) == 4

    # Disjointness check
    all_assigned = (
        contract.train_ids | contract.val_ids | contract.cal_ids | contract.test_ids
    )
    assert len(all_assigned) == 20
