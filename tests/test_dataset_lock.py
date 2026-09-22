"""
Unit tests for two-step DatasetLock creation, verification, and performance metric exclusion (Corrections 5, 20, 21).
"""

import json
from pathlib import Path
import pytest
from src.data.dataset_lock import (
    DatasetLock,
    verify_dataset_lock,
    create_and_verify_dataset_lock,
    compute_file_sha256,
)


@pytest.fixture
def mock_manifest_files(tmp_path):
    """Create three decoupled dummy manifest files for testing lock protocols."""
    s_path = tmp_path / "sampling_manifest.json"
    e_path = tmp_path / "evidence_manifest.json"
    a_path = tmp_path / "annotation_manifest.json"

    s_path.write_text(json.dumps({"cohort": "coco_600", "splits": {"train": 300, "test": 120}}), encoding="utf-8")
    e_path.write_text(json.dumps({"provider": "llava_15_hf", "records": []}), encoding="utf-8")
    a_path.write_text(json.dumps({"annotators": ["A", "B"], "records": []}), encoding="utf-8")

    return s_path, e_path, a_path


def test_two_step_lock_creation_and_verification(tmp_path, mock_manifest_files):
    """
    Verify successful execution of the mandatory two-step lock protocol:
    1. Pre-lock validation of manifests.
    2. Generation of lock file.
    3. Independent re-reading from disk and verification of SHA-256 hashes.
    """
    s_path, e_path, a_path = mock_manifest_files
    lock_path = tmp_path / "dataset_lock.json"

    lock = create_and_verify_dataset_lock(
        sampling_manifest_path=s_path,
        evidence_manifest_path=e_path,
        annotation_manifest_path=a_path,
        output_lock_path=lock_path,
        code_sha="abc123def456",
        cohort_counts={"total_images": 600, "train": 300, "test": 120},
    )

    assert lock_path.exists()
    assert lock.lock_hash != ""
    assert len(lock.lock_hash) == 64
    assert lock.sampling_manifest_hash == compute_file_sha256(s_path)
    assert lock.evidence_manifest_hash == compute_file_sha256(e_path)
    assert lock.annotation_manifest_hash == compute_file_sha256(a_path)

    # Verify independently
    is_valid, issues = verify_dataset_lock(lock_path)
    assert is_valid is True
    assert len(issues) == 0


def test_performance_metrics_exclusion_from_lock(tmp_path, mock_manifest_files):
    """
    Verify that injecting performance metrics (F1, AUROC, robust width)
    into a dataset lock is strictly forbidden and raises ValueError (Correction 21).
    """
    s_path, e_path, a_path = mock_manifest_files
    lock_path = tmp_path / "contaminated_lock.json"

    contaminated_payload = {
        "schema_version": "1.0.0",
        "lock_id": "test_lock",
        "code_sha": "abc123def456",
        "sampling_manifest_path": str(s_path),
        "sampling_manifest_hash": compute_file_sha256(s_path),
        "evidence_manifest_path": str(e_path),
        "evidence_manifest_hash": compute_file_sha256(e_path),
        "annotation_manifest_path": str(a_path),
        "annotation_manifest_hash": compute_file_sha256(a_path),
        "cohort_counts": {"total_images": 600},
        "test_f1_score": 0.88,  # FORBIDDEN PERFORMANCE METRIC
    }

    with pytest.raises(ValueError, match="contaminated with performance metric 'test_f1_score'"):
        DatasetLock.from_dict(contaminated_payload)
