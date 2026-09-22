"""
Unit tests for cryptographic immutability enforcement of locked datasets.
"""

import json
from pathlib import Path
import pytest
from src.data.dataset_lock import (
    create_and_verify_dataset_lock,
    verify_dataset_lock,
)


@pytest.fixture
def locked_dataset_setup(tmp_path):
    """Create a verified locked dataset on disk."""
    s_path = tmp_path / "sampling_manifest.json"
    e_path = tmp_path / "evidence_manifest.json"
    a_path = tmp_path / "annotation_manifest.json"
    lock_path = tmp_path / "dataset_lock.json"

    s_path.write_text(json.dumps({"images": ["img1", "img2"]}), encoding="utf-8")
    e_path.write_text(json.dumps({"evidence": ["ev1", "ev2"]}), encoding="utf-8")
    a_path.write_text(json.dumps({"labels": ["supported", "hallucinated"]}), encoding="utf-8")

    lock = create_and_verify_dataset_lock(
        sampling_manifest_path=s_path,
        evidence_manifest_path=e_path,
        annotation_manifest_path=a_path,
        output_lock_path=lock_path,
        code_sha="commit_sha_123",
        cohort_counts={"total": 2},
    )
    return s_path, e_path, a_path, lock_path


def test_tampering_sampling_manifest_fails_lock_verification(locked_dataset_setup):
    """
    Verify that altering even a single byte in sampling_manifest.json
    after lock creation causes verification failure.
    """
    s_path, _, _, lock_path = locked_dataset_setup
    
    # Tamper with sampling manifest
    s_path.write_text(json.dumps({"images": ["img1", "img2", "img3_tampered"]}), encoding="utf-8")

    is_valid, issues = verify_dataset_lock(lock_path)
    assert is_valid is False
    assert any("Sampling manifest SHA-256 mismatch" in iss for iss in issues)


def test_tampering_evidence_manifest_fails_lock_verification(locked_dataset_setup):
    """
    Verify that altering evidence_manifest.json causes lock failure.
    """
    _, e_path, _, lock_path = locked_dataset_setup
    
    e_path.write_text(json.dumps({"evidence": ["tampered_evidence"]}), encoding="utf-8")

    is_valid, issues = verify_dataset_lock(lock_path)
    assert is_valid is False
    assert any("Evidence manifest SHA-256 mismatch" in iss for iss in issues)


def test_tampering_lock_file_hash_fails_verification(locked_dataset_setup):
    """
    Verify that tampering with stored lock_hash itself is detected.
    """
    _, _, _, lock_path = locked_dataset_setup

    data = json.loads(lock_path.read_text(encoding="utf-8"))
    data["lock_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
    lock_path.write_text(json.dumps(data), encoding="utf-8")

    is_valid, issues = verify_dataset_lock(lock_path)
    assert is_valid is False
    assert any("Lock hash mismatch" in iss for iss in issues)
