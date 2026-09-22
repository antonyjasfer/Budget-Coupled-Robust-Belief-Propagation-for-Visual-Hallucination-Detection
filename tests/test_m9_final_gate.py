"""Tests for Milestone 9 Final Dataset Gate."""

import json
from pathlib import Path
import pytest

from src.scientific.gate import FinalDatasetGate, GateStatus


def test_gate_rejects_pending_workspace_dataset():
    """Workspace dataset lock is PENDING_ANNOTATION, so gate should fail safely."""
    gate = FinalDatasetGate()
    res = gate.verify()
    assert not res.passed
    assert res.status == GateStatus.FAILED
    assert len(res.diagnostics) > 0


def test_gate_rejects_missing_file(tmp_path):
    """Gate should fail with NOT_AVAILABLE if lock file does not exist."""
    missing_lock = tmp_path / "non_existent_file_lock.json"
    gate = FinalDatasetGate(dataset_lock_path=missing_lock)
    res = gate.verify()
    assert not res.passed
    assert res.status == GateStatus.NOT_AVAILABLE
    assert any("missing" in d.lower() or "not found" in d.lower() for d in res.diagnostics)


def test_gate_detects_synthetic_data(tmp_path):
    """Gate must reject dataset containing synthetic / mocked flags in claims."""
    manifest_file = tmp_path / "m7_manifest.json"
    manifest_data = {
        "entries": [
            {"image": {"image_id": f"img_{i:03d}"}, "split": "test"}
            for i in range(600)
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    claims_file = tmp_path / "m7_claims.jsonl"
    claims_lines = [
        json.dumps({
            "claim_id": f"c_{i}",
            "image_id": f"img_{i:03d}",
            "split": "test",
            "is_synthetic": (i == 0),  # Flag synthetic on first claim
        })
        for i in range(600)
    ]
    claims_file.write_text("\n".join(claims_lines), encoding="utf-8")

    gt_file = tmp_path / "m7_ground_truth.jsonl"
    gt_lines = [
        json.dumps({
            "claim_id": f"c_{i}",
            "final_ground_truth": "supported",
            "has_disagreement": False,
        })
        for i in range(600)
    ]
    gt_file.write_text("\n".join(gt_lines), encoding="utf-8")

    lock_file = tmp_path / "m7_dataset_lock.json"
    lock_data = {
        "status": "LOCKED",
        "dataset_name": "m7_locked",
        "checksums": {
            "m7_manifest_sha256": FinalDatasetGate.compute_sha256(manifest_file),
            "m7_claims_sha256": FinalDatasetGate.compute_sha256(claims_file),
            "m7_ground_truth_sha256": FinalDatasetGate.compute_sha256(gt_file),
        },
    }
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    gate = FinalDatasetGate(
        manifest_path=manifest_file,
        claims_path=claims_file,
        ground_truth_path=gt_file,
        dataset_lock_path=lock_file,
        target_image_count=600,
    )
    res = gate.verify()
    assert not res.passed
    assert any("synthetic" in d.lower() for d in res.diagnostics)


def test_gate_detects_duplicate_claim_ids(tmp_path):
    """Gate must reject duplicate claim IDs."""
    manifest_file = tmp_path / "m7_manifest.json"
    manifest_data = {
        "entries": [
            {"image": {"image_id": f"img_{i:03d}"}, "split": "test"}
            for i in range(600)
        ]
    }
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    claims_file = tmp_path / "m7_claims.jsonl"
    claims_lines = [
        json.dumps({
            "claim_id": "duplicate_c_0" if i < 2 else f"c_{i}",
            "image_id": f"img_{i:03d}",
            "split": "test",
            "is_synthetic": False,
        })
        for i in range(600)
    ]
    claims_file.write_text("\n".join(claims_lines), encoding="utf-8")

    gt_file = tmp_path / "m7_ground_truth.jsonl"
    gt_lines = [
        json.dumps({
            "claim_id": "duplicate_c_0" if i < 2 else f"c_{i}",
            "final_ground_truth": "supported",
            "has_disagreement": False,
        })
        for i in range(600)
    ]
    gt_file.write_text("\n".join(gt_lines), encoding="utf-8")

    lock_file = tmp_path / "m7_dataset_lock.json"
    lock_data = {
        "status": "LOCKED",
        "dataset_name": "m7_locked",
        "checksums": {
            "m7_manifest_sha256": FinalDatasetGate.compute_sha256(manifest_file),
            "m7_claims_sha256": FinalDatasetGate.compute_sha256(claims_file),
            "m7_ground_truth_sha256": FinalDatasetGate.compute_sha256(gt_file),
        },
    }
    lock_file.write_text(json.dumps(lock_data), encoding="utf-8")

    gate = FinalDatasetGate(
        manifest_path=manifest_file,
        claims_path=claims_file,
        ground_truth_path=gt_file,
        dataset_lock_path=lock_file,
        target_image_count=600,
    )
    res = gate.verify()
    assert not res.passed
    assert any("duplicate" in d.lower() for d in res.diagnostics)
