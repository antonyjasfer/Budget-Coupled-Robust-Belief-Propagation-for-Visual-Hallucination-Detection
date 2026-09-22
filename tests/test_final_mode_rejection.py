"""
Unit tests for validate_final_dataset.py distinguishing DEVELOPMENT vs FINAL modes (Corrections 6, 22).
"""

import json
from pathlib import Path
import pytest
from scripts.validate_final_dataset import validate_dataset
from src.data.provenance import ValidationMode
from src.data.dataset_lock import create_and_verify_dataset_lock


def test_development_mode_passes_without_lock(tmp_path):
    """
    Verify DEVELOPMENT mode succeeds with sampling manifest present,
    even without dataset_lock.json (Correction 6).
    """
    sampling_path = tmp_path / "sampling_manifest.json"
    sampling_path.write_text(json.dumps({
        "selected_image_ids": ["img1", "img2"],
        "split_assignments": {"img1": "train", "img2": "test"},
        "split_counts": {"train": 1, "test": 1},
    }), encoding="utf-8")

    res = validate_dataset(tmp_path, mode=ValidationMode.DEVELOPMENT)
    assert res["is_valid"] is True
    assert res["checks"]["sampling_manifest"] == "VALID"
    assert res["checks"]["dataset_lock"] == "NOT LOCKED"


def test_final_mode_fails_without_lock_and_annotations(tmp_path):
    """
    Verify FINAL mode strictly requires complete evidence, annotations,
    and a valid dataset_lock.json.
    """
    sampling_path = tmp_path / "sampling_manifest.json"
    sampling_path.write_text(json.dumps({
        "selected_image_ids": [f"img_{i}" for i in range(600)],
        "split_assignments": {f"img_{i}": "train" for i in range(600)},
        "split_counts": {"train": 300, "validation": 90, "calibration": 90, "test": 120},
    }), encoding="utf-8")

    res = validate_dataset(tmp_path, mode=ValidationMode.FINAL)
    assert res["is_valid"] is False
    assert any("evidence_manifest.json is required" in iss for iss in res["issues"])
    assert any("annotation_manifest.json is required" in iss for iss in res["issues"])
    assert any("dataset_lock.json is missing" in iss for iss in res["issues"])


def test_final_mode_passes_when_all_criteria_met(tmp_path):
    """
    Verify FINAL mode passes when all final criteria, valid manifests,
    and verified dataset lock are in place.
    """
    # 1. 600-image sampling manifest
    img_ids = [f"coco_{i:04d}" for i in range(600)]
    splits = {}
    for i in range(300):
        splits[img_ids[i]] = "train"
    for i in range(300, 390):
        splits[img_ids[i]] = "validation"
    for i in range(390, 480):
        splits[img_ids[i]] = "calibration"
    for i in range(480, 600):
        splits[img_ids[i]] = "test"

    s_path = tmp_path / "sampling_manifest.json"
    s_path.write_text(json.dumps({
        "dataset_version": "v2",
        "selected_image_ids": img_ids,
        "split_assignments": splits,
        "split_counts": {"train": 300, "validation": 90, "calibration": 90, "test": 120},
    }), encoding="utf-8")

    # 2. Complete evidence manifest
    e_records = [
        {
            "claim_id": f"claim_{i}",
            "image_id": img_ids[i % 600],
            "detector_score": 0.85,
            "detector_available": True,
            "clip_score": 0.72,
            "similarity_available": True,
            "provenance": "real_human_annotated",
        }
        for i in range(600)
    ]
    e_path = tmp_path / "evidence_manifest.json"
    e_path.write_text(json.dumps({"dataset_version": "v2", "records": e_records}), encoding="utf-8")

    # 3. Complete annotation manifest
    a_records = [
        {
            "claim_id": f"claim_{i}",
            "ground_truth_status": "supported",
            "provenance": "real_adjudicated",
        }
        for i in range(600)
    ]
    a_path = tmp_path / "annotation_manifest.json"
    a_path.write_text(json.dumps({"records": a_records}), encoding="utf-8")

    # 4. Verified dataset lock
    lock_path = tmp_path / "dataset_lock.json"
    create_and_verify_dataset_lock(
        sampling_manifest_path=s_path,
        evidence_manifest_path=e_path,
        annotation_manifest_path=a_path,
        output_lock_path=lock_path,
        code_sha="clean_tested_sha_123",
        cohort_counts={"total_images": 600, "train": 300, "validation": 90, "calibration": 90, "test": 120},
    )

    res = validate_dataset(tmp_path, mode=ValidationMode.FINAL)
    assert res["is_valid"] is True
    assert res["checks"]["dataset_lock"] == "LOCKED_VALID"
    assert res["checks"]["provenance_isolation"] == "PASS"
    assert len(res["issues"]) == 0
