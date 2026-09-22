"""Unit tests for ParameterBundle artifact serialization and provenance validation."""

import json
from pathlib import Path

import pytest

from src.calibration.bundle import ParameterBundle


def test_bundle_initialization_defaults():
    """Verify bundle default state is DEVELOPMENT mode."""
    bundle = ParameterBundle()
    assert bundle.mode == "DEVELOPMENT"
    assert not bundle.is_scientifically_final()
    assert bundle.version == "1.0.0"


def test_bundle_serialization_roundtrip(tmp_path: Path):
    """Verify save, load, and checksum validation."""
    bundle = ParameterBundle(
        mode="DEVELOPMENT",
        theta_model={"feature_type": "combined"},
        probability_calibration={"method": "platt"},
        epsilon_model={"epsilon": 0.25},
        budget_model={"budget": 0.60},
        coupling_model={"lambda": 0.20},
        train_ids_hash="hash_train_123",
        validation_ids_hash="hash_val_456",
        calibration_ids_hash="hash_cal_789",
        dataset_hash="coco_m6_canonical",
        code_sha="d3b7f57",
    )

    save_path = tmp_path / "params.json"
    checksum = bundle.save(save_path)
    assert len(checksum) == 64  # SHA-256 hex string

    loaded = ParameterBundle.load(save_path)
    assert loaded.mode == "DEVELOPMENT"
    assert loaded.theta_model == bundle.theta_model
    assert loaded.train_ids_hash == "hash_train_123"
    assert loaded.compute_checksum() == checksum


def test_bundle_tamper_detection(tmp_path: Path):
    """Verify that tampering with JSON content triggers checksum mismatch error."""
    bundle = ParameterBundle(mode="DEVELOPMENT", epsilon_model={"epsilon": 0.20})
    save_path = tmp_path / "tampered.json"
    bundle.save(save_path)

    # Tamper with file
    with open(save_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["epsilon_model"]["epsilon"] = 0.99  # Malicious edit
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    with pytest.raises(ValueError, match="ParameterBundle checksum mismatch"):
        ParameterBundle.load(save_path, verify_checksum=True)


def test_invalid_mode_rejected():
    """Verify mode validation prevents arbitrary status strings."""
    with pytest.raises(ValueError, match="Mode must be 'DEVELOPMENT' or 'FINAL'"):
        ParameterBundle(mode="PRODUCTION_READY")
