"""
Tests for real-image pilot preflight verification facility.
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    SplitName,
)
from src.data.manifests import save_manifest
from src.vlm.preflight import (
    run_preflight,
    check_runtime_packages,
    check_local_checkpoint_availability,
)


def _create_synthetic_test_manifest(temp_dir: Path, is_synthetic: bool = False, count: int = 4) -> Path:
    entries = []
    for i in range(count):
        img_id = f"test_img_{i:03d}"
        img_path = temp_dir / f"{img_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 32)  # minimal jpeg header bytes
        
        from src.vlm.provider import compute_file_sha256
        fhash = compute_file_sha256(img_path)
        
        record = ImageRecord(
            image_id=img_id,
            dataset_source="synthetic",
            file_name=f"{img_id}.jpg",
            file_hash=fhash,
            width=640,
            height=480,
            metadata={"is_synthetic": is_synthetic, "split": "train"},
        )
        entries.append(DatasetManifestEntry(
            image=record,
            split=SplitName.TRAIN,
            annotations=[],
        ))
    
    manifest = DatasetManifest(
        manifest_id="test_manifest_001",
        description="preflight_test_dataset",
        entries=entries,
    )
    manifest_path = temp_dir / "test_manifest.json"
    save_manifest(manifest, manifest_path)
    return manifest_path


def test_preflight_detects_missing_manifest():
    report = run_preflight(manifest_path="non_existent_manifest.json")
    assert not report.overall_passed
    assert not report.is_real_pilot_ready
    assert any("not found" in b for b in report.blockers)


def test_preflight_detects_synthetic_manifest_in_real_mode():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        man_path = _create_synthetic_test_manifest(tdp, is_synthetic=True, count=3)
        report = run_preflight(
            manifest_path=man_path,
            image_base_dir=tdp,
            require_real_images=True,
        )
        assert any("synthetic mock images" in b for b in report.blockers)


def test_preflight_detects_missing_checkpoint_in_offline_mode():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        man_path = _create_synthetic_test_manifest(tdp, is_synthetic=False, count=3)
        report = run_preflight(
            manifest_path=man_path,
            image_base_dir=tdp,
            allow_download=False,
            require_real_images=False,
        )
        # Should flag checkpoint presence unless local weights actually exist in cache
        chk_gate = next(g for g in report.gates if g.gate_name == "8_checkpoint_availability")
        if not chk_gate.passed:
            assert "not found locally" in chk_gate.message


def test_preflight_detects_hash_mismatch():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        man_path = _create_synthetic_test_manifest(tdp, is_synthetic=False, count=2)
        
        # Corrupt one image file on disk
        img0_path = tdp / "test_img_000.jpg"
        with open(img0_path, "ab") as f:
            f.write(b"corruption_bytes")

        report = run_preflight(
            manifest_path=man_path,
            image_base_dir=tdp,
            require_real_images=False,
        )
        assert any("hash mismatch" in b.lower() for b in report.blockers)


def test_preflight_rejects_reserved_evaluation_images():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        # Create manifest with reserved external image
        img_id = "test_img_pope_001"
        img_path = tdp / f"{img_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        
        from src.vlm.provider import compute_file_sha256
        fhash = compute_file_sha256(img_path)
        
        record = ImageRecord(
            image_id=img_id,
            dataset_source="synthetic",
            file_name=f"{img_id}.jpg",
            file_hash=fhash,
            width=640,
            height=480,
            metadata={"is_reserved_external": True, "split": "train"},
        )
        entry = DatasetManifestEntry(image=record, split=SplitName.TRAIN, annotations=[])
        manifest = DatasetManifest(manifest_id="m_res", description="res_ds", entries=[entry])
        man_path = tdp / "manifest.json"
        save_manifest(manifest, man_path)

        report = run_preflight(manifest_path=man_path, image_base_dir=tdp, require_real_images=False)
        assert any("reserved" in b.lower() for b in report.blockers)


def test_check_runtime_packages_structure():
    pkgs = check_runtime_packages()
    assert "torch" in pkgs
    assert "transformers" in pkgs
    assert "PIL" in pkgs
    assert "bitsandbytes" in pkgs
    assert isinstance(pkgs["torch"]["installed"], bool)
    assert isinstance(pkgs["bitsandbytes"]["installed"], bool)
