"""
Tests for the VLM CLI commands (run-pilot, generate-caption, preflight, export-bundle, validate-bundle).
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.data.coco import load_coco_instances
from src.data.manifests import create_manifest, save_manifest
from src.data.schemas import SplitName
from src.vlm.cli import main


def _create_sample_manifest(tmp_path: Path, count: int = 4) -> Path:
    fixture_dir = Path(__file__).parent / "fixtures"
    coco_path = fixture_dir / "coco_instances_synthetic.json"
    adapter = load_coco_instances(coco_path)
    entries = adapter.to_manifest_entries()
    
    for i, e in enumerate(entries):
        if i < count:
            e.split = SplitName.TRAIN
            e.image.metadata["split"] = "train"
        else:
            e.split = SplitName.TEST
            e.image.metadata["split"] = "test"

    manifest = create_manifest("test_manifest", entries=entries)
    manifest_path = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_path)
    return manifest_path


def test_vlm_cli_pilot_synthetic(tmp_path):
    """CLI run-pilot in synthetic mode runs end-to-end and exports bundle."""
    manifest_path = _create_sample_manifest(tmp_path, count=4)
    cache_dir = tmp_path / "cache"
    bundle_out = tmp_path / "exported_bundle.json"

    cmd = [
        "run-pilot",
        "--manifest", str(manifest_path),
        "--cache-dir", str(cache_dir),
        "--output-bundle", str(bundle_out),
        "--synthetic",
        "--sample-size", "4",
    ]

    ret = main(cmd)
    assert ret == 0
    assert bundle_out.exists()

    with open(bundle_out, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_responses"] == 4
    assert len(data["entries"]) == 4


def test_vlm_cli_generate_caption_synthetic(tmp_path, capsys):
    """CLI generate-caption in synthetic mode prints caption and response ID."""
    cache_dir = tmp_path / "cache"
    fake_img = tmp_path / "test_image.jpg"
    fake_img.write_text("dummy image bytes")

    cmd = [
        "generate-caption",
        "--image-path", str(fake_img),
        "--cache-dir", str(cache_dir),
        "--synthetic",
    ]

    ret = main(cmd)
    assert ret == 0
    captured = capsys.readouterr()
    assert "Generated caption:" in captured.out
    assert "Response ID:" in captured.out


def test_vlm_cli_preflight_command(tmp_path, capsys):
    """CLI preflight executes and reports structured gate checks."""
    manifest_path = _create_sample_manifest(tmp_path, count=3)
    cmd = [
        "preflight",
        "--manifest", str(manifest_path),
        "--sample-size", "3",
    ]
    ret = main(cmd)
    # Synthetic fixture manifest on machine without torch/weights should return 1 (BLOCKED)
    assert ret == 1
    captured = capsys.readouterr()
    assert "REAL-PILOT PREFLIGHT VERIFICATION REPORT" in captured.out
    assert "Gates Passed" in captured.out


def test_vlm_cli_export_and_validate_bundle(tmp_path):
    """CLI export-bundle and validate-bundle work seamlessly."""
    manifest_path = _create_sample_manifest(tmp_path, count=3)
    bundle_out = tmp_path / "cli_bundle"

    export_cmd = [
        "export-bundle",
        "--manifest", str(manifest_path),
        "--output-dir", str(bundle_out),
        "--sample-size", "3",
    ]
    ret_exp = main(export_cmd)
    assert ret_exp == 0
    assert (bundle_out / "manifest.json").exists()

    val_cmd = [
        "validate-bundle",
        "--bundle-dir", str(bundle_out),
    ]
    ret_val = main(val_cmd)
    assert ret_val == 0
