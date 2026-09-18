"""
Unit tests for data pipeline CLI subcommands.
"""

from pathlib import Path
import json
import pytest

from src.data.cli import main
from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    AtomicObjectExistenceClaim,
    AnnotationRecord,
    GroundTruthStatus,
    AnnotationSource,
    DatasetSource,
)
from src.data.manifests import save_manifest, create_manifest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def test_cli_validate_manifest(tmp_path):
    """Test CLI validate-manifest command."""
    img = ImageRecord(image_id="img_cli", dataset_source=DatasetSource.COCO)
    entry = DatasetManifestEntry(image=img)
    manifest = create_manifest(manifest_id="m_cli", entries=[entry])

    m_path = tmp_path / "valid_manifest.json"
    save_manifest(manifest, m_path)

    exit_code = main(["validate-manifest", str(m_path)])
    assert exit_code == 0


def test_cli_generate_splits_and_audit(tmp_path):
    """Test CLI generate-splits and audit-overlap workflow."""
    entries = []
    for i in range(1, 11):
        img = ImageRecord(image_id=f"coco_{200 + i}", dataset_source=DatasetSource.COCO)
        entries.append(DatasetManifestEntry(image=img))

    manifest = create_manifest(manifest_id="m_split_test", entries=entries)
    m_path = tmp_path / "manifest_for_split.json"
    save_manifest(manifest, m_path)

    out_dir = tmp_path / "splits_out"

    # 1. Run generate-splits
    split_exit = main([
        "generate-splits",
        str(m_path),
        "--output-dir", str(out_dir),
        "--seed", "42",
        "--train-prop", "0.5",
        "--val-prop", "0.2",
        "--calib-prop", "0.1",
        "--test-prop", "0.2",
    ])
    assert split_exit == 0

    train_path = out_dir / "manifest_for_split_train.json"
    val_path = out_dir / "manifest_for_split_validation.json"
    assert train_path.exists()
    assert val_path.exists()

    # 2. Audit overlap between train and validation (must be 0 overlap)
    audit_exit = main([
        "audit-overlap",
        str(train_path),
        str(val_path),
    ])
    assert audit_exit == 0
