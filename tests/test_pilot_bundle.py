"""
Tests for portable real-image pilot bundle creation and validation.
"""

from pathlib import Path
import shutil
import tempfile
from typing import Tuple
import pytest

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    SplitName,
)
from src.data.manifests import save_manifest
from src.vlm.pilot_bundle import (
    create_portable_pilot_bundle,
    validate_portable_pilot_bundle,
)
from src.vlm.provider import compute_file_sha256


def _create_sample_manifest_with_images(temp_dir: Path, count: int = 5) -> Tuple[Path, Path]:
    images_dir = temp_dir / "raw_images"
    images_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for i in range(count):
        img_id = f"train_coco_{i:03d}"
        img_path = images_dir / f"{img_id}.jpg"
        with open(img_path, "wb") as f:
            f.write(f"jpeg_dummy_data_{i}".encode("utf-8"))
        
        fhash = compute_file_sha256(img_path)
        record = ImageRecord(
            image_id=img_id,
            dataset_source="coco",
            file_name=f"{img_id}.jpg",
            file_hash=fhash,
            width=640,
            height=480,
            metadata={"split": "train"},
        )
        entries.append(DatasetManifestEntry(image=record, split=SplitName.TRAIN, annotations=[]))

    manifest = DatasetManifest(
        manifest_id="train_manifest_test",
        description="coco_train_sample",
        entries=entries,
    )
    manifest_path = temp_dir / "sample_manifest.json"
    save_manifest(manifest, manifest_path)
    return manifest_path, images_dir


def test_create_and_validate_portable_bundle():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        man_path, img_dir = _create_sample_manifest_with_images(tdp, count=4)
        bundle_dir = tdp / "exported_pilot_bundle"

        created_dir = create_portable_pilot_bundle(
            manifest_path=man_path,
            output_dir=bundle_dir,
            image_base_dir=img_dir,
            sample_size=4,
            seed=42,
            copy_images=True,
        )

        assert created_dir.exists()
        assert (created_dir / "manifest.json").exists()
        assert (created_dir / "generation_config.json").exists()
        assert (created_dir / "run_manifest.json").exists()
        assert (created_dir / "README_RUN.md").exists()
        assert (created_dir / "images").exists()

        # Validate bundle
        val_res = validate_portable_pilot_bundle(created_dir)
        assert val_res["valid"], f"Validation failed with errors: {val_res['errors']}"
        assert val_res["image_count"] == 4


def test_bundle_relocation_preserves_hashes_and_validity():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        man_path, img_dir = _create_sample_manifest_with_images(tdp, count=3)
        bundle_dir = tdp / "orig_bundle"

        create_portable_pilot_bundle(
            manifest_path=man_path,
            output_dir=bundle_dir,
            image_base_dir=img_dir,
            sample_size=3,
            seed=42,
            copy_images=True,
        )

        # Relocate bundle to new location
        relocated_dir = tdp / "relocated_bundle"
        shutil.copytree(bundle_dir, relocated_dir)

        # Validation on relocated bundle must still pass
        val_res = validate_portable_pilot_bundle(relocated_dir)
        assert val_res["valid"], f"Relocated validation failed: {val_res['errors']}"
        assert val_res["image_count"] == 3


def test_bundle_validation_catches_tampered_image():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        man_path, img_dir = _create_sample_manifest_with_images(tdp, count=2)
        bundle_dir = tdp / "bundle_to_corrupt"

        create_portable_pilot_bundle(
            manifest_path=man_path,
            output_dir=bundle_dir,
            image_base_dir=img_dir,
            sample_size=2,
            seed=42,
            copy_images=True,
        )

        # Tamper with an image file inside bundle
        target_img = bundle_dir / "images" / "train_coco_000.jpg"
        with open(target_img, "wb") as f:
            f.write(b"corrupt_tampered_content")

        val_res = validate_portable_pilot_bundle(bundle_dir)
        assert not val_res["valid"]
        assert any("Hash mismatch" in err for err in val_res["errors"])
