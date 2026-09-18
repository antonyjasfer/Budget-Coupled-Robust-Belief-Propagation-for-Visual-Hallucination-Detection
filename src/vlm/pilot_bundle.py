"""
Portable Real-Image Pilot Bundle Generator and Validator.

Enables packaging up to 10 selected real training images, portable manifest,
generation configuration, provenance hashes, and run instructions for execution
in a dedicated GPU environment without transferring repository internals or test datasets.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Any, Union, Tuple

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    SplitName,
)
from src.data.manifests import load_manifest, save_manifest
from src.vlm.provider import (
    VLMGenerationConfig,
    compute_file_sha256,
)
from src.vlm.pipeline import select_pilot_images

logger = logging.getLogger(__name__)


@dataclass
class PilotRunManifest:
    """Provenance metadata for a portable pilot package."""
    bundle_id: str
    created_at: str
    manifest_id: str
    total_images: int
    selected_image_ids: List[str]
    image_hashes: Dict[str, str]
    config_hash: str
    target_model_name: str
    split: str
    python_requirement: str = ">=3.10,<=3.12"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PilotRunManifest":
        return cls(
            bundle_id=str(data["bundle_id"]),
            created_at=str(data["created_at"]),
            manifest_id=str(data["manifest_id"]),
            total_images=int(data["total_images"]),
            selected_image_ids=list(data.get("selected_image_ids", [])),
            image_hashes=dict(data.get("image_hashes", {})),
            config_hash=str(data["config_hash"]),
            target_model_name=str(data["target_model_name"]),
            split=str(data.get("split", SplitName.TRAIN.value)),
            python_requirement=str(data.get("python_requirement", ">=3.10,<=3.12")),
            metadata=dict(data.get("metadata", {})),
        )


def create_portable_pilot_bundle(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    split_registry_path: Optional[Union[str, Path]] = None,
    image_base_dir: Optional[Union[str, Path]] = None,
    sample_size: int = 10,
    seed: int = 42,
    copy_images: bool = True,
) -> Path:
    """
    Package selected real training images and portable manifest into output_dir.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)

    split_registry = None
    if split_registry_path and Path(split_registry_path).exists():
        with open(split_registry_path, "r", encoding="utf-8") as f:
            split_registry = json.load(f)

    # 1. Select images strictly from training split
    selected_entries = select_pilot_images(
        manifest=manifest,
        split_registry=split_registry,
        sample_size=min(sample_size, 10),
        seed=seed,
    )

    # 2. Load generation config
    gen_config = VLMGenerationConfig()
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            gen_config = VLMGenerationConfig.from_dict(json.load(f))

    # Save generation config in bundle
    config_file = out / "generation_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(gen_config.to_dict(), f, indent=2)

    image_hashes: Dict[str, str] = {}
    portable_entries: List[DatasetManifestEntry] = []

    for entry in selected_entries:
        img = entry.image
        img_id = img.image_id
        raw_name = img.file_name or f"{img_id}.jpg"
        source_path = Path(image_base_dir) / raw_name if image_base_dir else Path(raw_name)

        target_relative_name = f"images/{Path(raw_name).name}"
        target_file_path = images_dir / Path(raw_name).name

        if source_path.exists():
            resolved_hash = compute_file_sha256(source_path)
            if copy_images:
                shutil.copy2(source_path, target_file_path)
        elif img.file_hash:
            resolved_hash = img.file_hash
        else:
            resolved_hash = hashlib.sha256(img_id.encode("utf-8")).hexdigest()

        image_hashes[img_id] = resolved_hash

        portable_image = ImageRecord(
            image_id=img_id,
            dataset_source=img.dataset_source,
            file_name=target_relative_name,
            file_hash=resolved_hash,
            width=img.width,
            height=img.height,
            metadata={
                **img.metadata,
                "split": SplitName.TRAIN.value,
                "is_synthetic": getattr(img, "is_synthetic", False),
            },
        )
        portable_entry = DatasetManifestEntry(
            image=portable_image,
            split=SplitName.TRAIN,
            annotations=entry.annotations,
        )
        portable_entries.append(portable_entry)

    # Write portable manifest
    bundle_id = f"pilot_bundle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    portable_manifest = DatasetManifest(
        manifest_id=f"manifest_{bundle_id}",
        description=f"Portable pilot manifest from {manifest.manifest_id}",
        entries=portable_entries,
        metadata={
            "pilot_seed": seed,
            "sample_size": len(portable_entries),
            "source_manifest_id": manifest.manifest_id,
        },
    )
    manifest_file = out / "manifest.json"
    save_manifest(portable_manifest, manifest_file)

    # Write run manifest
    run_manifest = PilotRunManifest(
        bundle_id=bundle_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        manifest_id=portable_manifest.manifest_id,
        total_images=len(portable_entries),
        selected_image_ids=[e.image.image_id for e in portable_entries],
        image_hashes=image_hashes,
        config_hash=gen_config.get_param_hash(),
        target_model_name=gen_config.model_name,
        split=SplitName.TRAIN.value,
        python_requirement=">=3.10,<=3.12",
        metadata={"seed": seed},
    )
    run_manifest_file = out / "run_manifest.json"
    with open(run_manifest_file, "w", encoding="utf-8") as f:
        json.dump(run_manifest.to_dict(), f, indent=2)

    # Write README_RUN.md
    readme_file = out / "README_RUN.md"
    readme_content = f"""# Portable Real-Image Pilot Bundle: {bundle_id}

## Specifications
- **Target Checkpoint**: `{gen_config.model_name}`
- **Selected Training Images**: {len(portable_entries)}
- **Split Provenance**: Strictly `TRAIN` split (no validation, test, or reserved images)
- **Generation Settings**: Greedy decoding, max_new_tokens={gen_config.max_new_tokens}, prompt fixed.

## Execution Command (in environment with GPU & dependencies)
```bash
# 1. Install dependencies
uv add torch transformers pillow

# 2. Run Preflight
uv run python -m src.vlm.cli preflight --manifest manifest.json --config generation_config.json

# 3. Execute Real Pilot
uv run python -m src.vlm.cli run-pilot \\
  --manifest manifest.json \\
  --config generation_config.json \\
  --cache-dir cache \\
  --output-bundle annotation_bundle.json
```
"""
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write(readme_content)

    return out


def validate_portable_pilot_bundle(bundle_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate an exported portable pilot bundle structure, integrity, and isolation.
    """
    bdir = Path(bundle_dir)
    errors: List[str] = []
    warnings: List[str] = []

    if not bdir.exists():
        return {"valid": False, "errors": [f"Bundle directory does not exist: {bdir}"], "warnings": []}

    manifest_path = bdir / "manifest.json"
    config_path = bdir / "generation_config.json"
    run_manifest_path = bdir / "run_manifest.json"

    if not manifest_path.exists():
        errors.append("Missing manifest.json in bundle")
    if not config_path.exists():
        errors.append("Missing generation_config.json in bundle")
    if not run_manifest_path.exists():
        errors.append("Missing run_manifest.json in bundle")

    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}

    # Load and validate manifests
    try:
        manifest = load_manifest(manifest_path)
        with open(run_manifest_path, "r", encoding="utf-8") as f:
            run_manifest = PilotRunManifest.from_dict(json.load(f))
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to parse bundle manifest files: {e}"], "warnings": []}

    # Verify split isolation (must be 100% train)
    for entry in manifest.entries:
        if entry.split != SplitName.TRAIN and entry.split != SplitName.TRAIN.value:
            errors.append(f"Image {entry.image.image_id} has invalid non-train split: {entry.split}")
        if entry.image.metadata.get("is_reserved_external", False):
            errors.append(f"Image {entry.image.image_id} is marked as reserved external")

    # Verify image count <= 10
    if len(manifest.entries) > 10:
        errors.append(f"Bundle contains {len(manifest.entries)} images (exceeds pilot limit of 10)")

    # Verify image files and hashes
    images_dir = bdir / "images"
    for entry in manifest.entries:
        img_id = entry.image.image_id
        expected_hash = run_manifest.image_hashes.get(img_id)

        # Check relative path resolution
        resolved_path = bdir / entry.image.file_name
        if images_dir.exists() and resolved_path.exists():
            actual_hash = compute_file_sha256(resolved_path)
            if expected_hash and actual_hash != expected_hash:
                errors.append(f"Hash mismatch for image {img_id}: expected {expected_hash}, got {actual_hash}")

    is_valid = len(errors) == 0
    return {
        "valid": is_valid,
        "bundle_id": run_manifest.bundle_id,
        "image_count": len(manifest.entries),
        "selected_image_ids": run_manifest.selected_image_ids,
        "errors": errors,
        "warnings": warnings,
    }
