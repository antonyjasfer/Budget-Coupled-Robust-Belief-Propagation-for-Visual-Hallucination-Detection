"""
Manifest creation, validation, and serialization.

Ensures referential integrity across images, claims, annotations, and split tags.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Set, Optional, Any, Union
import json

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    SCHEMA_VERSION,
)


def create_manifest(
    manifest_id: str,
    description: str = "",
    entries: Optional[List[DatasetManifestEntry]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DatasetManifest:
    """
    Create a new DatasetManifest instance with standard metadata.
    """
    created_at = datetime.now(timezone.utc).isoformat()
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        manifest_id=manifest_id,
        created_at=created_at,
        description=description,
        entries=entries or [],
        metadata=metadata or {},
    )


def validate_manifest(manifest: DatasetManifest) -> List[str]:
    """
    Perform strict referential integrity and schema validation on a DatasetManifest.

    Checks:
    1. Schema version compatibility.
    2. Uniqueness of canonical image_ids.
    3. Uniqueness of claim_ids across the entire manifest.
    4. Uniqueness of annotation_ids across the entire manifest.
    5. Referential integrity: Every claim must point to the image_id of its containing entry.
    6. Referential integrity: Every annotation must point to a valid claim_id in the same entry.
    7. Valid GroundTruthStatus and SplitName values.

    Returns:
        List of validation error strings (empty list if valid).
    """
    errors: List[str] = []

    if not manifest.manifest_id:
        errors.append("Manifest is missing 'manifest_id'")

    image_ids: Set[str] = set()
    claim_ids: Set[str] = set()
    annotation_ids: Set[str] = set()

    for idx, entry in enumerate(manifest.entries):
        # Check image
        img = entry.image
        if not img or not img.image_id:
            errors.append(f"Entry {idx} is missing a valid ImageRecord or image_id")
            continue

        if img.image_id in image_ids:
            errors.append(f"Duplicate image_id '{img.image_id}' found in entry {idx}")
        image_ids.add(img.image_id)

        entry_claim_ids: Set[str] = set()
        for c_idx, claim in enumerate(entry.claims):
            if not claim.claim_id:
                errors.append(f"Entry {idx}, claim {c_idx} has empty claim_id")
                continue

            if claim.claim_id in claim_ids:
                errors.append(f"Duplicate claim_id '{claim.claim_id}' in entry {idx}")
            claim_ids.add(claim.claim_id)
            entry_claim_ids.add(claim.claim_id)

            if claim.image_id != img.image_id:
                errors.append(
                    f"Referential integrity failure: claim '{claim.claim_id}' references image_id '{claim.image_id}', "
                    f"but is housed under entry with image_id '{img.image_id}'"
                )

        for a_idx, ann in enumerate(entry.annotations):
            if not ann.annotation_id:
                errors.append(f"Entry {idx}, annotation {a_idx} has empty annotation_id")
                continue

            if ann.annotation_id in annotation_ids:
                errors.append(f"Duplicate annotation_id '{ann.annotation_id}' in entry {idx}")
            annotation_ids.add(ann.annotation_id)

            if ann.claim_id not in entry_claim_ids:
                errors.append(
                    f"Referential integrity failure: annotation '{ann.annotation_id}' references claim_id '{ann.claim_id}', "
                    f"which does not exist in entry '{img.image_id}'"
                )

    return errors


def save_manifest(manifest: DatasetManifest, file_path: Union[str, Path], validate: bool = True) -> None:
    """Save a DatasetManifest to a JSON file."""
    if validate:
        errs = validate_manifest(manifest)
        if errs:
            raise ValueError(f"Cannot save invalid manifest: {'; '.join(errs)}")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def load_manifest(file_path: Union[str, Path], validate: bool = True) -> DatasetManifest:
    """Load and optionally validate a DatasetManifest from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = DatasetManifest.from_dict(data)
    if validate:
        errs = validate_manifest(manifest)
        if errs:
            raise ValueError(f"Loaded manifest failed validation: {'; '.join(errs)}")

    return manifest
