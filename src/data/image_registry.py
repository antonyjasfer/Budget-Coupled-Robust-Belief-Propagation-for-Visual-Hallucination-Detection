"""
Image registry and overlap / identity management.

Maintains canonical image identifiers, aliases, normalized filenames, and content SHA-256 hashes.
Groups duplicate or aliased images into unified identity groups to prevent data leakage across splits.

CRITICAL METHODOLOGICAL LIMITATION:
Content SHA-256 hashes detect byte-for-byte identical image files. They DO NOT detect
resized, cropped, color-shifted, or re-compressed near-duplicate images.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
import hashlib

from src.data.schemas import ImageRecord, DatasetManifest


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class ImageIdentityGroup:
    """
    Unified cluster of canonical image IDs and hashes representing a single visual entity.

    Attributes:
        group_id: Unique identifier for this identity group (e.g. 'group_coco_123').
        canonical_image_ids: Set of image_id strings in this group.
        file_hashes: Set of SHA-256 content hashes in this group.
        file_names: Set of associated file names.
    """
    group_id: str
    canonical_image_ids: Set[str] = field(default_factory=set)
    file_hashes: Set[str] = field(default_factory=set)
    file_names: Set[str] = field(default_factory=set)


@dataclass
class OverlapReport:
    """
    Results of an overlap audit between two manifests, splits, or dataset sources.

    Attributes:
        has_overlap: Boolean indicating whether any identity overlap exists.
        overlapping_image_ids: Set of overlapping canonical image IDs.
        overlapping_file_hashes: Set of overlapping file content hashes.
        overlapping_groups: List of group IDs present in both sets.
        summary: Human-readable audit narrative.
    """
    has_overlap: bool
    overlapping_image_ids: Set[str]
    overlapping_file_hashes: Set[str]
    overlapping_groups: List[str]
    summary: str


class ImageRegistry:
    """
    Central registry for image identities, aliases, and duplicate tracking.
    """

    def __init__(self):
        self.groups: Dict[str, ImageIdentityGroup] = {}
        self.image_id_to_group_id: Dict[str, str] = {}
        self.hash_to_group_id: Dict[str, str] = {}
        self.records: Dict[str, ImageRecord] = {}

    def register_image(self, record: ImageRecord, compute_hash_if_path_exists: bool = False) -> str:
        """
        Register an ImageRecord into the registry. Links to an existing group if
        the image_id or file_hash matches an existing entry.

        Returns:
            group_id: The identifier of the assigned identity group.
        """
        img_id = record.image_id
        file_hash = record.file_hash

        if not file_hash and compute_hash_if_path_exists and record.file_name:
            path = Path(record.file_name)
            if path.exists():
                file_hash = compute_file_sha256(path)
                record.file_hash = file_hash

        # Check existing group by image_id or file_hash
        existing_group_id = self.image_id_to_group_id.get(img_id)
        if not existing_group_id and file_hash:
            existing_group_id = self.hash_to_group_id.get(file_hash)

        if existing_group_id is None:
            # Create new group
            group_id = f"group_{img_id}"
            group = ImageIdentityGroup(
                group_id=group_id,
                canonical_image_ids={img_id},
                file_hashes={file_hash} if file_hash else set(),
                file_names={record.file_name} if record.file_name else set(),
            )
            self.groups[group_id] = group
            self.image_id_to_group_id[img_id] = group_id
            if file_hash:
                self.hash_to_group_id[file_hash] = group_id
        else:
            # Merge into existing group
            group = self.groups[existing_group_id]
            group.canonical_image_ids.add(img_id)
            if file_hash:
                group.file_hashes.add(file_hash)
                self.hash_to_group_id[file_hash] = existing_group_id
            if record.file_name:
                group.file_names.add(record.file_name)
            self.image_id_to_group_id[img_id] = existing_group_id
            group_id = existing_group_id

        self.records[img_id] = record
        return group_id

    def register_manifest(self, manifest: DatasetManifest) -> None:
        """Register all images from a DatasetManifest."""
        for entry in manifest.entries:
            self.register_image(entry.image)

    def get_group_id_for_image(self, image_id: str) -> Optional[str]:
        """Get group ID for a canonical image ID."""
        return self.image_id_to_group_id.get(image_id)

    def audit_overlap(
        self,
        set_a_image_ids: Set[str],
        set_b_image_ids: Set[str],
        label_a: str = "Set A",
        label_b: str = "Set B"
    ) -> OverlapReport:
        """
        Audit overlap between two sets of image IDs using both canonical IDs and registered file hashes.
        """
        direct_id_overlap = set_a_image_ids.intersection(set_b_image_ids)

        groups_a = {self.image_id_to_group_id[iid] for iid in set_a_image_ids if iid in self.image_id_to_group_id}
        groups_b = {self.image_id_to_group_id[iid] for iid in set_b_image_ids if iid in self.image_id_to_group_id}
        overlapping_groups = list(groups_a.intersection(groups_b))

        hashes_a = {
            self.records[iid].file_hash
            for iid in set_a_image_ids
            if iid in self.records and self.records[iid].file_hash
        }
        hashes_b = {
            self.records[iid].file_hash
            for iid in set_b_image_ids
            if iid in self.records and self.records[iid].file_hash
        }
        overlapping_hashes = hashes_a.intersection(hashes_b)

        has_overlap = bool(direct_id_overlap or overlapping_groups or overlapping_hashes)

        if has_overlap:
            summary = (
                f"OVERLAP DETECTED between {label_a} and {label_b}: "
                f"{len(direct_id_overlap)} direct ID matches, "
                f"{len(overlapping_hashes)} SHA-256 content matches, "
                f"{len(overlapping_groups)} overlapping identity groups."
            )
        else:
            summary = f"NO OVERLAP detected between {label_a} ({len(set_a_image_ids)} images) and {label_b} ({len(set_b_image_ids)} images)."

        return OverlapReport(
            has_overlap=has_overlap,
            overlapping_image_ids=direct_id_overlap,
            overlapping_file_hashes=overlapping_hashes,
            overlapping_groups=overlapping_groups,
            summary=summary,
        )
