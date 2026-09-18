"""
Deterministic, leakage-safe dataset splitting.

Splits dataset entries strictly by ImageIdentityGroup, ensuring all claims,
responses, and duplicate/aliased images reside in the same split partition.

CRITICAL METHODOLOGICAL SAFEGUARDS:
1. Split unit is the IMAGE-IDENTITY GROUP, never the claim.
2. Explicitly reserved external-evaluation images (e.g., POPE eval images) are
   strictly segregated and never moved into development splits to satisfy ratios.
3. Splits are fully deterministic given the random seed and canonical sorted ordering.
4. Input manifest SHA-256 hash and split configuration are preserved in metadata.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, Tuple
import hashlib
import json
import numpy as np

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    SplitName,
)
from src.data.image_registry import ImageRegistry, ImageIdentityGroup


DEFAULT_PROPORTIONS: Dict[SplitName, float] = {
    SplitName.TRAIN: 0.50,
    SplitName.VALIDATION: 0.15,
    SplitName.CALIBRATION: 0.15,
    SplitName.TEST: 0.20,
}


@dataclass
class SplitMetadata:
    """
    Metadata recording split configuration, input provenance, and achieved counts.

    Attributes:
        seed: Random seed used for shuffling.
        target_proportions: Configured target partition ratios.
        input_manifest_hash: SHA-256 of the input manifest JSON string.
        total_images: Total number of canonical images in manifest.
        total_groups: Total number of ImageIdentityGroups.
        achieved_counts: Dict mapping split name -> count of images.
        achieved_group_counts: Dict mapping split name -> count of identity groups.
        achieved_proportions: Dict mapping split name -> fraction of images.
        reserved_external_count: Count of images reserved for external evaluation.
        assignments: Dict mapping canonical image_id -> split name.
    """
    seed: int
    target_proportions: Dict[str, float]
    input_manifest_hash: str
    total_images: int
    total_groups: int
    achieved_counts: Dict[str, int]
    achieved_group_counts: Dict[str, int]
    achieved_proportions: Dict[str, float]
    reserved_external_count: int
    assignments: Dict[str, str]


@dataclass
class SplitResult:
    """
    Container for partitioned manifests and split execution metadata.

    Attributes:
        manifests_by_split: Dict mapping SplitName -> DatasetManifest.
        unified_manifest: DatasetManifest with updated entry.split assignments.
        metadata: SplitMetadata instance.
    """
    manifests_by_split: Dict[SplitName, DatasetManifest]
    unified_manifest: DatasetManifest
    metadata: SplitMetadata


def split_manifest_by_image_groups(
    manifest: DatasetManifest,
    seed: int = 42,
    proportions: Optional[Dict[Union[SplitName, str], float]] = None,
    reserved_external_ids: Optional[Union[Set[str], List[str]]] = None,
    forced_assignments: Optional[Dict[str, Union[SplitName, str]]] = None,
) -> SplitResult:
    """
    Partition a DatasetManifest deterministically into train, validation, calibration, and test splits.

    Args:
        manifest: Input DatasetManifest.
        seed: Random seed for deterministic shuffling.
        proportions: Dict mapping SplitName (or str) to float proportion.
        reserved_external_ids: Set/List of image IDs that must be reserved exclusively for external evaluation (e.g. test).
        forced_assignments: Optional manual assignments (image_id -> SplitName).

    Returns:
        SplitResult containing split manifests and reproducibility metadata.
    """
    # 1. Normalize proportions
    if proportions is None:
        props: Dict[SplitName, float] = dict(DEFAULT_PROPORTIONS)
    else:
        props = {}
        for k, v in proportions.items():
            s_name = SplitName(k) if isinstance(k, str) else k
            props[s_name] = float(v)

    total_p = sum(props.values())
    if not (0.999 <= total_p <= 1.001):
        raise ValueError(f"Split proportions must sum to 1.0, got sum={total_p:.4f}")

    reserved_set = set(reserved_external_ids) if reserved_external_ids is not None else set()
    forced_map: Dict[str, SplitName] = {}
    if forced_assignments:
        for iid, s in forced_assignments.items():
            s_enum = SplitName(s) if isinstance(s, str) else s
            forced_map[iid] = s_enum

    # Check for conflicting reservations
    for iid in reserved_set:
        if iid in forced_map and forced_map[iid] in (SplitName.TRAIN, SplitName.VALIDATION, SplitName.CALIBRATION):
            raise ValueError(
                f"Conflicting reservation: image '{iid}' is marked as reserved external evaluation, "
                f"but forced into development split '{forced_map[iid]}'"
            )

    # 2. Build ImageRegistry and identify groups
    registry = ImageRegistry()
    registry.register_manifest(manifest)

    # 3. Canonical sort of group IDs for deterministic seeding
    all_group_ids = sorted(list(registry.groups.keys()))
    num_groups = len(all_group_ids)

    # Separate reserved groups, forced groups, and unassigned groups
    reserved_groups: Set[str] = set()
    forced_groups: Dict[str, SplitName] = {}
    free_groups: List[str] = []

    for gid in all_group_ids:
        grp = registry.groups[gid]
        # If any image in group is reserved external
        if grp.canonical_image_ids.intersection(reserved_set):
            reserved_groups.add(gid)
        elif any(iid in forced_map for iid in grp.canonical_image_ids):
            # Pick first forced assignment
            first_forced = next(forced_map[iid] for iid in grp.canonical_image_ids if iid in forced_map)
            forced_groups[gid] = first_forced
        else:
            free_groups.append(gid)

    # 4. Deterministic shuffle of free groups
    rng = np.random.default_rng(seed)
    shuffled_free_groups = list(rng.permutation(free_groups))

    # 5. Compute split allocations for free groups
    # Available split names in order
    split_order = [SplitName.TRAIN, SplitName.VALIDATION, SplitName.CALIBRATION, SplitName.TEST]
    group_assignments: Dict[str, SplitName] = {}

    # Assign reserved groups directly to TEST split
    for gid in reserved_groups:
        group_assignments[gid] = SplitName.TEST

    # Assign forced groups
    for gid, s_name in forced_groups.items():
        group_assignments[gid] = s_name

    # Allocate free groups proportionally
    num_free = len(shuffled_free_groups)
    if num_free > 0:
        # Calculate target counts
        free_counts: Dict[SplitName, int] = {}
        cum_allocated = 0
        for i, s_name in enumerate(split_order):
            if i == len(split_order) - 1:
                free_counts[s_name] = num_free - cum_allocated
            else:
                cnt = int(np.round(props.get(s_name, 0.0) * num_free))
                free_counts[s_name] = cnt
                cum_allocated += cnt

        # Assign slices
        cursor = 0
        for s_name in split_order:
            cnt = free_counts[s_name]
            for gid in shuffled_free_groups[cursor : cursor + cnt]:
                group_assignments[gid] = s_name
            cursor += cnt

    # 6. Apply assignments to manifest entries
    assignments_by_image: Dict[str, str] = {}
    manifests_by_split: Dict[SplitName, DatasetManifest] = {
        s: DatasetManifest(
            schema_version=manifest.schema_version,
            manifest_id=f"{manifest.manifest_id}_{s.value}",
            created_at=manifest.created_at,
            description=f"{s.value.capitalize()} partition of {manifest.manifest_id}",
            entries=[],
            metadata={"split_name": s.value, "parent_manifest_id": manifest.manifest_id},
        )
        for s in split_order
    }

    updated_entries: List[DatasetManifestEntry] = []
    achieved_counts: Dict[str, int] = {s.value: 0 for s in split_order}
    achieved_group_counts: Dict[str, int] = {s.value: 0 for s in split_order}

    for entry in manifest.entries:
        iid = entry.image.image_id
        gid = registry.get_group_id_for_image(iid)
        assigned_split = group_assignments.get(gid, SplitName.TEST)

        # Clone entry with assigned split
        new_entry = DatasetManifestEntry(
            image=entry.image,
            claims=entry.claims,
            annotations=entry.annotations,
            responses=entry.responses,
            split=assigned_split,
        )
        updated_entries.append(new_entry)
        manifests_by_split[assigned_split].entries.append(new_entry)
        assignments_by_image[iid] = assigned_split.value
        achieved_counts[assigned_split.value] += 1

    for gid, s_name in group_assignments.items():
        achieved_group_counts[s_name.value] += 1

    total_images = len(manifest.entries)
    achieved_props = {
        s.value: (achieved_counts[s.value] / total_images) if total_images > 0 else 0.0
        for s in split_order
    }

    # Hash input manifest
    input_hash = hashlib.sha256(manifest.to_json().encode("utf-8")).hexdigest()

    meta = SplitMetadata(
        seed=seed,
        target_proportions={k.value: v for k, v in props.items()},
        input_manifest_hash=input_hash,
        total_images=total_images,
        total_groups=num_groups,
        achieved_counts=achieved_counts,
        achieved_group_counts=achieved_group_counts,
        achieved_proportions=achieved_props,
        reserved_external_count=len(reserved_set),
        assignments=assignments_by_image,
    )

    unified_manifest = DatasetManifest(
        schema_version=manifest.schema_version,
        manifest_id=manifest.manifest_id,
        created_at=manifest.created_at,
        description=manifest.description,
        entries=updated_entries,
        metadata={
            **manifest.metadata,
            "split_seed": seed,
            "split_achieved_counts": achieved_counts,
        },
    )

    return SplitResult(
        manifests_by_split=manifests_by_split,
        unified_manifest=unified_manifest,
        metadata=meta,
    )


@dataclass
class DatasetSplitsCompatibilityWrapper:
    split_result: SplitResult

    @property
    def registry(self) -> Dict[str, str]:
        return self.split_result.metadata.assignments

    @property
    def split_indices(self) -> Dict[str, List[str]]:
        res: Dict[str, List[str]] = {s.value: [] for s in SplitName}
        for img_id, s_name in self.split_result.metadata.assignments.items():
            if s_name not in res:
                res[s_name] = []
            res[s_name].append(img_id)
        return res

    @property
    def unified_manifest(self) -> DatasetManifest:
        return self.split_result.unified_manifest

    @property
    def manifests_by_split(self) -> Dict[SplitName, DatasetManifest]:
        return self.split_result.manifests_by_split

    @property
    def metadata(self) -> SplitMetadata:
        return self.split_result.metadata


def create_dataset_splits(
    manifest: DatasetManifest,
    reserved_external_image_ids: Optional[Union[Set[str], List[str]]] = None,
    train_ratio: float = 0.50,
    val_ratio: float = 0.15,
    calib_ratio: float = 0.15,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> DatasetSplitsCompatibilityWrapper:
    """Convenience wrapper creating deterministic dataset splits."""
    proportions = {
        SplitName.TRAIN: train_ratio,
        SplitName.VALIDATION: val_ratio,
        SplitName.CALIBRATION: calib_ratio,
        SplitName.TEST: test_ratio,
    }
    split_res = split_manifest_by_image_groups(
        manifest=manifest,
        seed=seed,
        proportions=proportions,
        reserved_external_ids=reserved_external_image_ids,
    )
    return DatasetSplitsCompatibilityWrapper(split_res)

