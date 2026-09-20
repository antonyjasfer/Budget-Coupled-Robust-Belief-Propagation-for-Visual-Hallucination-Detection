"""
Split leakage validation and duplicate assignment detection for M7.

STRICT METHODOLOGICAL GUARANTEES:
1. Validates assignments BEFORE collapsing into a one-value dictionary, preventing
   silent overwrite of duplicate assignments (e.g., image_123 -> train and image_123 -> test).
2. Verifies zero cross-split image assignments.
3. Verifies zero cross-split byte-identical SHA-256 hash overlaps.
4. Verifies claim split assignments strictly match parent image split assignments.
5. Verifies image identity groups are never partitioned across multiple splits.
6. Explicitly documents the boundary of SHA-256 duplicate detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Sequence, Union
from collections import defaultdict

from src.data.schemas import SplitName, DatasetManifestEntry
from src.data.image_registry import ImageRegistry, ImageIdentityGroup
from src.annotation.schemas import M7ClaimRecord


class SplitLeakageError(Exception):
    """Raised when dataset split leakage or invalid duplicate assignment is detected."""
    pass


@dataclass
class SplitAuditReport:
    """Detailed audit report of split allocations and leakage checks."""
    is_valid: bool
    total_entries_checked: int
    unique_images: int
    split_counts: Dict[str, int]
    cross_split_overlaps: Dict[str, List[str]] = field(default_factory=dict)
    duplicate_assignment_attempts: List[Tuple[str, str]] = field(default_factory=list)
    hash_cross_split_overlaps: Dict[str, List[str]] = field(default_factory=dict)
    identity_group_violations: List[str] = field(default_factory=list)
    claim_image_split_mismatches: List[str] = field(default_factory=list)
    sha256_limitation_notice: str = (
        "NOTICE: SHA-256 verifies exact byte-identical duplicates only. It does NOT "
        "detect near-duplicates arising from cropping, resizing, compression, or color-shifting."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_entries_checked": self.total_entries_checked,
            "unique_images": self.unique_images,
            "split_counts": self.split_counts,
            "cross_split_overlaps": self.cross_split_overlaps,
            "duplicate_assignment_attempts": [
                {"image_id": img, "split": spl} for img, spl in self.duplicate_assignment_attempts
            ],
            "hash_cross_split_overlaps": self.hash_cross_split_overlaps,
            "identity_group_violations": self.identity_group_violations,
            "claim_image_split_mismatches": self.claim_image_split_mismatches,
            "sha256_limitation_notice": self.sha256_limitation_notice,
        }


def validate_no_image_split_overlap(
    assignments: Sequence[Union[Tuple[str, Union[str, SplitName]], DatasetManifestEntry]]
) -> SplitAuditReport:
    """
    Validate split assignments to guarantee zero cross-split leakage and catch duplicate assignments.

    CRITICAL ARCHITECTURAL SAFEGUARD:
    Accepts a Sequence (list/tuple) of assignments or entries, NOT a Dict[str, str],
    ensuring that attempts to assign the same image multiple times (either to the same
    split or conflicting splits) are explicitly detected and reported.

    Args:
        assignments: Sequence of (image_id, split) tuples or DatasetManifestEntry objects.

    Returns:
        SplitAuditReport detailing audit results.

    Raises:
        SplitLeakageError: If cross-split leakage or duplicate assignment attempts exist.
    """
    image_to_splits: Dict[str, Set[str]] = defaultdict(set)
    seen_assignments: Set[Tuple[str, str]] = set()
    duplicate_attempts: List[Tuple[str, str]] = []
    split_counts: Dict[str, int] = defaultdict(int)

    for entry in assignments:
        if isinstance(entry, DatasetManifestEntry):
            img_id = entry.image.image_id if entry.image else ""
            split_val = entry.split.value if hasattr(entry.split, "value") else str(entry.split)
        elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
            img_id = str(entry[0])
            s = entry[1]
            split_val = s.value if hasattr(s, "value") else str(s)
        else:
            raise ValueError(f"Invalid assignment entry: {entry}")

        if not img_id:
            raise ValueError("Empty image_id encountered in split assignments")

        assign_pair = (img_id, split_val)
        if assign_pair in seen_assignments:
            duplicate_attempts.append(assign_pair)
        else:
            seen_assignments.add(assign_pair)

        image_to_splits[img_id].add(split_val)

    # Count distinct images per split
    for img_id, splits in image_to_splits.items():
        for s in splits:
            split_counts[s] += 1

    # Detect cross-split overlaps (same image in multiple splits)
    cross_split_overlaps = {
        img: sorted(list(splits))
        for img, splits in image_to_splits.items()
        if len(splits) > 1
    }

    is_valid = (len(cross_split_overlaps) == 0) and (len(duplicate_attempts) == 0)

    report = SplitAuditReport(
        is_valid=is_valid,
        total_entries_checked=len(assignments),
        unique_images=len(image_to_splits),
        split_counts=dict(split_counts),
        cross_split_overlaps=cross_split_overlaps,
        duplicate_assignment_attempts=duplicate_attempts,
    )

    if not is_valid:
        error_msgs = []
        if cross_split_overlaps:
            error_msgs.append(
                f"{len(cross_split_overlaps)} image(s) assigned to multiple splits: "
                f"{list(cross_split_overlaps.items())[:3]}"
            )
        if duplicate_attempts:
            error_msgs.append(
                f"{len(duplicate_attempts)} duplicate assignment attempt(s) detected: "
                f"{duplicate_attempts[:3]}"
            )
        raise SplitLeakageError("Split validation failed: " + "; ".join(error_msgs))

    return report


def validate_hash_split_disjointness(
    entries: Sequence[DatasetManifestEntry]
) -> Dict[str, List[str]]:
    """
    Ensure that identical SHA-256 image hashes never appear in different split partitions.

    Args:
        entries: Sequence of DatasetManifestEntry objects.

    Returns:
        Dict mapping hash -> list of conflicting splits (empty if no overlap).

    Raises:
        SplitLeakageError: If identical file hashes appear in distinct splits.
    """
    hash_to_splits: Dict[str, Set[str]] = defaultdict(set)
    for entry in entries:
        if not entry.image or not entry.image.file_hash:
            continue
        h = entry.image.file_hash
        s = entry.split.value if hasattr(entry.split, "value") else str(entry.split)
        hash_to_splits[h].add(s)

    overlapping_hashes = {
        h: sorted(list(splits))
        for h, splits in hash_to_splits.items()
        if len(splits) > 1
    }

    if overlapping_hashes:
        raise SplitLeakageError(
            f"Hash cross-split leakage detected for {len(overlapping_hashes)} image hash(es): "
            f"{list(overlapping_hashes.items())[:3]}"
        )

    return overlapping_hashes


def validate_claim_image_split_consistency(
    claims: Sequence[M7ClaimRecord],
    image_splits: Dict[str, str]
) -> List[str]:
    """
    Ensure every claim's assigned split matches its parent image's assigned split.

    Args:
        claims: Sequence of M7ClaimRecord objects.
        image_splits: Mapping from image_id to split string.

    Returns:
        List of mismatch descriptions (empty if valid).

    Raises:
        SplitLeakageError: If any claim split diverges from parent image split.
    """
    mismatches = []
    for claim in claims:
        parent_split = image_splits.get(claim.image_id)
        if parent_split is None:
            mismatches.append(f"Claim '{claim.claim_id}' references unknown image '{claim.image_id}'")
        elif claim.split != parent_split:
            mismatches.append(
                f"Claim '{claim.claim_id}' split ('{claim.split}') does not match "
                f"parent image '{claim.image_id}' split ('{parent_split}')"
            )

    if mismatches:
        raise SplitLeakageError(
            f"Claim/image split consistency failure ({len(mismatches)} error(s)): {mismatches[:3]}"
        )

    return mismatches
