"""
Representative Cohort Sampling and Frozen Split Assignments for MS COCO Benchmark.

Strictly enforces:
1. Primary representative cohort sampled strictly from pre-label image-level metadata.
2. Zero dependence on generated claim count, graph size, model evidence, or human labels.
3. Frozen 300 / 90 / 90 / 120 image split (Train / Val / Cal / Test).
4. All claims inherit their parent image's split partition.
5. Distinct, explicit Graph-Stress Supplement with separate manifest and provenance.
6. Predeclared corruption cohort specifications inheriting source image identities.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Set
import hashlib
import json
import numpy as np

from src.data.schemas import SplitName

SAMPLING_SCHEMA_VERSION: str = "1.0.0"

DEFAULT_SPLIT_COUNTS: Dict[str, int] = {
    SplitName.TRAIN.value: 300,
    SplitName.VALIDATION.value: 90,
    SplitName.CALIBRATION.value: 90,
    SplitName.TEST.value: 120,
}


class CohortType(str, Enum):
    """
    Designation of the benchmark cohort.
    """
    PRIMARY_REPRESENTATIVE = "primary_representative"
    GRAPH_STRESS_SUPPLEMENT = "graph_stress_supplement"


@dataclass
class SamplingManifest:
    """
    Manifest A: Cryptographically frozen sampling manifest.
    """
    schema_version: str
    manifest_id: str
    cohort_type: CohortType
    candidate_population_size: int
    selected_image_ids: List[str]
    sampling_rule: str
    seed: int
    split_assignments: Dict[str, str]  # image_id -> split name
    split_counts: Dict[str, int]
    image_identity_groups: Dict[str, List[str]]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    manifest_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "cohort_type": self.cohort_type.value,
            "candidate_population_size": self.candidate_population_size,
            "selected_image_ids": self.selected_image_ids,
            "sampling_rule": self.sampling_rule,
            "seed": self.seed,
            "split_assignments": self.split_assignments,
            "split_counts": self.split_counts,
            "image_identity_groups": self.image_identity_groups,
            "created_at": self.created_at,
        }
        if self.manifest_hash:
            d["manifest_hash"] = self.manifest_hash
        return d

    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of canonical serialized content excluding manifest_hash.
        """
        d = self.to_dict()
        d.pop("manifest_hash", None)
        canonical_json = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingManifest":
        m = cls(
            schema_version=data["schema_version"],
            manifest_id=data["manifest_id"],
            cohort_type=CohortType(data["cohort_type"]),
            candidate_population_size=data["candidate_population_size"],
            selected_image_ids=data["selected_image_ids"],
            sampling_rule=data["sampling_rule"],
            seed=data["seed"],
            split_assignments=data["split_assignments"],
            split_counts=data["split_counts"],
            image_identity_groups=data["image_identity_groups"],
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            manifest_hash=data.get("manifest_hash", ""),
        )
        return m


def filter_eligible_coco_universe(
    candidate_images: List[Dict[str, Any]],
    min_width: int = 100,
    min_height: int = 100,
) -> List[Dict[str, Any]]:
    """
    Filter candidate COCO images strictly using pre-label image-level metadata.

    Permitted filtering criteria:
    - Valid integer/string image_id.
    - Image dimensions >= min_width and >= min_height.
    - Presence of image file_name.

    STRICTLY FORBIDDEN CRITERIA:
    - Number of generated claims or objects.
    - Graph connectivity or degree.
    - Detector / CLIP confidence.
    - Human labels or ambiguity.
    """
    eligible: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for img in candidate_images:
        img_id = str(img.get("id") or img.get("image_id", ""))
        if not img_id or img_id in seen_ids:
            continue
        
        # Validate dimensions if present
        w = img.get("width", 0)
        h = img.get("height", 0)
        if (w > 0 and w < min_width) or (h > 0 and h < min_height):
            continue

        seen_ids.add(img_id)
        eligible.append(img)

    # Sort deterministically by image_id
    eligible.sort(key=lambda x: str(x.get("id") or x.get("image_id", "")))
    return eligible


def sample_primary_representative_cohort(
    eligible_images: List[Dict[str, Any]],
    cohort_size: int = 600,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Sample exactly cohort_size images from eligible universe using a reproducible seed.

    Guarantees:
    - Deterministic order based on canonical sorting before pseudorandom shuffle.
    - Selection does NOT inspect claim counts, captions, or evidence.
    """
    if len(eligible_images) < cohort_size:
        raise ValueError(
            f"Eligible universe has only {len(eligible_images)} images, "
            f"which is less than requested primary cohort size {cohort_size}."
        )

    # Deterministic sort
    sorted_images = sorted(eligible_images, key=lambda x: str(x.get("id") or x.get("image_id", "")))

    rng = np.random.RandomState(seed)
    indices = np.arange(len(sorted_images))
    rng.shuffle(indices)

    selected_indices = indices[:cohort_size]
    # Re-sort selected images canonically by ID for reproducibility
    selected = [sorted_images[i] for i in sorted(selected_indices)]
    return selected


def assign_frozen_splits(
    sampled_images: List[Dict[str, Any]],
    split_counts: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Partition sampled images into frozen splits: 300 Train, 90 Val, 90 Cal, 120 Test.

    Guarantees:
    - Split assignment is frozen BEFORE human annotation.
    - All claims generated from an image inherit its split assignment.
    """
    if split_counts is None:
        split_counts = DEFAULT_SPLIT_COUNTS.copy()

    total_expected = sum(split_counts.values())
    if len(sampled_images) != total_expected:
        raise ValueError(
            f"Sampled images count ({len(sampled_images)}) does not match "
            f"total split counts requested ({total_expected})."
        )

    sorted_images = sorted(sampled_images, key=lambda x: str(x.get("id") or x.get("image_id", "")))
    rng = np.random.RandomState(seed)
    shuffled_indices = np.arange(len(sorted_images))
    rng.shuffle(shuffled_indices)

    assignments: Dict[str, str] = {}
    achieved: Dict[str, int] = {k: 0 for k in split_counts}

    current_idx = 0
    # Assign sequentially according to fixed partition blocks
    for split_name, count in split_counts.items():
        partition_indices = shuffled_indices[current_idx : current_idx + count]
        for idx in partition_indices:
            img = sorted_images[idx]
            img_id = str(img.get("id") or img.get("image_id", ""))
            assignments[img_id] = split_name
            achieved[split_name] += 1
        current_idx += count

    return assignments, achieved


def create_sampling_manifest(
    eligible_universe: List[Dict[str, Any]],
    sampled_images: List[Dict[str, Any]],
    split_assignments: Dict[str, str],
    split_counts: Dict[str, int],
    seed: int = 42,
    cohort_type: CohortType = CohortType.PRIMARY_REPRESENTATIVE,
    manifest_id: str = "coco_600_primary_sampling_manifest",
    sampling_rule: str = "deterministic_sorted_prng_shuffle_v1",
) -> SamplingManifest:
    """
    Construct cryptographically hashed SamplingManifest.
    """
    selected_ids = sorted([str(img.get("id") or img.get("image_id", "")) for img in sampled_images])
    
    # Image identity groups (each primary image forms its own root identity group)
    identity_groups = {img_id: [img_id] for img_id in selected_ids}

    manifest = SamplingManifest(
        schema_version=SAMPLING_SCHEMA_VERSION,
        manifest_id=manifest_id,
        cohort_type=cohort_type,
        candidate_population_size=len(eligible_universe),
        selected_image_ids=selected_ids,
        sampling_rule=sampling_rule,
        seed=seed,
        split_assignments=split_assignments,
        split_counts=split_counts,
        image_identity_groups=identity_groups,
    )
    manifest.manifest_hash = manifest.compute_hash()
    return manifest


def create_predeclared_corruption_manifest(
    source_image_ids: List[str],
    corruption_families: Optional[List[str]] = None,
    severities: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Pre-declare corruption cohort parameters.
    Guarantees:
    - Source image identity is preserved.
    - Selection is not conditioned on model error or interval growth.
    """
    if corruption_families is None:
        corruption_families = ["gaussian_noise", "gaussian_blur", "jpeg_compression", "contrast_reduction"]
    if severities is None:
        severities = [1, 2, 3, 4, 5]

    entries: List[Dict[str, Any]] = []
    for img_id in sorted(source_image_ids):
        for family in sorted(corruption_families):
            for sev in sorted(severities):
                entries.append({
                    "source_image_id": img_id,
                    "corruption_family": family,
                    "severity": sev,
                    "derived_image_id": f"{img_id}_{family}_sev{sev}",
                    "seed": seed,
                })

    content = {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "total_corrupted_variants": len(entries),
        "source_images_count": len(source_image_ids),
        "families": corruption_families,
        "severities": severities,
        "entries": entries,
    }
    content_hash = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    content["manifest_hash"] = content_hash
    return content
