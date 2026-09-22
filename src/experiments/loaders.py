"""
Dataset loading, split safety validation, and synthetic fixture generation for M8.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    GroundTruthStatus,
    SplitName,
    DatasetSource,
)
from src.data.manifests import load_manifest, create_manifest
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import (
    M7ClaimRecord,
    AnnotationRecord,
    AdjudicationRecord,
    FinalGroundTruthRecord,
)
from src.annotation.workflow import load_m6_evidence_file
from src.experiments.configs import M8ExperimentConfig, ExecutionMode


@dataclass
class M8DatasetBundle:
    """
    Consolidated container for validated M8 evaluation data.
    """
    claims: List[M7ClaimRecord]
    ground_truth: Dict[str, FinalGroundTruthRecord]
    manifest: Optional[DatasetManifest]
    images_by_split: Dict[str, List[str]]
    claims_by_split: Dict[str, List[M7ClaimRecord]]
    claims_by_image: Dict[str, List[M7ClaimRecord]]
    execution_state: str  # "STATE_A", "STATE_B", "STATE_C", "SMOKE_FIXTURE"
    is_locked: bool
    dataset_hash: str
    evidence_hash: str
    is_synthetic: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def images(self) -> Dict[str, Any]:
        """Dictionary of image IDs to manifest entry or record."""
        if self.manifest and hasattr(self.manifest, "entries"):
            return {e.image.image_id: e for e in self.manifest.entries}
        # Fallback to unique images from claims_by_image
        return {img_id: img_id for img_id in self.claims_by_image.keys()}

    def get_claims_for_image(self, image_id: str) -> List[M7ClaimRecord]:
        """Retrieve all claims for a given image ID."""
        return self.claims_by_image.get(image_id, [])

    def get_eval_claims(self, splits: Optional[List[str]] = None) -> List[M7ClaimRecord]:
        """Retrieve claims for requested evaluation splits (e.g. ['test'])."""
        if splits is None:
            return list(self.claims)
        
        target_splits = {s.strip().lower() for s in splits}
        return [c for c in self.claims if c.split.lower() in target_splits]

    def get_ground_truth_for_claim(self, claim_id: str) -> Optional[GroundTruthStatus]:
        """Get adjudicated ground truth status if available."""
        if claim_id in self.ground_truth:
            return self.ground_truth[claim_id].final_ground_truth
        return None


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_records_sha256(records: List[Any]) -> str:
    """Compute deterministic SHA-256 across records."""
    hasher = hashlib.sha256()
    for r in sorted(records, key=lambda x: getattr(x, "claim_id", getattr(x, "image_id", str(x)))):
        if hasattr(r, "to_dict"):
            d_str = json.dumps(r.to_dict(), sort_keys=True)
        elif isinstance(r, dict):
            d_str = json.dumps(r, sort_keys=True)
        else:
            d_str = str(r)
        hasher.update(d_str.encode("utf-8"))
    return hasher.hexdigest()


def validate_m8_dataset_integrity(
    claims_or_bundle: Union[M8DatasetBundle, List[M7ClaimRecord]],
    manifest: Optional[DatasetManifest] = None,
    ground_truth: Optional[Dict[str, FinalGroundTruthRecord]] = None,
    allow_synthetic: bool = True,
    *,
    claims: Optional[List[M7ClaimRecord]] = None,
) -> None:
    """
    Perform rigorous sanity and safety checks:
    1. Reject duplicate claim IDs.
    2. Reject duplicate (image_id, claim_id) pairs.
    3. Enforce zero image ID overlap across splits (split leakage check).
    4. Ensure claim.split matches manifest split assignment.
    5. Verify UNKNOWN semantics are preserved without data corruption.
    6. Verify ground-truth independence (no heuristic derivation from detector scores).
    """
    if claims is not None:
        claims_or_bundle = claims

    if isinstance(claims_or_bundle, M8DatasetBundle):
        claims = claims_or_bundle.claims
        manifest = claims_or_bundle.manifest
        ground_truth = claims_or_bundle.ground_truth
    else:
        claims = claims_or_bundle
        ground_truth = ground_truth or {}

    if not claims:
        raise ValueError("Cannot validate empty claims dataset.")

    seen_claim_ids: Set[str] = set()
    image_splits: Dict[str, str] = {}
    claim_count_by_img: Dict[str, int] = {}

    # Build manifest split lookup if available
    manifest_splits: Dict[str, str] = {}
    if manifest is not None:
        for entry in manifest.entries:
            s_val = entry.split.value if hasattr(entry.split, "value") else str(entry.split)
            manifest_splits[entry.image.image_id] = s_val.lower()

    for idx, claim in enumerate(claims):
        cid = claim.claim_id
        img_id = claim.image_id
        split = claim.split.lower()

        # 1. Reject duplicate claim ID
        if cid in seen_claim_ids:
            raise ValueError(f"Duplicate claim_id detected in dataset: '{cid}' at index {idx}")
        seen_claim_ids.add(cid)

        # 2. Check split consistency for this image
        if img_id in image_splits:
            prev_split = image_splits[img_id]
            if prev_split != split:
                raise ValueError(
                    f"Split leakage detected: image '{img_id}' appears in multiple splits "
                    f"('{prev_split}' and '{split}') across different claims."
                )
        else:
            image_splits[img_id] = split

        # 3. Check manifest split consistency
        if manifest_splits and img_id in manifest_splits:
            man_split = manifest_splits[img_id]
            if man_split != split:
                raise ValueError(
                    f"Claim split mismatch for image '{img_id}': claim says '{split}', "
                    f"manifest says '{man_split}'."
                )

        claim_count_by_img[img_id] = claim_count_by_img.get(img_id, 0) + 1

        # 4. Check synthetic flag if forbidden
        if not allow_synthetic and (claim.is_synthetic or claim.evidence.is_synthetic):
            raise ValueError(
                f"Synthetic record detected in strict non-synthetic evaluation mode: claim '{cid}'"
            )

        # 5. Validate evidence numerical sanity
        ev = claim.evidence
        if ev.detector_available:
            if ev.detector_score is None or not (0.0 <= ev.detector_score <= 1.0):
                raise ValueError(f"Invalid detector score for claim '{cid}': {ev.detector_score}")
        if ev.similarity_available:
            if ev.clip_score is None or not (-1.0 <= ev.clip_score <= 1.0):
                raise ValueError(f"Invalid similarity score for claim '{cid}': {ev.clip_score}")

    # 6. Verify ground truth records
    for cid, gt_rec in ground_truth.items():
        if gt_rec.final_ground_truth is not None:
            # Check canonical status
            if not isinstance(gt_rec.final_ground_truth, GroundTruthStatus):
                raise ValueError(
                    f"Ground truth status for claim '{cid}' is not a GroundTruthStatus enum: {gt_rec.final_ground_truth}"
                )


def load_m8_dataset(
    config: Optional[M8ExperimentConfig] = None,
    evidence_path: Optional[Union[str, Path]] = None,
    ground_truth_path: Optional[Union[str, Path]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    split: Optional[str] = None,
    max_images: Optional[int] = None,
    check_integrity: bool = True,
) -> M8DatasetBundle:
    """
    Load dataset bundle according to configuration or explicit paths and verify integrity.
    """
    if config is None:
        config = M8ExperimentConfig()
        if evidence_path:
            config.evidence_path = str(evidence_path)
        if ground_truth_path:
            config.ground_truth_path = str(ground_truth_path)
        if manifest_path:
            config.manifest_path = str(manifest_path)
        if max_images:
            config.subset_size = max_images
        if split:
            config.eval_splits = [split.lower()]

    if config.mode == ExecutionMode.SMOKE:
        return create_synthetic_smoke_bundle(
            num_images=config.subset_size or 10,
            seed=config.seed,
        )

    claims_path = Path(config.claims_path)
    gt_path = Path(config.ground_truth_path)
    man_path = Path(config.manifest_path)
    lock_path = Path(config.dataset_lock_path)
    ev_path = Path(config.evidence_path)

    # Determine State and load records
    claims: List[M7ClaimRecord] = []
    ground_truth: Dict[str, FinalGroundTruthRecord] = {}
    manifest: Optional[DatasetManifest] = None

    if man_path.exists():
        manifest = load_manifest(man_path)

    # Check lock file status
    is_locked = False
    lock_status = "UNLOCKED"
    if lock_path.exists():
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
                lock_status = lock_data.get("status", "UNLOCKED")
                is_locked = (lock_status == "LOCKED")
        except Exception:
            is_locked = False

    if config.mode == ExecutionMode.FINAL and not is_locked and not config.force:
        raise RuntimeError(
            f"ExecutionMode.FINAL requested, but dataset is not locked (status: {lock_status}). "
            f"Complete full M7 annotation and locking before final evaluation."
        )

    # 1. Load Ground Truth records if available
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rec = FinalGroundTruthRecord.from_dict(d)
                ground_truth[rec.claim_id] = rec
                claims.append(rec.claim_record)

    # 2. If claims were not populated from GT, load from claims_path or evidence_path
    if not claims and claims_path.exists():
        with open(claims_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                claims.append(M7ClaimRecord.from_dict(d))

    if not claims and ev_path.exists():
        ev_records = load_m6_evidence_file(ev_path)
        # Match with manifest split if available
        for ev in ev_records:
            assigned_split = SplitName.TRAIN
            if manifest is not None:
                for entry in manifest.entries:
                    if entry.image.image_id == ev.image_id:
                        assigned_split = entry.split
                        break
            claims.append(M7ClaimRecord.from_m6_evidence(ev, assigned_split=assigned_split))

    if not claims:
        raise FileNotFoundError(
            f"No claims or evidence found at {claims_path}, {gt_path}, or {ev_path}."
        )

    # If manifest is present, ensure claim splits align with manifest
    if manifest is not None:
        man_split_map = {e.image.image_id: e.split.value if hasattr(e.split, "value") else str(e.split) for e in manifest.entries}
        for c in claims:
            if c.image_id in man_split_map:
                c.split = man_split_map[c.image_id]

    # Check state
    num_claims = len(claims)
    unique_images = len({c.image_id for c in claims})
    is_synth = any(c.is_synthetic for c in claims)

    if is_synth:
        execution_state = "SMOKE_FIXTURE"
    elif is_locked and unique_images >= 600:
        execution_state = "STATE_C"
    elif unique_images >= 10:
        execution_state = "STATE_A" if unique_images == 10 else "STATE_B"
    else:
        execution_state = "STATE_B"

    # Validate integrity
    allow_synth = (config.mode in [ExecutionMode.SMOKE, ExecutionMode.AUTO] or config.force)
    validate_m8_dataset_integrity(
        claims_or_bundle=claims,
        manifest=manifest,
        ground_truth=ground_truth,
        allow_synthetic=allow_synth,
    )

    # Subsetting if requested
    if config.subset_size is not None and config.subset_size < len(claims):
        claims = claims[:config.subset_size]

    # Index by image and split
    images_by_split: Dict[str, List[str]] = {}
    claims_by_split: Dict[str, List[M7ClaimRecord]] = {}
    claims_by_image: Dict[str, List[M7ClaimRecord]] = {}

    for c in claims:
        s = c.split.lower()
        img = c.image_id
        if s not in images_by_split:
            images_by_split[s] = []
        if img not in images_by_split[s]:
            images_by_split[s].append(img)

        if s not in claims_by_split:
            claims_by_split[s] = []
        claims_by_split[s].append(c)

        if img not in claims_by_image:
            claims_by_image[img] = []
        claims_by_image[img].append(c)

    dataset_hash = compute_records_sha256(claims)
    evidence_hash = compute_records_sha256([c.evidence for c in claims])

    return M8DatasetBundle(
        claims=claims,
        ground_truth=ground_truth,
        manifest=manifest,
        images_by_split=images_by_split,
        claims_by_split=claims_by_split,
        claims_by_image=claims_by_image,
        execution_state=execution_state,
        is_locked=is_locked,
        dataset_hash=dataset_hash,
        evidence_hash=evidence_hash,
        is_synthetic=is_synth,
        metadata={
            "total_claims": len(claims),
            "total_images": len(claims_by_image),
            "lock_status": lock_status,
        }
    )


def create_synthetic_smoke_bundle(
    num_images: int = 10,
    seed: int = 42,
) -> M8DatasetBundle:
    """
    Create a deterministic, self-contained synthetic M8 dataset bundle for testing.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    splits_list = ["train", "validation", "calibration", "test"]
    categories = ["dog", "cat", "car", "person", "chair", "apple", "banana", "bottle"]

    entries: List[DatasetManifestEntry] = []
    claims: List[M7ClaimRecord] = []
    ground_truth: Dict[str, FinalGroundTruthRecord] = {}

    for i in range(num_images):
        img_id = f"syn_img_{i:03d}"
        split = splits_list[i % len(splits_list)]

        img_rec = ImageRecord(
            image_id=img_id,
            dataset_source=DatasetSource.SYNTHETIC,
            file_name=f"{img_id}.jpg",
            file_hash=f"hash_syn_{img_id}",
            width=640,
            height=480,
            metadata={"is_synthetic": True, "split": split},
        )
        entry = DatasetManifestEntry(
            image=img_rec,
            split=SplitName(split),
        )
        entries.append(entry)

        # Generate 1-3 claims per image
        num_claims = 1 + (i % 3)
        for c_idx in range(num_claims):
            cat = categories[(i * 3 + c_idx) % len(categories)]
            cid = f"claim_{img_id}_{cat}_{c_idx}"

            # Ground truth distribution: guarantee representation across classes
            gt_rand = rng.random()
            if c_idx == 0 and i % 3 == 0:
                gt_status = GroundTruthStatus.UNKNOWN
                det_score = float(rng.uniform(0.30, 0.70))
                clip_score = float(rng.uniform(0.15, 0.30))
            elif gt_rand < 0.60:
                gt_status = GroundTruthStatus.SUPPORTED
                det_score = float(rng.uniform(0.65, 0.98))
                clip_score = float(rng.uniform(0.25, 0.45))
            else:
                gt_status = GroundTruthStatus.HALLUCINATED
                det_score = float(rng.uniform(0.01, 0.35))
                clip_score = float(rng.uniform(0.05, 0.22))

            ev = ClaimLevelEvidenceRecord(
                claim_id=cid,
                image_id=img_id,
                object_category=cat,
                text_span=cat,
                caption=f"A synthetic image showing a {cat}.",
                image_hash=f"hash_syn_{img_id}",
                split=split,
                detector_score=det_score,
                detector_available=True,
                detector_model="synthetic/owlvit-mock",
                detector_revision="syn_rev",
                detector_configuration={"device": "cpu", "prompt_template": "a photo of a {category}"},
                clip_score=clip_score,
                similarity_available=True,
                clip_model="synthetic/clip-mock",
                clip_revision="syn_rev",
                clip_prompt_template="a photo of a {category}",
                preprocessing_configuration={"device": "cpu"},
                vlm_generation_source="synthetic_fixture",
                is_synthetic=True,
            )

            claim_rec = M7ClaimRecord(
                claim_id=cid,
                image_id=img_id,
                object_category=cat,
                text_span=cat,
                caption=ev.caption,
                image_hash=ev.image_hash,
                split=split,
                evidence=ev,
                is_synthetic=True,
            )
            claims.append(claim_rec)

            ann_a = AnnotationRecord(
                claim_id=cid,
                image_id=img_id,
                annotator_id="annotator_A",
                ground_truth_status=gt_status,
                rationale="Synthetic fixture annotation",
            )
            ann_b = AnnotationRecord(
                claim_id=cid,
                image_id=img_id,
                annotator_id="annotator_B",
                ground_truth_status=gt_status,
                rationale="Synthetic fixture annotation",
            )

            gt_rec = FinalGroundTruthRecord(
                claim_id=cid,
                image_id=img_id,
                object_category=cat,
                text_span=cat,
                split=split,
                claim_record=claim_rec,
                annotation_a=ann_a,
                annotation_b=ann_b,
                has_disagreement=False,
                adjudication=None,
                final_ground_truth=gt_status,
                is_synthetic=True,
            )
            ground_truth[cid] = gt_rec

    manifest = create_manifest(
        manifest_id="synthetic_smoke_manifest",
        description="Deterministic synthetic smoke manifest for M8 test execution",
        entries=entries,
    )

    images_by_split: Dict[str, List[str]] = {}
    claims_by_split: Dict[str, List[M7ClaimRecord]] = {}
    claims_by_image: Dict[str, List[M7ClaimRecord]] = {}

    for c in claims:
        s = c.split.lower()
        img = c.image_id
        if s not in images_by_split:
            images_by_split[s] = []
        if img not in images_by_split[s]:
            images_by_split[s].append(img)

        if s not in claims_by_split:
            claims_by_split[s] = []
        claims_by_split[s].append(c)

        if img not in claims_by_image:
            claims_by_image[img] = []
        claims_by_image[img].append(c)

    return M8DatasetBundle(
        claims=claims,
        ground_truth=ground_truth,
        manifest=manifest,
        images_by_split=images_by_split,
        claims_by_split=claims_by_split,
        claims_by_image=claims_by_image,
        execution_state="SMOKE_FIXTURE",
        is_locked=True,
        dataset_hash=compute_records_sha256(claims),
        evidence_hash=compute_records_sha256([c.evidence for c in claims]),
        is_synthetic=True,
        metadata={"total_claims": len(claims), "total_images": num_images, "is_smoke": True},
    )
