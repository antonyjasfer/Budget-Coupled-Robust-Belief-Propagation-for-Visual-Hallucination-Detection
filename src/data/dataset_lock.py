"""
Scientific Dataset Lock and Three-Manifest Verification Pipeline.

Strictly enforces:
1. Three-manifest decoupled architecture:
   - Sampling Manifest (population, image IDs, sampling rule, seed, splits)
   - Evidence Manifest (claims, model provenance, feature hashes)
   - Annotation Manifest (human A/B labels, disagreements, adjudication)
2. Two-step lock protocol (validate pre-lock -> write lock -> independently verify from disk).
3. Hard exclusion of experiment/performance metrics (no F1, AUROC, robust width, accuracy).
4. Cryptographic SHA-256 immutability enforcement.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import hashlib
import json

DATASET_LOCK_SCHEMA_VERSION: str = "1.0.0"

# Explicitly forbidden performance metrics that must never enter a dataset lock
FORBIDDEN_PERFORMANCE_KEYS: Tuple[str, ...] = (
    "f1",
    "roc_auc",
    "aurc",
    "e_aurc",
    "accuracy",
    "precision",
    "recall",
    "robust_width",
    "mean_width",
    "model_correctness",
    "bp_result",
    "standard_bp",
    "robust_bp",
)


@dataclass
class DatasetLock:
    """
    Cryptographic lock referencing the three decoupled manifests.
    Describes data provenance ONLY; contains zero model performance metrics.
    """
    schema_version: str
    lock_id: str
    code_sha: str
    sampling_manifest_path: str
    sampling_manifest_hash: str
    evidence_manifest_path: str
    evidence_manifest_hash: str
    annotation_manifest_path: str
    annotation_manifest_hash: str
    cohort_counts: Dict[str, int]
    lock_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    lock_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "schema_version": self.schema_version,
            "lock_id": self.lock_id,
            "code_sha": self.code_sha,
            "sampling_manifest_path": self.sampling_manifest_path,
            "sampling_manifest_hash": self.sampling_manifest_hash,
            "evidence_manifest_path": self.evidence_manifest_path,
            "evidence_manifest_hash": self.evidence_manifest_hash,
            "annotation_manifest_path": self.annotation_manifest_path,
            "annotation_manifest_hash": self.annotation_manifest_hash,
            "cohort_counts": self.cohort_counts,
            "lock_timestamp": self.lock_timestamp,
        }
        if self.lock_hash:
            d["lock_hash"] = self.lock_hash
        return d

    def compute_lock_hash(self) -> str:
        """
        Compute SHA-256 of canonical serialized representation excluding lock_hash.
        """
        d = self.to_dict()
        d.pop("lock_hash", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetLock":
        # Check for forbidden performance metrics
        for key in data:
            for forbidden in FORBIDDEN_PERFORMANCE_KEYS:
                if forbidden in key.lower():
                    raise ValueError(
                        f"Dataset lock is contaminated with performance metric '{key}'. "
                        "Dataset locks must define data provenance only."
                    )

        return cls(
            schema_version=data["schema_version"],
            lock_id=data["lock_id"],
            code_sha=data["code_sha"],
            sampling_manifest_path=data["sampling_manifest_path"],
            sampling_manifest_hash=data["sampling_manifest_hash"],
            evidence_manifest_path=data["evidence_manifest_path"],
            evidence_manifest_hash=data["evidence_manifest_hash"],
            annotation_manifest_path=data["annotation_manifest_path"],
            annotation_manifest_hash=data["annotation_manifest_hash"],
            cohort_counts=data["cohort_counts"],
            lock_timestamp=data.get("lock_timestamp", datetime.now(timezone.utc).isoformat()),
            lock_hash=data.get("lock_hash", ""),
        )


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file on disk."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_lock(
    lock_file_path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, List[str]]:
    """
    Independently verify an existing dataset lock against artifacts on disk.

    Verification steps:
    1. Read and parse lock JSON.
    2. Check for contamination by performance metrics.
    3. Verify lock_hash matches canonical content hash.
    4. Re-read sampling, evidence, and annotation manifests from disk.
    5. Recompute SHA-256 for each manifest and verify match with lock entries.
    """
    lock_path = Path(lock_file_path)
    if not lock_path.exists():
        return False, [f"Dataset lock file does not exist: {lock_path}"]

    if base_dir is None:
        base_dir = lock_path.parent
    else:
        base_dir = Path(base_dir)

    issues: List[str] = []

    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as err:
        return False, [f"Failed to parse dataset lock JSON: {err}"]

    # Contamination check
    for key in data:
        for forbidden in FORBIDDEN_PERFORMANCE_KEYS:
            if forbidden in key.lower():
                issues.append(f"Lock contaminated with performance key: {key}")

    try:
        lock_obj = DatasetLock.from_dict(data)
    except ValueError as err:
        return False, [str(err)]

    # Verify self-hash
    expected_lock_hash = lock_obj.compute_lock_hash()
    if lock_obj.lock_hash and lock_obj.lock_hash != expected_lock_hash:
        issues.append(
            f"Lock hash mismatch: stored '{lock_obj.lock_hash}', expected '{expected_lock_hash}'"
        )

    # Check manifest files
    manifests_to_check = [
        ("sampling", lock_obj.sampling_manifest_path, lock_obj.sampling_manifest_hash),
        ("evidence", lock_obj.evidence_manifest_path, lock_obj.evidence_manifest_hash),
        ("annotation", lock_obj.annotation_manifest_path, lock_obj.annotation_manifest_hash),
    ]

    for name, rel_path, expected_hash in manifests_to_check:
        full_path = Path(rel_path)
        if not full_path.is_absolute():
            full_path = base_dir / full_path

        if not full_path.exists():
            issues.append(f"{name.capitalize()} manifest file not found: {full_path}")
            continue

        try:
            actual_hash = compute_file_sha256(full_path)
            if actual_hash != expected_hash:
                issues.append(
                    f"{name.capitalize()} manifest SHA-256 mismatch! Stored: {expected_hash}, Actual: {actual_hash}"
                )
        except Exception as err:
            issues.append(f"Failed to read {name} manifest: {err}")

    is_valid = len(issues) == 0
    return is_valid, issues


def create_and_verify_dataset_lock(
    sampling_manifest_path: Union[str, Path],
    evidence_manifest_path: Union[str, Path],
    annotation_manifest_path: Union[str, Path],
    output_lock_path: Union[str, Path],
    code_sha: str,
    cohort_counts: Dict[str, int],
    lock_id: str = "coco_600_primary_lock",
) -> DatasetLock:
    """
    Execute mandatory two-step lock protocol:
    1. Pre-lock validation: verify manifests exist and are non-empty.
    2. Compute independent SHA-256 for all three manifests.
    3. Generate lock object and write to disk.
    4. Independently re-read from disk and verify cryptographic consistency.
    """
    s_path = Path(sampling_manifest_path)
    e_path = Path(evidence_manifest_path)
    a_path = Path(annotation_manifest_path)
    out_path = Path(output_lock_path)

    # Step 1: Pre-lock validation
    for name, p in [("Sampling", s_path), ("Evidence", e_path), ("Annotation", a_path)]:
        if not p.exists():
            raise FileNotFoundError(f"Pre-lock validation failed: {name} manifest does not exist at {p}")
        if p.stat().st_size == 0:
            raise ValueError(f"Pre-lock validation failed: {name} manifest at {p} is empty.")

    # Step 2: Compute hashes
    s_hash = compute_file_sha256(s_path)
    e_hash = compute_file_sha256(e_path)
    a_hash = compute_file_sha256(a_path)

    # Step 3: Instantiate lock and compute lock_hash
    lock = DatasetLock(
        schema_version=DATASET_LOCK_SCHEMA_VERSION,
        lock_id=lock_id,
        code_sha=code_sha,
        sampling_manifest_path=str(s_path),
        sampling_manifest_hash=s_hash,
        evidence_manifest_path=str(e_path),
        evidence_manifest_hash=e_hash,
        annotation_manifest_path=str(a_path),
        annotation_manifest_hash=a_hash,
        cohort_counts=cohort_counts,
    )
    lock.lock_hash = lock.compute_lock_hash()

    # Write lock to disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(lock.to_dict(), f, indent=2)

    # Step 4: Independently re-read and verify
    is_valid, issues = verify_dataset_lock(out_path)
    if not is_valid:
        # Catch errors in lock generation itself and clean up
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"Lock verification step failed: {issues}")

    return lock
