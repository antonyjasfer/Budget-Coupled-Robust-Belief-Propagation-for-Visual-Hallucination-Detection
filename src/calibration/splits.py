"""Strict data split contracts and leakage prevention guards for Phase 9B/9C parameterization.

Roles:
- TRAIN: Fit evidence-to-probability feature models.
- VALIDATION: Choose regularization, model family, coupling strength lambda, and topology.
- CALIBRATION: Calibrate probabilities (Platt / Isotonic) and estimate perturbation bounds (epsilon, B).
- TEST: Final benchmark evaluation only. ZERO parameters or thresholds may be derived from test.

STRICT METHODOLOGICAL GUARANTEES:
1. Claim-level isolation: Disjoint claim ID sets across splits.
2. Image-level / identity-group isolation: All claims belonging to a given image or identity-group
   MUST belong to exactly ONE split. An image cannot contribute claims to multiple splits.
3. Cryptographic provenance hashes for auditability.
4. Hard runtime guards against test claim or test image leakage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Set, Union


class SplitRole(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    CALIBRATION = "CALIBRATION"
    TEST = "TEST"


class SplitLeakageError(Exception):
    """Raised when test data, test IDs, or cross-split image assignments contaminate parameter fitting."""

    pass


@dataclass
class SplitContract:
    """Manages partition sets and enforces strict split isolation at both claim and image levels."""

    train_ids: Set[str] = field(default_factory=set)
    val_ids: Set[str] = field(default_factory=set)
    cal_ids: Set[str] = field(default_factory=set)
    test_ids: Set[str] = field(default_factory=set)

    # Image-level / Identity-group isolation
    claim_to_image: Dict[str, str] = field(default_factory=dict)
    train_image_ids: Set[str] = field(default_factory=set)
    val_image_ids: Set[str] = field(default_factory=set)
    cal_image_ids: Set[str] = field(default_factory=set)
    test_image_ids: Set[str] = field(default_factory=set)

    def __init__(
        self,
        train_ids: Optional[Iterable[str]] = None,
        val_ids: Optional[Iterable[str]] = None,
        cal_ids: Optional[Iterable[str]] = None,
        test_ids: Optional[Iterable[str]] = None,
        validation_ids: Optional[Iterable[str]] = None,
        calibration_ids: Optional[Iterable[str]] = None,
        claim_to_image: Optional[Dict[str, str]] = None,
        train_image_ids: Optional[Iterable[str]] = None,
        val_image_ids: Optional[Iterable[str]] = None,
        cal_image_ids: Optional[Iterable[str]] = None,
        test_image_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.train_ids = set(train_ids or [])
        self.val_ids = set(val_ids or validation_ids or [])
        self.cal_ids = set(cal_ids or calibration_ids or [])
        self.test_ids = set(test_ids or [])

        self.claim_to_image = dict(claim_to_image or {})
        self.train_image_ids = set(train_image_ids or [])
        self.val_image_ids = set(val_image_ids or [])
        self.cal_image_ids = set(cal_image_ids or [])
        self.test_image_ids = set(test_image_ids or [])

        # Auto-populate image IDs from claim_to_image mapping
        for c in self.train_ids:
            if c in self.claim_to_image:
                self.train_image_ids.add(self.claim_to_image[c])
        for c in self.val_ids:
            if c in self.claim_to_image:
                self.val_image_ids.add(self.claim_to_image[c])
        for c in self.cal_ids:
            if c in self.claim_to_image:
                self.cal_image_ids.add(self.claim_to_image[c])
        for c in self.test_ids:
            if c in self.claim_to_image:
                self.test_image_ids.add(self.claim_to_image[c])

        self.validate_isolation()

    @property
    def validation_ids(self) -> Set[str]:
        return self.val_ids

    @property
    def calibration_ids(self) -> Set[str]:
        return self.cal_ids

    @property
    def train_hash(self) -> str:
        return self.get_hash(self.train_ids)

    @property
    def validation_hash(self) -> str:
        return self.get_hash(self.val_ids)

    @property
    def calibration_hash(self) -> str:
        return self.get_hash(self.cal_ids)

    @property
    def test_hash(self) -> str:
        return self.get_hash(self.test_ids)

    @property
    def train_image_hash(self) -> str:
        return self.get_hash(self.train_image_ids)

    @property
    def test_image_hash(self) -> str:
        return self.get_hash(self.test_image_ids)

    def get_hash(self, id_collection: Iterable[str]) -> str:
        """Deterministic SHA-256 fingerprint of an ID collection."""
        sorted_list = sorted(list(id_collection))
        return hashlib.sha256(json.dumps(sorted_list).encode("utf-8")).hexdigest()

    def compute_split_hash(self, ids: Set[str]) -> str:
        return self.get_hash(ids)

    def validate_isolation(self) -> None:
        """Verify that all splits are mutually disjoint at both claim and image levels."""
        # 1. Claim-level disjointness
        claim_splits = {
            "TRAIN": self.train_ids,
            "VALIDATION": self.val_ids,
            "CALIBRATION": self.cal_ids,
            "TEST": self.test_ids,
        }
        names = list(claim_splits.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                s1, s2 = claim_splits[n1], claim_splits[n2]
                overlap = s1.intersection(s2)
                if overlap:
                    raise SplitLeakageError(
                        f"Leakage detected between {n1} and {n2}! Overlapping IDs: {sorted(list(overlap))[:5]}"
                    )

        # 2. Image-level disjointness
        image_splits = {
            "TRAIN": self.train_image_ids,
            "VALIDATION": self.val_image_ids,
            "CALIBRATION": self.cal_image_ids,
            "TEST": self.test_image_ids,
        }
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                s1, s2 = image_splits[n1], image_splits[n2]
                overlap = s1.intersection(s2)
                if overlap:
                    raise SplitLeakageError(
                        f"Image-level leakage detected: images {sorted(list(overlap))[:5]} "
                        f"assigned across multiple splits ({n1} and {n2})!"
                    )

    def assert_no_leakage(
        self,
        records: List[Dict[str, Any]],
        role: Union[str, SplitRole] = SplitRole.TRAIN,
    ) -> None:
        """Assert that no TEST claim IDs or TEST image IDs are present in candidate records."""
        role_name = role.value if isinstance(role, SplitRole) else str(role)

        # Claim ID check
        rec_ids = {r.get("claim_id", r.get("id", "")) for r in records if r.get("claim_id") or r.get("id")}
        leaked_claims = rec_ids.intersection(self.test_ids)
        if leaked_claims:
            raise SplitLeakageError(
                f"TEST IDs found in {role_name} records! Leaked claims: {sorted(list(leaked_claims))[:5]}"
            )

        # Image ID check
        rec_images = {
            r.get("image_id", r.get("identity_group", r.get("image_hash", "")))
            for r in records
            if r.get("image_id") or r.get("identity_group") or r.get("image_hash")
        }
        rec_images.discard("")
        leaked_images = rec_images.intersection(self.test_image_ids)
        if leaked_images:
            raise SplitLeakageError(
                f"TEST image IDs found in {role_name} records! Leaked images: {sorted(list(leaked_images))[:5]}"
            )

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Summary of split sizes and cryptographic fingerprints."""
        self.validate_isolation()
        return {
            "num_train_claims": len(self.train_ids),
            "num_validation_claims": len(self.val_ids),
            "num_calibration_claims": len(self.cal_ids),
            "num_test_claims": len(self.test_ids),
            "num_train_images": len(self.train_image_ids),
            "num_validation_images": len(self.val_image_ids),
            "num_calibration_images": len(self.cal_image_ids),
            "num_test_images": len(self.test_image_ids),
            "train_hash": self.train_hash,
            "validation_hash": self.validation_hash,
            "calibration_hash": self.calibration_hash,
            "test_hash": self.test_hash,
            "train_image_hash": self.train_image_hash,
            "test_image_hash": self.test_image_hash,
            "image_isolation_verified": True,
            "test_isolation_verified": True,
        }


def create_deterministic_splits(
    ids: List[str],
    train_ratio: float = 0.40,
    val_ratio: float = 0.20,
    cal_ratio: float = 0.20,
    salt: str = "robust_bp_m9b_seed_42",
) -> SplitContract:
    """Deterministically partition a list of unique claim IDs."""
    sorted_ids = sorted(list(set(ids)))
    n = len(sorted_ids)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_cal = int(round(n * cal_ratio))

    train_ids = set(sorted_ids[:n_train])
    val_ids = set(sorted_ids[n_train : n_train + n_val])
    cal_ids = set(sorted_ids[n_train + n_val : n_train + n_val + n_cal])
    test_ids = set(sorted_ids[n_train + n_val + n_cal :])

    return SplitContract(
        train_ids=train_ids,
        val_ids=val_ids,
        cal_ids=cal_ids,
        test_ids=test_ids,
    )


def create_image_level_deterministic_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.40,
    val_ratio: float = 0.20,
    cal_ratio: float = 0.20,
    salt: str = "robust_bp_m9c_image_seed_42",
) -> SplitContract:
    """Deterministically partition records at the IMAGE / IDENTITY-GROUP level.

    Guarantees:
    - All claims from the same image belong to the exact same split.
    - Zero image cross-contamination across TRAIN, VALIDATION, CALIBRATION, TEST.
    """
    image_to_claims: Dict[str, List[str]] = {}
    claim_to_image: Dict[str, str] = {}

    for r in records:
        cid = r.get("claim_id", r.get("id", ""))
        img_id = r.get("image_id", r.get("identity_group", r.get("image_hash", "")))
        if not cid or not img_id:
            continue
        image_to_claims.setdefault(img_id, []).append(cid)
        claim_to_image[cid] = img_id

    distinct_images = sorted(list(image_to_claims.keys()))
    n_img = len(distinct_images)

    n_tr = int(round(n_img * train_ratio))
    n_va = int(round(n_img * val_ratio))
    n_ca = int(round(n_img * cal_ratio))

    train_imgs = set(distinct_images[:n_tr])
    val_imgs = set(distinct_images[n_tr : n_tr + n_va])
    cal_imgs = set(distinct_images[n_tr + n_va : n_tr + n_va + n_ca])
    test_imgs = set(distinct_images[n_tr + n_va + n_ca :])

    train_claims: Set[str] = set()
    val_claims: Set[str] = set()
    cal_claims: Set[str] = set()
    test_claims: Set[str] = set()

    for img in train_imgs:
        train_claims.update(image_to_claims[img])
    for img in val_imgs:
        val_claims.update(image_to_claims[img])
    for img in cal_imgs:
        cal_claims.update(image_to_claims[img])
    for img in test_imgs:
        test_claims.update(image_to_claims[img])

    return SplitContract(
        train_ids=train_claims,
        val_ids=val_claims,
        cal_ids=cal_claims,
        test_ids=test_claims,
        claim_to_image=claim_to_image,
        train_image_ids=train_imgs,
        val_image_ids=val_imgs,
        cal_image_ids=cal_imgs,
        test_image_ids=test_imgs,
    )


def partition_records_by_split(records: List[Any]) -> Dict[str, List[Any]]:
    """Partition claim or image records into standard split buckets."""
    splits: Dict[str, List[Any]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
    for rec in records:
        sp = getattr(rec, "split", None) if not isinstance(rec, dict) else rec.get("split")
        if isinstance(sp, str):
            sp_clean = sp.lower()
            if sp_clean in ["val", "validation"]:
                splits["validation"].append(rec)
            elif sp_clean in ["cal", "calibration"]:
                splits["calibration"].append(rec)
            elif sp_clean in ["train"]:
                splits["train"].append(rec)
            elif sp_clean in ["test"]:
                splits["test"].append(rec)
            else:
                splits["train"].append(rec)
        else:
            splits["train"].append(rec)

    return splits
