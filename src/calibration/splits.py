"""Strict data split contracts and leakage prevention guards for Phase 9B parameterization.

Roles:
- TRAIN: Fit evidence-to-probability feature models.
- VALIDATION: Choose regularization, model family, coupling strength lambda, and topology.
- CALIBRATION: Calibrate probabilities (Platt / Isotonic) and estimate perturbation bounds (epsilon, B).
- TEST: Final benchmark evaluation only. ZERO parameters or thresholds may be derived from test.
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
    """Raised when test data or test IDs contaminate parameter fitting or splits overlap."""

    pass


@dataclass
class SplitContract:
    """Manages partition sets and enforces strict split isolation."""

    train_ids: Set[str] = field(default_factory=set)
    val_ids: Set[str] = field(default_factory=set)
    cal_ids: Set[str] = field(default_factory=set)
    test_ids: Set[str] = field(default_factory=set)

    def __init__(
        self,
        train_ids: Optional[Iterable[str]] = None,
        val_ids: Optional[Iterable[str]] = None,
        cal_ids: Optional[Iterable[str]] = None,
        test_ids: Optional[Iterable[str]] = None,
        validation_ids: Optional[Iterable[str]] = None,
        calibration_ids: Optional[Iterable[str]] = None,
    ) -> None:
        self.train_ids = set(train_ids or [])
        self.val_ids = set(val_ids or validation_ids or [])
        self.cal_ids = set(cal_ids or calibration_ids or [])
        self.test_ids = set(test_ids or [])
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

    def get_hash(self, id_collection: Iterable[str]) -> str:
        """Deterministic SHA-256 fingerprint of an ID collection."""
        sorted_list = sorted(list(id_collection))
        return hashlib.sha256(json.dumps(sorted_list).encode("utf-8")).hexdigest()

    def compute_split_hash(self, ids: Set[str]) -> str:
        return self.get_hash(ids)

    def validate_isolation(self) -> None:
        """Verify that all splits are mutually disjoint, and test IDs are strictly isolated."""
        splits = {
            "TRAIN": self.train_ids,
            "VALIDATION": self.val_ids,
            "CALIBRATION": self.cal_ids,
            "TEST": self.test_ids,
        }
        names = list(splits.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                s1, s2 = splits[n1], splits[n2]
                overlap = s1.intersection(s2)
                if overlap:
                    raise SplitLeakageError(
                        f"Leakage detected between {n1} and {n2}! Overlapping IDs: {sorted(list(overlap))[:5]}"
                    )

    def assert_no_leakage(
        self,
        records: List[Dict[str, Any]],
        role: Union[str, SplitRole] = SplitRole.TRAIN,
    ) -> None:
        """Assert that no TEST IDs are present in candidate records."""
        role_name = role.value if isinstance(role, SplitRole) else str(role)
        rec_ids = {r.get("claim_id", r.get("id", "")) for r in records}
        leaked = rec_ids.intersection(self.test_ids)
        if leaked:
            raise SplitLeakageError(
                f"TEST IDs found in {role_name} records! Leaked: {sorted(list(leaked))[:5]}"
            )

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Summary of split sizes and cryptographic fingerprints."""
        self.validate_isolation()
        return {
            "num_train": len(self.train_ids),
            "num_validation": len(self.val_ids),
            "num_calibration": len(self.cal_ids),
            "num_test": len(self.test_ids),
            "train_hash": self.train_hash,
            "validation_hash": self.validation_hash,
            "calibration_hash": self.calibration_hash,
            "test_hash": self.test_hash,
            "test_isolation_verified": True,
        }


def create_deterministic_splits(
    ids: List[str],
    train_ratio: float = 0.40,
    val_ratio: float = 0.20,
    cal_ratio: float = 0.20,
    salt: str = "robust_bp_m9b_seed_42",
) -> SplitContract:
    """Deterministically partition a list of unique IDs using hash assignment.

    The remainder (1 - train_ratio - val_ratio - cal_ratio) goes to TEST.
    """
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
