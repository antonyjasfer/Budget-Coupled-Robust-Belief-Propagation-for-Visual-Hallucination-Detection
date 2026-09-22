"""Parameter bundle representation and persistence for robust Ising hallucination detection.

Stores fitted/calibrated parameter configurations, feature extractors, uncertainty set
bounds, coupling parameters, and strict provenance hashes (train, validation, calibration,
dataset, code SHA) to prevent leakage and guarantee auditability.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParameterBundle:
    """Versioned parameter bundle containing all calibration artifacts and provenance."""

    version: str = "1.0.0"
    mode: str = "DEVELOPMENT"  # "DEVELOPMENT" or "FINAL"
    theta_model: Dict[str, Any] = field(default_factory=dict)
    probability_calibration: Dict[str, Any] = field(default_factory=dict)
    epsilon_model: Dict[str, Any] = field(default_factory=dict)
    budget_model: Dict[str, Any] = field(default_factory=dict)
    coupling_model: Dict[str, Any] = field(default_factory=dict)
    topology_model: Dict[str, Any] = field(default_factory=dict)

    # Provenance metadata
    train_ids_hash: str = ""
    validation_ids_hash: str = ""
    calibration_ids_hash: str = ""
    test_ids_hash: str = ""  # For record keeping only - must NOT match train/val/cal
    dataset_hash: str = ""
    code_sha: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("DEVELOPMENT", "FINAL"):
            raise ValueError(f"Mode must be 'DEVELOPMENT' or 'FINAL', got: {self.mode}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert bundle to serializable dictionary."""
        return asdict(self)

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of bundle contents (excluding any pre-existing checksum)."""
        d = self.to_dict()
        d.pop("checksum", None)
        canonical_json = json.dumps(d, sort_keys=True, indent=2)
        return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> str:
        """Save bundle to JSON file with embedded SHA-256 checksum."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        checksum = self.compute_checksum()
        data["checksum"] = checksum
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return checksum

    @classmethod
    def load(cls, path: str | Path, verify_checksum: bool = True) -> ParameterBundle:
        """Load bundle from JSON file and optionally verify SHA-256 checksum."""
        target_path = Path(path)
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stored_checksum = data.pop("checksum", None)
        bundle = cls(**data)

        if verify_checksum and stored_checksum is not None:
            computed = bundle.compute_checksum()
            if computed != stored_checksum:
                raise ValueError(
                    f"ParameterBundle checksum mismatch! Stored: {stored_checksum}, "
                    f"Computed: {computed}. The file may have been modified or corrupted."
                )

        return bundle

    def is_scientifically_final(self) -> bool:
        """Check if bundle represents a certified final calibration."""
        return self.mode == "FINAL"
