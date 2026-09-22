"""
Final Dataset Gate for Milestone 9.

Enforces strict integrity, security, and completeness criteria before
allowing final scientific evaluation on the benchmark dataset.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple

logger = logging.getLogger("m9_gate")


class GateStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass
class GateCheckResult:
    """Detailed result of the FinalDatasetGate inspection."""
    passed: bool
    status: GateStatus
    checks: Dict[str, bool]
    diagnostics: List[str]
    image_count: int
    claim_count: int
    resolved_ground_truth_count: int
    is_dataset_locked: bool
    lock_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinalDatasetGate:
    """
    Gatekeeper that validates whether the benchmark dataset satisfies
    all strict scientific requirements for FINAL publication evaluation.
    """

    def __init__(
        self,
        manifest_path: str | Path = "data/manifests/m7_image_manifest.json",
        claims_path: str | Path = "data/exports/m7_claims.jsonl",
        ground_truth_path: str | Path = "data/exports/m7_ground_truth.jsonl",
        dataset_lock_path: str | Path = "data/manifests/m7_dataset_lock.json",
        target_image_count: int = 600,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.claims_path = Path(claims_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.dataset_lock_path = Path(dataset_lock_path)
        self.target_image_count = target_image_count

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Compute standard SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def verify(self, strict_fail: bool = False) -> GateCheckResult:
        """
        Perform all 14 mandatory validation checks for the final dataset.

        Args:
            strict_fail: If True, raises RuntimeError when checks fail.

        Returns:
            GateCheckResult with fine-grained pass/fail status and diagnostics.
        """
        checks: Dict[str, bool] = {}
        diagnostics: List[str] = []

        # 1. Dataset lock file existence
        lock_exists = self.dataset_lock_path.exists()
        checks["lock_file_exists"] = lock_exists
        if not lock_exists:
            diagnostics.append(f"Missing dataset lock file at: {self.dataset_lock_path}")

        lock_data: Dict[str, Any] = {}
        lock_status = "MISSING"
        if lock_exists:
            try:
                with open(self.dataset_lock_path, "r", encoding="utf-8") as f:
                    lock_data = json.load(f)
                lock_status = lock_data.get("status", "UNKNOWN")
            except Exception as e:
                diagnostics.append(f"Failed to read dataset lock file: {e}")

        # 2. Lock status strictly LOCKED
        is_locked = (lock_status == "LOCKED")
        checks["is_dataset_locked"] = is_locked
        if not is_locked:
            diagnostics.append(
                f"Dataset lock status is '{lock_status}', expected 'LOCKED'. "
                f"Reason from lock: {lock_data.get('status_reason', 'None')}"
            )

        # 3. File existence
        manifest_exists = self.manifest_path.exists()
        claims_exist = self.claims_path.exists()
        gt_exists = self.ground_truth_path.exists()

        checks["manifest_file_exists"] = manifest_exists
        checks["claims_file_exists"] = claims_exist
        checks["ground_truth_file_exists"] = gt_exists

        if not manifest_exists:
            diagnostics.append(f"Missing image manifest: {self.manifest_path}")
        if not claims_exist:
            diagnostics.append(f"Missing claims file: {self.claims_path}")
        if not gt_exists:
            diagnostics.append(f"Missing ground truth file: {self.ground_truth_path}")

        # If files are missing, fail early
        if not (manifest_exists and claims_exist and gt_exists):
            result = GateCheckResult(
                passed=False,
                status=GateStatus.NOT_AVAILABLE if not lock_exists else GateStatus.FAILED,
                checks=checks,
                diagnostics=diagnostics,
                image_count=0,
                claim_count=0,
                resolved_ground_truth_count=0,
                is_dataset_locked=is_locked,
                lock_status=lock_status,
            )
            if strict_fail:
                raise RuntimeError(f"FinalDatasetGate failed: {'; '.join(diagnostics)}")
            return result

        # 4. Checksum verification against lock
        checksums = lock_data.get("checksums", {})
        man_hash = self.compute_sha256(self.manifest_path)
        claims_hash = self.compute_sha256(self.claims_path)
        gt_hash = self.compute_sha256(self.ground_truth_path)

        expected_man = checksums.get("m7_manifest_sha256")
        expected_claims = checksums.get("m7_claims_sha256")
        expected_gt = checksums.get("m7_ground_truth_sha256")

        checks["manifest_checksum_match"] = (expected_man is not None and man_hash == expected_man)
        checks["claims_checksum_match"] = (expected_claims is not None and claims_hash == expected_claims)
        checks["ground_truth_checksum_match"] = (expected_gt is not None and gt_hash == expected_gt)

        if expected_man and man_hash != expected_man:
            diagnostics.append(f"Manifest checksum mismatch: {man_hash} != {expected_man}")
        if expected_claims and claims_hash != expected_claims:
            diagnostics.append(f"Claims checksum mismatch: {claims_hash} != {expected_claims}")
        if expected_gt and gt_hash != expected_gt:
            diagnostics.append(f"Ground truth checksum mismatch: {gt_hash} != {expected_gt}")

        # 5. Load manifest and verify image count and split isolation
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest_json = json.load(f)

        entries = manifest_json.get("entries", [])
        image_splits: Dict[str, str] = {}
        for e in entries:
            img_id = e.get("image", {}).get("image_id")
            s_val = e.get("split", "train")
            if img_id:
                image_splits[img_id] = s_val.lower()

        actual_image_count = len(image_splits)
        has_target_size = (actual_image_count >= self.target_image_count)
        checks["has_target_image_size"] = has_target_size
        if not has_target_size:
            diagnostics.append(
                f"Manifest contains {actual_image_count} images, below target of {self.target_image_count}."
            )

        # 6. Load claims and ground truth
        claims: List[Dict[str, Any]] = []
        with open(self.claims_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    claims.append(json.loads(line))

        gt_records: Dict[str, Dict[str, Any]] = {}
        with open(self.ground_truth_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    gt_records[d.get("claim_id")] = d

        # 7. Check uniqueness of claim IDs
        claim_ids: Set[str] = set()
        has_duplicates = False
        for c in claims:
            cid = c.get("claim_id")
            if cid in claim_ids:
                has_duplicates = True
                diagnostics.append(f"Duplicate claim_id detected: '{cid}'")
            claim_ids.add(cid)
        checks["unique_claim_ids"] = not has_duplicates

        # 8. Check split leakage
        has_split_leakage = False
        for c in claims:
            img_id = c.get("image_id")
            c_split = c.get("split", "").lower()
            m_split = image_splits.get(img_id, "")
            if m_split and c_split != m_split:
                has_split_leakage = True
                diagnostics.append(
                    f"Split mismatch for image '{img_id}': claim={c_split}, manifest={m_split}"
                )
        checks["zero_split_leakage"] = not has_split_leakage

        # 9. Non-synthetic data check
        has_synthetic = False
        for c in claims:
            if c.get("is_synthetic", False) or c.get("evidence", {}).get("is_synthetic", False):
                has_synthetic = True
                diagnostics.append(f"Synthetic record found in candidate final data: '{c.get('claim_id')}'")
                break
        checks["zero_synthetic_data"] = not has_synthetic

        # 10. Annotation completeness & dispute resolution
        resolved_gt_count = 0
        has_unresolved = False
        for cid, gt in gt_records.items():
            final_status = gt.get("final_ground_truth")
            has_dis = gt.get("has_disagreement", False)
            adj = gt.get("adjudication")
            if final_status is not None and final_status in ["supported", "hallucinated", "unknown"]:
                resolved_gt_count += 1
            if has_dis and adj is None:
                has_unresolved = True

        checks["all_annotations_resolved"] = (resolved_gt_count == len(claims) and len(claims) > 0)
        checks["zero_unresolved_disputes"] = not has_unresolved

        if resolved_gt_count < len(claims):
            diagnostics.append(
                f"Incomplete annotation: {resolved_gt_count}/{len(claims)} claims have resolved ground truth."
            )
        if has_unresolved:
            diagnostics.append("Unresolved inter-annotator disputes detected in ground truth dataset.")

        # 11. UNKNOWN semantics preservation
        unknown_semantics_valid = True
        for cid, gt in gt_records.items():
            final_status = gt.get("final_ground_truth")
            if final_status == "unknown":
                pass  # Strictly valid
            elif final_status not in ["supported", "hallucinated", None]:
                unknown_semantics_valid = False
                diagnostics.append(f"Invalid ground truth status value: '{final_status}' on claim '{cid}'")
        checks["valid_ground_truth_semantics"] = unknown_semantics_valid

        # Overall verdict
        all_passed = all(checks.values())
        if not lock_exists:
            overall_status = GateStatus.NOT_AVAILABLE
        elif all_passed:
            overall_status = GateStatus.PASSED
        else:
            overall_status = GateStatus.FAILED

        result = GateCheckResult(
            passed=all_passed,
            status=overall_status,
            checks=checks,
            diagnostics=diagnostics,
            image_count=actual_image_count,
            claim_count=len(claims),
            resolved_ground_truth_count=resolved_gt_count,
            is_dataset_locked=is_locked,
            lock_status=lock_status,
            metadata={
                "manifest_sha256": man_hash,
                "claims_sha256": claims_hash,
                "ground_truth_sha256": gt_hash,
                "target_image_count": self.target_image_count,
            },
        )

        if strict_fail and not all_passed:
            diag_str = "\n  - ".join(diagnostics)
            raise RuntimeError(f"FinalDatasetGate failed {len(diagnostics)} checks:\n  - {diag_str}")

        return result
