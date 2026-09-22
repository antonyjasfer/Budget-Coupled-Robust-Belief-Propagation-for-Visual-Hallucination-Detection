"""
Data Provenance and Isolation Enforcements for Visual Hallucination Benchmark.

Defines:
1. Seven-level data provenance taxonomy.
2. Evidence failure states and typed failure records.
3. Hard mock/pseudo-label isolation validators for DEVELOPMENT vs FINAL modes.
4. Predeclared evidence failure policies preventing default numeric-zero substitution.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Set


class DataProvenanceState(str, Enum):
    """
    Seven-level canonical data provenance taxonomy.
    """
    REAL_UNLABELED = "real_unlabeled"
    REAL_HUMAN_ANNOTATED = "real_human_annotated"
    REAL_ADJUDICATED = "real_adjudicated"
    SYNTHETIC_FIXTURE = "synthetic_fixture"
    MOCK_ANNOTATION = "mock_annotation"
    PSEUDO_LABEL = "pseudo_label"
    DEVELOPMENT_ONLY = "development_only"


class EvidenceFailureState(str, Enum):
    """
    Predeclared states for multi-modal evidence extraction attempts.
    """
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    INVALID_PROVENANCE = "invalid_provenance"


class ValidationMode(str, Enum):
    """
    Validation gate modes.
    """
    DEVELOPMENT = "development"
    FINAL = "final"


FORBIDDEN_FINAL_PROVENANCES: Set[DataProvenanceState] = {
    DataProvenanceState.SYNTHETIC_FIXTURE,
    DataProvenanceState.MOCK_ANNOTATION,
    DataProvenanceState.PSEUDO_LABEL,
    DataProvenanceState.DEVELOPMENT_ONLY,
}


@dataclass
class EvidenceFailureRecord:
    """
    Audit record for unavailable or failed evidence extraction.
    Ensures missing evidence is never silently substituted with numeric zero.
    """
    claim_id: str
    provider: str
    failure_state: EvidenceFailureState
    reason_code: str
    attempt_count: int = 1
    last_error_class: Optional[str] = None
    provenance_status: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "provider": self.provider,
            "failure_state": self.failure_state.value,
            "reason_code": self.reason_code,
            "attempt_count": self.attempt_count,
            "last_error_class": self.last_error_class,
            "provenance_status": self.provenance_status,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceFailureRecord":
        return cls(
            claim_id=data["claim_id"],
            provider=data["provider"],
            failure_state=EvidenceFailureState(data["failure_state"]),
            reason_code=data["reason_code"],
            attempt_count=data.get("attempt_count", 1),
            last_error_class=data.get("last_error_class"),
            provenance_status=data.get("provenance_status"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ProvenanceAuditResult:
    """
    Summary of provenance audit across a collection of records.
    """
    mode: ValidationMode
    is_valid: bool
    total_records: int
    counts_by_provenance: Dict[str, int]
    violations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "is_valid": self.is_valid,
            "total_records": self.total_records,
            "counts_by_provenance": self.counts_by_provenance,
            "violations": self.violations,
            "details": self.details,
        }


def parse_provenance_state(value: Union[str, DataProvenanceState]) -> DataProvenanceState:
    """
    Safely parse raw input into canonical DataProvenanceState enum.
    """
    if isinstance(value, DataProvenanceState):
        return value
    if isinstance(value, str):
        cleaned = value.strip().lower()
        for state in DataProvenanceState:
            if state.value == cleaned:
                return state
    raise ValueError(f"Unknown data provenance state: '{value}'. Expected one of: {[s.value for s in DataProvenanceState]}")


def validate_provenance_isolation(
    records: List[Any],
    mode: Union[ValidationMode, str] = ValidationMode.DEVELOPMENT,
    allow_unlabeled_in_final: bool = False,
) -> ProvenanceAuditResult:
    """
    Enforce mock/pseudo-label isolation rules based on validation mode.

    Rules:
    - In DEVELOPMENT mode:
        All provenance states are permitted. Mock and synthetic records are logged and counted.
    - In FINAL mode:
        1. SYNTHETIC_FIXTURE, MOCK_ANNOTATION, PSEUDO_LABEL, and DEVELOPMENT_ONLY are STRICTLY FORBIDDEN.
        2. REAL_UNLABELED is rejected unless allow_unlabeled_in_final is explicitly True.
        3. Only REAL_HUMAN_ANNOTATED and REAL_ADJUDICATED are accepted as valid final ground truth.

    Args:
        records: List of objects (dicts or dataclasses) possessing a provenance attribute or dict key.
        mode: ValidationMode.DEVELOPMENT or ValidationMode.FINAL.
        allow_unlabeled_in_final: Whether unresolved REAL_UNLABELED is tolerated in final mode.

    Returns:
        ProvenanceAuditResult with detailed counts and violation messages.
    """
    if isinstance(mode, str):
        mode = ValidationMode(mode.strip().lower())

    counts: Dict[str, int] = {s.value: 0 for s in DataProvenanceState}
    violations: List[str] = []

    for idx, rec in enumerate(records):
        # Extract provenance attribute or key
        raw_prov = None
        if isinstance(rec, dict):
            raw_prov = rec.get("provenance") or rec.get("provenance_state") or rec.get("data_provenance")
            # Check nested metadata if not found at root
            if raw_prov is None and "metadata" in rec and isinstance(rec["metadata"], dict):
                raw_prov = rec["metadata"].get("provenance") or rec["metadata"].get("provenance_state")
        else:
            raw_prov = getattr(rec, "provenance", None) or getattr(rec, "provenance_state", None)
            if raw_prov is None and hasattr(rec, "metadata") and isinstance(rec.metadata, dict):
                raw_prov = rec.metadata.get("provenance") or rec.metadata.get("provenance_state")

        if raw_prov is None:
            # If completely missing provenance tag
            violations.append(f"Record at index {idx} has missing provenance metadata.")
            continue

        try:
            prov_state = parse_provenance_state(raw_prov)
            counts[prov_state.value] += 1
        except ValueError as err:
            violations.append(f"Record at index {idx} has invalid provenance: {err}")
            continue

        if mode == ValidationMode.FINAL:
            # Check forbidden states
            if prov_state in FORBIDDEN_FINAL_PROVENANCES:
                violations.append(
                    f"FINAL MODE VIOLATION: Record index {idx} contains forbidden provenance '{prov_state.value}'."
                )
            elif prov_state == DataProvenanceState.REAL_UNLABELED and not allow_unlabeled_in_final:
                violations.append(
                    f"FINAL MODE VIOLATION: Record index {idx} contains unresolved 'real_unlabeled' claim in final benchmark."
                )

    is_valid = len(violations) == 0
    return ProvenanceAuditResult(
        mode=mode,
        is_valid=is_valid,
        total_records=len(records),
        counts_by_provenance=counts,
        violations=violations,
    )


def validate_evidence_status(
    record: Dict[str, Any],
    required_providers: Optional[List[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that multi-modal evidence records adhere to the predeclared failure policy.
    Ensures missing evidence is never substituted with numeric zero.

    Args:
        record: Evidence record dictionary.
        required_providers: List of evidence providers required to be AVAILABLE (e.g. ['owl_vit', 'clip']).

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    if required_providers is None:
        required_providers = ["owl_vit", "clip"]

    issues: List[str] = []
    
    # Check detector evidence
    det_score = record.get("detector_score")
    det_avail = record.get("detector_available", False)
    if not det_avail and det_score is not None:
        if det_score == 0.0:
            issues.append("ZeroSubstitutionViolation: detector_score is 0.0 while detector_available is False. Missing evidence must be None.")
    
    # Check CLIP evidence
    clip_score = record.get("clip_score")
    clip_avail = record.get("similarity_available", False)
    if not clip_avail and clip_score is not None:
        if clip_score == 0.0:
            issues.append("ZeroSubstitutionViolation: clip_score is 0.0 while similarity_available is False. Missing evidence must be None.")

    # Check required providers
    for prov in required_providers:
        if prov == "owl_vit" and not det_avail:
            issues.append("ProviderMissing: required evidence provider 'owl_vit' is unavailable.")
        elif prov == "clip" and not clip_avail:
            issues.append("ProviderMissing: required evidence provider 'clip' is unavailable.")

    return len(issues) == 0, issues
