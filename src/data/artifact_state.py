"""
Artifact State Machine for Phase 10A-R2 Evidence Acquisition Pipeline.

Defines explicit, typed artifact lifecycle states to distinguish:
- CPU-placeholder artifacts from GPU-finalized artifacts
- Pre-GPU execution states from post-GPU completion states
- Intermediate checkpoints from final science-ready manifests

State Transitions (valid):
    SAMPLING_FROZEN  →  GPU_PENDING  →  GPU_IN_PROGRESS  →  GPU_COMPLETE  →  EVIDENCE_FROZEN
                                     →  GPU_PARTIAL      →  GPU_IN_PROGRESS (resume)
                                     →  GPU_FAILED       →  GPU_IN_PROGRESS (retry)

    EVIDENCE_FROZEN  →  TASKS_POPULATED  →  PRE_ANNOTATION_SEALED

Invalid state transitions raise ArtifactStateError.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
import hashlib
import json


class ArtifactState(str, Enum):
    """Canonical lifecycle states for the Phase 10A-R2 evidence acquisition pipeline."""

    # Phase 1: Sampling complete, images verified, ready for GPU
    SAMPLING_FROZEN = "SAMPLING_FROZEN"

    # Phase 2: GPU execution has been requested but not started
    GPU_PENDING = "GPU_PENDING"

    # Phase 3: GPU inference actively running (checkpoint exists)
    GPU_IN_PROGRESS = "GPU_IN_PROGRESS"

    # Phase 3b: GPU inference partially completed (interrupted/resumable)
    GPU_PARTIAL = "GPU_PARTIAL"

    # Phase 3c: GPU inference failed (retryable)
    GPU_FAILED = "GPU_FAILED"

    # Phase 4: All 600 images processed by GPU, evidence manifest finalized
    GPU_COMPLETE = "GPU_COMPLETE"

    # Phase 5: Evidence manifest sealed with cryptographic hash
    EVIDENCE_FROZEN = "EVIDENCE_FROZEN"

    # Phase 6: Annotation tasks populated with N > 0 real claims
    TASKS_POPULATED = "TASKS_POPULATED"

    # Phase 7: Pre-annotation freeze sealed referencing all upstream hashes
    PRE_ANNOTATION_SEALED = "PRE_ANNOTATION_SEALED"


# Valid state transitions
VALID_TRANSITIONS: Dict[ArtifactState, Set[ArtifactState]] = {
    ArtifactState.SAMPLING_FROZEN: {ArtifactState.GPU_PENDING},
    ArtifactState.GPU_PENDING: {ArtifactState.GPU_IN_PROGRESS},
    ArtifactState.GPU_IN_PROGRESS: {
        ArtifactState.GPU_COMPLETE,
        ArtifactState.GPU_PARTIAL,
        ArtifactState.GPU_FAILED,
    },
    ArtifactState.GPU_PARTIAL: {ArtifactState.GPU_IN_PROGRESS},
    ArtifactState.GPU_FAILED: {ArtifactState.GPU_IN_PROGRESS},
    ArtifactState.GPU_COMPLETE: {ArtifactState.EVIDENCE_FROZEN},
    ArtifactState.EVIDENCE_FROZEN: {ArtifactState.TASKS_POPULATED},
    ArtifactState.TASKS_POPULATED: {ArtifactState.PRE_ANNOTATION_SEALED},
    ArtifactState.PRE_ANNOTATION_SEALED: set(),  # Terminal state
}


class ArtifactStateError(Exception):
    """Raised on invalid state transitions or state contract violations."""

    def __init__(self, message: str, current_state: Optional[ArtifactState] = None,
                 requested_state: Optional[ArtifactState] = None):
        self.current_state = current_state
        self.requested_state = requested_state
        super().__init__(message)


@dataclass
class ArtifactStateRecord:
    """Audit log entry for a single state transition."""
    from_state: Optional[str]
    to_state: str
    timestamp: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUAcquisitionCheckpoint:
    """
    Intermediate checkpoint for GPU evidence acquisition.

    DISTINCT from final_evidence_manifest: this is a resumable execution
    checkpoint, not a sealed science artifact. Contains mutable execution
    state that is NOT valid for downstream scientific use.
    """
    schema_version: str = "2.0.0"
    checkpoint_type: str = "gpu_acquisition_checkpoint"
    state: str = ArtifactState.GPU_PENDING.value
    sampling_manifest_hash: str = ""
    total_target_images: int = 600
    completed_images: int = 0
    failed_images: int = 0
    genuine_zero_claim_images: int = 0
    completed_image_ids: List[str] = field(default_factory=list)
    failed_image_ids: List[str] = field(default_factory=list)
    evidence_records: List[Dict[str, Any]] = field(default_factory=list)
    failure_records: List[Dict[str, Any]] = field(default_factory=list)
    execution_environment: Dict[str, Any] = field(default_factory=dict)
    last_checkpoint_timestamp: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checkpoint_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 of checkpoint content (excluding hash field itself)."""
        d = self.to_dict()
        d.pop("checkpoint_hash", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_type": self.checkpoint_type,
            "state": self.state,
            "sampling_manifest_hash": self.sampling_manifest_hash,
            "total_target_images": self.total_target_images,
            "completed_images": self.completed_images,
            "failed_images": self.failed_images,
            "genuine_zero_claim_images": self.genuine_zero_claim_images,
            "completed_image_ids": self.completed_image_ids,
            "failed_image_ids": self.failed_image_ids,
            "evidence_records": self.evidence_records,
            "failure_records": self.failure_records,
            "execution_environment": self.execution_environment,
            "last_checkpoint_timestamp": self.last_checkpoint_timestamp,
            "created_at": self.created_at,
            "checkpoint_hash": self.checkpoint_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPUAcquisitionCheckpoint":
        return cls(
            schema_version=data.get("schema_version", "2.0.0"),
            checkpoint_type=data.get("checkpoint_type", "gpu_acquisition_checkpoint"),
            state=data.get("state", ArtifactState.GPU_PENDING.value),
            sampling_manifest_hash=data.get("sampling_manifest_hash", ""),
            total_target_images=data.get("total_target_images", 600),
            completed_images=data.get("completed_images", 0),
            failed_images=data.get("failed_images", 0),
            genuine_zero_claim_images=data.get("genuine_zero_claim_images", 0),
            completed_image_ids=data.get("completed_image_ids", []),
            failed_image_ids=data.get("failed_image_ids", []),
            evidence_records=data.get("evidence_records", []),
            failure_records=data.get("failure_records", []),
            execution_environment=data.get("execution_environment", {}),
            last_checkpoint_timestamp=data.get("last_checkpoint_timestamp", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            checkpoint_hash=data.get("checkpoint_hash", ""),
        )


class ArtifactStateMachine:
    """
    Manages artifact lifecycle state with strict transition validation.

    Enforces:
    1. Only valid state transitions are permitted.
    2. GPU checkpoints are explicitly NOT interchangeable with final evidence manifests.
    3. CPU-refusal states cannot advance past GPU_PENDING.
    4. V2 artifacts are incompatible with v1/10A-partial artifacts.
    """

    def __init__(self, initial_state: ArtifactState = ArtifactState.SAMPLING_FROZEN):
        self._state = initial_state
        self._history: List[ArtifactStateRecord] = [
            ArtifactStateRecord(
                from_state=None,
                to_state=initial_state.value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason="State machine initialized",
            )
        ]

    @property
    def state(self) -> ArtifactState:
        return self._state

    @property
    def history(self) -> List[ArtifactStateRecord]:
        return list(self._history)

    def can_transition(self, target: ArtifactState) -> bool:
        """Check whether the given transition is valid without performing it."""
        return target in VALID_TRANSITIONS.get(self._state, set())

    def transition(self, target: ArtifactState, reason: str = "",
                   metadata: Optional[Dict[str, Any]] = None) -> ArtifactState:
        """
        Perform a validated state transition.

        Args:
            target: The target state to transition to.
            reason: Human-readable reason for this transition.
            metadata: Optional metadata to record with the transition.

        Returns:
            The new state after transition.

        Raises:
            ArtifactStateError: If the transition is invalid.
        """
        if not self.can_transition(target):
            valid = VALID_TRANSITIONS.get(self._state, set())
            raise ArtifactStateError(
                f"Invalid state transition: {self._state.value} → {target.value}. "
                f"Valid transitions from {self._state.value}: {[s.value for s in valid]}",
                current_state=self._state,
                requested_state=target,
            )

        old_state = self._state
        self._state = target
        self._history.append(
            ArtifactStateRecord(
                from_state=old_state.value,
                to_state=target.value,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=reason or f"Transition {old_state.value} → {target.value}",
                metadata=metadata or {},
            )
        )
        return self._state

    def require_state(self, *expected: ArtifactState) -> None:
        """Assert the state machine is in one of the expected states."""
        if self._state not in expected:
            raise ArtifactStateError(
                f"Expected state(s) {[s.value for s in expected]}, "
                f"but current state is {self._state.value}",
                current_state=self._state,
            )

    def is_gpu_ready(self) -> bool:
        """Check if the state machine is ready for GPU execution."""
        return self._state in {
            ArtifactState.GPU_PENDING,
            ArtifactState.GPU_PARTIAL,
            ArtifactState.GPU_FAILED,
        }

    def is_evidence_finalized(self) -> bool:
        """Check if evidence acquisition is complete and frozen."""
        return self._state in {
            ArtifactState.EVIDENCE_FROZEN,
            ArtifactState.TASKS_POPULATED,
            ArtifactState.PRE_ANNOTATION_SEALED,
        }


def validate_checkpoint_not_evidence(checkpoint: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a GPU acquisition checkpoint is NOT confused with
    a final evidence manifest.

    Returns:
        (is_valid_checkpoint, issues)
    """
    issues = []

    ctype = checkpoint.get("checkpoint_type")
    if ctype != "gpu_acquisition_checkpoint":
        issues.append(
            f"Expected checkpoint_type='gpu_acquisition_checkpoint', got '{ctype}'. "
            "This may be a final evidence manifest, not a checkpoint."
        )

    state = checkpoint.get("state")
    terminal_states = {ArtifactState.EVIDENCE_FROZEN.value, ArtifactState.TASKS_POPULATED.value,
                       ArtifactState.PRE_ANNOTATION_SEALED.value}
    if state in terminal_states:
        issues.append(
            f"Checkpoint state '{state}' is a terminal evidence state. "
            "Checkpoints must remain in GPU_PENDING/GPU_IN_PROGRESS/GPU_PARTIAL/GPU_COMPLETE."
        )

    # A checkpoint with 0 completed images but claiming GPU_COMPLETE is invalid
    if state == ArtifactState.GPU_COMPLETE.value and checkpoint.get("completed_images", 0) == 0:
        issues.append(
            "Checkpoint claims GPU_COMPLETE but completed_images is 0. "
            "GPU_COMPLETE requires at least one successfully processed image."
        )

    return len(issues) == 0, issues


def validate_version_compatibility(artifact: Dict[str, Any], required_version: str = "v2") -> Tuple[bool, List[str]]:
    """
    Validate that an artifact is version-compatible with the Phase 10A-R2 pipeline.
    V1 and 10A-partial artifacts are explicitly rejected.

    Returns:
        (is_compatible, issues)
    """
    issues = []
    version = artifact.get("dataset_version", "unknown")
    rejected_versions = {"v1", "10A_partial", "10A_recovery"}

    if version in rejected_versions:
        issues.append(
            f"Artifact version '{version}' is incompatible with Phase 10A-R2 pipeline. "
            f"Required version: '{required_version}'."
        )

    if version != required_version and version not in rejected_versions:
        issues.append(
            f"Artifact version '{version}' does not match required '{required_version}'."
        )

    return len(issues) == 0, issues


def validate_annotation_task_readiness(
    tasks: List[Dict[str, Any]],
    require_real_claims: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate annotation task readiness:
    1. N > 0 real claims present.
    2. All label fields are null (masked).
    3. All claims have valid provenance (not synthetic/mock).

    Returns:
        (is_ready, issues)
    """
    issues = []

    if len(tasks) == 0:
        issues.append(
            "ANNOTATION TASK READINESS FAILED: tasks_count is 0. "
            "Cannot begin annotation with zero claims. GPU inference must "
            "complete first to populate task packages with real claims."
        )
        return False, issues

    # Check label masking
    for idx, task in enumerate(tasks):
        label = task.get("expected_label")
        if label is not None:
            issues.append(
                f"LABEL MASKING VIOLATION: Task {idx} ({task.get('task_id', 'unknown')}) "
                f"has expected_label={label!r}, must be null before annotation."
            )

    # Check no synthetic/mock claims leaked
    forbidden_markers = {"SYNTHETIC", "MOCK", "PSEUDO", "DEVELOPMENT_ONLY", "PLACEHOLDER"}
    for idx, task in enumerate(tasks):
        claim_text = str(task.get("claim_surface", "")).upper()
        task_id = str(task.get("task_id", "")).upper()
        for marker in forbidden_markers:
            if marker in claim_text or marker in task_id:
                issues.append(
                    f"PROVENANCE VIOLATION: Task {idx} contains forbidden marker '{marker}' "
                    f"in claim_surface or task_id."
                )

    return len(issues) == 0, issues
