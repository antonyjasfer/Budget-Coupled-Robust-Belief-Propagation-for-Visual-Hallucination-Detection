"""
Regression tests for the Phase 10A-R2 Artifact State Machine.

Tests:
1. Valid state transitions follow the defined graph.
2. Invalid transitions raise ArtifactStateError.
3. GPU checkpoint is not confused with final evidence manifest.
4. Version compatibility rejects v1/10A-partial artifacts.
5. Annotation task readiness validates N > 0, label masking, and provenance.
6. State machine history is correctly recorded.
"""

import pytest

from src.data.artifact_state import (
    ArtifactState,
    ArtifactStateMachine,
    ArtifactStateError,
    GPUAcquisitionCheckpoint,
    VALID_TRANSITIONS,
    validate_checkpoint_not_evidence,
    validate_version_compatibility,
    validate_annotation_task_readiness,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# State Machine Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestArtifactStateMachine:
    """Tests for ArtifactStateMachine lifecycle."""

    def test_initial_state(self):
        sm = ArtifactStateMachine()
        assert sm.state == ArtifactState.SAMPLING_FROZEN

    def test_custom_initial_state(self):
        sm = ArtifactStateMachine(initial_state=ArtifactState.GPU_PENDING)
        assert sm.state == ArtifactState.GPU_PENDING

    def test_valid_full_pipeline_transitions(self):
        """Full pipeline: SAMPLING_FROZEN → ... → PRE_ANNOTATION_SEALED."""
        sm = ArtifactStateMachine()
        sm.transition(ArtifactState.GPU_PENDING, reason="Request GPU run")
        assert sm.state == ArtifactState.GPU_PENDING

        sm.transition(ArtifactState.GPU_IN_PROGRESS, reason="Starting inference")
        assert sm.state == ArtifactState.GPU_IN_PROGRESS

        sm.transition(ArtifactState.GPU_COMPLETE, reason="All images processed")
        assert sm.state == ArtifactState.GPU_COMPLETE

        sm.transition(ArtifactState.EVIDENCE_FROZEN, reason="Evidence manifest sealed")
        assert sm.state == ArtifactState.EVIDENCE_FROZEN

        sm.transition(ArtifactState.TASKS_POPULATED, reason="Task packages populated")
        assert sm.state == ArtifactState.TASKS_POPULATED

        sm.transition(ArtifactState.PRE_ANNOTATION_SEALED, reason="Freeze sealed")
        assert sm.state == ArtifactState.PRE_ANNOTATION_SEALED

    def test_partial_resume_transitions(self):
        """GPU_IN_PROGRESS → GPU_PARTIAL → GPU_IN_PROGRESS (resume)."""
        sm = ArtifactStateMachine(initial_state=ArtifactState.GPU_IN_PROGRESS)
        sm.transition(ArtifactState.GPU_PARTIAL, reason="Interrupted")
        assert sm.state == ArtifactState.GPU_PARTIAL

        sm.transition(ArtifactState.GPU_IN_PROGRESS, reason="Resuming")
        assert sm.state == ArtifactState.GPU_IN_PROGRESS

    def test_failed_retry_transitions(self):
        """GPU_IN_PROGRESS → GPU_FAILED → GPU_IN_PROGRESS (retry)."""
        sm = ArtifactStateMachine(initial_state=ArtifactState.GPU_IN_PROGRESS)
        sm.transition(ArtifactState.GPU_FAILED, reason="OOM error")
        assert sm.state == ArtifactState.GPU_FAILED

        sm.transition(ArtifactState.GPU_IN_PROGRESS, reason="Retrying")
        assert sm.state == ArtifactState.GPU_IN_PROGRESS

    def test_invalid_skip_transition(self):
        """Cannot skip from SAMPLING_FROZEN to GPU_COMPLETE."""
        sm = ArtifactStateMachine()
        with pytest.raises(ArtifactStateError):
            sm.transition(ArtifactState.GPU_COMPLETE)

    def test_invalid_backward_transition(self):
        """Cannot go backwards from EVIDENCE_FROZEN to GPU_IN_PROGRESS."""
        sm = ArtifactStateMachine(initial_state=ArtifactState.EVIDENCE_FROZEN)
        with pytest.raises(ArtifactStateError):
            sm.transition(ArtifactState.GPU_IN_PROGRESS)

    def test_terminal_state_no_transitions(self):
        """PRE_ANNOTATION_SEALED is terminal — no transitions allowed."""
        sm = ArtifactStateMachine(initial_state=ArtifactState.PRE_ANNOTATION_SEALED)
        for state in ArtifactState:
            with pytest.raises(ArtifactStateError):
                sm.transition(state)

    def test_can_transition_method(self):
        sm = ArtifactStateMachine()
        assert sm.can_transition(ArtifactState.GPU_PENDING) is True
        assert sm.can_transition(ArtifactState.GPU_COMPLETE) is False

    def test_require_state_passes(self):
        sm = ArtifactStateMachine()
        sm.require_state(ArtifactState.SAMPLING_FROZEN)  # should not raise

    def test_require_state_fails(self):
        sm = ArtifactStateMachine()
        with pytest.raises(ArtifactStateError):
            sm.require_state(ArtifactState.GPU_COMPLETE, ArtifactState.EVIDENCE_FROZEN)

    def test_is_gpu_ready(self):
        for state in [ArtifactState.GPU_PENDING, ArtifactState.GPU_PARTIAL, ArtifactState.GPU_FAILED]:
            sm = ArtifactStateMachine(initial_state=state)
            assert sm.is_gpu_ready() is True

        for state in [ArtifactState.SAMPLING_FROZEN, ArtifactState.GPU_COMPLETE, ArtifactState.EVIDENCE_FROZEN]:
            sm = ArtifactStateMachine(initial_state=state)
            assert sm.is_gpu_ready() is False

    def test_is_evidence_finalized(self):
        for state in [ArtifactState.EVIDENCE_FROZEN, ArtifactState.TASKS_POPULATED, ArtifactState.PRE_ANNOTATION_SEALED]:
            sm = ArtifactStateMachine(initial_state=state)
            assert sm.is_evidence_finalized() is True

        for state in [ArtifactState.SAMPLING_FROZEN, ArtifactState.GPU_PENDING, ArtifactState.GPU_IN_PROGRESS]:
            sm = ArtifactStateMachine(initial_state=state)
            assert sm.is_evidence_finalized() is False

    def test_history_recording(self):
        sm = ArtifactStateMachine()
        sm.transition(ArtifactState.GPU_PENDING, reason="test")
        sm.transition(ArtifactState.GPU_IN_PROGRESS, reason="test2")

        history = sm.history
        assert len(history) == 3  # init + 2 transitions
        assert history[0].to_state == ArtifactState.SAMPLING_FROZEN.value
        assert history[1].to_state == ArtifactState.GPU_PENDING.value
        assert history[2].to_state == ArtifactState.GPU_IN_PROGRESS.value


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GPU Checkpoint Separation Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestCheckpointSeparation:
    """Ensure GPU checkpoints are not confused with evidence manifests."""

    def test_valid_checkpoint_passes(self):
        ckpt = {
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "state": "GPU_IN_PROGRESS",
            "completed_images": 50,
        }
        ok, issues = validate_checkpoint_not_evidence(ckpt)
        assert ok is True
        assert len(issues) == 0

    def test_evidence_manifest_rejected_as_checkpoint(self):
        """An evidence manifest must NOT pass checkpoint validation."""
        manifest = {
            "checkpoint_type": "evidence_manifest",
            "state": "EVIDENCE_FROZEN",
            "completed_images": 600,
        }
        ok, issues = validate_checkpoint_not_evidence(manifest)
        assert ok is False
        assert any("checkpoint_type" in i for i in issues)

    def test_gpu_complete_zero_images_rejected(self):
        ckpt = {
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "state": "GPU_COMPLETE",
            "completed_images": 0,
        }
        ok, issues = validate_checkpoint_not_evidence(ckpt)
        assert ok is False
        assert any("GPU_COMPLETE" in i and "0" in i for i in issues)

    def test_checkpoint_hash_computation(self):
        ckpt = GPUAcquisitionCheckpoint(
            sampling_manifest_hash="abc123",
            completed_images=10,
        )
        h1 = ckpt.compute_hash()
        assert len(h1) == 64  # SHA-256 hex digest
        ckpt.completed_images = 11
        h2 = ckpt.compute_hash()
        assert h1 != h2  # Different content → different hash

    def test_checkpoint_roundtrip(self):
        ckpt = GPUAcquisitionCheckpoint(
            sampling_manifest_hash="test_hash",
            completed_images=5,
            completed_image_ids=["img_1", "img_2", "img_3", "img_4", "img_5"],
        )
        d = ckpt.to_dict()
        ckpt2 = GPUAcquisitionCheckpoint.from_dict(d)
        assert ckpt2.completed_images == 5
        assert ckpt2.sampling_manifest_hash == "test_hash"
        assert len(ckpt2.completed_image_ids) == 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Version Compatibility Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestVersionCompatibility:
    """V2 required, v1 and 10A-partial explicitly rejected."""

    def test_v2_passes(self):
        ok, issues = validate_version_compatibility({"dataset_version": "v2"})
        assert ok is True

    def test_v1_rejected(self):
        ok, issues = validate_version_compatibility({"dataset_version": "v1"})
        assert ok is False
        assert any("incompatible" in i.lower() for i in issues)

    def test_10a_partial_rejected(self):
        ok, issues = validate_version_compatibility({"dataset_version": "10A_partial"})
        assert ok is False

    def test_10a_recovery_rejected(self):
        ok, issues = validate_version_compatibility({"dataset_version": "10A_recovery"})
        assert ok is False

    def test_unknown_version_flagged(self):
        ok, issues = validate_version_compatibility({"dataset_version": "v3"})
        assert ok is False
        assert any("v3" in i for i in issues)

    def test_missing_version_flagged(self):
        ok, issues = validate_version_compatibility({})
        assert ok is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Annotation Task Readiness Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestAnnotationTaskReadiness:
    """Validate N > 0, label masking, and provenance isolation."""

    def test_valid_tasks_pass(self):
        tasks = [
            {"task_id": "task_A_001", "claim_surface": "a cat", "label": None},
            {"task_id": "task_A_002", "claim_surface": "a dog", "label": None},
        ]
        ok, issues = validate_annotation_task_readiness(tasks)
        assert ok is True
        assert len(issues) == 0

    def test_zero_tasks_rejected(self):
        ok, issues = validate_annotation_task_readiness([])
        assert ok is False
        assert any("0" in i for i in issues)

    def test_unmasked_label_rejected(self):
        tasks = [
            {"task_id": "task_A_001", "claim_surface": "a cat", "label": "supported"},
        ]
        ok, issues = validate_annotation_task_readiness(tasks)
        assert ok is False
        assert any("LABEL MASKING" in i for i in issues)

    def test_v2_rejects_expected_label(self):
        """V2 validation MUST strictly reject legacy expected_label field."""
        tasks = [
            {"task_id": "task_A_001", "claim_surface": "a cat", "expected_label": None},
        ]
        ok, issues = validate_annotation_task_readiness(tasks, dataset_version="v2")
        assert ok is False
        assert any("expected_label" in i for i in issues)

    def test_migrate_legacy_tasks_to_v2(self):
        """Explicit migration converts expected_label to label."""
        from src.data.artifact_state import migrate_legacy_tasks_to_v2
        legacy = [
            {"task_id": "task_A_001", "claim_surface": "a cat", "expected_label": None},
        ]
        migrated = migrate_legacy_tasks_to_v2(legacy)
        assert "expected_label" not in migrated[0]
        assert "label" in migrated[0]
        assert migrated[0]["label"] is None
        ok, issues = validate_annotation_task_readiness(migrated, dataset_version="v2")
        assert ok is True

    def test_synthetic_marker_rejected(self):
        tasks = [
            {"task_id": "task_A_SYNTHETIC_001", "claim_surface": "a cat", "label": None},
        ]
        ok, issues = validate_annotation_task_readiness(tasks)
        assert ok is False
        assert any("SYNTHETIC" in i for i in issues)

    def test_mock_marker_rejected(self):
        tasks = [
            {"task_id": "task_A_001", "claim_surface": "MOCK cat", "label": None},
        ]
        ok, issues = validate_annotation_task_readiness(tasks)
        assert ok is False
        assert any("MOCK" in i for i in issues)

    def test_placeholder_marker_rejected(self):
        tasks = [
            {"task_id": "task_A_PLACEHOLDER_001", "claim_surface": "a cat", "label": None},
        ]
        ok, issues = validate_annotation_task_readiness(tasks)
        assert ok is False
        assert any("PLACEHOLDER" in i for i in issues)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Transition Graph Coverage Tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTransitionGraphCoverage:
    """Verify the transition graph is complete and correct."""

    def test_all_states_in_transition_graph(self):
        """Every ArtifactState must appear as a key in VALID_TRANSITIONS."""
        for state in ArtifactState:
            assert state in VALID_TRANSITIONS, f"State {state.value} missing from transition graph"

    def test_no_self_transitions(self):
        """No state should transition to itself."""
        for state, targets in VALID_TRANSITIONS.items():
            assert state not in targets, f"Self-transition detected: {state.value}"

    def test_terminal_state_has_no_transitions(self):
        """PRE_ANNOTATION_SEALED must have empty transition set."""
        assert len(VALID_TRANSITIONS[ArtifactState.PRE_ANNOTATION_SEALED]) == 0
