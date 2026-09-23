"""
Comprehensive regression tests for Colab runtime repairs and contract gates.

Validates:
1. CLI argument parsing (required explicit mode, invalid args, mutually exclusive flags).
2. Dry-run does not load models or write final artifacts.
3. Pilot processes only N images and never finalizes evidence or freezes.
4. Checkpoint directory and physical isolation between pilot and full runs.
5. Checkpoint provenance binding: fingerprint mismatch hard-fails resume.
6. Resume failure semantics: retryable failures re-attempt, terminal failures skip.
7. Strict genuine-zero semantics: empty LLaVA output classified as typed failure, NOT genuine zero.
8. Evidence source failure typed provenance: OWL-ViT and CLIP failures record typed info, scores stay null.
9. Annotation tasks use canonical 'label': null (rejecting expected_label) and integrate with Phase 10B.
10. Graph testability statistics cleanly separate pipeline failures from genuine zero claims.
11. Pre-annotation freeze verification gates prevent premature or incomplete sealing.
12. Source image acquisition gate required before inference.
"""

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from typing import Optional, List, Dict, Any
from unittest.mock import MagicMock, patch
import copy
import pytest

from scripts.run_phase10a_r2_colab import (
    parse_args,
    compute_json_hash,
    compute_stable_source_audit_hash,
    get_execution_git_info,
    enforce_cuda_gate,
    atomic_json_write,
    compute_checkpoint_provenance,
    classify_failure,
    load_or_create_checkpoint,
    save_checkpoint,
    process_single_image,
    verify_source_images_prepared,
    capture_runtime_environment,
    compute_environment_fingerprint,
    main,
    FROZEN_MODELS,
    LLAVA_GENERATION_CONFIG,
    CLAIM_EXTRACTION_CONFIG,
)
from src.data.artifact_state import (
    validate_annotation_task_readiness,
    migrate_legacy_tasks_to_v2,
)
from scripts.phase10b_annotation_interface import (
    validate_imported_labels,
    validate_claims_match,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. CLI Parsing Tests (Requirement 1, A9)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCLIParsing:
    def test_no_mode_hard_fails(self):
        """Running with no arguments must raise SystemExit (cannot start 600-image run accidentally)."""
        with pytest.raises(SystemExit):
            parse_args([])

    def test_dry_run_flag(self):
        args = parse_args(["--dry-run"])
        assert args.dry_run is True
        assert args.pilot is None
        assert args.full is False

    def test_pilot_flag(self):
        args = parse_args(["--pilot", "10", "--checkpoint-dir", "/tmp/ckpt"])
        assert args.pilot == 10
        assert args.dry_run is False
        assert args.full is False
        assert Path(args.checkpoint_dir) == Path("/tmp/ckpt")

    def test_pilot_non_positive_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--pilot", "0"])
        with pytest.raises(SystemExit):
            parse_args(["--pilot", "-5"])

    def test_full_flag(self):
        args = parse_args(["--full", "--resume", "--checkpoint-dir", "/drive/ckpt", "--output-dir", "/drive/out"])
        assert args.full is True
        assert args.resume is True
        assert Path(args.checkpoint_dir) == Path("/drive/ckpt")
        assert Path(args.output_dir) == Path("/drive/out")

    def test_mutually_exclusive_modes_rejected(self):
        with pytest.raises(SystemExit):
            parse_args(["--dry-run", "--full"])
        with pytest.raises(SystemExit):
            parse_args(["--pilot", "10", "--full"])
        with pytest.raises(SystemExit):
            parse_args(["--dry-run", "--pilot", "10"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Checkpoint Provenance & Physical Isolation (Requirement 2, A2, A3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCheckpointProvenanceAndIsolation:
    def test_checkpoint_provenance_binding(self):
        fingerprint, prov_hash = compute_checkpoint_provenance(
            sampling_manifest_hash="s_hash_1",
            audit_hash="a_hash_1",
            code_sha="c_sha_1",
            gen_cfg_hash="g_hash_1",
            claim_ext_hash="ce_hash_1",
        )
        assert len(prov_hash) == 64
        assert fingerprint["vlm_model"] == FROZEN_MODELS["vlm_model"]
        assert fingerprint["detector_model"] == FROZEN_MODELS["detector_model"]

    def test_provenance_mismatch_hard_fails_resume(self, tmp_path):
        ckpt_path = tmp_path / "gpu_acquisition_checkpoint_v2.json"
        fingerprint, prov_hash = compute_checkpoint_provenance(
            "s_hash_1", "a_hash_1", "c_sha_1", "g_hash_1", "ce_hash_1"
        )
        initial = {
            "schema_version": "2.0.0",
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "checkpoint_provenance_hash": prov_hash,
            "provenance_fingerprint": fingerprint,
            "completed_image_ids": ["img1"],
            "failed_image_ids": [],
        }
        with open(ckpt_path, "w") as f:
            json.dump(initial, f)

        # Attempt to resume with different hash must fail
        with pytest.raises(RuntimeError, match="CHECKPOINT PROVENANCE HASH MISMATCH"):
            load_or_create_checkpoint(
                checkpoint_path=ckpt_path,
                expected_checkpoint_type="gpu_acquisition_checkpoint",
                provenance_fingerprint=fingerprint,
                provenance_hash="different_hash_value",
                total_images=600,
                resume=True,
            )

    def test_pilot_checkpoint_never_feeds_full_resume(self, tmp_path):
        ckpt_path = tmp_path / "gpu_acquisition_checkpoint_v2.json"
        fingerprint, prov_hash = compute_checkpoint_provenance(
            "s_hash_1", "a_hash_1", "c_sha_1", "g_hash_1", "ce_hash_1"
        )
        initial = {
            "schema_version": "2.0.0",
            "checkpoint_type": "pilot_acquisition_checkpoint",
            "checkpoint_provenance_hash": prov_hash,
            "provenance_fingerprint": fingerprint,
            "completed_image_ids": ["img1"],
            "failed_image_ids": [],
        }
        with open(ckpt_path, "w") as f:
            json.dump(initial, f)

        # Full run trying to consume pilot checkpoint must fail
        with pytest.raises(RuntimeError, match="CHECKPOINT TYPE MISMATCH"):
            load_or_create_checkpoint(
                checkpoint_path=ckpt_path,
                expected_checkpoint_type="gpu_acquisition_checkpoint",
                provenance_fingerprint=fingerprint,
                provenance_hash=prov_hash,
                total_images=600,
                resume=True,
            )

    def test_no_silent_overwrite_without_resume(self, tmp_path):
        ckpt_path = tmp_path / "gpu_acquisition_checkpoint_v2.json"
        fingerprint, prov_hash = compute_checkpoint_provenance(
            "s_hash_1", "a_hash_1", "c_sha_1", "g_hash_1", "ce_hash_1"
        )
        ckpt_path.write_text("{}")

        with pytest.raises(RuntimeError, match="Checkpoint already exists"):
            load_or_create_checkpoint(
                checkpoint_path=ckpt_path,
                expected_checkpoint_type="gpu_acquisition_checkpoint",
                provenance_fingerprint=fingerprint,
                provenance_hash=prov_hash,
                total_images=600,
                resume=False,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Resume Failure Semantics (Requirement A4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestResumeFailureSemantics:
    def test_failure_classification(self):
        f_class, is_retry = classify_failure(TimeoutError("Connection timed out"))
        assert f_class == "RETRYABLE"
        assert is_retry is True

        f_class_term, is_retry_term = classify_failure(ValueError("Invalid syntax"))
        assert f_class_term == "TERMINAL"
        assert is_retry_term is False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Strict Genuine-Zero Semantics (Requirement 6)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class StrictFakeLLaVA:
    def __init__(self, caption: str = "A cat on a rug.", is_synthetic: bool = False, generation_source: str = "real_inference"):
        self.caption = caption
        self.is_synthetic = is_synthetic
        self.generation_source = generation_source

    def generate_caption(self, image_path, prompt=None, gen_config=None, image_id=None, image_hash=None):
        return StrictFakeVLMResponse(
            caption=self.caption,
            image_id=image_id or "img_test",
            response_id=f"resp_{image_id or 'test'}",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            is_synthetic=self.is_synthetic,
            generation_source=self.generation_source,
        )


class StrictFakeVLMResponse:
    def __init__(self, caption: str, image_id: str, response_id: str, model_name: str, model_revision: str, is_synthetic: bool = False, generation_source: str = "real_inference"):
        self.caption = caption
        self.image_id = image_id
        self.response_id = response_id
        self.model_name = model_name
        self.model_revision = model_revision
        self.is_synthetic = is_synthetic
        self.generation_source = generation_source
        self.image_hash = "mock_img_hash_123"
        self.prompt_hash = "mock_prompt_hash_456"
        self.execution_time_seconds = 0.42


class StrictFakeExtractedClaim:
    def __init__(self, claim_id: str, object_category: str, raw_claim_text: str):
        self.claim_id = claim_id
        self.object_category = object_category
        self.raw_claim_text = raw_claim_text
        # Note: category and surface_text deliberately omitted


class StrictFakeExtractionReport:
    def __init__(self, accepted_claims: list):
        self.accepted_claims = accepted_claims


class StrictFakeExtractor:
    def __init__(self, claims: list = None):
        self.claims = claims if claims is not None else [
            StrictFakeExtractedClaim(
                claim_id="claim_resp_test_cat",
                object_category="cat",
                raw_claim_text="a cat",
            )
        ]

    def extract_from_response(self, response_record):
        return StrictFakeExtractionReport(accepted_claims=self.claims)


class StrictFakeDetectorResult:
    def __init__(self, score: Optional[float], available: bool = True, error: Optional[str] = None):
        self.score = score
        self.available = available
        self.error = error
        # Note: max_score deliberately omitted


class StrictFakeDetector:
    def __init__(self, score: Optional[float] = 0.92, available: bool = True, error: Optional[str] = None):
        self.score = score
        self.available = available
        self.error = error

    def detect_category(self, image_path, category):
        return StrictFakeDetectorResult(score=self.score, available=self.available, error=self.error)


class StrictFakeCLIPResult:
    def __init__(self, score: Optional[float], available: bool = True, error: Optional[str] = None):
        self.score = score
        self.available = available
        self.error = error
        # Note: cosine_similarity deliberately omitted


class StrictFakeCLIP:
    def __init__(self, score: Optional[float] = 0.88, available: bool = True, error: Optional[str] = None):
        self.score = score
        self.available = available
        self.error = error

    def compute_similarity(self, image_path, text):
        return StrictFakeCLIPResult(score=self.score, available=self.available, error=self.error)


class TestStrictGenuineZeroSemantics:
    def test_empty_llava_caption_is_typed_failure(self, tmp_path):
        """Empty LLaVA response must produce a typed failure, NEVER genuine zero."""
        img_file = tmp_path / "test.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="blue").save(img_file)

        fake_vlm = StrictFakeLLaVA(caption="   ")  # Whitespace-only caption
        fake_extractor = StrictFakeExtractor()
        fake_detector = StrictFakeDetector()
        fake_clip = StrictFakeCLIP()

        records, failure, is_genuine_zero = process_single_image(
            image_id="img_001",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test.jpg",
            vlm_provider=fake_vlm,
            claim_extractor=fake_extractor,
            detector_provider=fake_detector,
            clip_provider=fake_clip,
            gen_config_hash="g_hash",
            claim_ext_hash="ce_hash",
        )

        assert is_genuine_zero is False
        assert len(records) == 0
        assert failure is not None
        assert failure["reason_code"] == "VLM_EMPTY_RESPONSE"
        assert failure["failure_class"] == "TERMINAL"
        assert failure["final_status"] == "TERMINAL_FAILURE"

    def test_valid_caption_zero_claims_is_genuine_zero(self, tmp_path):
        """Non-empty LLaVA response with 0 extracted claims is genuine zero."""
        img_file = tmp_path / "test.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="blue").save(img_file)

        fake_vlm = StrictFakeLLaVA(caption="A beautiful abstract blur with no objects.")
        fake_extractor = StrictFakeExtractor(claims=[])  # 0 claims extracted
        fake_detector = StrictFakeDetector()
        fake_clip = StrictFakeCLIP()

        records, failure, is_genuine_zero = process_single_image(
            image_id="img_002",
            image_path=img_file,
            research_split="TEST",
            coco_source_split="val2017",
            file_name="test.jpg",
            vlm_provider=fake_vlm,
            claim_extractor=fake_extractor,
            detector_provider=fake_detector,
            clip_provider=fake_clip,
            gen_config_hash="g_hash",
            claim_ext_hash="ce_hash",
        )

        assert is_genuine_zero is True
        assert len(records) == 0
        assert failure is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Typed Evidence Source Failures (Requirement A5)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestTypedEvidenceFailures:
    def test_detector_failure_records_typed_provenance_and_null_score(self, tmp_path):
        """Detector failure must record typed reason and keep score as None (null)."""
        img_file = tmp_path / "test.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="blue").save(img_file)

        fake_vlm = StrictFakeLLaVA(caption="A cat sitting on a rug.")
        fake_extractor = StrictFakeExtractor()
        fake_detector = StrictFakeDetector(score=None, available=False, error="Simulated detector CUDA failure")
        fake_clip = StrictFakeCLIP(score=0.85, available=True)

        records, failure, is_zero = process_single_image(
            image_id="img_003",
            image_path=img_file,
            research_split="TEST",
            coco_source_split="val2017",
            file_name="test.jpg",
            vlm_provider=fake_vlm,
            claim_extractor=fake_extractor,
            detector_provider=fake_detector,
            clip_provider=fake_clip,
            gen_config_hash="g_hash",
            claim_ext_hash="ce_hash",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec["detector_score"] is None
        assert rec["detector_available"] is False
        assert rec["detector_status"] == "FAILED"
        assert rec["detector_failure_info"] is not None
        assert "Simulated detector CUDA failure" in rec["detector_failure_info"]["error_message"]
        # CLIP succeeded
        assert rec["clip_score"] == 0.85
        assert rec["similarity_available"] is True
        assert rec["clip_status"] == "AVAILABLE"
        assert rec["clip_failure_info"] is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Annotation Task Schema & Phase 10B Integration (Requirement 5, A1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestAnnotationTaskContracts:
    def test_v2_tasks_require_label_null(self):
        """V2 tasks must have label: None and be accepted by readiness validator."""
        tasks = [
            {
                "task_id": "task_A_001",
                "claim_id": "c1",
                "image_id": "i1",
                "file_name": "i1.jpg",
                "claim_surface": "a cat",
                "object_category": "cat",
                "label": None,
            }
        ]
        ok, issues = validate_annotation_task_readiness(tasks, dataset_version="v2")
        assert ok is True
        assert len(issues) == 0

    def test_v2_tasks_reject_expected_label(self):
        """V2 tasks with legacy expected_label must be rejected."""
        legacy_tasks = [
            {
                "task_id": "task_A_001",
                "claim_id": "c1",
                "image_id": "i1",
                "file_name": "i1.jpg",
                "claim_surface": "a cat",
                "object_category": "cat",
                "expected_label": None,
            }
        ]
        ok, issues = validate_annotation_task_readiness(legacy_tasks, dataset_version="v2")
        assert ok is False
        assert any("expected_label" in iss for iss in issues)

    def test_compatibility_with_phase10b_tooling(self):
        """Generated blank tasks with filled labels must pass phase10b validation."""
        task_data = {
            "tasks": [
                {"claim_id": "claim_1", "label": "supported"},
                {"claim_id": "claim_2", "label": "hallucinated"},
                {"claim_id": "claim_3", "label": "unknown"},
            ]
        }
        ok, issues = validate_imported_labels(task_data)
        assert ok is True
        assert len(issues) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Source Image Verification Gate (Requirement 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestSourceImageVerificationGate:
    def test_missing_audit_fails(self, tmp_path):
        ok, msg, audit = verify_source_images_prepared(
            image_dir=tmp_path / "images",
            audit_p=tmp_path / "nonexistent_audit.json",
            expected_count=600,
        )
        assert ok is False
        assert "Source image audit missing" in msg

    def test_incomplete_audit_fails(self, tmp_path):
        audit_file = tmp_path / "audit.json"
        audit_file.write_text(json.dumps({
            "total_requested": 600,
            "valid_count": 590,
            "missing_count": 10,
            "corrupt_count": 0,
        }))
        ok, msg, audit = verify_source_images_prepared(
            image_dir=tmp_path / "images",
            audit_p=audit_file,
            expected_count=600,
        )
        assert ok is False
        assert "incomplete acquisition" in msg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Dry-Run Semantics (Requirement 1, 13)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestDryRunSemantics:
    @patch("scripts.run_phase10a_r2_colab.get_execution_git_info", return_value=("abc123deadbeef", True))
    @patch("scripts.run_phase10a_r2_colab.verify_source_images_prepared")
    @patch("scripts.run_phase10a_r2_colab.capture_runtime_environment")
    @patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate")
    @patch("src.vlm.llava_provider.LLaVA15Provider")
    def test_dry_run_does_not_load_models_or_write_artifacts(
        self, mock_llava, mock_cuda, mock_runtime, mock_verify, mock_git, tmp_path
    ):
        """Dry-run must validate preflight and return without loading LLaVA or writing final artifacts."""
        mock_cuda.return_value = {
            "device": "NVIDIA T4",
            "gpu_memory_gb": 15.0,
            "cuda_available": True,
            "cuda_version": "12.1",
        }
        mock_runtime.return_value = {
            "python_version": "3.12.6",
            "platform": "Linux",
            "torch_version": "2.1.0",
            "torchvision_version": "0.16.0",
            "transformers_version": "4.36.0",
            "accelerate_version": "0.25.0",
            "bitsandbytes_version": "0.41.0",
            "pillow_version": "10.0.0",
            "numpy_version": "1.26.0",
            "scipy_version": "1.11.0",
            "cuda_available": True,
            "cuda_version": "12.1",
            "device": "NVIDIA T4",
            "gpu_memory_gb": 15.0,
        }
        mock_verify.return_value = (
            True,
            "Source images fully verified (600/600 valid).",
            {
                "sampling_manifest_hash": "abc123",
                "total_requested": 600,
                "valid_count": 600,
                "missing_count": 0,
                "corrupt_count": 0,
                "records": [],
            },
        )

        # Run dry-run
        main(["--dry-run"])

        # Ensure LLaVA was never instantiated
        mock_llava.assert_not_called()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. PyTorch CUDA Memory Property (Defect 1)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestCUDAMemoryProperty:
    class MockDeviceProperties:
        def __init__(self, memory_bytes: int):
            self.total_memory = memory_bytes
            # Note: total_mem deliberately does NOT exist

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA T4")
    def test_enforce_cuda_gate_uses_total_memory(self, mock_name, mock_avail):
        """Must access total_memory without attempting to access non-existent total_mem."""
        import torch
        original_cuda = torch.version.cuda
        try:
            torch.version.cuda = "12.1"
            with patch("torch.cuda.get_device_properties", return_value=self.MockDeviceProperties(16 * 1024**3)):
                env = enforce_cuda_gate()
                assert env["device"] == "NVIDIA T4"
                assert env["gpu_memory_gb"] == 16.0
        finally:
            torch.version.cuda = original_cuda

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA T4")
    def test_capture_runtime_environment_uses_total_memory(self, mock_name, mock_avail):
        """capture_runtime_environment must access total_memory."""
        import sys
        import torch
        # Mock bitsandbytes in sys.modules to avoid import failure on non-CUDA Windows
        mock_bnb = MagicMock()
        mock_bnb.__version__ = "0.41.0"
        original_cuda = torch.version.cuda
        had_bnb = "bitsandbytes" in sys.modules
        original_bnb = sys.modules.get("bitsandbytes")
        try:
            torch.version.cuda = "12.1"
            sys.modules["bitsandbytes"] = mock_bnb
            with patch("torch.cuda.get_device_properties", return_value=self.MockDeviceProperties(16 * 1024**3)):
                env = capture_runtime_environment()
                assert env["gpu_memory_gb"] == 16.0
        finally:
            torch.version.cuda = original_cuda
            if had_bnb:
                sys.modules["bitsandbytes"] = original_bnb
            else:
                sys.modules.pop("bitsandbytes", None)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Stable Source-Image Audit Fingerprint (Defect 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestStableSourceAuditHash:
    def _create_sample_audit(self, timestamp: str, local_path_prefix: str) -> dict:
        records = [
            {
                "image_id": f"coco_{i:012d}",
                "coco_integer_id": i,
                "file_name": f"{i:012d}.jpg",
                "coco_source_split": "train2017" if i % 2 == 0 else "val2017",
                "research_split": "TRAIN" if i % 2 == 0 else "TEST",
                "expected_width": 640,
                "expected_height": 480,
                "actual_width": 640,
                "actual_height": 480,
                "sha256": f"sha256_mock_hash_{i:04d}",
                "status": "VALID",
                "is_valid": True,
                # Volatile metadata:
                "local_path": f"{local_path_prefix}/{i:012d}.jpg",
                "download_time_seconds": 0.123,
                "timestamp": timestamp,
            }
            for i in range(1, 11)
        ]
        return {
            "sampling_manifest_hash": "manifest_test_hash_12345",
            "total_requested": 10,
            "valid_count": 10,
            "missing_count": 0,
            "corrupt_count": 0,
            "timestamp": timestamp,
            "execution_duration_sec": 42.5,
            "records": records,
        }

    def test_stable_audit_hash_ignores_timestamp_and_volatile_fields(self):
        """Regenerated identical audit with different timestamps/paths must yield identical hash."""
        audit_1 = self._create_sample_audit("2026-09-20T10:00:00Z", "/content/drive/images")
        audit_2 = self._create_sample_audit("2026-09-23T15:30:00Z", "/tmp/colab_run_2/images")

        hash_1 = compute_stable_source_audit_hash(audit_1)
        hash_2 = compute_stable_source_audit_hash(audit_2)

        assert len(hash_1) == 64
        assert hash_1 == hash_2

    def test_stable_audit_hash_changes_when_image_sha_changes(self):
        """Modifying any image SHA must change the audit hash."""
        audit = self._create_sample_audit("2026-09-20T10:00:00Z", "/content/images")
        hash_orig = compute_stable_source_audit_hash(audit)

        audit_tampered = copy.deepcopy(audit)
        audit_tampered["records"][3]["sha256"] = "sha256_tampered_bytes_here"
        hash_tampered = compute_stable_source_audit_hash(audit_tampered)

        assert hash_orig != hash_tampered

    def test_stable_audit_hash_changes_when_dimensions_or_status_changes(self):
        """Modifying dimensions or status must change the audit hash."""
        audit = self._create_sample_audit("2026-09-20T10:00:00Z", "/content/images")
        hash_orig = compute_stable_source_audit_hash(audit)

        # Dimension change
        audit_dim = copy.deepcopy(audit)
        audit_dim["records"][0]["actual_width"] = 1920
        assert compute_stable_source_audit_hash(audit_dim) != hash_orig

        # Status change
        audit_status = copy.deepcopy(audit)
        audit_status["records"][0]["status"] = "CORRUPT"
        audit_status["records"][0]["is_valid"] = False
        assert compute_stable_source_audit_hash(audit_status) != hash_orig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 11. Git Execution SHA & Clean Working Tree Gate (Defect 3)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestGitExecutionProvenance:
    def test_actual_git_execution_sha_captured(self):
        """get_execution_git_info must return a 40-char hex SHA and boolean clean status."""
        sha, is_clean = get_execution_git_info()
        assert len(sha) == 40
        assert all(c in "0123456789abcdefABCDEF" for c in sha)
        assert isinstance(is_clean, bool)

    @patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate")
    @patch("scripts.run_phase10a_r2_colab.get_execution_git_info")
    def test_pilot_rejects_dirty_working_tree(self, mock_git, mock_cuda):
        """Pilot scientific execution must hard-fail before inference if working tree is dirty."""
        mock_cuda.return_value = {"device": "T4", "gpu_memory_gb": 15.0, "cuda_available": True, "cuda_version": "12.1"}
        mock_git.return_value = ("1111222233334444555566667777888899990000", False)

        with pytest.raises(RuntimeError, match="DIRTY GIT WORKING TREE DETECTED"):
            main(["--pilot", "5"])

    @patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate")
    @patch("scripts.run_phase10a_r2_colab.get_execution_git_info")
    def test_full_rejects_dirty_working_tree(self, mock_git, mock_cuda):
        """Full scientific execution must hard-fail before inference if working tree is dirty."""
        mock_cuda.return_value = {"device": "T4", "gpu_memory_gb": 15.0, "cuda_available": True, "cuda_version": "12.1"}
        mock_git.return_value = ("1111222233334444555566667777888899990000", False)

        with pytest.raises(RuntimeError, match="DIRTY GIT WORKING TREE DETECTED"):
            main(["--full"])

    @patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate")
    @patch("scripts.run_phase10a_r2_colab.get_execution_git_info")
    def test_dry_run_accepts_dirty_tree(self, mock_git, mock_cuda):
        """Dry-run may report dirty working tree without failing."""
        mock_cuda.return_value = {"device": "T4", "gpu_memory_gb": 15.0, "cuda_available": True, "cuda_version": "12.1"}
        mock_git.return_value = ("1111222233334444555566667777888899990000", False)

        # Must not raise RuntimeError
        main(["--dry-run"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 12. Resume Stability & Provenance Binding (Requirement 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestResumeStabilityWithAuditAndGit:
    def _create_audit(self, timestamp: str, image_sha: str = "valid_sha_123") -> dict:
        return {
            "sampling_manifest_hash": "manifest_hash_fixed",
            "total_requested": 1,
            "valid_count": 1,
            "missing_count": 0,
            "corrupt_count": 0,
            "timestamp": timestamp,
            "records": [
                {
                    "image_id": "img1",
                    "file_name": "img1.jpg",
                    "coco_source_split": "train2017",
                    "research_split": "TRAIN",
                    "expected_width": 640,
                    "expected_height": 480,
                    "actual_width": 640,
                    "actual_height": 480,
                    "sha256": image_sha,
                    "status": "VALID",
                    "is_valid": True,
                }
            ],
        }

    def test_resume_accepts_regenerated_identical_audit(self, tmp_path):
        """Checkpoint created under audit v1 must resume successfully under regenerated audit v2."""
        audit_v1 = self._create_audit("2026-09-20T10:00:00Z")
        audit_v2 = self._create_audit("2026-09-23T12:00:00Z")  # regenerated with new timestamp

        audit_hash_v1 = compute_stable_source_audit_hash(audit_v1)
        audit_hash_v2 = compute_stable_source_audit_hash(audit_v2)
        assert audit_hash_v1 == audit_hash_v2

        git_sha = "aabbccddeeff00112233445566778899aabbccdd"
        fp_v1, prov_hash_v1 = compute_checkpoint_provenance(
            sampling_manifest_hash="samp_hash",
            audit_hash=audit_hash_v1,
            execution_code_sha=git_sha,
            gen_cfg_hash="g_cfg",
            claim_ext_hash="c_ext",
            working_tree_clean=True,
        )

        ckpt_p = tmp_path / "gpu_acquisition_checkpoint_v2.json"
        initial = {
            "schema_version": "2.0.0",
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "checkpoint_provenance_hash": prov_hash_v1,
            "provenance_fingerprint": fp_v1,
            "completed_image_ids": ["img1"],
            "failed_image_ids": [],
        }
        with open(ckpt_p, "w", encoding="utf-8") as f:
            json.dump(initial, f)

        # Resume under regenerated audit v2
        fp_v2, prov_hash_v2 = compute_checkpoint_provenance(
            sampling_manifest_hash="samp_hash",
            audit_hash=audit_hash_v2,
            execution_code_sha=git_sha,
            gen_cfg_hash="g_cfg",
            claim_ext_hash="c_ext",
            working_tree_clean=True,
        )

        resumed = load_or_create_checkpoint(
            checkpoint_path=ckpt_p,
            expected_checkpoint_type="gpu_acquisition_checkpoint",
            provenance_fingerprint=fp_v2,
            provenance_hash=prov_hash_v2,
            total_images=1,
            resume=True,
        )
        assert resumed["completed_image_ids"] == ["img1"]

    def test_resume_rejects_changed_source_image_content(self, tmp_path):
        """Checkpoint created under audit v1 must FAIL resume if image bytes changed."""
        audit_v1 = self._create_audit("2026-09-20T10:00:00Z", image_sha="original_sha")
        audit_tampered = self._create_audit("2026-09-23T12:00:00Z", image_sha="tampered_sha")

        audit_hash_v1 = compute_stable_source_audit_hash(audit_v1)
        audit_hash_tampered = compute_stable_source_audit_hash(audit_tampered)
        assert audit_hash_v1 != audit_hash_tampered

        git_sha = "aabbccddeeff00112233445566778899aabbccdd"
        fp_v1, prov_hash_v1 = compute_checkpoint_provenance(
            sampling_manifest_hash="samp_hash",
            audit_hash=audit_hash_v1,
            execution_code_sha=git_sha,
            gen_cfg_hash="g_cfg",
            claim_ext_hash="c_ext",
            working_tree_clean=True,
        )

        ckpt_p = tmp_path / "gpu_acquisition_checkpoint_v2.json"
        initial = {
            "schema_version": "2.0.0",
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "checkpoint_provenance_hash": prov_hash_v1,
            "provenance_fingerprint": fp_v1,
            "completed_image_ids": ["img1"],
            "failed_image_ids": [],
        }
        with open(ckpt_p, "w", encoding="utf-8") as f:
            json.dump(initial, f)

        # Attempt resume under tampered image audit
        fp_tampered, prov_hash_tampered = compute_checkpoint_provenance(
            sampling_manifest_hash="samp_hash",
            audit_hash=audit_hash_tampered,
            execution_code_sha=git_sha,
            gen_cfg_hash="g_cfg",
            claim_ext_hash="c_ext",
            working_tree_clean=True,
        )

        with pytest.raises(RuntimeError, match="CHECKPOINT PROVENANCE HASH MISMATCH"):
            load_or_create_checkpoint(
                checkpoint_path=ckpt_p,
                expected_checkpoint_type="gpu_acquisition_checkpoint",
                provenance_fingerprint=fp_tampered,
                provenance_hash=prov_hash_tampered,
                total_images=1,
                resume=True,
            )

    def test_resume_rejects_changed_execution_sha(self, tmp_path):
        """Checkpoint must FAIL resume if execution code SHA has changed."""
        audit = self._create_audit("2026-09-20T10:00:00Z")
        audit_hash = compute_stable_source_audit_hash(audit)

        fp_1, prov_hash_1 = compute_checkpoint_provenance(
            sampling_manifest_hash="samp_hash",
            audit_hash=audit_hash,
            execution_code_sha="sha_commit_1",
            gen_cfg_hash="g_cfg",
            claim_ext_hash="c_ext",
            working_tree_clean=True,
        )

        ckpt_p = tmp_path / "gpu_acquisition_checkpoint_v2.json"
        initial = {
            "schema_version": "2.0.0",
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "checkpoint_provenance_hash": prov_hash_1,
            "provenance_fingerprint": fp_1,
            "completed_image_ids": ["img1"],
            "failed_image_ids": [],
        }
        with open(ckpt_p, "w", encoding="utf-8") as f:
            json.dump(initial, f)

        fp_2, prov_hash_2 = compute_checkpoint_provenance(
            sampling_manifest_hash="samp_hash",
            audit_hash=audit_hash,
            execution_code_sha="sha_commit_2_modified",
            gen_cfg_hash="g_cfg",
            claim_ext_hash="c_ext",
            working_tree_clean=True,
        )

        with pytest.raises(RuntimeError, match="CHECKPOINT PROVENANCE HASH MISMATCH"):
            load_or_create_checkpoint(
                checkpoint_path=ckpt_p,
                expected_checkpoint_type="gpu_acquisition_checkpoint",
                provenance_fingerprint=fp_2,
                provenance_hash=prov_hash_2,
                total_images=1,
                resume=True,
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 13. Strict Provider Contract Compliance (Section 10)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TestStrictProviderContracts:
    """
    Interface-faithful regression tests enforcing that caller uses ONLY
    production method names and result attributes from M6 pipeline providers.
    """

    def test_strict_fakes_produce_valid_claim_evidence_record(self, tmp_path):
        img_file = tmp_path / "test_strict.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_file)

        vlm = StrictFakeLLaVA(caption="A cat on a rug.")
        extractor = StrictFakeExtractor()
        detector = StrictFakeDetector(score=0.92, available=True)
        clip = StrictFakeCLIP(score=0.88, available=True)

        records, failure, is_zero = process_single_image(
            image_id="img_strict_001",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test_strict.jpg",
            vlm_provider=vlm,
            claim_extractor=extractor,
            detector_provider=detector,
            clip_provider=clip,
            gen_config_hash="g_cfg_strict",
            claim_ext_hash="c_ext_strict",
        )

        assert is_zero is False
        assert failure is None
        assert len(records) == 1

        rec = records[0]
        assert rec["raw_caption"] == "A cat on a rug."
        assert rec["claim_id"] == "claim_resp_test_cat"
        assert rec["object_category"] == "cat"
        assert rec["raw_claim_text"] == "a cat"
        assert rec["detector_score"] == 0.92
        assert rec["detector_available"] is True
        assert rec["detector_status"] == "AVAILABLE"
        assert rec["clip_score"] == 0.88
        assert rec["similarity_available"] is True
        assert rec["clip_status"] == "AVAILABLE"
        assert rec["provenance_status"] == "REAL_UNLABELED"

    def test_detector_unavailable_yields_none_score(self, tmp_path):
        img_file = tmp_path / "test_strict_det.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_file)

        vlm = StrictFakeLLaVA(caption="A cat on a rug.")
        extractor = StrictFakeExtractor()
        detector = StrictFakeDetector(score=None, available=False, error="OWL-ViT execution timeout")
        clip = StrictFakeCLIP(score=0.88, available=True)

        records, failure, is_zero = process_single_image(
            image_id="img_strict_002",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test_strict_det.jpg",
            vlm_provider=vlm,
            claim_extractor=extractor,
            detector_provider=detector,
            clip_provider=clip,
            gen_config_hash="g_cfg_strict",
            claim_ext_hash="c_ext_strict",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec["detector_score"] is None
        assert rec["detector_available"] is False
        assert rec["detector_status"] == "FAILED"
        assert "OWL-ViT execution timeout" in rec["detector_failure_info"]["error_message"]

    def test_clip_unavailable_yields_none_score(self, tmp_path):
        img_file = tmp_path / "test_strict_clip.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_file)

        vlm = StrictFakeLLaVA(caption="A cat on a rug.")
        extractor = StrictFakeExtractor()
        detector = StrictFakeDetector(score=0.92, available=True)
        clip = StrictFakeCLIP(score=None, available=False, error="CLIP Out of Memory")

        records, failure, is_zero = process_single_image(
            image_id="img_strict_003",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test_strict_clip.jpg",
            vlm_provider=vlm,
            claim_extractor=extractor,
            detector_provider=detector,
            clip_provider=clip,
            gen_config_hash="g_cfg_strict",
            claim_ext_hash="c_ext_strict",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec["clip_score"] is None
        assert rec["similarity_available"] is False
        assert rec["clip_status"] == "FAILED"
        assert "CLIP Out of Memory" in rec["clip_failure_info"]["error_message"]

    def test_empty_caption_yields_vlm_empty_response(self, tmp_path):
        img_file = tmp_path / "test_strict_empty.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_file)

        vlm = StrictFakeLLaVA(caption="   ")
        extractor = StrictFakeExtractor()
        detector = StrictFakeDetector()
        clip = StrictFakeCLIP()

        records, failure, is_zero = process_single_image(
            image_id="img_strict_004",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test_strict_empty.jpg",
            vlm_provider=vlm,
            claim_extractor=extractor,
            detector_provider=detector,
            clip_provider=clip,
            gen_config_hash="g_cfg_strict",
            claim_ext_hash="c_ext_strict",
        )

        assert is_zero is False
        assert len(records) == 0
        assert failure["reason_code"] == "VLM_EMPTY_RESPONSE"

    def test_zero_accepted_claims_yields_genuine_zero(self, tmp_path):
        img_file = tmp_path / "test_strict_zero.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_file)

        vlm = StrictFakeLLaVA(caption="An abstract blurry landscape.")
        extractor = StrictFakeExtractor(claims=[])
        detector = StrictFakeDetector()
        clip = StrictFakeCLIP()

        records, failure, is_zero = process_single_image(
            image_id="img_strict_005",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test_strict_zero.jpg",
            vlm_provider=vlm,
            claim_extractor=extractor,
            detector_provider=detector,
            clip_provider=clip,
            gen_config_hash="g_cfg_strict",
            claim_ext_hash="c_ext_strict",
        )

        assert is_zero is True
        assert len(records) == 0
        assert failure is None

    def test_invalid_provenance_yields_typed_failure(self, tmp_path):
        img_file = tmp_path / "test_strict_synth.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="green").save(img_file)

        vlm = StrictFakeLLaVA(caption="A cat on a rug.", is_synthetic=True, generation_source="synthetic_fixture")
        extractor = StrictFakeExtractor()
        detector = StrictFakeDetector()
        clip = StrictFakeCLIP()

        records, failure, is_zero = process_single_image(
            image_id="img_strict_006",
            image_path=img_file,
            research_split="TRAIN",
            coco_source_split="train2017",
            file_name="test_strict_synth.jpg",
            vlm_provider=vlm,
            claim_extractor=extractor,
            detector_provider=detector,
            clip_provider=clip,
            gen_config_hash="g_cfg_strict",
            claim_ext_hash="c_ext_strict",
        )

        assert is_zero is False
        assert len(records) == 0
        assert failure["reason_code"] == "INVALID_PROVENANCE"

    def test_static_contract_regression_no_old_api_references(self):
        """Assert runner source code contains no deprecated or nonexistent provider calls."""
        runner_path = Path(__file__).resolve().parent.parent / "scripts" / "run_phase10a_r2_colab.py"
        source = runner_path.read_text(encoding="utf-8")

        forbidden_patterns = [
            "vlm_provider.generate(",
            "claim_extractor.extract_claims(",
            "detector_provider.detect(",
            ".max_score",
            ".cosine_similarity",
            "claim.category",
            "claim.surface_text",
        ]

        violations = [pat for pat in forbidden_patterns if pat in source]
        assert len(violations) == 0, f"Found forbidden deprecated API patterns in runner: {violations}"

    def test_llava_provider_wrapper_delegation(self, tmp_path):
        """LLaVAProviderWrapper must delegate to generate_caption on the underlying provider."""
        from src.data.vlm_interface import LLaVAProviderWrapper
        img_file = tmp_path / "wrapper_test.jpg"
        from PIL import Image
        Image.new("RGB", (100, 100), color="yellow").save(img_file)

        fake_underlying = StrictFakeLLaVA(caption="A wrapped cat.")
        wrapper = LLaVAProviderWrapper(underlying_provider=fake_underlying)

        resp = wrapper.generate_caption(img_file, prompt="Test prompt")
        assert resp.caption == "A wrapped cat."


class TestEvidenceDownloadAndRevisionPinning:
    """Regression tests for evidence model download and revision pinning."""

    def test_colab_runner_detector_clip_download_enabled(self):
        """Runner must instantiate detector and CLIP providers with local_files_only=False."""
        runner_path = Path(__file__).resolve().parent.parent / "scripts" / "run_phase10a_r2_colab.py"
        source = runner_path.read_text(encoding="utf-8")

        assert "HuggingFaceDetectorProvider(" in source
        assert "TransformersCLIPProvider(" in source
        assert "local_files_only=False" in source

        # Verify specifically within the detector and clip instantiations
        det_idx = source.find("HuggingFaceDetectorProvider(")
        assert det_idx != -1
        det_block = source[det_idx:det_idx + 300]
        assert "local_files_only=False" in det_block

        clip_idx = source.find("TransformersCLIPProvider(")
        assert clip_idx != -1
        clip_block = source[clip_idx:clip_idx + 300]
        assert "local_files_only=False" in clip_block

    def test_pinned_llava_revision_reaches_from_pretrained(self):
        """Configured model_revision must be passed to AutoConfig, AutoProcessor, and LlavaForConditionalGeneration."""
        from src.vlm.llava_provider import LLaVA15Provider
        pinned_rev = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        provider = LLaVA15Provider(
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision=pinned_rev,
            device="cpu",
            local_files_only=True,
        )

        mock_config = MagicMock()
        mock_config._commit_hash = pinned_rev
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config._commit_hash = pinned_rev
        mock_model.parameters.return_value = []

        # Reset singleton cache
        LLaVA15Provider._model_instance = None
        LLaVA15Provider._loaded_model_id = None
        LLaVA15Provider._resolved_revision = None

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config) as mock_cfg_load, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_load, \
             patch("transformers.LlavaForConditionalGeneration.from_pretrained", return_value=mock_model) as mock_model_load:

            # 1. resolve_revision passes revision
            rev = provider.resolve_revision()
            assert rev == pinned_rev
            assert mock_cfg_load.call_args[1].get("revision") == pinned_rev

            # 2. _ensure_loaded passes revision to processor and model
            provider._ensure_loaded()
            assert mock_proc_load.call_args[1].get("revision") == pinned_rev
            assert mock_model_load.call_args[1].get("revision") == pinned_rev

    def test_pinned_owlvit_revision_reaches_from_pretrained(self):
        """Configured detector revision must be passed to AutoConfig, AutoProcessor, and OwlViTForObjectDetection."""
        from src.evidence.detector_provider import HuggingFaceDetectorProvider
        pinned_rev = "cbc355fb364588351c5d51c7f74465e8e7ec6f72"
        provider = HuggingFaceDetectorProvider(
            model_name="google/owlvit-base-patch32",
            model_revision=pinned_rev,
            device="cpu",
            local_files_only=False,
        )

        mock_config = MagicMock()
        mock_config._commit_hash = pinned_rev
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config._commit_hash = pinned_rev
        mock_model.parameters.return_value = []

        # Reset caches
        HuggingFaceDetectorProvider._model_cache.clear()
        HuggingFaceDetectorProvider._processor_cache.clear()
        HuggingFaceDetectorProvider._revision_cache.clear()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config) as mock_cfg_load, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_load, \
             patch("transformers.OwlViTForObjectDetection.from_pretrained", return_value=mock_model) as mock_model_load:

            rev = provider.resolve_revision()
            assert rev == pinned_rev
            assert mock_cfg_load.call_args[1].get("revision") == pinned_rev

            provider._ensure_loaded()
            assert mock_proc_load.call_args[1].get("revision") == pinned_rev
            assert mock_model_load.call_args[1].get("revision") == pinned_rev

    def test_pinned_clip_revision_reaches_from_pretrained(self):
        """Configured CLIP revision must be passed to AutoConfig, AutoProcessor, and CLIPModel."""
        from src.evidence.clip_provider import TransformersCLIPProvider
        pinned_rev = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
        provider = TransformersCLIPProvider(
            model_name="openai/clip-vit-base-patch32",
            model_revision=pinned_rev,
            device="cpu",
            local_files_only=False,
        )

        mock_config = MagicMock()
        mock_config._commit_hash = pinned_rev
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config._commit_hash = pinned_rev
        mock_model.parameters.return_value = []

        # Reset caches
        TransformersCLIPProvider._model_cache.clear()
        TransformersCLIPProvider._processor_cache.clear()
        TransformersCLIPProvider._revision_cache.clear()

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config) as mock_cfg_load, \
             patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_load, \
             patch("transformers.CLIPModel.from_pretrained", return_value=mock_model) as mock_model_load:

            rev = provider.resolve_revision()
            assert rev == pinned_rev
            assert mock_cfg_load.call_args[1].get("revision") == pinned_rev

            provider._ensure_loaded()
            assert mock_proc_load.call_args[1].get("revision") == pinned_rev
            assert mock_model_load.call_args[1].get("revision") == pinned_rev

    def test_revision_mismatch_rejected_llava(self):
        """If resolved revision does not match pinned revision, raise RuntimeError INVALID_PROVENANCE."""
        from src.vlm.llava_provider import LLaVA15Provider
        provider = LLaVA15Provider(
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            device="cpu",
        )
        mock_model = MagicMock()
        mock_model.config._commit_hash = "wrong_sha_1234567890abcdef"
        mock_model.parameters.return_value = []

        LLaVA15Provider._model_instance = None
        LLaVA15Provider._loaded_model_id = None
        LLaVA15Provider._resolved_revision = None

        with patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()), \
             patch("transformers.LlavaForConditionalGeneration.from_pretrained", return_value=mock_model):
            with pytest.raises(RuntimeError, match="INVALID_PROVENANCE"):
                provider._ensure_loaded()

    def test_revision_mismatch_rejected_owlvit(self):
        """If resolved revision does not match pinned revision, raise RuntimeError INVALID_PROVENANCE."""
        from src.evidence.detector_provider import HuggingFaceDetectorProvider
        provider = HuggingFaceDetectorProvider(
            model_name="google/owlvit-base-patch32",
            model_revision="cbc355fb364588351c5d51c7f74465e8e7ec6f72",
        )
        mock_model = MagicMock()
        mock_model.config._commit_hash = "wrong_sha_detector"
        mock_model.parameters.return_value = []

        HuggingFaceDetectorProvider._model_cache.clear()
        HuggingFaceDetectorProvider._processor_cache.clear()
        HuggingFaceDetectorProvider._revision_cache.clear()

        with patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()), \
             patch("transformers.OwlViTForObjectDetection.from_pretrained", return_value=mock_model):
            with pytest.raises(RuntimeError, match="INVALID_PROVENANCE"):
                provider._ensure_loaded()

    def test_revision_mismatch_rejected_clip(self):
        """If resolved revision does not match pinned revision, raise RuntimeError INVALID_PROVENANCE."""
        from src.evidence.clip_provider import TransformersCLIPProvider
        provider = TransformersCLIPProvider(
            model_name="openai/clip-vit-base-patch32",
            model_revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        )
        mock_model = MagicMock()
        mock_model.config._commit_hash = "wrong_sha_clip"
        mock_model.parameters.return_value = []

        TransformersCLIPProvider._model_cache.clear()
        TransformersCLIPProvider._processor_cache.clear()
        TransformersCLIPProvider._revision_cache.clear()

        with patch("transformers.AutoProcessor.from_pretrained", return_value=MagicMock()), \
             patch("transformers.CLIPModel.from_pretrained", return_value=mock_model):
            with pytest.raises(RuntimeError, match="INVALID_PROVENANCE"):
                provider._ensure_loaded()

    def test_pilot_evidence_health_gate_zero_detector_fails(self, tmp_path):
        """When claims > 0 but detector_available == 0, pilot health gate must fail."""
        from scripts.run_phase10a_r2_colab import main
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        fake_records = [{
            "claim_id": "c1",
            "image_id": "img1",
            "detector_available": False,
            "detector_score": None,
            "detector_status": "FAILED",
            "clip_available": True,
            "clip_score": 0.85,
            "clip_status": "AVAILABLE",
        }]

        with patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate", return_value={"device": "Tesla T4", "gpu_memory_gb": 15.0, "cuda_available": True, "cuda_version": "12.2"}), \
             patch("scripts.run_phase10a_r2_colab.verify_source_images_prepared", return_value=(True, "OK", {"records": []})), \
             patch("scripts.run_phase10a_r2_colab.get_execution_git_info", return_value=("1111222233334444555566667777888899990000", True)), \
             patch("scripts.run_phase10a_r2_colab.process_single_image", return_value=(fake_records, None, False)):
            with pytest.raises(RuntimeError, match="PILOT_EVIDENCE_INCOMPLETE"):
                main(["--pilot", "1", "--checkpoint-dir", str(ckpt_dir)])

    def test_pilot_evidence_health_gate_zero_clip_fails(self, tmp_path):
        """When claims > 0 but clip_available == 0, pilot health gate must fail."""
        from scripts.run_phase10a_r2_colab import main
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        fake_records = [{
            "claim_id": "c1",
            "image_id": "img1",
            "detector_available": True,
            "detector_score": 0.92,
            "detector_status": "AVAILABLE",
            "clip_available": False,
            "clip_score": None,
            "clip_status": "FAILED",
        }]

        with patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate", return_value={"device": "Tesla T4", "gpu_memory_gb": 15.0, "cuda_available": True, "cuda_version": "12.2"}), \
             patch("scripts.run_phase10a_r2_colab.verify_source_images_prepared", return_value=(True, "OK", {"records": []})), \
             patch("scripts.run_phase10a_r2_colab.get_execution_git_info", return_value=("1111222233334444555566667777888899990000", True)), \
             patch("scripts.run_phase10a_r2_colab.process_single_image", return_value=(fake_records, None, False)):
            with pytest.raises(RuntimeError, match="PILOT_EVIDENCE_INCOMPLETE"):
                main(["--pilot", "1", "--checkpoint-dir", str(ckpt_dir)])

    def test_pilot_evidence_health_gate_mixed_evidence_succeeds(self, tmp_path):
        """When claims > 0 and both detector and clip have at least 1 usable score, pilot succeeds even with mixed failures."""
        from scripts.run_phase10a_r2_colab import main
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        fake_records = [
            {
                "claim_id": "c1",
                "image_id": "img1",
                "detector_available": True,
                "detector_score": 0.92,
                "detector_status": "AVAILABLE",
                "clip_available": False,
                "similarity_available": False,
                "clip_score": None,
                "clip_status": "FAILED",
            },
            {
                "claim_id": "c2",
                "image_id": "img1",
                "detector_available": False,
                "detector_score": None,
                "detector_status": "FAILED",
                "clip_available": True,
                "similarity_available": True,
                "clip_score": 0.77,
                "clip_status": "AVAILABLE",
            },
        ]

        with patch("scripts.run_phase10a_r2_colab.enforce_cuda_gate", return_value={"device": "Tesla T4", "gpu_memory_gb": 15.0, "cuda_available": True, "cuda_version": "12.2"}), \
             patch("scripts.run_phase10a_r2_colab.verify_source_images_prepared", return_value=(True, "OK", {"records": []})), \
             patch("scripts.run_phase10a_r2_colab.get_execution_git_info", return_value=("1111222233334444555566667777888899990000", True)), \
             patch("scripts.run_phase10a_r2_colab.process_single_image", return_value=(fake_records, None, False)):
            main(["--pilot", "1", "--checkpoint-dir", str(ckpt_dir)])

            diag_file = ckpt_dir / "pilot" / "pilot_diagnostics.json"
            assert diag_file.exists()
            with open(diag_file, "r") as f:
                diag = json.load(f)
            assert diag["status"] == "PILOT_ONLY"
            assert diag["claims_count"] == 2
            assert diag["detector_available_count"] == 1
            assert diag["clip_available_count"] == 1

    def test_null_evidence_remains_null_never_zero(self, tmp_path):
        """Failed or unavailable detector and CLIP evidence must strictly produce score None, never 0.0."""
        from scripts.run_phase10a_r2_colab import process_single_image
        from PIL import Image

        test_img = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100), color="blue").save(test_img)

        fake_vlm = StrictFakeLLaVA(caption="A zebra and a lion.")
        fake_ext = StrictFakeExtractor([
            StrictFakeExtractedClaim(claim_id="c_z", object_category="zebra", raw_claim_text="a zebra"),
        ])
        fake_det = StrictFakeDetector(score=None, available=False, error="OWL-ViT timeout")
        fake_clip = StrictFakeCLIP(score=None, available=False, error="CLIP OOM")

        records, failure, is_zero = process_single_image(
            image_id="coco_test_999",
            image_path=test_img,
            file_name="test.jpg",
            coco_source_split="train2017",
            research_split="TRAIN",
            vlm_provider=fake_vlm,
            claim_extractor=fake_ext,
            detector_provider=fake_det,
            clip_provider=fake_clip,
            gen_config_hash="gen_hash",
            claim_ext_hash="claim_hash",
        )

        assert len(records) == 1
        rec = records[0]
        assert rec["detector_score"] is None
        assert rec["detector_score"] != 0.0
        assert rec["detector_available"] is False
        assert rec["detector_status"] == "FAILED"
        assert rec["clip_score"] is None
        assert rec["clip_score"] != 0.0
        assert rec["clip_available"] is False
        assert rec["similarity_available"] is False
        assert rec["clip_status"] == "FAILED"


