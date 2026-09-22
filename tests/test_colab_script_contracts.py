"""
Regression tests for Colab script configuration and contract validation.

These tests validate the CONFIGURATION and CONTRACT aspects of the Colab
execution script WITHOUT requiring an actual GPU. They verify:

1. Frozen model revision constants match expected values.
2. CUDA gate function raises RuntimeError on CPU.
3. Checkpoint load/create/save semantics.
4. Atomic JSON write semantics.
5. Evidence manifest structure contracts.
6. Script importability.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Import the Colab script module
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import specific functions/constants from the Colab script
from scripts.run_phase10a_r2_colab import (
    FROZEN_MODELS,
    LLAVA_GENERATION_CONFIG,
    CLAIM_EXTRACTION_CONFIG,
    compute_json_hash,
    atomic_json_write,
    load_or_create_checkpoint,
    enforce_cuda_gate,
)


class TestFrozenModelRevisions:
    """Verify frozen model revisions match expected values."""

    def test_llava_model_id(self):
        assert FROZEN_MODELS["vlm_model"] == "llava-hf/llava-1.5-7b-hf"

    def test_llava_revision(self):
        assert FROZEN_MODELS["vlm_revision"] == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"

    def test_owlvit_model_id(self):
        assert FROZEN_MODELS["detector_model"] == "google/owlvit-base-patch32"

    def test_owlvit_revision(self):
        assert FROZEN_MODELS["detector_revision"] == "cbc355fb364588351c5d51c7f74465e8e7ec6f72"

    def test_clip_model_id(self):
        assert FROZEN_MODELS["clip_model"] == "openai/clip-vit-base-patch32"

    def test_clip_revision(self):
        assert FROZEN_MODELS["clip_revision"] == "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class TestGenerationConfig:
    """Verify generation config is correctly frozen."""

    def test_greedy_decoding(self):
        assert LLAVA_GENERATION_CONFIG["do_sample"] is False

    def test_temperature(self):
        assert LLAVA_GENERATION_CONFIG["temperature"] == 0.2

    def test_max_tokens(self):
        assert LLAVA_GENERATION_CONFIG["max_new_tokens"] == 128

    def test_quantization(self):
        assert LLAVA_GENERATION_CONFIG["quantization"] == "4bit_nf4"

    def test_seed(self):
        assert LLAVA_GENERATION_CONFIG["random_seed"] == 42


class TestCUDAGate:
    """Verify CUDA gate rejects CPU execution."""

    def test_cuda_gate_raises_on_cpu(self):
        """The CUDA gate MUST raise RuntimeError when no GPU is available."""
        import torch
        if torch.cuda.is_available():
            pytest.skip("GPU is available — cannot test CPU rejection")

        with pytest.raises(RuntimeError, match="CUDA GPU REQUIRED"):
            enforce_cuda_gate()


class TestComputeJsonHash:
    """Verify deterministic hash computation."""

    def test_deterministic(self):
        data = {"a": 1, "b": [2, 3]}
        h1 = compute_json_hash(data)
        h2 = compute_json_hash(data)
        assert h1 == h2

    def test_key_order_independent(self):
        h1 = compute_json_hash({"a": 1, "b": 2})
        h2 = compute_json_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_content_different_hash(self):
        h1 = compute_json_hash({"a": 1})
        h2 = compute_json_hash({"a": 2})
        assert h1 != h2

    def test_sha256_format(self):
        h = compute_json_hash({"test": True})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestAtomicJsonWrite:
    """Verify atomic write semantics."""

    def test_atomic_write_creates_file(self, tmp_path):
        target = tmp_path / "test.json"
        data = {"key": "value"}
        atomic_json_write(target, data)

        assert target.exists()
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_write_overwrites(self, tmp_path):
        target = tmp_path / "test.json"
        atomic_json_write(target, {"version": 1})
        atomic_json_write(target, {"version": 2})

        with open(target) as f:
            loaded = json.load(f)
        assert loaded["version"] == 2

    def test_atomic_write_creates_parents(self, tmp_path):
        target = tmp_path / "nested" / "dir" / "test.json"
        atomic_json_write(target, {"deep": True})
        assert target.exists()

    def test_no_partial_writes(self, tmp_path):
        """If write fails, original file should remain unchanged."""
        target = tmp_path / "test.json"
        atomic_json_write(target, {"original": True})

        # Attempt to write a non-serializable object should fail
        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            atomic_json_write(target, {"bad": NotSerializable()})

        # Original file should still be intact
        with open(target) as f:
            loaded = json.load(f)
        assert loaded == {"original": True}


class TestCheckpointManagement:
    """Verify checkpoint load/create/save semantics."""

    def test_fresh_checkpoint_creation(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.json"
        ckpt = load_or_create_checkpoint(ckpt_path, "test_hash", 600)

        assert ckpt["checkpoint_type"] == "gpu_acquisition_checkpoint"
        assert ckpt["state"] == "GPU_IN_PROGRESS"
        assert ckpt["sampling_manifest_hash"] == "test_hash"
        assert ckpt["total_target_images"] == 600
        assert ckpt["completed_images"] == 0
        assert len(ckpt["completed_image_ids"]) == 0

    def test_checkpoint_resume(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.json"

        # Create initial checkpoint
        initial = {
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "state": "GPU_IN_PROGRESS",
            "sampling_manifest_hash": "test_hash",
            "total_target_images": 600,
            "completed_images": 50,
            "completed_image_ids": [f"img_{i}" for i in range(50)],
            "failed_image_ids": [],
            "evidence_records": [],
            "failure_records": [],
        }
        with open(ckpt_path, "w") as f:
            json.dump(initial, f)

        # Load should resume
        ckpt = load_or_create_checkpoint(ckpt_path, "test_hash", 600)
        assert ckpt["completed_images"] == 50
        assert len(ckpt["completed_image_ids"]) == 50

    def test_checkpoint_hash_mismatch_creates_fresh(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.json"

        old = {
            "checkpoint_type": "gpu_acquisition_checkpoint",
            "sampling_manifest_hash": "old_hash",
            "completed_images": 100,
            "completed_image_ids": [f"img_{i}" for i in range(100)],
        }
        with open(ckpt_path, "w") as f:
            json.dump(old, f)

        # Different hash → fresh checkpoint
        ckpt = load_or_create_checkpoint(ckpt_path, "new_hash", 600)
        assert ckpt["completed_images"] == 0
        assert ckpt["sampling_manifest_hash"] == "new_hash"


class TestEvidenceManifestContracts:
    """Verify evidence manifest v2 structure expectations."""

    def test_generation_config_hash_stability(self):
        """The generation config hash should not change between runs."""
        h = compute_json_hash(LLAVA_GENERATION_CONFIG)
        assert len(h) == 64
        # Verify it's deterministic across calls
        assert h == compute_json_hash(LLAVA_GENERATION_CONFIG)

    def test_claim_extraction_hash_stability(self):
        h = compute_json_hash(CLAIM_EXTRACTION_CONFIG)
        assert len(h) == 64
        assert h == compute_json_hash(CLAIM_EXTRACTION_CONFIG)
