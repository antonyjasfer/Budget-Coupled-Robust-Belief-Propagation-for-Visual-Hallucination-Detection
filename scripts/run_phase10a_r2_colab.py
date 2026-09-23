"""
Phase 10A-R2 Canonical Colab GPU Execution Script.

ONE-COMMAND runner for Google Colab. This is the ONLY authorized entry point
for producing real multi-modal evidence from the frozen 600-image primary cohort.

Usage in Colab:
    # 1. Dry run validation (no models loaded, no final artifacts written)
    !python scripts/run_phase10a_r2_colab.py --dry-run

    # 2. Pilot run (10 images, persistent Drive checkpoint, PILOT_ONLY status)
    !python scripts/run_phase10a_r2_colab.py --pilot 10 --checkpoint-dir /content/drive/MyDrive/m10a_r2_checkpoints

    # 3. Full scientific run (600 images, resumable, persistent Drive checkpoint)
    !python scripts/run_phase10a_r2_colab.py --full --resume --checkpoint-dir /content/drive/MyDrive/m10a_r2_checkpoints

Requirements:
    - NVIDIA GPU with CUDA (minimum T4 with 15GB VRAM for 4-bit LLaVA)
    - bitsandbytes, transformers, torch with CUDA
    - All 600 source images in data/coco/images/ verified via source_image_audit_v2.json

Execution Flow:
    1. CLI Argument Parsing & Strict Mode Selection (--dry-run | --pilot N | --full)
    2. HARD CUDA GATE — abort immediately if no GPU (except mocked in unit tests)
    3. Load & Verify Frozen Artifacts (manifest, candidate universe, source image audit)
    4. Runtime Environment Capture & Fingerprint Verification
    5. Load or Create Checkpoint (with strict provenance hash and pilot/full namespace isolation)
    6. If --dry-run: report preflight success and exit cleanly without loading models
    7. Load Models according to T4 Memory-Safe Policy:
       - LLaVA-1.5-7B: CUDA, 4-bit NF4, fp16 compute
       - OWL-ViT: CPU
       - CLIP: CPU
    8. Process Images with Bounded Retry Policy & Strict Genuine-Zero Semantics
    9. GPU Complete Gate (attempted == 600 required for GPU_COMPLETE)
    10. Final Evidence Manifest V2 with separated failure vs zero-claim graph metrics
    11. Human Annotation Task Packages V2 (canonical field 'label': null)
    12. Pre-Annotation Freeze V2 with comprehensive verification gates
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Dict, List, Optional, Any, Tuple, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.artifact_state import (
    validate_annotation_task_readiness,
)
from src.vlm.provider import VLMGenerationConfig, VLMResponse
from src.data.schemas import GeneratedResponseRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase10a_r2_colab")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FROZEN MODEL REVISIONS — DO NOT MODIFY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FROZEN_MODELS = {
    "vlm_model": "llava-hf/llava-1.5-7b-hf",
    "vlm_revision": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
    "detector_model": "google/owlvit-base-patch32",
    "detector_revision": "cbc355fb364588351c5d51c7f74465e8e7ec6f72",
    "clip_model": "openai/clip-vit-base-patch32",
    "clip_revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
}

LLAVA_GENERATION_CONFIG = {
    "model_name": FROZEN_MODELS["vlm_model"],
    "prompt": "Describe this image in detail focusing on the main objects present.",
    "max_new_tokens": 128,
    "temperature": 0.2,
    "do_sample": False,
    "seed": 42,
    "random_seed": 42,
    "device": "cuda",
    "dtype": "float16",
    "local_files_only": False,
    "load_in_4bit": True,
    "quantization_type": "nf4",
    "quantization": "4bit_nf4",
    "bnb_4bit_use_double_quant": True,
    "compute_dtype": "float16",
    "device_map": "auto",
}

CLAIM_EXTRACTION_CONFIG = {
    "extractor_version": "conservative_coco80_v2",
    "vocabulary": "coco_80_categories",
    "normalization": "lowercase_lemma_existence",
    "canonicalization": "exact_coco_category_map",
    "duplicate_policy": "unique_per_image",
    "allowed_claim_types": ["object_existence"],
    "filter_rules": ["min_length_3", "no_negations", "direct_syntactic_subject_or_object"],
}

MAX_RETRIES = 3
RETRYABLE_ERROR_CLASSES = {
    "OutOfMemoryError",
    "CUDAOutOfMemoryError",
    "TimeoutError",
    "ConnectionError",
    "HTTPError",
    "URLError",
    "IOError",
    "OSError",
}


def compute_json_hash(data: dict) -> str:
    """Compute SHA-256 of JSON-serialized data with canonical formatting."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_json_write(path: Path, data: Any) -> None:
    """Write JSON atomically: write to temp file in same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem + "_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if path.exists():
            path.unlink()
        shutil.move(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA HARD GATE & RUNTIME ENVIRONMENT FINGERPRINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def enforce_cuda_gate() -> Dict[str, Any]:
    """
    Hard abort if CUDA is not available.

    Returns:
        GPU environment metadata dict.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU REQUIRED. This script must run on a CUDA-enabled device "
            "(e.g., Google Colab with GPU runtime). No model loading on CPU is permitted. "
            "Aborting."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cuda_ver = torch.version.cuda
    torch_ver = torch.__version__

    env = {
        "device": gpu_name,
        "gpu_memory_gb": round(gpu_mem, 2),
        "cuda_available": True,
        "cuda_version": cuda_ver,
        "torch_version": torch_ver,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    logger.info(f"GPU Gate PASSED: {gpu_name} ({gpu_mem:.1f} GB), CUDA {cuda_ver}")
    return env


def capture_runtime_environment(cuda_env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Capture exact versions of runtime dependencies for environment lock and provenance."""
    import torch
    import PIL
    import numpy as np
    import scipy

    def get_mod_version(name: str) -> str:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except ImportError:
            return "not_installed"

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": get_mod_version("torchvision"),
        "transformers_version": get_mod_version("transformers"),
        "accelerate_version": get_mod_version("accelerate"),
        "bitsandbytes_version": get_mod_version("bitsandbytes"),
        "pillow_version": PIL.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0,
    }
    if cuda_env:
        env.update({k: v for k, v in cuda_env.items() if v is not None})
    return env


def compute_environment_fingerprint(env: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 of locked runtime dependencies."""
    locked_keys = [
        "python_version", "torch_version", "torchvision_version",
        "transformers_version", "accelerate_version", "bitsandbytes_version",
        "pillow_version", "numpy_version", "scipy_version",
    ]
    locked_dict = {k: env.get(k) for k in locked_keys}
    return compute_json_hash(locked_dict)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SOURCE IMAGE VERIFICATION GATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def verify_source_images_prepared(
    image_dir: Path,
    audit_p: Path,
    expected_count: int = 600,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Verify that all 600 source images are present and pass the acquisition gate.
    Requires requested=600, valid=600, missing=0, corrupt=0.
    """
    if not audit_p.exists():
        msg = (
            f"Source image audit missing: {audit_p}. "
            "COCO source images are gitignored and must be acquired before GPU inference. "
            "Please run: python scripts/acquire_and_verify_source_images_v2.py"
        )
        return False, msg, {}

    try:
        with open(audit_p, "r", encoding="utf-8") as f:
            audit = json.load(f)
    except Exception as e:
        return False, f"Failed to read source image audit {audit_p}: {e}", {}

    valid_count = audit.get("valid_count", 0)
    missing_count = audit.get("missing_count", 0)
    corrupt_count = audit.get("corrupt_count", 0)
    total_requested = audit.get("total_requested", 0)

    if (
        total_requested != expected_count
        or valid_count != expected_count
        or missing_count > 0
        or corrupt_count > 0
    ):
        msg = (
            f"Source image audit shows incomplete acquisition: "
            f"Valid={valid_count}/{expected_count}, Missing={missing_count}, Corrupt={corrupt_count}. "
            "Please run: python scripts/acquire_and_verify_source_images_v2.py"
        )
        return False, msg, audit

    if not image_dir.exists():
        msg = f"Image directory does not exist: {image_dir}"
        return False, msg, audit

    return True, "Source images fully verified (600/600 valid).", audit


def compute_stable_source_audit_hash(audit_data: Dict[str, Any]) -> str:
    """
    Compute a deterministic scientific fingerprint of source image audit data.
    Excludes volatile fields like timestamp, local paths, or download timing.
    Includes sampling_manifest_hash, counts, and per-image canonical metadata/sha256.
    """
    records = audit_data.get("records", [])
    stable_records = []
    for r in records:
        rec_entry = {
            "image_id": r.get("image_id"),
            "coco_integer_id": r.get("coco_integer_id"),
            "file_name": r.get("file_name"),
            "coco_source_split": r.get("coco_source_split"),
            "research_split": r.get("research_split"),
            "expected_width": r.get("expected_width"),
            "expected_height": r.get("expected_height"),
            "actual_width": r.get("actual_width"),
            "actual_height": r.get("actual_height"),
            "sha256": r.get("sha256"),
            "status": r.get("status"),
            "is_valid": r.get("is_valid"),
        }
        stable_records.append(rec_entry)

    # Sort deterministically by image_id
    stable_records.sort(key=lambda x: str(x.get("image_id", "")))

    payload = {
        "sampling_manifest_hash": audit_data.get("sampling_manifest_hash", ""),
        "total_requested": audit_data.get("total_requested", 0),
        "valid_count": audit_data.get("valid_count", 0),
        "missing_count": audit_data.get("missing_count", 0),
        "corrupt_count": audit_data.get("corrupt_count", 0),
        "records": stable_records,
    }
    return compute_json_hash(payload)


def get_execution_git_info(cwd: Optional[Path] = None) -> Tuple[str, bool]:
    """
    Get actual execution Git commit SHA and working tree cleanliness.

    Returns:
        (execution_code_sha, working_tree_clean)
    """
    import subprocess

    target_dir = str(cwd or PROJECT_ROOT)
    try:
        sha_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        head_sha = sha_proc.stdout.strip()

        diff_proc = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=target_dir,
            capture_output=True,
        )
        unstaged_clean = (diff_proc.returncode == 0)

        cached_proc = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=target_dir,
            capture_output=True,
        )
        staged_clean = (cached_proc.returncode == 0)

        working_tree_clean = unstaged_clean and staged_clean
        return head_sha, working_tree_clean
    except Exception as e:
        logger.warning(f"Failed to query git status: {e}")
        return "unknown", False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINT PROVENANCE & MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_checkpoint_provenance(
    sampling_manifest_hash: str,
    audit_hash: str,
    execution_code_sha: str = "",
    gen_cfg_hash: str = "",
    claim_ext_hash: str = "",
    working_tree_clean: bool = True,
    sampling_manifest_creation_sha: str = "",
    code_sha: str = "",
) -> Tuple[Dict[str, Any], str]:
    """
    Compute exact scientific provenance fingerprint for checkpoint binding.
    Checkpoints cannot be resumed if any component of this fingerprint differs.
    """
    if not execution_code_sha and code_sha:
        execution_code_sha = code_sha
    fingerprint = {
        "dataset_version": "v2",
        "sampling_manifest_hash": sampling_manifest_hash,
        "source_image_audit_hash": audit_hash,
        "vlm_model": FROZEN_MODELS["vlm_model"],
        "vlm_revision": FROZEN_MODELS["vlm_revision"],
        "processor_revision": FROZEN_MODELS["vlm_revision"],
        "generation_config_hash": gen_cfg_hash,
        "claim_extractor_hash": claim_ext_hash,
        "detector_model": FROZEN_MODELS["detector_model"],
        "detector_revision": FROZEN_MODELS["detector_revision"],
        "clip_model": FROZEN_MODELS["clip_model"],
        "clip_revision": FROZEN_MODELS["clip_revision"],
        "execution_code_sha": execution_code_sha,
        "working_tree_clean": working_tree_clean,
        "sampling_manifest_creation_sha": sampling_manifest_creation_sha,
    }
    prov_hash = compute_json_hash(fingerprint)
    return fingerprint, prov_hash


def classify_failure(err: Exception) -> Tuple[str, bool]:
    """Classify failure as ('RETRYABLE', True) or ('TERMINAL', False)."""
    err_class = type(err).__name__
    err_msg = str(err).lower()
    if (
        err_class in RETRYABLE_ERROR_CLASSES
        or "cuda out of memory" in err_msg
        or "timed out" in err_msg
        or "connection" in err_msg
    ):
        return "RETRYABLE", True
    return "TERMINAL", False


def load_or_create_checkpoint(
    checkpoint_path: Path,
    expected_checkpoint_type: str,
    provenance_fingerprint: Dict[str, Any],
    provenance_hash: str,
    total_images: int,
    resume: bool,
) -> Dict[str, Any]:
    """
    Load existing checkpoint or create a fresh one with strict provenance binding.

    On resume:
        - Must match expected_checkpoint_type (pilot vs full isolation)
        - Must match checkpoint_provenance_hash (no config drift)
        - If mismatch: HARD FAIL
    """
    if checkpoint_path.exists():
        if not resume:
            raise RuntimeError(
                f"Checkpoint already exists at {checkpoint_path}. "
                "To resume this run, supply --resume. "
                "To start a fresh acquisition, specify a different --checkpoint-dir."
            )

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        # 1. Type isolation check (pilot vs full)
        found_type = ckpt.get("checkpoint_type")
        if found_type != expected_checkpoint_type:
            raise RuntimeError(
                f"CHECKPOINT TYPE MISMATCH: Found '{found_type}', expected '{expected_checkpoint_type}'. "
                "Pilot checkpoints must NEVER be consumed by --full --resume."
            )

        # 2. Strict provenance fingerprint check
        found_prov_hash = ckpt.get("checkpoint_provenance_hash")
        if found_prov_hash != provenance_hash:
            raise RuntimeError(
                f"CHECKPOINT PROVENANCE HASH MISMATCH: "
                f"Expected {provenance_hash}, found {found_prov_hash}. "
                "The scientific configuration, model revisions, code SHA, or dataset manifests have changed. "
                "Cannot merge or resume into a divergent checkpoint namespace."
            )

        completed = len(ckpt.get("completed_image_ids", []))
        failed = len(ckpt.get("failed_image_ids", []))
        logger.info(
            f"Resuming from verified checkpoint: {completed} completed, "
            f"{failed} failed out of {total_images} target images."
        )
        return ckpt

    # Fresh checkpoint
    return {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "checkpoint_type": expected_checkpoint_type,
        "state": "GPU_IN_PROGRESS" if expected_checkpoint_type == "gpu_acquisition_checkpoint" else "PILOT_IN_PROGRESS",
        "checkpoint_provenance_hash": provenance_hash,
        "provenance_fingerprint": provenance_fingerprint,
        "sampling_manifest_hash": provenance_fingerprint.get("sampling_manifest_hash", ""),
        "total_target_images": total_images,
        "completed_images": 0,
        "failed_images": 0,
        "genuine_zero_claim_images": 0,
        "completed_image_ids": [],
        "genuine_zero_image_ids": [],
        "failed_image_ids": [],
        "evidence_records": [],
        "failure_records": [],
        "execution_environment": {},
        "last_checkpoint_timestamp": datetime.now(timezone.utc).isoformat(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_checkpoint(checkpoint_path: Path, checkpoint: Dict[str, Any]) -> None:
    """Atomically save checkpoint to disk."""
    checkpoint["last_checkpoint_timestamp"] = datetime.now(timezone.utc).isoformat()
    atomic_json_write(checkpoint_path, checkpoint)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SINGLE-IMAGE PROCESSING (STRICT GENUINE-ZERO & TYPED EVIDENCE FAILURES)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_single_image(
    image_id: str,
    image_path: Path,
    research_split: str,
    coco_source_split: str,
    file_name: str,
    vlm_provider: Any,
    claim_extractor: Any,
    detector_provider: Any,
    clip_provider: Any,
    gen_config_hash: str,
    claim_ext_hash: str,
    attempt_count: int = 1,
    vlm_gen_config: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """
    Process a single image through the full evidence pipeline using canonical provider contracts.

    Strict genuine-zero semantics:
        - Image load SUCCESS
        - LLaVA inference SUCCESS with NON-EMPTY response
        - Claim extraction SUCCESS
        - EXACTLY 0 eligible object-existence claims
        => is_genuine_zero = True, failure_record = None, evidence_records = []

    If LLaVA returns empty/invalid text:
        => Typed failure (EmptyCaptionError / VLM_EMPTY_RESPONSE), NOT genuine zero.

    Returns:
        (evidence_records, failure_record_or_none, is_genuine_zero)
    """
    from PIL import Image

    evidence_records = []
    failure_record = None

    try:
        # 1. Load & verify image on disk
        if not image_path.exists():
            failure_record = {
                "claim_id": f"img_level_{image_id}",
                "image_id": image_id,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "provider": FROZEN_MODELS["vlm_model"],
                "failure_state": "FAILED",
                "reason_code": "SOURCE_IMAGE_NOT_FOUND",
                "attempt_count": attempt_count,
                "failure_class": "TERMINAL",
                "retryable": False,
                "last_error_class": "FileNotFoundError",
                "last_error_message": f"Image file not found: {image_path}",
                "final_status": "TERMINAL_FAILURE",
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "generation_config_hash": gen_config_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return [], failure_record, False

        try:
            with Image.open(str(image_path)) as img_check:
                img_check.verify()
        except Exception as img_err:
            f_class, is_retry = classify_failure(img_err)
            failure_record = {
                "claim_id": f"img_level_{image_id}",
                "image_id": image_id,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "provider": FROZEN_MODELS["vlm_model"],
                "failure_state": "FAILED",
                "reason_code": f"IMAGE_LOAD_FAILED: {type(img_err).__name__}",
                "attempt_count": attempt_count,
                "failure_class": f_class,
                "retryable": is_retry,
                "last_error_class": type(img_err).__name__,
                "last_error_message": str(img_err)[:500],
                "final_status": "RETRY_EXHAUSTED" if (not is_retry or attempt_count >= MAX_RETRIES) else "FAILED",
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "generation_config_hash": gen_config_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return [], failure_record, False

        # 2. Canonical LLaVA forward inference (takes image_path, returns VLMResponse)
        prompt_text = LLAVA_GENERATION_CONFIG.get("prompt", "Describe this image in detail focusing on the main objects present.")
        vlm_response = vlm_provider.generate_caption(
            image_path=image_path,
            prompt=prompt_text,
            gen_config=vlm_gen_config,
            image_id=image_id,
        )

        caption_text = vlm_response.caption if vlm_response is not None else ""

        # Validate provenance: must be real_inference and not synthetic
        is_synth = getattr(vlm_response, "is_synthetic", False)
        gen_source = getattr(vlm_response, "generation_source", "real_inference")
        if is_synth or gen_source != "real_inference":
            failure_record = {
                "claim_id": f"img_level_{image_id}",
                "image_id": image_id,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "provider": FROZEN_MODELS["vlm_model"],
                "failure_state": "FAILED",
                "reason_code": "INVALID_PROVENANCE",
                "attempt_count": attempt_count,
                "failure_class": "TERMINAL",
                "retryable": False,
                "last_error_class": "InvalidProvenanceError",
                "last_error_message": f"Expected real_inference with is_synthetic=False, got source={gen_source}, is_synthetic={is_synth}",
                "final_status": "TERMINAL_FAILURE",
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "generation_config_hash": gen_config_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.warning(f"VLM returned invalid provenance for {image_id}; recorded as TYPED FAILURE.")
            return [], failure_record, False

        # Strict validation: empty response is a typed failure, NEVER genuine zero
        if not caption_text or not str(caption_text).strip():
            failure_record = {
                "claim_id": f"img_level_{image_id}",
                "image_id": image_id,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "provider": FROZEN_MODELS["vlm_model"],
                "failure_state": "FAILED",
                "reason_code": "VLM_EMPTY_RESPONSE",
                "attempt_count": attempt_count,
                "failure_class": "TERMINAL",
                "retryable": False,
                "last_error_class": "EmptyCaptionError",
                "last_error_message": "LLaVA generated empty or whitespace-only caption",
                "final_status": "TERMINAL_FAILURE",
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "generation_config_hash": gen_config_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.warning(f"VLM generated empty response for {image_id}; recorded as TYPED FAILURE.")
            return [], failure_record, False

        # 3. Canonical conservative claim extraction via GeneratedResponseRecord
        try:
            resp_record = GeneratedResponseRecord(
                response_id=vlm_response.response_id,
                image_id=image_id,
                model_name=getattr(vlm_response, "model_name", FROZEN_MODELS["vlm_model"]),
                response_text=caption_text,
            )
            report = claim_extractor.extract_from_response(resp_record)
            claims = report.accepted_claims
        except Exception as claim_err:
            f_class, is_retry = classify_failure(claim_err)
            failure_record = {
                "claim_id": f"img_level_{image_id}",
                "image_id": image_id,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "provider": FROZEN_MODELS["vlm_model"],
                "failure_state": "FAILED",
                "reason_code": f"CLAIM_EXTRACTION_EXCEPTION: {type(claim_err).__name__}",
                "attempt_count": attempt_count,
                "failure_class": f_class,
                "retryable": is_retry,
                "last_error_class": type(claim_err).__name__,
                "last_error_message": str(claim_err)[:500],
                "final_status": "RETRY_EXHAUSTED" if (not is_retry or attempt_count >= MAX_RETRIES) else "FAILED",
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "generation_config_hash": gen_config_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return [], failure_record, False

        # 4. Genuine Zero Claim Check
        if len(claims) == 0:
            logger.info(f"Image {image_id}: Valid caption produced, but 0 COCO claims (GENUINE_ZERO_CLAIM).")
            return [], None, True

        # 5. Extract OWL-ViT and CLIP evidence with canonical provider methods
        for claim in claims:
            claim_id = claim.claim_id
            cat = claim.object_category
            raw_claim_text = claim.raw_claim_text

            # OWL-ViT detection: detect_category(image_path=image_path, category=cat)
            det_score = None
            det_available = False
            det_status = "UNAVAILABLE"
            det_failure_info = None

            try:
                det_result = detector_provider.detect_category(
                    image_path=image_path,
                    category=cat,
                )
                if det_result is not None and det_result.available and det_result.score is not None:
                    det_score = float(det_result.score)
                    det_available = True
                    det_status = "AVAILABLE"
                else:
                    det_status = "FAILED"
                    err_msg = det_result.error if (det_result and det_result.error) else "OWL-ViT detector returned unavailable or None score"
                    det_failure_info = {
                        "provider": FROZEN_MODELS["detector_model"],
                        "revision": FROZEN_MODELS["detector_revision"],
                        "attempt_count": 1,
                        "reason_code": "DETECTOR_FAILED",
                        "error_class": "DetectorError",
                        "error_message": str(err_msg)[:500],
                    }
            except Exception as det_err:
                det_status = "FAILED"
                det_failure_info = {
                    "provider": FROZEN_MODELS["detector_model"],
                    "revision": FROZEN_MODELS["detector_revision"],
                    "attempt_count": 1,
                    "reason_code": f"DETECTOR_EXCEPTION: {type(det_err).__name__}",
                    "error_class": type(det_err).__name__,
                    "error_message": str(det_err)[:500],
                }
                logger.warning(f"OWL-ViT failed for {claim_id}: {det_err}")

            # CLIP similarity: compute_similarity(image_path=image_path, text=cat)
            clip_score = None
            clip_available = False
            clip_status = "UNAVAILABLE"
            clip_failure_info = None

            try:
                clip_result = clip_provider.compute_similarity(
                    image_path=image_path,
                    text=cat,
                )
                if clip_result is not None and clip_result.available and clip_result.score is not None:
                    clip_score = float(clip_result.score)
                    clip_available = True
                    clip_status = "AVAILABLE"
                else:
                    clip_status = "FAILED"
                    err_msg = clip_result.error if (clip_result and clip_result.error) else "CLIP returned unavailable or None score"
                    clip_failure_info = {
                        "provider": FROZEN_MODELS["clip_model"],
                        "revision": FROZEN_MODELS["clip_revision"],
                        "attempt_count": 1,
                        "reason_code": "CLIP_FAILED",
                        "error_class": "CLIPError",
                        "error_message": str(err_msg)[:500],
                    }
            except Exception as clip_err:
                clip_status = "FAILED"
                clip_failure_info = {
                    "provider": FROZEN_MODELS["clip_model"],
                    "revision": FROZEN_MODELS["clip_revision"],
                    "attempt_count": 1,
                    "reason_code": f"CLIP_EXCEPTION: {type(clip_err).__name__}",
                    "error_class": type(clip_err).__name__,
                    "error_message": str(clip_err)[:500],
                }
                logger.warning(f"CLIP failed for {claim_id}: {clip_err}")

            record = {
                "claim_id": claim_id,
                "image_id": image_id,
                "file_name": file_name,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "object_category": cat,
                "raw_claim_text": raw_claim_text,
                "raw_caption": caption_text,
                "response_id": vlm_response.response_id,
                "detector_score": det_score,  # float or None (null in JSON)
                "detector_available": det_available,
                "detector_status": det_status,
                "detector_failure_info": det_failure_info,
                "clip_score": clip_score,  # float or None (null in JSON)
                "similarity_available": clip_available,
                "clip_available": clip_available,
                "clip_status": clip_status,
                "clip_failure_info": clip_failure_info,
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "detector_revision": FROZEN_MODELS["detector_revision"],
                "clip_revision": FROZEN_MODELS["clip_revision"],
                "generation_config_hash": gen_config_hash,
                "claim_extractor_hash": claim_ext_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if hasattr(vlm_response, "image_hash") and vlm_response.image_hash:
                record["image_hash"] = vlm_response.image_hash
            if hasattr(vlm_response, "prompt_hash") and vlm_response.prompt_hash:
                record["prompt_hash"] = vlm_response.prompt_hash
            if hasattr(vlm_response, "execution_time_seconds"):
                record["vlm_execution_time_seconds"] = vlm_response.execution_time_seconds

            evidence_records.append(record)

    except Exception as err:
        f_class, is_retry = classify_failure(err)
        failure_record = {
            "claim_id": f"img_level_{image_id}",
            "image_id": image_id,
            "coco_source_split": coco_source_split,
            "research_split": research_split,
            "provider": FROZEN_MODELS["vlm_model"],
            "failure_state": "FAILED",
            "reason_code": f"PIPELINE_EXCEPTION: {type(err).__name__}",
            "attempt_count": attempt_count,
            "failure_class": f_class,
            "retryable": is_retry,
            "last_error_class": type(err).__name__,
            "last_error_message": str(err)[:500],
            "final_status": "RETRY_EXHAUSTED" if (not is_retry or attempt_count >= MAX_RETRIES) else "FAILED",
            "provenance_status": "REAL_UNLABELED",
            "model_revision": FROZEN_MODELS["vlm_revision"],
            "generation_config_hash": gen_config_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(f"Pipeline failure for {image_id}: {err}")
        return [], failure_record, False

    return evidence_records, None, False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse and validate CLI arguments with mutually exclusive required mode.
    """
    parser = argparse.ArgumentParser(
        description="Phase 10A-R2 Canonical Colab GPU Execution Script."
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform environment, artifact, and source image validation only. Do not load models or write final artifacts.",
    )
    mode_group.add_argument(
        "--pilot",
        type=int,
        metavar="N",
        help="Process exactly N frozen-manifest images. Write checkpoint/preview with PILOT_ONLY status. Do not finalize scientific artifacts.",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Process all 600 images in the primary cohort.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Reuse valid checkpoint entries and process only remaining/retryable images.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Write persistent checkpoint files under this path (e.g. Google Drive mount).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output root for final transferable artifacts.",
    )

    args = parser.parse_args(argv)

    if args.pilot is not None and args.pilot <= 0:
        parser.error("--pilot N must be a positive integer greater than 0.")

    return args


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("Phase 10A-R2 Colab GPU Execution — START")
    logger.info(f"Mode: {'DRY-RUN' if args.dry_run else ('PILOT (' + str(args.pilot) + ' images)' if args.pilot else 'FULL (600 images)')}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Checkpoint Dir: {args.checkpoint_dir or 'DEFAULT (data/manifests)'}")
    logger.info(f"Output Dir: {args.output_dir or 'DEFAULT'}")
    logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # Step 1: CUDA Gate & Environment Verification
    # ──────────────────────────────────────────────────────────────
    gpu_env = enforce_cuda_gate()
    runtime_env = capture_runtime_environment(gpu_env)
    env_fingerprint = compute_environment_fingerprint(runtime_env)
    logger.info(f"Runtime environment fingerprint: {env_fingerprint[:16]}...")

    # ──────────────────────────────────────────────────────────────
    # Step 2: Load Frozen Artifacts & Hashes
    # ──────────────────────────────────────────────────────────────
    manifest_p = PROJECT_ROOT / "data" / "manifests" / "final_sampling_manifest_v2.json"
    universe_p = PROJECT_ROOT / "data" / "manifests" / "coco_candidate_universe_v2.json"
    corruption_p = PROJECT_ROOT / "data" / "manifests" / "final_corruption_manifest_v2.json"
    audit_p = PROJECT_ROOT / "reports" / "m10a_recovery2" / "source_image_audit_v2.json"
    image_dir = PROJECT_ROOT / "data" / "coco" / "images"

    for required in [manifest_p, universe_p]:
        if not required.exists():
            raise FileNotFoundError(f"Required artifact missing: {required}")

    with open(manifest_p, "r", encoding="utf-8") as f:
        sampling_data = json.load(f)
    with open(universe_p, "r", encoding="utf-8") as f:
        universe_data = json.load(f)

    sampling_hash = sampling_data.get("manifest_hash", "")
    universe_hash = universe_data.get("universe_hash", "")
    gen_cfg_hash = compute_json_hash(LLAVA_GENERATION_CONFIG)
    claim_ext_hash = compute_json_hash(CLAIM_EXTRACTION_CONFIG)

    image_ids = sampling_data.get("selected_image_ids", [])
    research_splits = sampling_data.get("research_splits", {})
    coco_source_splits = sampling_data.get("coco_source_splits", {})
    images_meta = {img["image_id"]: img for img in sampling_data.get("images", [])}

    if len(image_ids) != 600:
        raise ValueError(f"Expected exactly 600 images in sampling manifest, found {len(image_ids)}")

    sampling_manifest_creation_sha = sampling_data.get("code_sha", "unknown")

    # Actual Execution Git HEAD and Working Tree Verification
    execution_sha, working_tree_clean = get_execution_git_info()
    logger.info(f"Execution Git SHA: {execution_sha} (Clean: {working_tree_clean})")

    # Clean working tree gate for scientific execution (PILOT / FULL)
    if not args.dry_run and not working_tree_clean:
        raise RuntimeError(
            f"DIRTY GIT WORKING TREE DETECTED: Execution Git SHA is {execution_sha}, "
            "but uncommitted or staged changes are present in the working tree. "
            "Scientific GPU execution (--pilot / --full) strictly requires a clean Git working tree. "
            "Commit or stash all changes and verify 'git status' before proceeding."
        )

    # ──────────────────────────────────────────────────────────────
    # Step 3: Source Image Preparation Gate
    # ──────────────────────────────────────────────────────────────
    images_ok, images_msg, audit_data = verify_source_images_prepared(
        image_dir=image_dir,
        audit_p=audit_p,
        expected_count=600,
    )
    if not images_ok:
        logger.error(images_msg)
        raise RuntimeError(images_msg)

    # Compute deterministic scientific fingerprint of source image audit data
    audit_hash = compute_stable_source_audit_hash(audit_data)
    logger.info(f"Stable source image audit hash: {audit_hash[:16]}...")

    corruption_hash = ""
    if corruption_p.exists():
        with open(corruption_p, "r", encoding="utf-8") as f:
            corruption_data = json.load(f)
            corruption_hash = corruption_data.get("manifest_hash", "")

    # Compute checkpoint provenance fingerprint
    prov_fingerprint, prov_hash = compute_checkpoint_provenance(
        sampling_manifest_hash=sampling_hash,
        audit_hash=audit_hash,
        execution_code_sha=execution_sha,
        gen_cfg_hash=gen_cfg_hash,
        claim_ext_hash=claim_ext_hash,
        working_tree_clean=working_tree_clean,
        sampling_manifest_creation_sha=sampling_manifest_creation_sha,
    )
    logger.info(f"Checkpoint provenance hash: {prov_hash[:16]}...")

    # ──────────────────────────────────────────────────────────────
    # Step 4: Resolve Checkpoint Paths (Physical Isolation)
    # ──────────────────────────────────────────────────────────────
    base_checkpoint_root = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else PROJECT_ROOT / "data" / "manifests"
    )

    if args.pilot:
        ckpt_sub = base_checkpoint_root / "pilot"
        ckpt_path = ckpt_sub / "gpu_acquisition_checkpoint_v2.json"
        expected_ckpt_type = "pilot_acquisition_checkpoint"
        target_image_ids = image_ids[: args.pilot]
    else:
        ckpt_sub = base_checkpoint_root / "full"
        ckpt_path = ckpt_sub / "gpu_acquisition_checkpoint_v2.json"
        expected_ckpt_type = "gpu_acquisition_checkpoint"
        target_image_ids = image_ids

    # ──────────────────────────────────────────────────────────────
    # Step 5: DRY-RUN Mode Execution
    # ──────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY-RUN VALIDATION COMPLETE")
        logger.info("  CUDA Gate: PASSED")
        logger.info(f"  Device: {gpu_env['device']} ({gpu_env['gpu_memory_gb']} GB)")
        logger.info(f"  Execution Git SHA: {execution_sha}")
        logger.info(f"  Working Tree Clean: {working_tree_clean}")
        logger.info(f"  Source Images: 600/600 verified (stable audit hash: {audit_hash[:16]}...)")
        logger.info(f"  Sampling Manifest: 600 images (hash: {sampling_hash[:16]}...)")
        logger.info(f"  Provenance Hash: {prov_hash[:16]}...")
        logger.info(f"  Target Checkpoint Path: {ckpt_path}")
        logger.info("DRY-RUN SUCCESS: No models loaded, no final artifacts written.")
        logger.info("=" * 60)
        return

    # ──────────────────────────────────────────────────────────────
    # Step 6: Environment Lock Verification (A8)
    # ──────────────────────────────────────────────────────────────
    lock_file = base_checkpoint_root / "colab_runtime_environment_lock.json"
    if args.full and lock_file.exists():
        with open(lock_file, "r", encoding="utf-8") as f:
            locked_env = json.load(f)
        locked_fp = compute_environment_fingerprint(locked_env)
        if locked_fp != env_fingerprint:
            raise RuntimeError(
                f"RUNTIME ENVIRONMENT DRIFT DETECTED: "
                f"Current fingerprint {env_fingerprint} differs from locked pilot {locked_fp}. "
                "Reinstall exact recorded package versions before running full acquisition."
            )
        logger.info("Runtime environment lock verified against pilot fingerprint.")

    # ──────────────────────────────────────────────────────────────
    # Step 7: Load or Create Checkpoint
    # ──────────────────────────────────────────────────────────────
    checkpoint = load_or_create_checkpoint(
        checkpoint_path=ckpt_path,
        expected_checkpoint_type=expected_ckpt_type,
        provenance_fingerprint=prov_fingerprint,
        provenance_hash=prov_hash,
        total_images=len(target_image_ids),
        resume=args.resume,
    )
    checkpoint["execution_environment"] = runtime_env

    completed_set = set(checkpoint.get("completed_image_ids", []))
    failed_records_by_img = {
        r["image_id"]: r for r in checkpoint.get("failure_records", []) if "image_id" in r
    }

    # Determine remaining images, considering retry policy
    remaining: List[str] = []
    for iid in target_image_ids:
        if iid in completed_set:
            continue
        if iid in failed_records_by_img:
            f_rec = failed_records_by_img[iid]
            if f_rec.get("failure_class") == "RETRYABLE" and f_rec.get("attempt_count", 1) < MAX_RETRIES:
                logger.info(f"Image {iid} marked RETRYABLE (attempt {f_rec.get('attempt_count')}). Scheduling retry.")
                remaining.append(iid)
            else:
                # Terminal failure or max retries exhausted -> skip
                continue
        else:
            remaining.append(iid)

    logger.info(
        f"Target Images: {len(target_image_ids)}, "
        f"Completed: {len(completed_set)}, "
        f"Failed/Terminal: {len(checkpoint.get('failed_image_ids', [])) - (len(target_image_ids) - len(completed_set) - len(remaining))}, "
        f"Remaining To Process: {len(remaining)}"
    )

    # ──────────────────────────────────────────────────────────────
    # Step 8: Initialize Models (T4 Memory-Safe Device Policy)
    # ──────────────────────────────────────────────────────────────
    logger.info("Initializing models under T4 Memory-Safe Configuration...")
    from src.vlm.llava_provider import LLaVA15Provider
    from src.claims.extraction import ConservativeClaimExtractor
    from src.claims.vocabulary import create_coco_category_registry
    from src.evidence.detector_provider import HuggingFaceDetectorProvider
    from src.evidence.clip_provider import TransformersCLIPProvider

    vlm_gen_config = VLMGenerationConfig.from_dict(LLAVA_GENERATION_CONFIG)

    # LLaVA on CUDA with 4-bit NF4 (lazy-loaded on first inference)
    vlm_provider = LLaVA15Provider(
        model_name=FROZEN_MODELS["vlm_model"],
        model_revision=FROZEN_MODELS["vlm_revision"],
        device="cuda",
        dtype="float16",
        load_in_4bit=True,
        quantization_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        device_map="auto",
        local_files_only=False,
        allow_download=True,
    )
    logger.info("LLaVA provider initialized; model weights will lazy-load on first inference.")

    claim_extractor = ConservativeClaimExtractor(create_coco_category_registry())
    logger.info("Conservative claim extractor initialized.")

    # OWL-ViT and CLIP on CPU by default (M6 memory-safe policy for T4 16GB)
    detector_provider = HuggingFaceDetectorProvider(
        model_name=FROZEN_MODELS["detector_model"],
        model_revision=FROZEN_MODELS["detector_revision"],
        device="cpu",
        local_files_only=False,
    )
    logger.info("OWL-ViT detector provider initialized (CPU, local_files_only=False); model weights will lazy-load on first inference.")

    clip_provider = TransformersCLIPProvider(
        model_name=FROZEN_MODELS["clip_model"],
        model_revision=FROZEN_MODELS["clip_revision"],
        device="cpu",
        local_files_only=False,
    )
    logger.info("CLIP provider initialized (CPU, local_files_only=False); model weights will lazy-load on first inference.")

    # ──────────────────────────────────────────────────────────────
    # Step 9: Process Images with Atomic Checkpointing
    # ──────────────────────────────────────────────────────────────
    CHECKPOINT_INTERVAL = 10
    start_time = time.time()

    for idx, img_id in enumerate(remaining):
        meta = images_meta.get(img_id, {})
        file_name = meta.get("file_name", f"{img_id}.jpg")
        r_split = research_splits.get(img_id, "TRAIN")
        c_split = coco_source_splits.get(img_id, "train2017")

        img_path = image_dir / file_name
        if not img_path.exists():
            numeric_id = img_id.replace("coco_", "")
            img_path = image_dir / f"{numeric_id}.jpg"

        prev_attempts = failed_records_by_img.get(img_id, {}).get("attempt_count", 0)
        current_attempt = prev_attempts + 1

        records, failure, is_genuine_zero = process_single_image(
            image_id=img_id,
            image_path=img_path,
            research_split=r_split,
            coco_source_split=c_split,
            file_name=file_name,
            vlm_provider=vlm_provider,
            claim_extractor=claim_extractor,
            detector_provider=detector_provider,
            clip_provider=clip_provider,
            gen_config_hash=gen_cfg_hash,
            claim_ext_hash=claim_ext_hash,
            attempt_count=current_attempt,
            vlm_gen_config=vlm_gen_config,
        )

        if not failure and len(checkpoint.get("completed_image_ids", [])) == 0:
            logger.info("First LLaVA inference successful; model weights active.")

        if failure:
            if img_id not in checkpoint["failed_image_ids"]:
                checkpoint["failed_image_ids"].append(img_id)
            checkpoint["failed_images"] = len(checkpoint["failed_image_ids"])
            # Update failure records
            checkpoint["failure_records"] = [
                r for r in checkpoint["failure_records"] if r.get("image_id") != img_id
            ]
            checkpoint["failure_records"].append(failure)
        else:
            if img_id not in checkpoint["completed_image_ids"]:
                checkpoint["completed_image_ids"].append(img_id)
            checkpoint["completed_images"] = len(checkpoint["completed_image_ids"])

            # Clean any old failure records if this was a retry
            checkpoint["failure_records"] = [
                r for r in checkpoint["failure_records"] if r.get("image_id") != img_id
            ]
            checkpoint["failed_image_ids"] = [
                iid for iid in checkpoint["failed_image_ids"] if iid != img_id
            ]
            checkpoint["failed_images"] = len(checkpoint["failed_image_ids"])

            if is_genuine_zero:
                if img_id not in checkpoint.get("genuine_zero_image_ids", []):
                    if "genuine_zero_image_ids" not in checkpoint:
                        checkpoint["genuine_zero_image_ids"] = []
                    checkpoint["genuine_zero_image_ids"].append(img_id)
                checkpoint["genuine_zero_claim_images"] = len(checkpoint["genuine_zero_image_ids"])
            else:
                checkpoint["evidence_records"].extend(records)

        # Periodic checkpoint
        processed_this_session = idx + 1
        if processed_this_session % CHECKPOINT_INTERVAL == 0:
            elapsed = time.time() - start_time
            rate = processed_this_session / elapsed if elapsed > 0 else 0
            logger.info(
                f"Progress: {processed_this_session}/{len(remaining)} "
                f"({rate:.1f} img/s), "
                f"claims={len(checkpoint['evidence_records'])}, "
                f"failures={checkpoint['failed_images']}"
            )
            save_checkpoint(ckpt_path, checkpoint)

    # ──────────────────────────────────────────────────────────────
    # Step 10: State Machine & GPU Complete Gate
    # ──────────────────────────────────────────────────────────────
    total_attempted = len(checkpoint["completed_image_ids"]) + len(checkpoint["failed_image_ids"])

    if args.pilot:
        evidence_records = checkpoint["evidence_records"]
        claims_count = len(evidence_records)
        detector_available_count = sum(
            1 for r in evidence_records
            if r.get("detector_available") is True and r.get("detector_score") is not None
        )
        detector_failed_count = claims_count - detector_available_count
        clip_available_count = sum(
            1 for r in evidence_records
            if (r.get("clip_available") is True or r.get("similarity_available") is True) and r.get("clip_score") is not None
        )
        clip_failed_count = claims_count - clip_available_count

        logger.info(
            f"Pilot evidence health check: claims={claims_count}, "
            f"detector_available={detector_available_count}/{claims_count}, "
            f"clip_available={clip_available_count}/{claims_count}"
        )

        is_evidence_incomplete = False
        if claims_count > 0 and (detector_available_count == 0 or clip_available_count == 0):
            is_evidence_incomplete = True
            pilot_status = "PILOT_EVIDENCE_INCOMPLETE"
            checkpoint["state"] = "PILOT_EVIDENCE_INCOMPLETE"
        else:
            pilot_status = "PILOT_ONLY"
            checkpoint["state"] = "PILOT_COMPLETE"

        save_checkpoint(ckpt_path, checkpoint)

        # Write pilot preview and diagnostics
        pilot_preview = {
            "status": pilot_status,
            "schema_version": "2.0.0",
            "pilot_n": args.pilot,
            "provenance_hash": prov_hash,
            "completed_images": len(checkpoint["completed_image_ids"]),
            "failed_images": len(checkpoint["failed_image_ids"]),
            "genuine_zero_images": checkpoint.get("genuine_zero_claim_images", 0),
            "claims_count": claims_count,
            "detector_available_count": detector_available_count,
            "detector_failed_count": detector_failed_count,
            "clip_available_count": clip_available_count,
            "clip_failed_count": clip_failed_count,
            "records_sample": evidence_records[:10],
            "failures": checkpoint["failure_records"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        pilot_diag_p = ckpt_sub / "pilot_diagnostics.json"
        atomic_json_write(pilot_diag_p, pilot_preview)

        if args.output_dir:
            out_root = Path(args.output_dir)
            atomic_json_write(out_root / "pilot" / "pilot_diagnostics.json", pilot_preview)

        logger.info(f"Pilot diagnostics written: {pilot_diag_p}")

        if is_evidence_incomplete:
            err_msg = (
                f"PILOT EVIDENCE HEALTH GATE FAILED: Systematic provider failure detected. "
                f"claims_count={claims_count}, detector_available={detector_available_count}, "
                f"clip_available={clip_available_count}. Status: PILOT_EVIDENCE_INCOMPLETE"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # Save runtime environment lock from successful pilot
        atomic_json_write(lock_file, runtime_env)
        atomic_json_write(PROJECT_ROOT / "reports" / "m10a_recovery2" / "colab_runtime_environment_lock.json", runtime_env)

        logger.info(f"Pilot acquisition complete ({args.pilot} images attempted).")
        logger.info("=" * 60)
        logger.info("Phase 10A-R2 Colab PILOT Execution — COMPLETE (PILOT_ONLY)")
        logger.info("=" * 60)
        return

    # FULL MODE: GPU Complete Gate
    if total_attempted == 600:
        checkpoint["state"] = "GPU_COMPLETE"
        logger.info("GPU Complete Gate PASSED (all 600 cohort images attempted).")
    else:
        checkpoint["state"] = "GPU_PARTIAL"
        logger.warning(f"GPU Complete Gate NOT MET: attempted {total_attempted}/600 images.")

    save_checkpoint(ckpt_path, checkpoint)

    # ──────────────────────────────────────────────────────────────
    # Step 11: Compute Scientific Graph Testability Statistics (A6)
    # ──────────────────────────────────────────────────────────────
    evidence_records = checkpoint["evidence_records"]
    failure_records = checkpoint["failure_records"]

    failed_img_ids = set(checkpoint["failed_image_ids"])
    completed_img_ids = set(checkpoint["completed_image_ids"])
    genuine_zero_ids = set(checkpoint.get("genuine_zero_image_ids", []))
    claim_bearing_ids = completed_img_ids - genuine_zero_ids

    total_images_count = len(image_ids)
    pipeline_failure_images = len(failed_img_ids)
    genuine_zero_images = len(genuine_zero_ids)
    claim_bearing_images = len(claim_bearing_ids)
    successful_claim_gen_images = len(completed_img_ids)

    claims_per_image: Counter = Counter()
    for rec in evidence_records:
        claims_per_image[rec["image_id"]] += 1

    n_one = sum(1 for c in claims_per_image.values() if c == 1)
    n_two = sum(1 for c in claims_per_image.values() if c == 2)
    n_three = sum(1 for c in claims_per_image.values() if c == 3)
    n_four_plus = sum(1 for c in claims_per_image.values() if c >= 4)
    n_multi = sum(1 for c in claims_per_image.values() if c >= 2)
    n_three_plus = sum(1 for c in claims_per_image.values() if c >= 3)
    n_graph_claims = sum(c for c in claims_per_image.values() if c >= 2)
    n_edges = sum(c - 1 for c in claims_per_image.values() if c >= 2)

    # TEST split specifics (120 frozen test images)
    test_ids = [iid for iid in image_ids if research_splits.get(iid) == "TEST"]
    test_total = len(test_ids)
    test_failed = sum(1 for iid in test_ids if iid in failed_img_ids)
    test_genuine_zero = sum(1 for iid in test_ids if iid in genuine_zero_ids)
    test_successful_gen = sum(1 for iid in test_ids if iid in completed_img_ids)
    test_one = sum(1 for iid in test_ids if claims_per_image.get(iid, 0) == 1)
    test_multi = sum(1 for iid in test_ids if claims_per_image.get(iid, 0) >= 2)
    test_three_plus = sum(1 for iid in test_ids if claims_per_image.get(iid, 0) >= 3)
    test_graph_claims = sum(claims_per_image.get(iid, 0) for iid in test_ids if claims_per_image.get(iid, 0) >= 2)

    multi_fraction_full = n_multi / total_images_count if total_images_count > 0 else 0.0
    multi_fraction_successful = n_multi / successful_claim_gen_images if successful_claim_gen_images > 0 else 0.0
    testability = "SUFFICIENT" if multi_fraction_successful >= 0.3 else ("MARGINAL" if multi_fraction_successful >= 0.15 else "INSUFFICIENT")

    # ──────────────────────────────────────────────────────────────
    # Step 12: Build Final Evidence Manifest V2
    # ──────────────────────────────────────────────────────────────
    out_evidence_p = PROJECT_ROOT / "data" / "manifests" / "final_evidence_manifest_v2.json"
    evidence_manifest = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "acquisition_version": "10A-R2",
        "manifest_id": "coco_600_primary_evidence_manifest_v2",
        "sampling_manifest_hash": sampling_hash,
        "source_image_audit_hash": audit_hash,
        "generation_config_hash": gen_cfg_hash,
        "claim_extractor_hash": claim_ext_hash,
        "checkpoint_provenance_hash": prov_hash,
        "execution_code_sha": execution_sha,
        "working_tree_clean": working_tree_clean,
        "sampling_manifest_creation_sha": sampling_manifest_creation_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_models": FROZEN_MODELS,
        "execution_environment": runtime_env,
        "statistics": {
            "total_images": total_images_count,
            "vlm_attempted": total_attempted,
            "successful_claim_generation_images": successful_claim_gen_images,
            "claim_bearing_images": claim_bearing_images,
            "genuine_zero_claim_images": genuine_zero_images,
            "pipeline_failure_images": pipeline_failure_images,
            "total_claims": len(evidence_records),
        },
        "graph_testability": {
            "overall": {
                "total_images_full_cohort": total_images_count,
                "successful_claim_gen_images": successful_claim_gen_images,
                "claim_bearing_images": claim_bearing_images,
                "genuine_zero_claim_images": genuine_zero_images,
                "pipeline_failure_images": pipeline_failure_images,
                "one_claim_images": n_one,
                "two_claim_images": n_two,
                "three_claim_images": n_three,
                "four_plus_claim_images": n_four_plus,
                "multi_claim_images": n_multi,
                "three_plus_claim_images": n_three_plus,
                "multi_claim_fraction_full_cohort": round(multi_fraction_full, 4),
                "multi_claim_fraction_successful_cohort": round(multi_fraction_successful, 4),
                "graph_participating_claims": n_graph_claims,
                "potential_tree_edges": n_edges,
            },
            "test_split": {
                "frozen_test_images_total": test_total,
                "successful_claim_generation_test_images": test_successful_gen,
                "genuine_zero_test_images": test_genuine_zero,
                "pipeline_failed_test_images": test_failed,
                "one_claim_test_images": test_one,
                "multi_claim_test_images": test_multi,
                "three_plus_claim_test_images": test_three_plus,
                "graph_participating_test_claims": test_graph_claims,
            },
            "status": testability,
        },
        "records": evidence_records,
        "failures": failure_records,
    }

    evidence_hash = compute_json_hash(evidence_manifest)
    evidence_manifest["manifest_hash"] = evidence_hash

    atomic_json_write(out_evidence_p, evidence_manifest)
    if args.output_dir:
        atomic_json_write(Path(args.output_dir) / "final_evidence_manifest_v2.json", evidence_manifest)
    logger.info(f"Evidence Manifest V2 written: {len(evidence_records)} claims, hash={evidence_hash[:16]}...")

    # ──────────────────────────────────────────────────────────────
    # Step 13: Populate Human Task Packages V2 (Canonical 'label': null)
    # ──────────────────────────────────────────────────────────────
    task_a_p = PROJECT_ROOT / "data" / "annotations" / "annotator_A_tasks_v2.json"
    task_b_p = PROJECT_ROOT / "data" / "annotations" / "annotator_B_tasks_v2.json"

    tasks_a = []
    tasks_b = []
    for rec in evidence_records:
        task_a = {
            "task_id": f"task_A_{rec['claim_id']}",
            "claim_id": rec["claim_id"],
            "image_id": rec["image_id"],
            "file_name": rec["file_name"],
            "claim_surface": rec["raw_claim_text"],
            "object_category": rec["object_category"],
            "label": None,  # CANONICAL — annotator must provide supported/hallucinated/unknown
        }
        tasks_a.append(task_a)
        task_b = dict(task_a)
        task_b["task_id"] = f"task_B_{rec['claim_id']}"
        tasks_b.append(task_b)

    task_payload_a = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "annotator_id": "ANNOTATOR_A",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_manifest_hash": evidence_hash,
        "tasks_count": len(tasks_a),
        "tasks": tasks_a,
    }
    task_payload_b = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "annotator_id": "ANNOTATOR_B",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_manifest_hash": evidence_hash,
        "tasks_count": len(tasks_b),
        "tasks": tasks_b,
    }

    atomic_json_write(task_a_p, task_payload_a)
    atomic_json_write(task_b_p, task_payload_b)
    if args.output_dir:
        atomic_json_write(Path(args.output_dir) / "annotator_A_tasks_v2.json", task_payload_a)
        atomic_json_write(Path(args.output_dir) / "annotator_B_tasks_v2.json", task_payload_b)
    logger.info(f"Annotation tasks written: A={len(tasks_a)}, B={len(tasks_b)} (canonical 'label': null)")

    # ──────────────────────────────────────────────────────────────
    # Step 14: Pre-Annotation Freeze V2 Verification Gate (A10, Req 7)
    # ──────────────────────────────────────────────────────────────
    freeze_issues: List[str] = []
    freeze_status = "TASKS_NOT_READY"

    if total_attempted != 600:
        freeze_issues.append(f"Cohort incomplete: attempted {total_attempted}/600 images.")
        freeze_status = "GPU_PARTIAL"

    if len(evidence_records) == 0:
        freeze_issues.append("Evidence manifest has 0 claims.")
        freeze_status = "EVIDENCE_INCOMPLETE"

    if len(tasks_a) == 0 or len(tasks_b) == 0:
        freeze_issues.append("Annotation tasks are empty.")
        freeze_status = "TASKS_NOT_READY"

    cids_a = [t["claim_id"] for t in tasks_a]
    cids_b = [t["claim_id"] for t in tasks_b]
    if cids_a != cids_b:
        freeze_issues.append("Annotator A and B claim sets do not match.")
        freeze_status = "TASKS_NOT_READY"

    # Validate task readiness
    ready_a, issues_a = validate_annotation_task_readiness(tasks_a, dataset_version="v2")
    if not ready_a:
        freeze_issues.extend(issues_a)
        freeze_status = "TASKS_NOT_READY"

    out_freeze_p = PROJECT_ROOT / "data" / "manifests" / "pre_annotation_freeze_v2.json"
    if len(freeze_issues) == 0 and checkpoint["state"] == "GPU_COMPLETE":
        freeze_status = "PRE_ANNOTATION_FROZEN_V2"
        task_claims_hash = compute_json_hash({
            "claim_ids": cids_a,
            "count": len(tasks_a),
        })

        freeze = {
            "schema_version": "2.0.0",
            "dataset_version": "v2",
            "acquisition_version": "10A-R2",
            "freeze_status": freeze_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "execution_code_sha": execution_sha,
            "working_tree_clean": working_tree_clean,
            "sampling_manifest_creation_sha": sampling_manifest_creation_sha,
            "code_sha": execution_sha,
            "checkpoint_provenance_hash": prov_hash,
            "hashes": {
                "candidate_universe_hash": universe_hash,
                "sampling_manifest_hash": sampling_hash,
                "source_image_audit_hash": audit_hash,
                "corruption_manifest_hash": corruption_hash,
                "evidence_manifest_hash": evidence_hash,
                "llava_generation_config_hash": gen_cfg_hash,
                "claim_extraction_config_hash": claim_ext_hash,
                "task_claims_set_hash": task_claims_hash,
            },
            "splits": sampling_data.get("split_counts"),
            "total_source_images_verified": len(image_ids),
            "annotation_tasks_ready": True,
            "annotation_tasks_count": len(tasks_a),
        }
        freeze_hash = compute_json_hash(freeze)
        freeze["freeze_hash"] = freeze_hash

        atomic_json_write(out_freeze_p, freeze)
        if args.output_dir:
            atomic_json_write(Path(args.output_dir) / "pre_annotation_freeze_v2.json", freeze)
        logger.info(f"Pre-Annotation Freeze V2 SEALED: hash={freeze_hash[:16]}...")
    else:
        logger.warning(
            f"Pre-Annotation Freeze V2 NOT SEALED (Status: {freeze_status}). "
            f"Issues: {freeze_issues}"
        )

    # ──────────────────────────────────────────────────────────────
    # Step 15: Acquisition Report
    # ──────────────────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    out_report_p = PROJECT_ROOT / "reports" / "m10a_recovery2" / "gpu_acquisition_report.md"

    report = f"""# Phase 10A-R2 GPU Acquisition Report

**Date:** {datetime.now(timezone.utc).isoformat()}
**Execution Git SHA:** `{execution_sha}`
**Working Tree Clean:** `{working_tree_clean}`
**Sampling Manifest Creation SHA:** `{sampling_manifest_creation_sha}`
**GPU Device:** {gpu_env['device']}
**GPU Memory:** {gpu_env['gpu_memory_gb']} GB
**CUDA Version:** {gpu_env['cuda_version']}
**Execution Time:** {elapsed_total:.1f} seconds
**Freeze Status:** `{freeze_status}`

---

## Evidence Summary

| Metric | Value |
| :--- | ---: |
| Total Images (Full Cohort) | {total_images_count} |
| Successfully Claim-Generated Images | {successful_claim_gen_images} |
| Claim-Bearing Images | {claim_bearing_images} |
| Genuine Zero-Claim Images | {genuine_zero_images} |
| Pipeline-Failed Images | {pipeline_failure_images} |
| Total Claims Extracted | {len(evidence_records)} |
| Multi-Claim Images | {n_multi} |
| Multi-Claim Fraction (Full Cohort) | {multi_fraction_full:.4f} |
| Multi-Claim Fraction (Successful Cohort) | {multi_fraction_successful:.4f} |
| Graph Testability | {testability} |

## TEST Split (120 Images)

| Metric | Value |
| :--- | ---: |
| Frozen Test Images | {test_total} |
| Successful Generation | {test_successful_gen} |
| Genuine Zero-Claim | {test_genuine_zero} |
| Pipeline Failed | {test_failed} |
| One-Claim Images | {test_one} |
| Multi-Claim Images | {test_multi} |
| 3+ Claim Images | {test_three_plus} |
| Graph-Participating Claims | {test_graph_claims} |

## Model Revisions (T4 Memory-Safe Policy)

| Model | Revision | Device |
| :--- | :--- | :--- |
| LLaVA-1.5-7B | `{FROZEN_MODELS['vlm_revision']}` | CUDA (4-bit NF4) |
| OWL-ViT | `{FROZEN_MODELS['detector_revision']}` | CPU |
| CLIP | `{FROZEN_MODELS['clip_revision']}` | CPU |

## Provenance & Artifact Hashes

| Artifact | Hash / SHA |
| :--- | :--- |
| Execution Git SHA | `{execution_sha}` |
| Working Tree Clean | `{working_tree_clean}` |
| Sampling Creation SHA | `{sampling_manifest_creation_sha}` |
| Checkpoint Provenance | `{prov_hash}` |
| Evidence Manifest V2 | `{evidence_hash}` |
| Sampling Manifest V2 | `{sampling_hash}` |
| Source Image Audit | `{audit_hash}` |
| Generation Config | `{gen_cfg_hash}` |
| Claim Extractor | `{claim_ext_hash}` |

## Annotation Readiness

- **Tasks Populated:** {len(tasks_a)} claims (A={len(tasks_a)}, B={len(tasks_b)})
- **Annotation Ready:** {'YES' if freeze_status == 'PRE_ANNOTATION_FROZEN_V2' else 'NO — freeze gates not sealed'}
- **Label Field:** Canonical `label = null` (strictly masking model evidence)
"""

    out_report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_p, "w", encoding="utf-8") as f:
        f.write(report)
    if args.output_dir:
        with open(Path(args.output_dir) / "gpu_acquisition_report.md", "w", encoding="utf-8") as f:
            f.write(report)

    logger.info("=" * 60)
    logger.info("Phase 10A-R2 Colab GPU Execution — COMPLETE")
    logger.info(f"  Claims extracted: {len(evidence_records)}")
    logger.info(f"  Genuine zero-claim images: {genuine_zero_images}")
    logger.info(f"  Pipeline failures: {pipeline_failure_images}")
    logger.info(f"  Annotation tasks: {len(tasks_a)}")
    logger.info(f"  Freeze status: {freeze_status}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
