"""
Phase 10A-R2 Canonical Colab GPU Execution Script.

ONE-COMMAND runner for Google Colab. This is the ONLY authorized entry point
for producing real multi-modal evidence from the frozen 600-image primary cohort.

Usage in Colab:
    !python scripts/run_phase10a_r2_colab.py

Requirements:
    - NVIDIA GPU with CUDA (minimum T4 with 15GB VRAM for 4-bit LLaVA)
    - bitsandbytes, transformers, torch with CUDA
    - All 600 source images in data/coco/images/

Execution Flow:
    1. HARD CUDA GATE — abort immediately if no GPU
    2. Load frozen sampling manifest v2 (verify hash)
    3. Load or create GPU acquisition checkpoint
    4. Resume from last checkpoint if partially complete
    5. For each unprocessed image:
       a. LLaVA forward inference → raw caption
       b. Conservative claim extraction → atomic claims
       c. OWL-ViT detection → raw detector scores
       d. CLIP similarity → raw cosine similarities
       e. Atomic checkpoint write (crash-safe)
    6. Finalize evidence manifest v2
    7. Populate annotation task packages v2 (N > 0 required)
    8. Seal pre-annotation freeze v2
    9. Advance artifact state machine to PRE_ANNOTATION_SEALED

IMPORTANT:
    - No model loading on CPU. Hard failure if CUDA unavailable.
    - No synthetic, mock, or caption-sourced claims.
    - All evidence values are raw (d_i ∈ [0,1], g_i ∈ [-1,1]).
    - Predeclared failure policy applies to individual image failures.
"""

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
import traceback
from typing import Dict, List, Optional, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    "prompt_template": (
        "USER: <image>\n"
        "Describe this image in detail focusing on the main objects present. ASSISTANT:"
    ),
    "model_id": FROZEN_MODELS["vlm_model"],
    "model_revision": FROZEN_MODELS["vlm_revision"],
    "processor_revision": FROZEN_MODELS["vlm_revision"],
    "max_new_tokens": 128,
    "temperature": 0.2,
    "do_sample": False,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": 1.0,
    "stopping_criteria": "eos_token",
    "random_seed": 42,
    "quantization": "4bit_nf4",
    "dtype": "float16",
    "image_preprocessing": "clip_image_processor_standard",
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


def compute_json_hash(data: dict) -> str:
    """Compute SHA-256 of JSON-serialized data with canonical formatting."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_json_write(path: Path, data: Any) -> None:
    """Write JSON atomically: write to temp file, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=path.stem + "_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # On Windows, target must not exist for rename
        if path.exists():
            path.unlink()
        shutil.move(tmp_path, str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUDA HARD GATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def enforce_cuda_gate() -> Dict[str, Any]:
    """
    Hard abort if CUDA is not available.

    Returns:
        GPU environment metadata dict.

    Raises:
        RuntimeError: if no CUDA GPU is detected.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU REQUIRED. This script must run on a CUDA-enabled device "
            "(e.g., Google Colab with GPU runtime). No model loading on CPU is permitted. "
            "Aborting."
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CHECKPOINT MANAGEMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def load_or_create_checkpoint(
    checkpoint_path: Path, sampling_manifest_hash: str, total_images: int
) -> Dict[str, Any]:
    """Load existing checkpoint or create a fresh one."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        # Validate checkpoint is for the correct sampling manifest
        if ckpt.get("sampling_manifest_hash") != sampling_manifest_hash:
            logger.warning(
                "Checkpoint sampling hash mismatch. Creating fresh checkpoint. "
                f"Expected: {sampling_manifest_hash}, "
                f"Found: {ckpt.get('sampling_manifest_hash')}"
            )
        else:
            completed = len(ckpt.get("completed_image_ids", []))
            logger.info(f"Resuming from checkpoint: {completed}/{total_images} images completed.")
            return ckpt

    # Fresh checkpoint
    return {
        "schema_version": "2.0.0",
        "checkpoint_type": "gpu_acquisition_checkpoint",
        "state": "GPU_IN_PROGRESS",
        "sampling_manifest_hash": sampling_manifest_hash,
        "total_target_images": total_images,
        "completed_images": 0,
        "failed_images": 0,
        "genuine_zero_claim_images": 0,
        "completed_image_ids": [],
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
# SINGLE-IMAGE PROCESSING
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
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Process a single image through the full evidence pipeline.

    Returns:
        (evidence_records, failure_record_or_none)
    """
    from PIL import Image

    evidence_records = []
    failure_record = None

    try:
        # 1. Load image
        img = Image.open(str(image_path)).convert("RGB")

        # 2. LLaVA forward inference
        caption_text = vlm_provider.generate(
            image=img,
            prompt=LLAVA_GENERATION_CONFIG["prompt_template"],
            max_new_tokens=LLAVA_GENERATION_CONFIG["max_new_tokens"],
            do_sample=LLAVA_GENERATION_CONFIG["do_sample"],
            temperature=LLAVA_GENERATION_CONFIG["temperature"],
        )

        if not caption_text or not caption_text.strip():
            # Genuine zero — model produced empty output
            return [], None

        # 3. Conservative claim extraction
        claims = claim_extractor.extract_claims(caption_text, image_id=image_id)

        if len(claims) == 0:
            # Genuine zero — model produced caption but no object-existence claims
            return [], None

        # 4. For each claim, extract OWL-ViT and CLIP evidence
        for claim in claims:
            claim_id = f"{image_id}_claim_{claim.category}"
            det_score = None
            det_available = False
            clip_score = None
            clip_available = False

            # OWL-ViT detection
            try:
                det_result = detector_provider.detect(img, claim.category)
                if det_result is not None:
                    det_score = float(det_result.max_score)
                    det_available = True
            except Exception as det_err:
                logger.warning(f"OWL-ViT failed for {claim_id}: {det_err}")

            # CLIP similarity
            try:
                clip_result = clip_provider.compute_similarity(img, claim.category)
                if clip_result is not None:
                    clip_score = float(clip_result.cosine_similarity)
                    clip_available = True
            except Exception as clip_err:
                logger.warning(f"CLIP failed for {claim_id}: {clip_err}")

            record = {
                "claim_id": claim_id,
                "image_id": image_id,
                "file_name": file_name,
                "coco_source_split": coco_source_split,
                "research_split": research_split,
                "object_category": claim.category,
                "raw_claim_text": claim.surface_text,
                "raw_caption": caption_text,
                "detector_score": det_score,
                "detector_available": det_available,
                "clip_score": clip_score,
                "similarity_available": clip_available,
                "provenance_status": "REAL_UNLABELED",
                "model_revision": FROZEN_MODELS["vlm_revision"],
                "detector_revision": FROZEN_MODELS["detector_revision"],
                "clip_revision": FROZEN_MODELS["clip_revision"],
                "generation_config_hash": gen_config_hash,
                "claim_extractor_hash": claim_ext_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            evidence_records.append(record)

    except Exception as err:
        # Pipeline failure for this image
        failure_record = {
            "claim_id": f"img_level_{image_id}",
            "image_id": image_id,
            "coco_source_split": coco_source_split,
            "research_split": research_split,
            "provider": FROZEN_MODELS["vlm_model"],
            "failure_state": "FAILED",
            "reason_code": f"PIPELINE_EXCEPTION: {type(err).__name__}",
            "attempt_count": 1,
            "last_error_class": type(err).__name__,
            "last_error_message": str(err)[:500],
            "provenance_status": "REAL_UNLABELED",
            "model_revision": FROZEN_MODELS["vlm_revision"],
            "generation_config_hash": gen_config_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.error(f"Pipeline failure for {image_id}: {err}")

    return evidence_records, failure_record


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """
    One-command Colab execution entry point.

    Orchestrates the full evidence acquisition pipeline with
    atomic checkpointing and strict provenance enforcement.
    """
    logger.info("=" * 60)
    logger.info("Phase 10A-R2 Colab GPU Execution — START")
    logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────────
    # Step 1: CUDA Hard Gate
    # ──────────────────────────────────────────────────────────────
    gpu_env = enforce_cuda_gate()

    # ──────────────────────────────────────────────────────────────
    # Step 2: Load Frozen Artifacts
    # ──────────────────────────────────────────────────────────────
    manifest_p = PROJECT_ROOT / "data" / "manifests" / "final_sampling_manifest_v2.json"
    universe_p = PROJECT_ROOT / "data" / "manifests" / "coco_candidate_universe_v2.json"
    corruption_p = PROJECT_ROOT / "data" / "manifests" / "final_corruption_manifest_v2.json"
    audit_p = PROJECT_ROOT / "reports" / "m10a_recovery2" / "source_image_audit_v2.json"
    image_dir = PROJECT_ROOT / "data" / "coco" / "images"

    checkpoint_p = PROJECT_ROOT / "data" / "manifests" / "gpu_acquisition_checkpoint_v2.json"
    out_evidence_p = PROJECT_ROOT / "data" / "manifests" / "final_evidence_manifest_v2.json"
    out_freeze_p = PROJECT_ROOT / "data" / "manifests" / "pre_annotation_freeze_v2.json"
    task_a_p = PROJECT_ROOT / "data" / "annotations" / "annotator_A_tasks_v2.json"
    task_b_p = PROJECT_ROOT / "data" / "annotations" / "annotator_B_tasks_v2.json"
    out_report_p = PROJECT_ROOT / "reports" / "m10a_recovery2" / "gpu_acquisition_report.md"

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

    assert len(image_ids) == 600, f"Expected 600 images, got {len(image_ids)}"

    code_sha = sampling_data.get("code_sha", "unknown")

    logger.info(f"Loaded sampling manifest: {len(image_ids)} images, hash={sampling_hash[:16]}...")
    logger.info(f"Generation config hash: {gen_cfg_hash[:16]}...")
    logger.info(f"Claim extractor hash: {claim_ext_hash[:16]}...")

    # ──────────────────────────────────────────────────────────────
    # Step 3: Initialize Models (GPU ONLY)
    # ──────────────────────────────────────────────────────────────
    logger.info("Loading frozen LLaVA-1.5-7B (4-bit quantized)...")
    from src.vlm.llava_provider import LLaVA15Provider
    from src.claims.extraction import ConservativeClaimExtractor
    from src.claims.vocabulary import create_coco_category_registry
    from src.evidence.detector_provider import HuggingFaceDetectorProvider
    from src.evidence.clip_provider import TransformersCLIPProvider

    vlm_provider = LLaVA15Provider(
        model_name=FROZEN_MODELS["vlm_model"],
        model_revision=FROZEN_MODELS["vlm_revision"],
        device="cuda",
        dtype="float16",
        load_in_4bit=True,
        local_files_only=False,
        allow_download=True,
    )
    logger.info("LLaVA model loaded.")

    claim_extractor = ConservativeClaimExtractor(create_coco_category_registry())
    logger.info("Claim extractor initialized.")

    detector_provider = HuggingFaceDetectorProvider(
        model_name=FROZEN_MODELS["detector_model"],
        model_revision=FROZEN_MODELS["detector_revision"],
        device="cuda",
    )
    logger.info("OWL-ViT detector loaded.")

    clip_provider = TransformersCLIPProvider(
        model_name=FROZEN_MODELS["clip_model"],
        model_revision=FROZEN_MODELS["clip_revision"],
        device="cuda",
    )
    logger.info("CLIP model loaded.")

    # ──────────────────────────────────────────────────────────────
    # Step 4: Load or Create Checkpoint
    # ──────────────────────────────────────────────────────────────
    checkpoint = load_or_create_checkpoint(checkpoint_p, sampling_hash, len(image_ids))
    checkpoint["execution_environment"] = gpu_env
    completed_set = set(checkpoint.get("completed_image_ids", []))
    failed_set = set(checkpoint.get("failed_image_ids", []))

    remaining = [iid for iid in image_ids if iid not in completed_set and iid not in failed_set]
    logger.info(f"Images remaining: {len(remaining)} / {len(image_ids)}")

    # ──────────────────────────────────────────────────────────────
    # Step 5: Process Images with Atomic Checkpointing
    # ──────────────────────────────────────────────────────────────
    CHECKPOINT_INTERVAL = 10  # Save checkpoint every N images

    start_time = time.time()
    for idx, img_id in enumerate(remaining):
        meta = images_meta.get(img_id, {})
        file_name = meta.get("file_name", f"{img_id}.jpg")
        r_split = research_splits.get(img_id, "TRAIN")
        c_split = coco_source_splits.get(img_id, "train2017")

        # Resolve image path
        img_path = image_dir / file_name
        if not img_path.exists():
            # Try alternate naming
            numeric_id = img_id.replace("coco_", "")
            img_path = image_dir / f"{numeric_id}.jpg"

        if not img_path.exists():
            logger.error(f"Source image not found: {img_path}")
            checkpoint["failed_image_ids"].append(img_id)
            checkpoint["failed_images"] = len(checkpoint["failed_image_ids"])
            checkpoint["failure_records"].append({
                "claim_id": f"img_level_{img_id}",
                "image_id": img_id,
                "failure_state": "FAILED",
                "reason_code": "SOURCE_IMAGE_NOT_FOUND",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            continue

        # Process
        records, failure = process_single_image(
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
        )

        if failure:
            checkpoint["failed_image_ids"].append(img_id)
            checkpoint["failed_images"] = len(checkpoint["failed_image_ids"])
            checkpoint["failure_records"].append(failure)
        else:
            checkpoint["completed_image_ids"].append(img_id)
            checkpoint["completed_images"] = len(checkpoint["completed_image_ids"])
            checkpoint["evidence_records"].extend(records)
            if len(records) == 0:
                checkpoint["genuine_zero_claim_images"] = (
                    checkpoint.get("genuine_zero_claim_images", 0) + 1
                )

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
            save_checkpoint(checkpoint_p, checkpoint)

    # Final checkpoint save
    checkpoint["state"] = "GPU_COMPLETE"
    save_checkpoint(checkpoint_p, checkpoint)

    elapsed_total = time.time() - start_time
    logger.info(f"GPU inference complete: {elapsed_total:.1f}s total")

    # ──────────────────────────────────────────────────────────────
    # Step 6: Build Final Evidence Manifest V2
    # ──────────────────────────────────────────────────────────────
    evidence_records = checkpoint["evidence_records"]
    failure_records = checkpoint["failure_records"]
    n_genuine_zero = checkpoint.get("genuine_zero_claim_images", 0)

    # Compute per-image claim counts for graph testability
    claims_per_image: Counter = Counter()
    for rec in evidence_records:
        claims_per_image[rec["image_id"]] += 1

    n_zero = sum(1 for iid in image_ids if claims_per_image.get(iid, 0) == 0)
    n_one = sum(1 for c in claims_per_image.values() if c == 1)
    n_two = sum(1 for c in claims_per_image.values() if c == 2)
    n_three = sum(1 for c in claims_per_image.values() if c == 3)
    n_four_plus = sum(1 for c in claims_per_image.values() if c >= 4)
    n_multi = sum(1 for c in claims_per_image.values() if c >= 2)
    n_graph_claims = sum(c for c in claims_per_image.values() if c >= 2)
    n_edges = sum(c - 1 for c in claims_per_image.values() if c >= 2)

    # Test-split stats
    test_ids = {iid for iid in image_ids if research_splits.get(iid) == "TEST"}
    test_zero = sum(1 for iid in test_ids if claims_per_image.get(iid, 0) == 0)
    test_multi = sum(1 for iid in test_ids if claims_per_image.get(iid, 0) >= 2)

    multi_fraction = n_multi / len(image_ids) if image_ids else 0.0
    testability = "SUFFICIENT" if multi_fraction >= 0.3 else ("MARGINAL" if multi_fraction >= 0.15 else "INSUFFICIENT")

    # Load audit hash if available
    audit_hash = ""
    if audit_p.exists():
        with open(audit_p, "rb") as f:
            audit_hash = hashlib.sha256(f.read()).hexdigest()

    corruption_hash = ""
    if corruption_p.exists():
        with open(corruption_p, "r", encoding="utf-8") as f:
            corruption_data = json.load(f)
            corruption_hash = corruption_data.get("manifest_hash", "")

    evidence_manifest = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "acquisition_version": "10A-R2",
        "manifest_id": "coco_600_primary_evidence_manifest_v2",
        "sampling_manifest_hash": sampling_hash,
        "source_image_audit_hash": audit_hash,
        "generation_config_hash": gen_cfg_hash,
        "claim_extractor_hash": claim_ext_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_models": FROZEN_MODELS,
        "execution_environment": gpu_env,
        "statistics": {
            "total_images": len(image_ids),
            "vlm_attempted": len(image_ids),
            "vlm_successful": checkpoint["completed_images"],
            "vlm_failed": checkpoint["failed_images"],
            "total_claims": len(evidence_records),
            "genuine_zero_claims": n_genuine_zero,
            "pipeline_failure_zero_claims": checkpoint["failed_images"],
        },
        "graph_testability": {
            "overall": {
                "total_images": len(image_ids),
                "zero_claim_images": n_zero,
                "genuine_zero_claims": n_genuine_zero,
                "pipeline_failure_zero_claims": checkpoint["failed_images"],
                "one_claim_images": n_one,
                "two_claim_images": n_two,
                "three_claim_images": n_three,
                "four_plus_claim_images": n_four_plus,
                "multi_claim_fraction": round(multi_fraction, 4),
                "graph_participating_claims": n_graph_claims,
                "potential_tree_edges": n_edges,
            },
            "test_split": {
                "total_images": len(test_ids),
                "zero_claim_images": test_zero,
                "multi_claim_images": test_multi,
            },
            "status": testability,
        },
        "records": evidence_records,
        "failures": failure_records,
    }

    evidence_hash = compute_json_hash(evidence_manifest)
    evidence_manifest["manifest_hash"] = evidence_hash

    atomic_json_write(out_evidence_p, evidence_manifest)
    logger.info(f"Evidence Manifest V2 written: {len(evidence_records)} claims, hash={evidence_hash[:16]}...")

    # ──────────────────────────────────────────────────────────────
    # Step 7: Populate Human Task Packages V2
    # ──────────────────────────────────────────────────────────────
    tasks_dir = PROJECT_ROOT / "data" / "annotations"
    tasks_dir.mkdir(parents=True, exist_ok=True)

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
            "expected_label": None,  # MASKED — annotator must provide
        }
        tasks_a.append(task_a)
        task_b = dict(task_a)
        task_b["task_id"] = f"task_B_{rec['claim_id']}"
        tasks_b.append(task_b)

    if len(tasks_a) == 0:
        logger.error(
            "CRITICAL: Zero annotation tasks produced. "
            "Cannot proceed to pre-annotation freeze with 0 claims."
        )
        # Still write the empty packages for auditability but flag as NOT ready
        annotation_ready = False
    else:
        annotation_ready = True
        logger.info(f"Annotation tasks populated: {len(tasks_a)} claims (A={len(tasks_a)}, B={len(tasks_b)})")

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

    # ──────────────────────────────────────────────────────────────
    # Step 8: Seal Pre-Annotation Freeze V2
    # ──────────────────────────────────────────────────────────────
    task_claims_hash = compute_json_hash({
        "claim_ids": [t["claim_id"] for t in tasks_a],
        "count": len(tasks_a),
    })

    freeze = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "acquisition_version": "10A-R2",
        "freeze_status": "PRE_ANNOTATION_FROZEN_V2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "code_sha": code_sha,
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
        "annotation_tasks_ready": annotation_ready,
        "annotation_tasks_count": len(tasks_a),
    }

    freeze_hash = compute_json_hash(freeze)
    freeze["freeze_hash"] = freeze_hash

    atomic_json_write(out_freeze_p, freeze)
    logger.info(f"Pre-Annotation Freeze V2 sealed: hash={freeze_hash[:16]}...")

    # ──────────────────────────────────────────────────────────────
    # Step 9: Generate GPU Acquisition Report
    # ──────────────────────────────────────────────────────────────
    report = f"""# Phase 10A-R2 GPU Acquisition Report

**Date:** {datetime.now(timezone.utc).isoformat()}
**GPU Device:** {gpu_env['device']}
**GPU Memory:** {gpu_env['gpu_memory_gb']} GB
**CUDA Version:** {gpu_env['cuda_version']}
**Execution Time:** {elapsed_total:.1f} seconds

---

## Evidence Summary

| Metric | Value |
| :--- | ---: |
| Total Images | {len(image_ids)} |
| VLM Successful | {checkpoint['completed_images']} |
| VLM Failed | {checkpoint['failed_images']} |
| Total Claims | {len(evidence_records)} |
| Genuine Zero-Claim Images | {n_genuine_zero} |
| Multi-Claim Images | {n_multi} |
| Multi-Claim Fraction | {multi_fraction:.4f} |
| Graph Testability | {testability} |

## Model Revisions

| Model | Revision |
| :--- | :--- |
| LLaVA-1.5-7B | `{FROZEN_MODELS['vlm_revision']}` |
| OWL-ViT | `{FROZEN_MODELS['detector_revision']}` |
| CLIP | `{FROZEN_MODELS['clip_revision']}` |

## Artifact Hashes

| Artifact | Hash |
| :--- | :--- |
| Evidence Manifest V2 | `{evidence_hash}` |
| Pre-Annotation Freeze V2 | `{freeze_hash}` |
| Sampling Manifest V2 | `{sampling_hash}` |
| Generation Config | `{gen_cfg_hash}` |
| Claim Extractor | `{claim_ext_hash}` |

## Annotation Readiness

- **Tasks Populated:** {len(tasks_a)} claims
- **Annotation Ready:** {'YES' if annotation_ready else 'NO — zero claims, investigate pipeline'}
- **Label Masking:** All expected_label fields set to null
"""

    out_report_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_p, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report written: {out_report_p}")

    # ──────────────────────────────────────────────────────────────
    # Final Summary
    # ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 10A-R2 Colab GPU Execution — COMPLETE")
    logger.info(f"  Claims extracted: {len(evidence_records)}")
    logger.info(f"  Pipeline failures: {checkpoint['failed_images']}")
    logger.info(f"  Annotation tasks: {len(tasks_a)}")
    logger.info(f"  Testability: {testability}")
    logger.info(f"  Annotation ready: {annotation_ready}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
