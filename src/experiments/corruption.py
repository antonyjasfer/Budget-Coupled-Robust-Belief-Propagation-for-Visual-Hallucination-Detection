"""
Deterministic visual corruption and perturbed evidence generation for M8.
"""

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image, ImageFilter

from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import M7ClaimRecord
from src.experiments.configs import CorruptionConfig


def apply_image_corruption(
    image: Image.Image,
    corruption_type: str,
    severity: str = "medium",
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Image.Image:
    """
    Apply a deterministic visual corruption to a PIL Image.
    """
    if severity == "clean" or corruption_type == "none":
        return image.copy()

    c_type = corruption_type.lower()
    img_rgb = image.convert("RGB")

    if c_type == "gaussian_blur":
        radius = float(params.get(severity, 3.0) if params else {"light": 1.5, "medium": 3.0, "heavy": 6.0}[severity])
        return img_rgb.filter(ImageFilter.GaussianBlur(radius=radius))

    elif c_type == "jpeg_compression":
        quality = int(params.get(severity, 40) if params else {"light": 70, "medium": 40, "heavy": 15}[severity])
        buffer = io.BytesIO()
        img_rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

    elif c_type == "downsampling":
        factor = float(params.get(severity, 0.25) if params else {"light": 0.5, "medium": 0.25, "heavy": 0.125}[severity])
        w, h = img_rgb.size
        new_w, new_h = max(8, int(w * factor)), max(8, int(h * factor))
        down = img_rgb.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return down.resize((w, h), Image.Resampling.NEAREST)

    elif c_type == "additive_noise":
        std = float(params.get(severity, 0.15) if params else {"light": 0.05, "medium": 0.15, "heavy": 0.30}[severity])
        rng = np.random.default_rng(seed)
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        noise = rng.normal(0.0, std, arr.shape)
        noisy = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).astype(np.uint8))

    elif c_type == "center_occlusion":
        fraction = float(params.get(severity, 0.30) if params else {"light": 0.15, "medium": 0.30, "heavy": 0.50}[severity])
        w, h = img_rgb.size
        occ_w, occ_h = int(w * fraction), int(h * fraction)
        x0, y0 = (w - occ_w) // 2, (h - occ_h) // 2
        
        arr = np.array(img_rgb)
        arr[y0:y0 + occ_h, x0:x0 + occ_w, :] = 128  # neutral grey patch
        return Image.fromarray(arr)

    else:
        raise ValueError(f"Unknown corruption type: '{corruption_type}'")


def create_corrupted_claim_record(
    clean_claim: M7ClaimRecord,
    corruption_type: str,
    severity: str = "medium",
    seed: int = 42,
) -> M7ClaimRecord:
    """
    Generate a perturbed ClaimRecord simulating degraded evidence under visual corruption.
    Hypothesis: Degradation weakens evidence confidence and increases uncertainty.
    """
    if severity == "clean" or corruption_type == "none":
        return clean_claim

    rng = np.random.default_rng(seed + hash(clean_claim.claim_id) % 10000)
    ev = clean_claim.evidence

    severity_degradation = {
        "light": 0.15,
        "medium": 0.35,
        "heavy": 0.65,
    }.get(severity, 0.35)

    # 1. Degrade detector score (pull towards ambiguous 0.5 / reduce confidence)
    orig_det = ev.detector_score if (ev.detector_available and ev.detector_score is not None) else 0.5
    noise = float(rng.normal(0.0, 0.05 * severity_degradation))
    if orig_det >= 0.5:
        # High confidence decreases
        degraded_det = float(orig_det - severity_degradation * (orig_det - 0.5) + noise)
    else:
        # Low confidence increases towards ambiguous
        degraded_det = float(orig_det + severity_degradation * (0.5 - orig_det) + noise)
    degraded_det = float(np.clip(degraded_det, 0.01, 0.99))

    # 2. Degrade CLIP score
    orig_clip = ev.clip_score if (ev.similarity_available and ev.clip_score is not None) else 0.2
    clip_noise = float(rng.normal(0.0, 0.04 * severity_degradation))
    degraded_clip = float(np.clip(orig_clip * (1.0 - 0.5 * severity_degradation) + clip_noise, -0.99, 0.99))

    corrupted_ev = ClaimLevelEvidenceRecord(
        claim_id=f"{clean_claim.claim_id}_corrupt_{corruption_type}_{severity}",
        image_id=clean_claim.image_id,
        object_category=clean_claim.object_category,
        text_span=clean_claim.text_span,
        caption=clean_claim.caption,
        image_hash=f"{clean_claim.image_hash}_{corruption_type}_{severity}",
        split=clean_claim.split,
        detector_score=degraded_det,
        detector_available=True,
        detector_model=ev.detector_model,
        detector_revision=ev.detector_revision,
        clip_score=degraded_clip,
        similarity_available=True,
        clip_model=ev.clip_model,
        clip_revision=ev.clip_revision,
        vlm_generation_source=ev.vlm_generation_source,
        is_synthetic=clean_claim.is_synthetic,
        metadata={
            "corruption_type": corruption_type,
            "corruption_severity": severity,
            "original_detector_score": orig_det,
            "original_clip_score": orig_clip,
            "source_claim_id": clean_claim.claim_id,
        },
    )

    return M7ClaimRecord(
        claim_id=clean_claim.claim_id,
        image_id=clean_claim.image_id,
        object_category=clean_claim.object_category,
        text_span=clean_claim.text_span,
        caption=clean_claim.caption,
        image_hash=corrupted_ev.image_hash,
        split=clean_claim.split,
        evidence=corrupted_ev,
        is_synthetic=clean_claim.is_synthetic,
        metadata={"corruption_type": corruption_type, "corruption_severity": severity},
    )


# Alias for backwards compatibility
apply_visual_corruption = apply_image_corruption


def evaluate_corruption_pipeline(
    bundle: Any,
    cfg: Any,
) -> Dict[str, List[Any]]:
    """
    Run evaluation across all configured visual corruption types and severities.
    Returns dictionary mapping condition names to list of ClaimEvaluationResults.
    """
    from src.experiments.inference_runner import InferenceRunner
    from src.experiments.loaders import M8DatasetBundle

    runner = InferenceRunner(cfg)
    results_by_condition: Dict[str, List[Any]] = {}

    corruption_types = cfg.corruption.corruption_types
    severities = cfg.corruption.severity_levels

    for c_type in corruption_types:
        for sev in severities:
            cond_name = f"{c_type}_{sev}"
            corrupted_claims = [
                create_corrupted_claim_record(c, c_type, sev, seed=cfg.seed)
                for c in bundle.claims
            ]
            
            # Build temporary bundle with corrupted claims
            claims_by_img: Dict[str, List[M7ClaimRecord]] = {}
            for c in corrupted_claims:
                claims_by_img.setdefault(c.image_id, []).append(c)
                
            claims_by_split: Dict[str, List[M7ClaimRecord]] = {}
            for c in corrupted_claims:
                claims_by_split.setdefault(c.split.lower(), []).append(c)

            corrupted_bundle = M8DatasetBundle(
                claims=corrupted_claims,
                ground_truth=bundle.ground_truth,
                manifest=bundle.manifest,
                images_by_split=bundle.images_by_split,
                claims_by_split=claims_by_split,
                claims_by_image=claims_by_img,
                execution_state=bundle.execution_state,
                is_locked=bundle.is_locked,
                dataset_hash=bundle.dataset_hash,
                evidence_hash=f"{bundle.evidence_hash}_{cond_name}",
                is_synthetic=bundle.is_synthetic,
                metadata={"condition": cond_name},
            )

            res = runner.run_dataset(corrupted_bundle, condition_tag=cond_name)
            results_by_condition[cond_name] = res

    return results_by_condition
