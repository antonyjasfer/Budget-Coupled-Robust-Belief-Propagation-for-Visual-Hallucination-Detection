"""
Milestone 6 Demonstration: End-to-End Visual Evidence Pipeline with Actual LLaVA-1.5 Inference.

Architecture:
10 Real COCO Images (TRAIN split) -> Actual LLaVA-1.5-7B -> Claims -> OWL-ViT Detector -> CLIP -> Claim-Level JSONL.

CRITICAL METHODOLOGICAL GUARANTEES:
1. ACTUAL VLM INFERENCE ONLY:
   - Captions MUST originate from actual LLaVA inference or a cache entry that was
     previously produced by actual LLaVA inference.
   - Hard-coded captions and manual cache pre-population are strictly prohibited.
2. RAW EVIDENCE ONLY:
   - Detector scores are raw bounding-box presence max-scores d_i in [0.0, 1.0].
   - Similarity scores are raw image-text cosine similarities g_i in [-1.0, 1.0].
   - Probability calibration is strictly NOT performed in Milestone 6.
3. NO PREMATURE PGM FITTING:
   - theta_i, epsilon_i, J_ij, posterior probabilities, and robust posterior bounds
     are strictly NOT introduced or computed in this extraction pipeline.
4. PRIMARY DETECTOR:
   - google/owlvit-base-patch32 (open-vocabulary zero-shot detection).
5. REAL DATA:
   - 10 genuine COCO training images strictly from the TRAIN split.
6. CONFIGURABLE FOR GPU EXECUTION:
   - Configurable device (e.g. cuda:0 or cpu) and dtype (float16 or float32).
"""

from pathlib import Path
import sys
import json
import time
import argparse
from typing import List, Dict, Optional, Any

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    DatasetSource,
    SplitName,
)
from src.data.manifests import create_manifest
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.vlm.provider import VLMGenerationConfig, compute_file_sha256
from src.vlm.llava_provider import LLaVA15Provider
from src.vlm.cache import VLMCache
from src.evidence.detector_provider import HuggingFaceDetectorProvider
from src.evidence.clip_provider import TransformersCLIPProvider
from src.evidence.pipeline import VisualEvidencePipeline


# 10 genuine COCO 2017 training image IDs (No captions or synthetic text)
REAL_COCO_TRAIN_DATA = [
    {"id": 9, "file_name": "000000000009.jpg"},
    {"id": 25, "file_name": "000000000025.jpg"},
    {"id": 30, "file_name": "000000000030.jpg"},
    {"id": 34, "file_name": "000000000034.jpg"},
    {"id": 36, "file_name": "000000000036.jpg"},
    {"id": 42, "file_name": "000000000042.jpg"},
    {"id": 49, "file_name": "000000000049.jpg"},
    {"id": 61, "file_name": "000000000061.jpg"},
    {"id": 64, "file_name": "000000000064.jpg"},
    {"id": 71, "file_name": "000000000071.jpg"},
]


def ensure_real_coco_images(base_dir: Path) -> List[Path]:
    """Ensure the 10 genuine COCO train images exist locally; download if needed."""
    import urllib.request
    from PIL import Image

    base_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in REAL_COCO_TRAIN_DATA:
        fn = item["file_name"]
        target = base_dir / fn
        if not target.exists():
            url = f"http://images.cocodataset.org/train2017/{fn}"
            print(f"Downloading genuine COCO image: {url}...")
            urllib.request.urlretrieve(url, target)
        # Verify decodability
        with Image.open(target) as img:
            img.verify()
        paths.append(target)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description="Run Milestone 6 Real Evidence Pipeline")
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    default_dtype = "float16" if torch.cuda.is_available() else "float32"

    parser.add_argument("--device", type=str, default=default_device, help="Compute device (e.g. cuda:0 or cpu)")
    parser.add_argument("--dtype", type=str, default=default_dtype, help="Model dtype (e.g. float16, float32, bfloat16)")
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf", help="LLaVA HuggingFace model ID")
    parser.add_argument("--detector-model", type=str, default="google/owlvit-base-patch32", help="Detector model ID")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model ID")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit downloading missing neural weights from HF")
    parser.add_argument("--output", type=str, default="data/exports/claim_level_evidence.jsonl", help="Output JSONL path")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of real COCO images to process")
    return parser.parse_args()


def run_pipeline_demonstration():
    args = parse_args()

    print("=" * 80)
    print("MILESTONE 6: REAL VISUAL EVIDENCE PIPELINE (ACTUAL LLaVA-1.5 INFERENCE)")
    print("=" * 80)
    print(f"  Configuration:")
    print(f"    Target Device    : {args.device}")
    print(f"    Target Dtype     : {args.dtype}")
    print(f"    CUDA Available   : {torch.cuda.is_available()}")
    print(f"    VLM Model        : {args.model_name}")
    print(f"    Detector Model   : {args.detector_model}")
    print(f"    CLIP Model       : {args.clip_model}")
    print(f"    Allow Download   : {args.allow_download}")
    print(f"    Output Path      : {args.output}")

    # 1. Prepare genuine COCO training images
    image_dir = Path("data/real_images")
    print("\n1. Ensuring 10 genuine COCO train2017 images are present...")
    image_paths = ensure_real_coco_images(image_dir)
    print(f"   Successfully verified {len(image_paths)} genuine COCO training images.")

    # 2. Build DatasetManifest strictly with TRAIN split
    manifest_entries = []
    for idx, item in enumerate(REAL_COCO_TRAIN_DATA):
        img_path = image_paths[idx]
        img_hash = compute_file_sha256(img_path)
        img_rec = ImageRecord(
            image_id=f"coco_{item['id']:012d}",
            dataset_source=DatasetSource.COCO,
            file_name=str(img_path.resolve()),
            file_hash=img_hash,
            coco_id=item["id"],
            metadata={"is_synthetic": False, "split": "train"},
        )
        entry = DatasetManifestEntry(
            image=img_rec,
            split=SplitName.TRAIN,
        )
        manifest_entries.append(entry)

    manifest = create_manifest(
        manifest_id="coco_train_10_evidence_manifest",
        description="Milestone 6 demonstration manifest with 10 genuine COCO training images",
        entries=manifest_entries,
    )
    print(f"   Constructed manifest with {len(manifest.entries)} entries strictly partitioned in TRAIN.")

    # 3. Setup VLM Cache and Genuine LLaVA Provider
    cache_dir = Path("data/cache/vlm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    vlm_cache = VLMCache(cache_dir)

    print("\n2. Initializing Genuine LLaVA-1.5 Vision-Language Provider...")
    vlm_provider = LLaVA15Provider(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        allow_download=args.allow_download,
        local_files_only=not args.allow_download,
    )
    cfg = VLMGenerationConfig(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )
    vlm_revision = vlm_provider.resolve_revision()
    print(f"   LLaVA-1.5 resolved revision: {vlm_revision}")

    # 4. Initialize Real Detector and Real CLIP Providers
    print("\n3. Initializing Real Neural Evidence Extractors...")
    print(f"   - Object Detector: {args.detector_model} (open-vocabulary zero-shot)")
    detector_provider = HuggingFaceDetectorProvider(
        model_name=args.detector_model,
        device=args.device,
    )
    print(f"     Detector revision: {detector_provider.resolve_revision()}")

    print(f"   - Image-Text Similarity: {args.clip_model} (cosine similarity)")
    clip_provider = TransformersCLIPProvider(
        model_name=args.clip_model,
        device=args.device,
    )
    print(f"     CLIP revision: {clip_provider.resolve_revision()}")

    # 5. Initialize and Run Evidence Pipeline
    print("\n4. Executing Visual Evidence Pipeline...")
    print("   Workflow: Image -> LLaVA-1.5 (Real Inference / Real Cache) -> Claims -> OWL-ViT -> CLIP -> JSONL")
    start_time = time.time()
    pipeline = VisualEvidencePipeline(
        vlm_provider=vlm_provider,
        vlm_cache=vlm_cache,
        claim_extractor=ConservativeClaimExtractor(create_coco_category_registry()),
        detector_provider=detector_provider,
        clip_provider=clip_provider,
        gen_config=cfg,
        enforce_train_split=True,
    )

    output_jsonl_path = Path(args.output)
    records, stats = pipeline.run(
        manifest=manifest,
        sample_size=args.sample_size,
        output_jsonl_path=output_jsonl_path,
    )
    elapsed = time.time() - start_time
    print(f"   Pipeline execution completed in {elapsed:.2f} seconds.")

    # 6. Audit and Display Results
    print("\n" + "=" * 80)
    print("AUDIT & EVIDENCE DATASET SUMMARY")
    print("=" * 80)
    print(f"  Exported JSONL Path              : {output_jsonl_path.resolve()}")
    print(f"  Real COCO Images Processed       : {stats.images_processed} / {len(manifest.entries)}")
    print(f"  Total Extracted Claims           : {stats.total_claims_extracted}")
    print(f"  Records with Detector Evidence   : {stats.detector_available_count}")
    print(f"  Records with CLIP Evidence       : {stats.clip_available_count}")
    print(f"  Unavailable / Failed Evidence    : {stats.unavailable_or_failed_count}")
    print(f"  Zero-Claim Images                : {stats.zero_claim_images}")
    print(f"  Raw Evidence Bound Verification  : {'PASSED' if stats.uncalibrated_raw_evidence_guaranteed else 'FAILED'}")
    print(f"  No PGM / Ising Params Inferred   : {'PASSED' if stats.no_pgm_parameters_inferred_guaranteed else 'FAILED'}")

    print("\nSample Claim-Level Evidence Records:")
    print("-" * 80)
    for r in records[:5]:
        print(f"Claim ID: {r.claim_id}")
        print(f"  Image ID        : {r.image_id}")
        print(f"  Category        : {r.object_category}")
        print(f"  Surface Span    : '{r.text_span}'")
        print(f"  Caption         : '{r.caption}'")
        print(f"  VLM Source      : {r.vlm_generation_source}")
        print(f"  Detector Score  : {r.detector_score:.4f} (model: {r.detector_model}, available: {r.detector_available})")
        print(f"  CLIP Score      : {r.clip_score:.4f} (model: {r.clip_model}, available: {r.similarity_available})")
        print(f"  Split / Synth   : {r.split} / is_synthetic={r.is_synthetic}")
        print("-" * 80)

    # 7. Post-Condition Assertions
    assert stats.images_processed == min(args.sample_size, len(manifest.entries))
    assert stats.total_claims_extracted > 0, "No claims extracted!"
    assert stats.detector_available_count == stats.total_claims_extracted, "Detector score missing on some claims!"
    assert stats.clip_available_count == stats.total_claims_extracted, "CLIP score missing on some claims!"
    assert stats.unavailable_or_failed_count == 0, "Some evidence computations failed!"

    # Verify JSONL lines on disk
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        file_records = [json.loads(line) for line in f]
    assert len(file_records) == len(records), "JSONL record count mismatch!"

    for item in file_records:
        d_val = item["detector_score"]
        c_val = item["clip_score"]
        assert 0.0 <= d_val <= 1.0, f"Detector score out of range: {d_val}"
        assert -1.0 <= c_val <= 1.0, f"CLIP score out of range: {c_val}"
        assert item["detector_available"] is True
        assert item["similarity_available"] is True
        assert item["split"] == "train"
        assert item["is_synthetic"] is False
        assert item["vlm_generation_source"] in ("real_inference", "cache")
        # Confirm PGM fields are NOT present
        assert "theta" not in item
        assert "epsilon" not in item
        assert "J_ij" not in item
        assert "posterior" not in item

    print("\nALL METHODOLOGICAL VERIFICATIONS PASSED: SUCCESS")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline_demonstration()
