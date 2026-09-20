"""
Scalable evidence acquisition experiment for Milestone 7.

Architecture:
M7 Manifest (600 reserved images) -> Image Universe Resolution ->
Real LLaVA-1.5 Inference (or verified real cache) -> Conservative Claim Extraction ->
OWL-ViT Open-Vocabulary Detection -> CLIP Cosine Similarity ->
ClaimLevelEvidenceRecord -> M7 Claim Dataset.

CRITICAL METHODOLOGICAL GUARANTEES:
1. Real neural inference or verified real-inference cache ONLY (is_synthetic=False).
2. Explicit missing evidence tracking without silent zero-mapping.
3. Supports checkpointing and resumption so long runs across 600 images can resume safely.
4. Preserves complete runtime provenance (model names, revisions, hashes, execution times).
"""

from pathlib import Path
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Optional, Any, Set

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
from src.data.manifests import load_manifest, create_manifest
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.vlm.provider import VLMGenerationConfig, compute_file_sha256
from src.vlm.llava_provider import LLaVA15Provider
from src.vlm.cache import VLMCache
from src.evidence.detector_provider import HuggingFaceDetectorProvider
from src.evidence.clip_provider import TransformersCLIPProvider
from src.evidence.pipeline import VisualEvidencePipeline
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import M7ClaimRecord
from src.annotation.workflow import ingest_m6_evidence_to_m7_claims

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m7_evidence_acquisition")


def parse_args():
    parser = argparse.ArgumentParser(description="Scalable Evidence Acquisition for M7 Benchmark")
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    default_dtype = "float16" if torch.cuda.is_available() else "float32"

    parser.add_argument("--manifest", type=str, default="data/manifests/m7_image_manifest.json", help="Path to M7 image manifest")
    parser.add_argument("--image-base-dir", type=str, default="data/real_images", help="Base directory where images are located")
    parser.add_argument("--cache-dir", type=str, default="data/cache/vlm", help="Directory for VLM response cache")
    parser.add_argument("--output-claims", type=str, default="data/exports/m7_claims.jsonl", help="Output path for M7 claims JSONL")
    parser.add_argument("--device", type=str, default=default_device, help="Compute device for VLM (e.g. cuda:0 or cpu)")
    parser.add_argument("--evidence-device", type=str, default="cpu", help="Compute device for detector and CLIP evidence models")
    parser.add_argument("--dtype", type=str, default=default_dtype, help="Model dtype (e.g. float16, float32)")
    parser.add_argument("--model-name", type=str, default="llava-hf/llava-1.5-7b-hf", help="LLaVA HuggingFace model ID")
    parser.add_argument("--detector-model", type=str, default="google/owlvit-base-patch32", help="Detector model ID")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32", help="CLIP model ID")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading weights from HuggingFace Hub")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit NF4 quantization for memory-safe GPU execution")
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit on number of images to process")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing claims output file if present")
    return parser.parse_args()


def run_evidence_acquisition(
    manifest_path: str,
    output_claims_path: str,
    vlm_provider: Any,
    vlm_cache: VLMCache,
    detector_provider: Any,
    clip_provider: Any,
    image_base_dir: Optional[str] = None,
    gen_config: Optional[VLMGenerationConfig] = None,
    max_images: Optional[int] = None,
    resume: bool = True,
) -> List[M7ClaimRecord]:
    """
    Execute scalable evidence acquisition across the manifest entries.
    """
    manifest = load_manifest(manifest_path)
    out_file = Path(output_claims_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Check existing claims if resuming
    processed_image_ids: Set[str] = set()
    existing_claims: List[M7ClaimRecord] = []
    if resume and out_file.exists():
        logger.info(f"Checking existing claims for resumption in {out_file}...")
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rec = M7ClaimRecord.from_dict(data)
                    existing_claims.append(rec)
                    processed_image_ids.add(rec.image_id)
                except Exception as e:
                    logger.warning(f"Skipping unparseable line during resume: {e}")
        logger.info(f"Resuming: found {len(existing_claims)} claims across {len(processed_image_ids)} images.")

    # 2. Select entries to process
    entries_to_process = [
        e for e in manifest.entries
        if e.image and e.image.image_id not in processed_image_ids
    ]
    if max_images is not None:
        entries_to_process = entries_to_process[:max_images]

    logger.info(f"Total manifest entries: {len(manifest.entries)} | Entries to process: {len(entries_to_process)}")

    # 3. Initialize VisualEvidencePipeline
    # In M7, evidence can be acquired across train, validation, calibration, and test splits
    pipeline = VisualEvidencePipeline(
        vlm_provider=vlm_provider,
        vlm_cache=vlm_cache,
        claim_extractor=ConservativeClaimExtractor(create_coco_category_registry()),
        detector_provider=detector_provider,
        clip_provider=clip_provider,
        image_base_dir=image_base_dir,
        gen_config=gen_config,
        enforce_train_split=False,  # Scalable M7 acquires across all partitions
    )

    new_claims: List[M7ClaimRecord] = []
    mode = "a" if resume and out_file.exists() else "w"
    with open(out_file, mode, encoding="utf-8") as f_out:
        for idx, entry in enumerate(entries_to_process, 1):
            img_id = entry.image.image_id
            logger.info(f"[{idx}/{len(entries_to_process)}] Processing image {img_id} (split={entry.split})...")
            try:
                evidence_records = pipeline.process_entry(entry)
                image_claims = ingest_m6_evidence_to_m7_claims(
                    evidence_records=evidence_records,
                    manifest=manifest,
                    allow_missing_images=False,
                )
                for c in image_claims:
                    f_out.write(json.dumps(c.to_dict()) + "\n")
                    new_claims.append(c)
                f_out.flush()
                logger.info(f"  Extracted {len(image_claims)} claims for {img_id}.")
            except Exception as e:
                logger.error(f"Failed processing image {img_id}: {e}")
                raise

    all_claims = existing_claims + new_claims
    logger.info(f"Evidence acquisition finished. Total claims in {out_file}: {len(all_claims)}")
    return all_claims


def main():
    args = parse_args()
    device_map = "auto" if args.load_in_4bit else None
    compute_dtype = "float16" if args.load_in_4bit else args.dtype

    logger.info("Initializing Scalable M7 Evidence Acquisition Pipeline...")
    vlm_cache = VLMCache(args.cache_dir)
    vlm_provider = LLaVA15Provider(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        allow_download=args.allow_download,
        local_files_only=not args.allow_download,
        load_in_4bit=args.load_in_4bit,
        device_map=device_map,
    )
    gen_config = VLMGenerationConfig(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        compute_dtype=compute_dtype,
        device_map=device_map,
    )
    detector_provider = HuggingFaceDetectorProvider(
        model_name=args.detector_model,
        device=args.evidence_device,
        local_files_only=not args.allow_download,
    )
    clip_provider = TransformersCLIPProvider(
        model_name=args.clip_model,
        device=args.evidence_device,
        local_files_only=not args.allow_download,
    )

    run_evidence_acquisition(
        manifest_path=args.manifest,
        output_claims_path=args.output_claims,
        vlm_provider=vlm_provider,
        vlm_cache=vlm_cache,
        detector_provider=detector_provider,
        clip_provider=clip_provider,
        image_base_dir=args.image_base_dir,
        gen_config=gen_config,
        max_images=args.max_images,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
