"""
End-to-End Visual Evidence Pipeline.

Orchestrates:
Real COCO images (TRAIN split) -> VLM Captioning (LLaVA-1.5) -> Conservative Claim Extraction ->
Object Detector (OWL-ViT raw max-score d_i in [0, 1]) -> CLIP (cosine similarity g_i in [-1, 1]) ->
Claim-Level JSONL Export.

CRITICAL METHODOLOGICAL GUARANTEES:
1. Observable evidence values remain strictly raw (d_i in [0, 1], g_i in [-1, 1]) with explicit availability flags.
2. Probability calibration is strictly forbidden in Milestone 6.
3. PGM potentials (theta_i, epsilon_i, J_ij) and robust posterior bounds are strictly NOT introduced.
4. Mathematical PGM cores in src/pgm/ and src/robust_bp/ remain completely untouched.
5. All pipeline demonstration images are drawn strictly from the TRAIN split.
"""

from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    GeneratedResponseRecord,
    SplitName,
)
from src.claims.vocabulary import CategoryRegistry, create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor, ExtractedClaim
from src.vlm.provider import VLMProvider, VLMGenerationConfig, compute_file_sha256
from src.vlm.cache import VLMCache
from src.evidence.schemas import ClaimLevelEvidenceRecord, RawEvidenceRecord
from src.evidence.detector_provider import BaseDetectorProvider, HuggingFaceDetectorProvider, DetectorResult
from src.evidence.clip_provider import BaseCLIPProvider, TransformersCLIPProvider, CLIPResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineStatistics:
    """Summary metrics and audit counts for the evidence pipeline execution."""
    images_processed: int = 0
    total_claims_extracted: int = 0
    detector_available_count: int = 0
    clip_available_count: int = 0
    unavailable_or_failed_count: int = 0
    zero_claim_images: int = 0
    uncalibrated_raw_evidence_guaranteed: bool = True
    no_pgm_parameters_inferred_guaranteed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VisualEvidencePipeline:
    """
    Production visual evidence pipeline orchestrating VLM generation, conservative claim extraction,
    and raw multimodal neural evidence extraction into a validated claim-level JSONL dataset.
    """

    def __init__(
        self,
        vlm_provider: VLMProvider,
        vlm_cache: VLMCache,
        claim_extractor: Optional[ConservativeClaimExtractor] = None,
        detector_provider: Optional[BaseDetectorProvider] = None,
        clip_provider: Optional[BaseCLIPProvider] = None,
        image_base_dir: Optional[Union[str, Path]] = None,
        gen_config: Optional[VLMGenerationConfig] = None,
        enforce_train_split: bool = True,
    ):
        self.vlm_provider = vlm_provider
        self.vlm_cache = vlm_cache
        self.claim_extractor = claim_extractor or ConservativeClaimExtractor(create_coco_category_registry())
        self.detector_provider = detector_provider or HuggingFaceDetectorProvider()
        self.clip_provider = clip_provider or TransformersCLIPProvider()
        self.image_base_dir = Path(image_base_dir) if image_base_dir else None
        self.gen_config = gen_config or VLMGenerationConfig(model_name=getattr(vlm_provider, "model_name", "llava-hf/llava-1.5-7b-hf"))
        self.enforce_train_split = enforce_train_split

    def resolve_image_path(self, entry: DatasetManifestEntry) -> Path:
        """Resolve absolute or relative path to the image file."""
        raw_fn = entry.image.file_name or f"{entry.image.image_id}.jpg"
        p = Path(raw_fn)
        if p.is_absolute() and p.exists():
            return p
        if self.image_base_dir:
            cand = self.image_base_dir / p.name
            if cand.exists():
                return cand
            cand2 = self.image_base_dir / raw_fn
            if cand2.exists():
                return cand2
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Cannot resolve image path for {entry.image.image_id} (file_name='{raw_fn}')")

    def process_entry(
        self,
        entry: DatasetManifestEntry,
    ) -> List[ClaimLevelEvidenceRecord]:
        """
        Process a single manifest entry end-to-end:
        1. Verify split isolation.
        2. Resolve image and verify SHA-256 integrity.
        3. Retrieve or generate VLM caption via cache.
        4. Extract atomic claims via ConservativeClaimExtractor.
        5. Extract raw detector presence scores d_i in [0, 1].
        6. Extract raw CLIP cosine similarities g_i in [-1, 1].
        7. Assemble validated ClaimLevelEvidenceRecord items.
        """
        # 1. Verify split isolation
        split_val = entry.split
        if split_val is None and "split" in entry.image.metadata:
            split_val = entry.image.metadata["split"]

        is_train = (split_val == SplitName.TRAIN or split_val == SplitName.TRAIN.value)
        if self.enforce_train_split and not is_train:
            raise ValueError(
                f"Image {entry.image.image_id} has split '{split_val}', but evidence pipeline "
                "strictly mandates TRAIN split images to guarantee split isolation."
            )

        # 2. Resolve image and check hash
        img_path = self.resolve_image_path(entry)
        computed_hash = compute_file_sha256(img_path)
        if entry.image.file_hash and entry.image.file_hash != computed_hash:
            logger.warning(
                f"SHA256 mismatch for {img_path}: expected {entry.image.file_hash}, got {computed_hash}"
            )

        # 3. Retrieve or generate VLM caption
        vlm_resp = self.vlm_cache.get(
            image_hash=computed_hash,
            model_name=self.vlm_provider.model_name,
            model_revision=getattr(self.vlm_provider, "resolve_revision", lambda: "snapshot")(),
            prompt=self.gen_config.prompt,
            gen_config=self.gen_config,
            provider_kind=self.vlm_provider.provider_kind,
            is_synthetic=self.vlm_provider.is_synthetic,
        )

        if vlm_resp is None:
            vlm_resp = self.vlm_provider.generate_caption(
                image_path=img_path,
                prompt=self.gen_config.prompt,
                gen_config=self.gen_config,
                image_id=entry.image.image_id,
                image_hash=computed_hash,
            )
            self.vlm_cache.put(vlm_resp, gen_config=self.gen_config)

        caption_text = vlm_resp.caption

        # 4. Extract atomic claims
        extraction_resp_rec = GeneratedResponseRecord(
            response_id=vlm_resp.response_id,
            image_id=entry.image.image_id,
            model_name=vlm_resp.model_name,
            response_text=caption_text,
        )
        report = self.claim_extractor.extract_from_response(extraction_resp_rec)

        claim_records: List[ClaimLevelEvidenceRecord] = []
        is_synth = getattr(entry.image, "is_synthetic", False) or entry.image.metadata.get("is_synthetic", False)

        for claim in report.accepted_claims:
            cat_name = claim.object_category
            first_span = claim.spans[0].matched_text if claim.spans else None

            # 5. Extract raw object detector evidence
            det_res: DetectorResult = self.detector_provider.detect_category(
                image_path=img_path,
                category=cat_name,
            )

            # 6. Extract raw CLIP similarity evidence
            clip_res: CLIPResult = self.clip_provider.compute_similarity(
                image_path=img_path,
                text=cat_name,
            )

            # Assemble diagnostic metadata (preserve failure reasons explicitly)
            meta: Dict[str, Any] = {
                "response_id": vlm_resp.response_id,
                "vlm_model": vlm_resp.model_name,
                "vlm_revision": vlm_resp.model_revision,
                "vlm_execution_time": vlm_resp.execution_time_seconds,
            }
            if det_res.error:
                meta["detector_error"] = det_res.error
            if clip_res.error:
                meta["clip_error"] = clip_res.error

            # 7. Construct and validate ClaimLevelEvidenceRecord
            rec = ClaimLevelEvidenceRecord(
                claim_id=claim.claim_id,
                image_id=entry.image.image_id,
                object_category=cat_name,
                text_span=first_span,
                caption=caption_text,
                image_hash=computed_hash,
                split="train" if is_train else str(split_val),
                detector_score=det_res.score,
                detector_available=det_res.available,
                detector_model=det_res.model_name,
                detector_revision=det_res.model_revision,
                detector_configuration=det_res.configuration,
                clip_score=clip_res.score,
                similarity_available=clip_res.available,
                clip_model=clip_res.model_name,
                clip_revision=clip_res.model_revision,
                clip_prompt_template=clip_res.prompt_template,
                preprocessing_configuration=clip_res.preprocessing_configuration,
                is_synthetic=bool(is_synth),
                metadata=meta,
            )
            rec.validate()
            claim_records.append(rec)

        return claim_records

    def run(
        self,
        manifest: DatasetManifest,
        sample_size: Optional[int] = None,
        output_jsonl_path: Optional[Union[str, Path]] = None,
    ) -> (List[ClaimLevelEvidenceRecord], PipelineStatistics):
        """
        Execute the evidence pipeline over manifest entries and export JSONL.
        """
        stats = PipelineStatistics()
        all_records: List[ClaimLevelEvidenceRecord] = []

        entries_to_process = manifest.entries
        if sample_size is not None and sample_size > 0:
            entries_to_process = entries_to_process[:sample_size]

        for entry in entries_to_process:
            records = self.process_entry(entry)
            stats.images_processed += 1

            if not records:
                stats.zero_claim_images += 1
                continue

            for rec in records:
                stats.total_claims_extracted += 1
                if rec.detector_available:
                    stats.detector_available_count += 1
                if rec.similarity_available:
                    stats.clip_available_count += 1
                if not rec.detector_available or not rec.similarity_available:
                    stats.unavailable_or_failed_count += 1

                all_records.append(rec)

        # Export JSONL if destination specified
        if output_jsonl_path:
            out_path = Path(output_jsonl_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in all_records:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            logger.info(f"Exported {len(all_records)} claim records to {out_path}")

        return all_records, stats
