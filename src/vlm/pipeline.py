"""
Development-only real-image pilot pipeline and annotation-ready bundle export.

Coordinates:
1. Training-split-only deterministic image selection.
2. Cached frozen-VLM caption acquisition.
3. Conservative claim extraction and rejected mention diagnostics.
4. Annotation bundle formatting with unreviewed status.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import random
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    GeneratedResponseRecord,
    AtomicObjectExistenceClaim,
    SplitName,
)
from src.claims.vocabulary import CategoryRegistry, create_coco_category_registry
from src.claims.extraction import (
    ConservativeClaimExtractor,
    ExtractionReport,
)
from src.vlm.provider import (
    VLMProvider,
    VLMGenerationConfig,
    VLMResponse,
    compute_file_sha256,
)
from src.vlm.cache import VLMCache

logger = logging.getLogger(__name__)


class AnnotationReviewStatus(str, Enum):
    """Explicit review state for exported annotation candidates."""
    NOT_YET_REVIEWED = "not_yet_reviewed"
    REVIEWED_SUPPORTED = "reviewed_supported"
    REVIEWED_HALLUCINATED = "reviewed_hallucinated"
    REVIEWED_UNKNOWN = "reviewed_unknown"


@dataclass
class AnnotationBundleEntry:
    """
    Export record for human or adjudicator review.
    Contains image reference, raw response, extracted claims, diagnostics, and unreviewed status.
    """
    image_id: str
    image_path: str
    image_hash: str
    split: str
    response_id: str
    caption: str
    model_name: str
    model_revision: str
    prompt: str
    is_synthetic: bool
    accepted_claims: List[Dict[str, Any]]
    rejected_mentions: List[Dict[str, Any]]
    provider_kind: str = "synthetic_fixture"
    review_status: str = AnnotationReviewStatus.NOT_YET_REVIEWED.value
    review_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationBundleEntry":
        is_synth = bool(data.get("is_synthetic", True))
        default_kind = "synthetic_fixture" if is_synth else "llava_15_hf"
        return cls(
            image_id=str(data["image_id"]),
            image_path=str(data["image_path"]),
            image_hash=str(data["image_hash"]),
            split=str(data["split"]),
            response_id=str(data["response_id"]),
            caption=str(data["caption"]),
            model_name=str(data["model_name"]),
            model_revision=str(data["model_revision"]),
            prompt=str(data["prompt"]),
            is_synthetic=is_synth,
            accepted_claims=list(data.get("accepted_claims", [])),
            rejected_mentions=list(data.get("rejected_mentions", [])),
            provider_kind=str(data.get("provider_kind", default_kind)),
            review_status=str(data.get("review_status", AnnotationReviewStatus.NOT_YET_REVIEWED.value)),
            review_notes=data.get("review_notes"),
        )


@dataclass
class AnnotationBundle:
    """
    Complete annotation bundle artifact ready for adjudication.
    """
    bundle_id: str
    created_at: str
    total_responses: int
    total_accepted_claims: int
    total_rejected_mentions: int
    entries: List[AnnotationBundleEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "total_responses": self.total_responses,
            "total_accepted_claims": self.total_accepted_claims,
            "total_rejected_mentions": self.total_rejected_mentions,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationBundle":
        return cls(
            bundle_id=str(data["bundle_id"]),
            created_at=str(data["created_at"]),
            total_responses=int(data["total_responses"]),
            total_accepted_claims=int(data["total_accepted_claims"]),
            total_rejected_mentions=int(data["total_rejected_mentions"]),
            entries=[AnnotationBundleEntry.from_dict(e) for e in data.get("entries", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PilotRunStats:
    """Execution statistics for a pilot run."""
    selected_image_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    generated_captions: int = 0
    accepted_claims_count: int = 0
    rejected_mentions_count: int = 0
    zero_claim_responses: int = 0
    total_execution_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_pilot_images(
    manifest: DatasetManifest,
    split_registry: Optional[Dict[str, str]] = None,
    sample_size: int = 10,
    seed: int = 42,
) -> List[DatasetManifestEntry]:
    """
    Deterministically select up to `sample_size` images strictly from the TRAIN split.
    Raises ValueError if non-training images or reserved evaluation images are encountered.
    """
    train_entries: List[DatasetManifestEntry] = []

    for entry in manifest.entries:
        img_id = entry.image.image_id
        split_val = entry.split
        if split_val is None and "split" in entry.image.metadata:
            split_val = entry.image.metadata["split"]
        if split_registry and img_id in split_registry:
            split_val = split_registry[img_id]

        if split_val == SplitName.TRAIN or split_val == SplitName.TRAIN.value:
            train_entries.append(entry)

    if not train_entries:
        raise ValueError("No images found in TRAIN split within the supplied manifest/registry")

    # Sort deterministically by image_id before sampling
    train_entries.sort(key=lambda e: e.image.image_id)

    rng = random.Random(seed)
    selected_count = min(sample_size, len(train_entries))
    return rng.sample(train_entries, selected_count)


def run_vlm_pilot(
    manifest: DatasetManifest,
    provider: VLMProvider,
    cache: VLMCache,
    extractor: Optional[ConservativeClaimExtractor] = None,
    gen_config: Optional[VLMGenerationConfig] = None,
    split_registry: Optional[Dict[str, str]] = None,
    sample_size: int = 10,
    seed: int = 42,
    image_base_dir: Optional[Union[str, Path]] = None,
) -> Tuple[AnnotationBundle, PilotRunStats]:
    """
    Execute the development-only pilot pipeline.
    """
    config = gen_config or VLMGenerationConfig()
    if extractor is None:
        registry = create_coco_category_registry()
        extractor = ConservativeClaimExtractor(category_registry=registry)

    # 1. Select images strictly from training split
    selected_entries = select_pilot_images(
        manifest=manifest,
        split_registry=split_registry,
        sample_size=sample_size,
        seed=seed,
    )

    stats = PilotRunStats(selected_image_count=len(selected_entries))
    bundle_entries: List[AnnotationBundleEntry] = []
    model_info = provider.get_model_info()
    provider_kind = getattr(provider, "provider_kind", "synthetic_fixture" if provider.is_synthetic else "llava_15_hf")

    for entry in selected_entries:
        img = entry.image
        img_id = img.image_id

        # Resolve image file path
        raw_name = img.file_name or f"{img_id}.jpg"
        if image_base_dir and not Path(raw_name).is_absolute():
            img_path = Path(image_base_dir) / raw_name
        else:
            img_path = Path(raw_name)

        # Compute or reuse image hash
        if img_path.exists():
            img_hash = compute_file_sha256(img_path)
        elif img.file_hash:
            img_hash = img.file_hash
        else:
            img_hash = hashlib.sha256(img_id.encode("utf-8")).hexdigest()

        # 2. Check cache with provider_kind and is_synthetic isolation
        cached_resp = cache.get(
            image_hash=img_hash,
            model_name=model_info.get("model_name", config.model_name),
            model_revision=model_info.get("model_revision", model_info.get("resolved_revision", "rev0")),
            prompt=config.prompt,
            gen_config=config,
            provider_kind=provider_kind,
            is_synthetic=provider.is_synthetic,
        )

        if cached_resp is not None:
            stats.cache_hits += 1
            vlm_resp = cached_resp
        else:
            stats.cache_misses += 1
            vlm_resp = provider.generate_caption(
                image_path=img_path,
                prompt=config.prompt,
                gen_config=config,
                image_id=img_id,
                image_hash=img_hash,
            )
            # Store to cache
            cache.put(vlm_resp, gen_config=config)
            stats.generated_captions += 1

        # Reject synthetic output in production mode
        if not provider.is_synthetic and vlm_resp.is_synthetic:
            raise RuntimeError(
                f"Production provider produced synthetic response for image {img_id}. Silent synthetic fallback is forbidden."
            )

        stats.total_execution_seconds += vlm_resp.execution_time_seconds

        # 3. Extract claims
        extraction_report: ExtractionReport = extractor.extract_claims_from_text(
            text=vlm_resp.caption,
            response_id=vlm_resp.response_id,
            image_id=img_id,
        )

        accepted_claims_dict = [c.to_dict() for c in extraction_report.accepted_claims]
        rejected_mentions_dict = [m.to_dict() for m in extraction_report.rejected_mentions]

        stats.accepted_claims_count += len(accepted_claims_dict)
        stats.rejected_mentions_count += len(rejected_mentions_dict)

        if len(accepted_claims_dict) == 0:
            stats.zero_claim_responses += 1

        bundle_entry = AnnotationBundleEntry(
            image_id=img_id,
            image_path=str(img_path),
            image_hash=img_hash,
            split=SplitName.TRAIN.value,
            response_id=vlm_resp.response_id,
            caption=vlm_resp.caption,
            model_name=vlm_resp.model_name,
            model_revision=vlm_resp.model_revision,
            prompt=vlm_resp.prompt,
            is_synthetic=vlm_resp.is_synthetic,
            provider_kind=vlm_resp.provider_kind,
            accepted_claims=accepted_claims_dict,
            rejected_mentions=rejected_mentions_dict,
            review_status=AnnotationReviewStatus.NOT_YET_REVIEWED.value,
            review_notes=None,
        )
        bundle_entries.append(bundle_entry)

    bundle_id = f"bundle_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    bundle = AnnotationBundle(
        bundle_id=bundle_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_responses=len(bundle_entries),
        total_accepted_claims=stats.accepted_claims_count,
        total_rejected_mentions=stats.rejected_mentions_count,
        entries=bundle_entries,
        metadata={
            "pilot_seed": seed,
            "sample_size": sample_size,
            "model_info": model_info,
            "gen_config": config.to_dict(),
        },
    )

    return bundle, stats


def export_annotation_bundle(bundle: AnnotationBundle, output_path: Union[str, Path]) -> Path:
    """Save an annotation bundle to a JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(bundle.to_dict(), f, indent=2)
    return out
