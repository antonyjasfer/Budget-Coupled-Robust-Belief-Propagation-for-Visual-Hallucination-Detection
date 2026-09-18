"""
Frozen-VLM caption generation, caching, and pilot pipeline package.
"""

from src.vlm.provider import (
    VLMProvider,
    VLMGenerationConfig,
    VLMResponse,
    SyntheticVLMProvider,
    compute_file_sha256,
)
from src.vlm.llava_provider import LLaVA15Provider
from src.vlm.cache import VLMCache, CacheStatistics
from src.vlm.pipeline import (
    AnnotationReviewStatus,
    AnnotationBundleEntry,
    AnnotationBundle,
    PilotRunStats,
    select_pilot_images,
    run_vlm_pilot,
    export_annotation_bundle,
)

__all__ = [
    "VLMProvider",
    "VLMGenerationConfig",
    "VLMResponse",
    "SyntheticVLMProvider",
    "compute_file_sha256",
    "LLaVA15Provider",
    "VLMCache",
    "CacheStatistics",
    "AnnotationReviewStatus",
    "AnnotationBundleEntry",
    "AnnotationBundle",
    "PilotRunStats",
    "select_pilot_images",
    "run_vlm_pilot",
    "export_annotation_bundle",
]
