"""
Data foundation package for visual hallucination detection.
Provides typed data contracts, metadata adapters, image registry, manifests, and deterministic splitting.
"""

from src.data.schemas import (
    GroundTruthStatus,
    DecisionStatus,
    AnnotationSource,
    DatasetSource,
    SplitName,
    ImageRecord,
    AtomicObjectExistenceClaim,
    AnnotationRecord,
    PredictionDecisionRecord,
    GeneratedResponseRecord,
    DatasetManifestEntry,
    DatasetManifest,
    ground_truth_to_pgm_label,
)
from src.data.image_registry import ImageRegistry, OverlapReport
from src.data.manifests import create_manifest, validate_manifest, load_manifest, save_manifest
from src.data.splits import split_manifest_by_image_groups, SplitResult

__all__ = [
    "GroundTruthStatus",
    "DecisionStatus",
    "AnnotationSource",
    "DatasetSource",
    "SplitName",
    "ImageRecord",
    "AtomicObjectExistenceClaim",
    "AnnotationRecord",
    "PredictionDecisionRecord",
    "GeneratedResponseRecord",
    "DatasetManifestEntry",
    "DatasetManifest",
    "ground_truth_to_pgm_label",
    "ImageRegistry",
    "OverlapReport",
    "create_manifest",
    "validate_manifest",
    "load_manifest",
    "save_manifest",
    "split_manifest_by_image_groups",
    "SplitResult",
]
