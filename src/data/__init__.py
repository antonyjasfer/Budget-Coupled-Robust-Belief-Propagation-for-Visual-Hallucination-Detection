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

from src.data.provenance import (
    DataProvenanceState,
    EvidenceFailureState,
    ValidationMode,
    EvidenceFailureRecord,
    ProvenanceAuditResult,
    parse_provenance_state,
    validate_provenance_isolation,
    validate_evidence_status,
)
from src.data.sampling import (
    CohortType,
    SamplingManifest,
    filter_eligible_coco_universe,
    sample_primary_representative_cohort,
    assign_frozen_splits,
    create_sampling_manifest,
    create_predeclared_corruption_manifest,
)
from src.data.dataset_adequacy import (
    GraphAdequacyRating,
    GraphStratificationCounts,
    GraphAdequacyReport,
    stratify_graph_adequacy,
    compute_binomial_precision,
    evaluate_test_event_adequacy,
)
from src.data.dataset_lock import (
    DatasetLock,
    verify_dataset_lock,
    create_and_verify_dataset_lock,
)
from src.data.external_benchmarks import (
    BenchmarkTaskType,
    BenchmarkAdapterStatus,
    BenchmarkMappingResult,
    BaseBenchmarkAdapter,
    POPEAdapter,
    AMBERAdapter,
)
from src.data.vlm_interface import (
    BaseVLMProvider,
    LLaVAProviderWrapper,
    SecondVLMProviderInterface,
)

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
    "DataProvenanceState",
    "EvidenceFailureState",
    "ValidationMode",
    "EvidenceFailureRecord",
    "ProvenanceAuditResult",
    "parse_provenance_state",
    "validate_provenance_isolation",
    "validate_evidence_status",
    "CohortType",
    "SamplingManifest",
    "filter_eligible_coco_universe",
    "sample_primary_representative_cohort",
    "assign_frozen_splits",
    "create_sampling_manifest",
    "create_predeclared_corruption_manifest",
    "GraphAdequacyRating",
    "GraphStratificationCounts",
    "GraphAdequacyReport",
    "stratify_graph_adequacy",
    "compute_binomial_precision",
    "evaluate_test_event_adequacy",
    "DatasetLock",
    "verify_dataset_lock",
    "create_and_verify_dataset_lock",
    "BenchmarkTaskType",
    "BenchmarkAdapterStatus",
    "BenchmarkMappingResult",
    "BaseBenchmarkAdapter",
    "POPEAdapter",
    "AMBERAdapter",
    "BaseVLMProvider",
    "LLaVAProviderWrapper",
    "SecondVLMProviderInterface",
]

