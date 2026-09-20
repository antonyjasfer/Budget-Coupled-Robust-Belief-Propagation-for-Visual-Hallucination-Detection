"""
M7 Annotation and Ground-Truth Package.
"""

from src.annotation.schemas import (
    parse_ground_truth_status,
    HumanAnnotationTask,
    AnnotationRecord,
    AdjudicationRecord,
    M7ClaimRecord,
    FinalGroundTruthRecord,
)
from src.annotation.masking import (
    create_masked_annotation_task,
    audit_masked_task,
    extract_minimal_context,
    FORBIDDEN_EVIDENCE_FIELDS,
)
from src.annotation.leakage import (
    validate_no_image_split_overlap,
    validate_hash_split_disjointness,
    validate_claim_image_split_consistency,
    SplitAuditReport,
    SplitLeakageError,
)
from src.annotation.agreement import (
    compute_cohens_kappa,
    merge_and_adjudicate_annotations,
    CohenKappaResult,
)
from src.annotation.workflow import (
    QualityReport,
    load_m6_evidence_file,
    ingest_m6_evidence_to_m7_claims,
    export_masked_templates,
    load_annotations_file,
    load_adjudications_file,
    build_final_ground_truth_dataset,
    generate_quality_report,
)

__all__ = [
    "parse_ground_truth_status",
    "HumanAnnotationTask",
    "AnnotationRecord",
    "AdjudicationRecord",
    "M7ClaimRecord",
    "FinalGroundTruthRecord",
    "create_masked_annotation_task",
    "audit_masked_task",
    "extract_minimal_context",
    "FORBIDDEN_EVIDENCE_FIELDS",
    "validate_no_image_split_overlap",
    "validate_hash_split_disjointness",
    "validate_claim_image_split_consistency",
    "SplitAuditReport",
    "SplitLeakageError",
    "compute_cohens_kappa",
    "merge_and_adjudicate_annotations",
    "CohenKappaResult",
    "QualityReport",
    "load_m6_evidence_file",
    "ingest_m6_evidence_to_m7_claims",
    "export_masked_templates",
    "load_annotations_file",
    "load_adjudications_file",
    "build_final_ground_truth_dataset",
    "generate_quality_report",
]
