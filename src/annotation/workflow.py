"""
Workflow orchestration and honest quality reporting for M7.

Provides:
- Manifest building and split auditing via src.data.splits
- Ingestion of M6 ClaimLevelEvidenceRecord exports
- Generation of independent masked annotation templates
- Merging and adjudication of completed annotations
- Granular dataset quality and honest coverage reporting
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Sequence, Union
from collections import defaultdict
import json

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    DatasetSource,
    SplitName,
    GroundTruthStatus,
)
from src.data.splits import split_manifest_by_image_groups, DEFAULT_PROPORTIONS
from src.data.image_registry import ImageRegistry, compute_file_sha256
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import (
    M7ClaimRecord,
    HumanAnnotationTask,
    AnnotationRecord,
    AdjudicationRecord,
    FinalGroundTruthRecord,
    parse_ground_truth_status,
)
from src.annotation.masking import create_masked_annotation_task, audit_masked_task
from src.annotation.leakage import (
    validate_no_image_split_overlap,
    validate_hash_split_disjointness,
    validate_claim_image_split_consistency,
    SplitAuditReport,
)
from src.annotation.agreement import (
    compute_cohens_kappa,
    merge_and_adjudicate_annotations,
    CohenKappaResult,
)


@dataclass
class QualityReport:
    """Comprehensive dataset quality, leakage, and coverage report."""
    # Image level
    target_reserved_images: int
    actual_images_in_manifest: int
    images_per_split: Dict[str, int]
    identity_groups_count: int
    split_leakage_audit: Dict[str, Any]

    # Evidence level
    evidence_covered_images: int
    evidence_covered_claims: int
    detector_available_claims: int
    clip_available_claims: int

    # Claim level
    total_claims: int
    claims_per_split: Dict[str, int]
    claims_per_image: Dict[str, int]
    zero_claim_images_count: int

    # Annotation level
    total_annotations: int
    claims_with_annotator_a: int
    claims_with_annotator_b: int
    jointly_annotated_claims: int
    adjudicated_claims: int
    supported_count: int
    hallucinated_count: int
    unknown_count: int
    annotation_coverage_pct: float

    # Agreement level
    agreement_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_level": {
                "target_reserved_images": self.target_reserved_images,
                "actual_images_in_manifest": self.actual_images_in_manifest,
                "images_per_split": self.images_per_split,
                "identity_groups_count": self.identity_groups_count,
                "split_leakage_audit": self.split_leakage_audit,
            },
            "evidence_level": {
                "evidence_covered_images": self.evidence_covered_images,
                "evidence_covered_claims": self.evidence_covered_claims,
                "detector_available_claims": self.detector_available_claims,
                "clip_available_claims": self.clip_available_claims,
            },
            "claim_level": {
                "total_claims": self.total_claims,
                "claims_per_split": self.claims_per_split,
                "claims_per_image": self.claims_per_image,
                "zero_claim_images_count": self.zero_claim_images_count,
            },
            "annotation_level": {
                "total_annotations": self.total_annotations,
                "claims_with_annotator_a": self.claims_with_annotator_a,
                "claims_with_annotator_b": self.claims_with_annotator_b,
                "jointly_annotated_claims": self.jointly_annotated_claims,
                "adjudicated_claims": self.adjudicated_claims,
                "supported_count": self.supported_count,
                "hallucinated_count": self.hallucinated_count,
                "unknown_count": self.unknown_count,
                "annotation_coverage_pct": self.annotation_coverage_pct,
            },
            "agreement_level": self.agreement_metrics,
        }

    def to_markdown(self) -> str:
        """Format a clear, honest markdown report."""
        lines = [
            "# Milestone 7 Dataset Quality and Coverage Report",
            "",
            "## 1. Image-Level Partitioning and Leakage Audit",
            f"- **Target Reserved Images**: {self.target_reserved_images}",
            f"- **Actual Images in Manifest**: {self.actual_images_in_manifest}",
            f"- **Identity Groups Count**: {self.identity_groups_count}",
            f"- **Images per Split**: {self.images_per_split}",
            f"- **Cross-Split Leakage Free**: {self.split_leakage_audit.get('is_valid', False)}",
            "",
            "## 2. Visual Evidence Coverage (M6 Ingestion)",
            f"- **Evidence-Covered Images**: {self.evidence_covered_images}",
            f"- **Evidence-Covered Claims**: {self.evidence_covered_claims}",
            f"- **Detector Available Claims**: {self.detector_available_claims}",
            f"- **CLIP Available Claims**: {self.clip_available_claims}",
            "",
            "## 3. Claim-Level Distribution",
            f"- **Total Extracted Claims**: {self.total_claims}",
            f"- **Claims per Split**: {self.claims_per_split}",
            f"- **Zero-Claim Images Count**: {self.zero_claim_images_count}",
            "",
            "## 4. Human Annotation and Coverage",
            f"- **Total Annotation Submissions**: {self.total_annotations}",
            f"- **Claims with Annotator A**: {self.claims_with_annotator_a}",
            f"- **Claims with Annotator B**: {self.claims_with_annotator_b}",
            f"- **Jointly Annotated Claims**: {self.jointly_annotated_claims}",
            f"- **Adjudicated Claims**: {self.adjudicated_claims}",
            f"- **Final SUPPORTED Labels**: {self.supported_count}",
            f"- **Final HALLUCINATED Labels**: {self.hallucinated_count}",
            f"- **Final UNKNOWN Labels**: {self.unknown_count}",
            f"- **Annotation Coverage**: {self.annotation_coverage_pct:.2f}%",
            "",
            "## 5. Inter-Annotator Agreement (Cohen's Kappa)",
            f"- **Observed Agreement (Po)**: {self.agreement_metrics.get('observed_agreement', 0.0):.4f}",
            f"- **Chance Agreement (Pe)**: {self.agreement_metrics.get('chance_agreement', 0.0):.4f}",
            f"- **Cohen's Kappa**: {self.agreement_metrics.get('cohens_kappa')}",
            f"- **Kappa Defined**: {self.agreement_metrics.get('kappa_defined')}",
            f"- **Status**: {self.agreement_metrics.get('status_message', 'N/A')}",
            "",
        ]
        return "\n".join(lines)


def load_m6_evidence_file(file_path: Union[str, Path]) -> List[ClaimLevelEvidenceRecord]:
    """Read an M6 ClaimLevelEvidenceRecord JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"M6 evidence file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(ClaimLevelEvidenceRecord.from_dict(data))
            except Exception as e:
                raise ValueError(f"Failed to parse M6 evidence record at line {line_num}: {e}")
    return records


def ingest_m6_evidence_to_m7_claims(
    evidence_records: Sequence[ClaimLevelEvidenceRecord],
    manifest: Optional[DatasetManifest] = None,
    allow_missing_images: bool = False,
) -> List[M7ClaimRecord]:
    """
    Transform M6 ClaimLevelEvidenceRecords into M7ClaimRecords linked to manifest splits.

    Args:
        evidence_records: Input M6 records.
        manifest: Optional DatasetManifest for split resolution and validation.
        allow_missing_images: If False, raises ValueError when an image is missing from manifest.

    Returns:
        List of M7ClaimRecord objects.
    """
    image_splits: Dict[str, str] = {}
    if manifest:
        for entry in manifest.entries:
            if entry.image:
                s_val = entry.split.value if hasattr(entry.split, "value") else str(entry.split)
                image_splits[entry.image.image_id] = s_val

    m7_claims = []
    for ev in evidence_records:
        if image_splits:
            if ev.image_id not in image_splits:
                if not allow_missing_images:
                    raise ValueError(
                        f"M6 evidence image '{ev.image_id}' not found in M7 manifest! "
                        "To prevent silent split contamination, all evidence images must be present in manifest."
                    )
                assigned_split = ev.split
            else:
                assigned_split = image_splits[ev.image_id]
        else:
            assigned_split = ev.split

        m7_claims.append(M7ClaimRecord.from_m6_evidence(ev, assigned_split=assigned_split))

    return m7_claims


def export_masked_templates(
    claims: Sequence[M7ClaimRecord],
    output_dir: Union[str, Path],
    image_path_lookup: Optional[Dict[str, str]] = None,
) -> Tuple[Path, Path]:
    """
    Export masked annotation templates for Annotator A and Annotator B.

    Returns:
        Tuple of (annotator_a_path, annotator_b_path).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    template_a_path = out_path / "m7_template_annotator_A.jsonl"
    template_b_path = out_path / "m7_template_annotator_B.jsonl"

    tasks_a = [
        create_masked_annotation_task(c, image_path_lookup=image_path_lookup, annotator_tag="A")
        for c in claims
    ]
    tasks_b = [
        create_masked_annotation_task(c, image_path_lookup=image_path_lookup, annotator_tag="B")
        for c in claims
    ]

    with open(template_a_path, "w", encoding="utf-8") as f:
        for t in tasks_a:
            f.write(json.dumps(t.to_dict()) + "\n")

    with open(template_b_path, "w", encoding="utf-8") as f:
        for t in tasks_b:
            f.write(json.dumps(t.to_dict()) + "\n")

    return template_a_path, template_b_path


def load_annotations_file(file_path: Union[str, Path]) -> Dict[Tuple[str, str], AnnotationRecord]:
    """
    Read completed annotations from a JSONL file, enforcing no duplicate submissions
    for the same (image_id, claim_id) pair.

    Returns:
        Dict mapping (image_id, claim_id) -> AnnotationRecord.
    """
    path = Path(file_path)
    if not path.exists():
        return {}

    annotations: Dict[Tuple[str, str], AnnotationRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rec = AnnotationRecord.from_dict(data)
                key = (rec.image_id, rec.claim_id)
                if key in annotations:
                    raise ValueError(
                        f"Duplicate annotation submission for image '{rec.image_id}', "
                        f"claim '{rec.claim_id}' on line {line_num} in {path}"
                    )
                annotations[key] = rec
            except Exception as e:
                raise ValueError(f"Error parsing annotation at line {line_num} in {path}: {e}")

    return annotations


def load_adjudications_file(file_path: Union[str, Path]) -> Dict[Tuple[str, str], AdjudicationRecord]:
    """
    Read completed adjudication records from JSONL file.

    Returns:
        Dict mapping (image_id, claim_id) -> AdjudicationRecord.
    """
    path = Path(file_path)
    if not path.exists():
        return {}

    adjudications: Dict[Tuple[str, str], AdjudicationRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                rec = AdjudicationRecord.from_dict(data)
                adjudications[(rec.image_id, rec.claim_id)] = rec
            except Exception as e:
                raise ValueError(f"Error parsing adjudication at line {line_num} in {path}: {e}")

    return adjudications


def build_final_ground_truth_dataset(
    claims: Sequence[M7ClaimRecord],
    annotations_a: Dict[Tuple[str, str], AnnotationRecord],
    annotations_b: Dict[Tuple[str, str], AnnotationRecord],
    adjudications: Dict[Tuple[str, str], AdjudicationRecord],
) -> List[FinalGroundTruthRecord]:
    """
    Merge M7 claims, dual annotations, and adjudications into the final ground-truth dataset.
    """
    final_records = []
    for claim in claims:
        key = (claim.image_id, claim.claim_id)
        ann_a = annotations_a.get(key)
        ann_b = annotations_b.get(key)
        adj = adjudications.get(key)

        record = merge_and_adjudicate_annotations(
            claim_record=claim,
            annotation_a=ann_a,
            annotation_b=ann_b,
            adjudication=adj,
        )
        final_records.append(record)

    return final_records


def generate_quality_report(
    manifest: Optional[DatasetManifest],
    claims: Sequence[M7ClaimRecord],
    final_records: Sequence[FinalGroundTruthRecord],
    target_reserved_count: int = 600,
) -> QualityReport:
    """
    Generate complete, honest dataset quality report across all 5 dimensions.
    """
    # 1. Image level
    actual_images = len(manifest.entries) if manifest else len(set(c.image_id for c in claims))
    images_per_split: Dict[str, int] = defaultdict(int)
    id_groups_count = 0
    split_leakage_dict: Dict[str, Any] = {"is_valid": True}

    if manifest:
        for entry in manifest.entries:
            s_val = entry.split.value if hasattr(entry.split, "value") else str(entry.split)
            images_per_split[s_val] += 1
        try:
            audit = validate_no_image_split_overlap(manifest.entries)
            split_leakage_dict = audit.to_dict()
        except Exception as e:
            split_leakage_dict = {"is_valid": False, "error": str(e)}

        # Count groups
        registry = ImageRegistry()
        for entry in manifest.entries:
            if entry.image:
                registry.register_image(entry.image)
        id_groups_count = len(registry.groups)
    else:
        for c in claims:
            images_per_split[c.split] += 1

    # 2. Evidence level
    ev_images = len(set(c.image_id for c in claims))
    ev_claims = len(claims)
    det_avail = sum(1 for c in claims if c.evidence.detector_available)
    clip_avail = sum(1 for c in claims if c.evidence.similarity_available)

    # 3. Claim level
    claims_per_split: Dict[str, int] = defaultdict(int)
    claims_per_image: Dict[str, int] = defaultdict(int)
    for c in claims:
        claims_per_split[c.split] += 1
        claims_per_image[c.image_id] += 1

    manifest_image_ids = {e.image.image_id for e in manifest.entries if e.image} if manifest else set(claims_per_image.keys())
    zero_claim_images = len(manifest_image_ids - set(claims_per_image.keys()))

    # 4. Annotation level
    ann_a_count = sum(1 for r in final_records if r.annotation_a is not None)
    ann_b_count = sum(1 for r in final_records if r.annotation_b is not None)
    joint_count = sum(1 for r in final_records if r.annotation_a is not None and r.annotation_b is not None)
    adj_count = sum(1 for r in final_records if r.adjudication is not None)

    supported_cnt = sum(1 for r in final_records if r.final_ground_truth == GroundTruthStatus.SUPPORTED)
    hallucinated_cnt = sum(1 for r in final_records if r.final_ground_truth == GroundTruthStatus.HALLUCINATED)
    unknown_cnt = sum(1 for r in final_records if r.final_ground_truth == GroundTruthStatus.UNKNOWN)

    total_claims_count = len(claims)
    annotated_resolved_count = supported_cnt + hallucinated_cnt + unknown_cnt
    coverage_pct = (annotated_resolved_count / total_claims_count * 100.0) if total_claims_count > 0 else 0.0

    # 5. Agreement level
    ann_a_statuses = [r.annotation_a.ground_truth_status for r in final_records if r.annotation_a and r.annotation_b]
    ann_b_statuses = [r.annotation_b.ground_truth_status for r in final_records if r.annotation_a and r.annotation_b]
    kappa_res = compute_cohens_kappa(ann_a_statuses, ann_b_statuses)

    return QualityReport(
        target_reserved_images=target_reserved_count,
        actual_images_in_manifest=actual_images,
        images_per_split=dict(images_per_split),
        identity_groups_count=id_groups_count,
        split_leakage_audit=split_leakage_dict,
        evidence_covered_images=ev_images,
        evidence_covered_claims=ev_claims,
        detector_available_claims=det_avail,
        clip_available_claims=clip_avail,
        total_claims=total_claims_count,
        claims_per_split=dict(claims_per_split),
        claims_per_image=dict(claims_per_image),
        zero_claim_images_count=zero_claim_images,
        total_annotations=ann_a_count + ann_b_count,
        claims_with_annotator_a=ann_a_count,
        claims_with_annotator_b=ann_b_count,
        jointly_annotated_claims=joint_count,
        adjudicated_claims=adj_count,
        supported_count=supported_cnt,
        hallucinated_count=hallucinated_cnt,
        unknown_count=unknown_cnt,
        annotation_coverage_pct=coverage_pct,
        agreement_metrics=kappa_res.to_dict(),
    )
