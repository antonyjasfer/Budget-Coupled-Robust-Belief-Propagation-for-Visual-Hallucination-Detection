"""
Inter-annotator agreement computation and adjudication logic for M7.

STRICT METHODOLOGICAL GUARANTEES:
1. Computes standard multiclass Cohen's kappa across exactly:
   SUPPORTED, HALLUCINATED, UNKNOWN.
2. Handles Pe == 1 boundary:
   When Pe == 1.0 (zero denominator 1 - Pe), returns cohens_kappa = None,
   kappa_defined = False, and an explanatory status message. NEVER divides by zero
   and NEVER silently returns zero.
3. Strict disagreement and adjudication policy:
   If Annotator A == Annotator B:
     agreement = True, consensus may be recorded as the shared label.
   If Annotator A != Annotator B:
     agreement = False, consensus = None, adjudication_required = True.
   NEVER automatically selects Annotator A or Annotator B.
   Preserves both raw independent annotations permanently.
4. UNKNOWN semantics:
   UNKNOWN is an observation status indicating visual indeterminacy.
   It is never conflated with HALLUCINATED or ABSTAIN.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Sequence
from collections import defaultdict
import math

from src.data.schemas import GroundTruthStatus
from src.annotation.schemas import (
    AnnotationRecord,
    AdjudicationRecord,
    FinalGroundTruthRecord,
    M7ClaimRecord,
    parse_ground_truth_status,
)

CANONICAL_STATUSES: Tuple[GroundTruthStatus, ...] = (
    GroundTruthStatus.SUPPORTED,
    GroundTruthStatus.HALLUCINATED,
    GroundTruthStatus.UNKNOWN,
)


@dataclass
class CohenKappaResult:
    """Detailed Cohen's Kappa evaluation metrics."""
    joint_count: int
    agreement_count: int
    observed_agreement: float
    chance_agreement: float
    cohens_kappa: Optional[float]
    kappa_defined: bool
    status_message: str
    confusion_matrix: Dict[str, Dict[str, int]]
    marginals_a: Dict[str, int]
    marginals_b: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "joint_count": self.joint_count,
            "agreement_count": self.agreement_count,
            "observed_agreement": self.observed_agreement,
            "chance_agreement": self.chance_agreement,
            "cohens_kappa": self.cohens_kappa,
            "kappa_defined": self.kappa_defined,
            "status_message": self.status_message,
            "confusion_matrix": self.confusion_matrix,
            "marginals_a": self.marginals_a,
            "marginals_b": self.marginals_b,
        }


def compute_cohens_kappa(
    annotations_a: Sequence[GroundTruthStatus],
    annotations_b: Sequence[GroundTruthStatus],
    tolerance: float = 1e-9,
) -> CohenKappaResult:
    """
    Compute multiclass Cohen's kappa across SUPPORTED, HALLUCINATED, and UNKNOWN.

    Args:
        annotations_a: Sequence of ground truth statuses from Annotator A.
        annotations_b: Sequence of ground truth statuses from Annotator B.
        tolerance: Numerical tolerance for floating point boundary checks.

    Returns:
        CohenKappaResult with complete contingency table and metrics.
    """
    if len(annotations_a) != len(annotations_b):
        raise ValueError(
            f"Mismatched annotation counts: Annotator A has {len(annotations_a)} "
            f"items, Annotator B has {len(annotations_b)} items."
        )

    n = len(annotations_a)

    # Initialize 3x3 contingency matrix over canonical statuses
    categories = [s.value for s in CANONICAL_STATUSES]
    matrix: Dict[str, Dict[str, int]] = {
        row: {col: 0 for col in categories}
        for row in categories
    }

    if n == 0:
        return CohenKappaResult(
            joint_count=0,
            agreement_count=0,
            observed_agreement=0.0,
            chance_agreement=0.0,
            cohens_kappa=None,
            kappa_defined=False,
            status_message="No jointly annotated items provided (N=0).",
            confusion_matrix=matrix,
            marginals_a={c: 0 for c in categories},
            marginals_b={c: 0 for c in categories},
        )

    # Populate contingency matrix
    agreements = 0
    for a, b in zip(annotations_a, annotations_b):
        a_status = parse_ground_truth_status(a)
        b_status = parse_ground_truth_status(b)
        matrix[a_status.value][b_status.value] += 1
        if a_status == b_status:
            agreements += 1

    observed_agreement = agreements / n

    # Compute marginals
    marginals_a = {
        cat: sum(matrix[cat][b_cat] for b_cat in categories)
        for cat in categories
    }
    marginals_b = {
        cat: sum(matrix[a_cat][cat] for a_cat in categories)
        for cat in categories
    }

    # Expected chance agreement Pe = sum_k (row_k / N) * (col_k / N)
    chance_agreement = sum(
        (marginals_a[cat] / n) * (marginals_b[cat] / n)
        for cat in categories
    )

    # Kappa = (Po - Pe) / (1 - Pe)
    denominator = 1.0 - chance_agreement

    if abs(denominator) <= tolerance:
        # Edge case: Pe == 1.0 (both annotators assigned 100% of claims to identical single category)
        return CohenKappaResult(
            joint_count=n,
            agreement_count=agreements,
            observed_agreement=observed_agreement,
            chance_agreement=chance_agreement,
            cohens_kappa=None,
            kappa_defined=False,
            status_message="Kappa is undefined because chance agreement Pe == 1.0 (zero denominator).",
            confusion_matrix=matrix,
            marginals_a=marginals_a,
            marginals_b=marginals_b,
        )

    kappa = (observed_agreement - chance_agreement) / denominator

    return CohenKappaResult(
        joint_count=n,
        agreement_count=agreements,
        observed_agreement=observed_agreement,
        chance_agreement=chance_agreement,
        cohens_kappa=kappa,
        kappa_defined=True,
        status_message="Success",
        confusion_matrix=matrix,
        marginals_a=marginals_a,
        marginals_b=marginals_b,
    )


def merge_and_adjudicate_annotations(
    claim_record: M7ClaimRecord,
    annotation_a: Optional[AnnotationRecord] = None,
    annotation_b: Optional[AnnotationRecord] = None,
    adjudication: Optional[AdjudicationRecord] = None,
) -> FinalGroundTruthRecord:
    """
    Merge dual annotations and optional adjudication into a FinalGroundTruthRecord.

    Strict rules:
    - If A and B are present and A == B:
        has_disagreement = False
        final_ground_truth = A.status
    - If A and B are present and A != B:
        has_disagreement = True
        If adjudication is present:
            final_ground_truth = adjudication.adjudicated_status
        Else:
            final_ground_truth = None (requires adjudication)
    - If only A is present:
        has_disagreement = False
        final_ground_truth = None (requires dual annotation)
    - If only B is present:
        has_disagreement = False
        final_ground_truth = None (requires dual annotation)
    - If neither is present:
        has_disagreement = False
        final_ground_truth = None (unannotated)

    NEVER automatically selects Annotator A or Annotator B on disagreement.
    """
    has_disagreement = False
    final_gt: Optional[GroundTruthStatus] = None

    if annotation_a and annotation_b:
        if annotation_a.ground_truth_status == annotation_b.ground_truth_status:
            has_disagreement = False
            final_gt = annotation_a.ground_truth_status
        else:
            has_disagreement = True
            if adjudication:
                final_gt = adjudication.adjudicated_status
            else:
                final_gt = None
    elif adjudication:
        final_gt = adjudication.adjudicated_status

    return FinalGroundTruthRecord(
        claim_id=claim_record.claim_id,
        image_id=claim_record.image_id,
        object_category=claim_record.object_category,
        text_span=claim_record.text_span,
        split=claim_record.split,
        claim_record=claim_record,
        annotation_a=annotation_a,
        annotation_b=annotation_b,
        has_disagreement=has_disagreement,
        adjudication=adjudication,
        final_ground_truth=final_gt,
        is_synthetic=claim_record.is_synthetic,
    )
