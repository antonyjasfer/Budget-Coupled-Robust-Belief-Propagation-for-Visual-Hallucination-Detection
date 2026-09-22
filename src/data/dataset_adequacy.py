"""
Dataset Adequacy, Graph Stratification, and Precision Planning.

Strictly enforces:
1. Subgroup stratification into G1 (1 claim), G2 (2 claims), G3 (3 claims), G4 (4+ claims).
2. Graph-evaluable subgroup analysis (G >= 2) without altering primary cohort membership.
3. Principled graph adequacy rating (SUFFICIENT, MARGINAL, INSUFFICIENT) with explicit scientific rationale.
4. Precision planning (binomial margin of error, CI width) without unsubstantiated power claims.
5. Post-hoc descriptive test event adequacy reporting (LOW EVENT COUNT / NOT EVALUABLE).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import numpy as np


class GraphAdequacyRating(str, Enum):
    """Scientific adequacy rating for graph-based inference."""
    SUFFICIENT = "sufficient"
    MARGINAL = "marginal"
    INSUFFICIENT = "insufficient"


@dataclass
class GraphStratificationCounts:
    """Detailed distribution of image claim counts and graph structure."""
    g1_images: int  # 1 claim (isolated unaries)
    g2_images: int  # 2 claims (single edge)
    g3_images: int  # 3 claims (small tree/triangle)
    g4_images: int  # 4+ claims (rich graph)
    total_images: int
    graph_evaluable_images: int  # G >= 2
    total_claims: int
    graph_evaluable_claims: int
    total_edges: int
    test_images_count: int
    test_multi_claim_images: int
    test_multi_claim_claims: int
    effective_image_clusters: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphAdequacyReport:
    """Scientific audit report for graph adequacy."""
    counts: GraphStratificationCounts
    rating: GraphAdequacyRating
    rationale: str
    downstream_analyses_evaluated: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "counts": self.counts.to_dict(),
            "rating": self.rating.value,
            "rationale": self.rationale,
            "downstream_analyses_evaluated": self.downstream_analyses_evaluated,
        }


def stratify_graph_adequacy(
    claims_by_image: Dict[str, List[Any]],
    split_assignments: Optional[Dict[str, str]] = None,
) -> GraphAdequacyReport:
    """
    Compute pre-label graph stratification within primary cohort.

    Stratification:
    G1: 1 claim (pure unary node)
    G2: 2 claims (1 edge in tree/chain)
    G3: 3 claims (2 edges in MST)
    G4: 4+ claims (3+ edges in MST)

    Evaluates downstream analyses:
    - Pairwise coupling parameterization (J > 0).
    - MST vs Independent topology comparison (M9C).
    - Image-level cluster bootstrap (M9D).
    """
    g1, g2, g3, g4 = 0, 0, 0, 0
    total_claims = 0
    graph_evaluable_claims = 0
    total_edges = 0

    test_imgs = 0
    test_multi_imgs = 0
    test_multi_claims = 0

    for img_id, claims in claims_by_image.items():
        k = len(claims)
        total_claims += k
        split = (split_assignments or {}).get(img_id, "").lower()

        if split == "test":
            test_imgs += 1
            if k >= 2:
                test_multi_imgs += 1
                test_multi_claims += k

        if k == 1:
            g1 += 1
        elif k == 2:
            g2 += 1
            graph_evaluable_claims += k
            total_edges += 1
        elif k == 3:
            g3 += 1
            graph_evaluable_claims += k
            total_edges += 2
        elif k >= 4:
            g4 += 1
            graph_evaluable_claims += k
            total_edges += (k - 1)  # spanning tree edges

    total_images = len(claims_by_image)
    graph_evaluable_images = g2 + g3 + g4
    effective_clusters = total_images

    counts = GraphStratificationCounts(
        g1_images=g1,
        g2_images=g2,
        g3_images=g3,
        g4_images=g4,
        total_images=total_images,
        graph_evaluable_images=graph_evaluable_images,
        total_claims=total_claims,
        graph_evaluable_claims=graph_evaluable_claims,
        total_edges=total_edges,
        test_images_count=test_imgs,
        test_multi_claim_images=test_multi_imgs,
        test_multi_claim_claims=test_multi_claims,
        effective_image_clusters=effective_clusters,
    )

    downstream = [
        "Pairwise coupling (J) estimation",
        "MST vs Independent topology ablation",
        "Local-vs-global interval width shrinkage",
        "Image-level cluster bootstrap error estimation",
    ]

    # Principled scientific evaluation referencing analyses:
    # Requires sufficient multi-claim images in the test set to evaluate topology differences
    if test_imgs > 0:
        test_multi_ratio = test_multi_imgs / test_imgs
    else:
        test_multi_ratio = graph_evaluable_images / max(1, total_images)

    if graph_evaluable_images >= 150 and (test_imgs == 0 or test_multi_imgs >= 30):
        rating = GraphAdequacyRating.SUFFICIENT
        rationale = (
            f"Cohort contains {graph_evaluable_images} multi-claim images ({graph_evaluable_claims} claims, "
            f"{total_edges} spanning edges) with {test_multi_imgs} in Test split ({test_multi_ratio:.1%}). "
            "Sufficient to estimate pairwise coupling J and detect topology differences between MST and Independent baseline."
        )
    elif graph_evaluable_images >= 60 and (test_imgs == 0 or test_multi_imgs >= 15):
        rating = GraphAdequacyRating.MARGINAL
        rationale = (
            f"Cohort contains {graph_evaluable_images} multi-claim images with {test_multi_imgs} in Test split. "
            "Pairwise coupling effects can be fitted, but power to distinguish MST from chain topology is limited."
        )
    else:
        rating = GraphAdequacyRating.INSUFFICIENT
        rationale = (
            f"Cohort contains only {graph_evaluable_images} multi-claim images ({test_multi_imgs} in Test split). "
            "Insufficient graph structure to perform reliable topology comparison; graph-stress supplement recommended."
        )

    return GraphAdequacyReport(
        counts=counts,
        rating=rating,
        rationale=rationale,
        downstream_analyses_evaluated=downstream,
    )


@dataclass
class PrecisionPlanningResult:
    """Binomial proportion precision estimate without unverified power claims."""
    target_proportion: float
    sample_size: int
    confidence_level: float
    z_score: float
    margin_of_error: float
    ci_lower: float
    ci_upper: float
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_binomial_precision(
    sample_size: int,
    target_proportion: float = 0.20,
    confidence_level: float = 0.95,
) -> PrecisionPlanningResult:
    """
    Compute estimation precision (Margin of Error) for a binomial rate.

    Formula:
        MoE = z_{1-alpha/2} * sqrt(p * (1 - p) / n)

    Does NOT claim statistical power because no alternative effect size or
    variance model is hypothesized.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    if not (0.0 < target_proportion < 1.0):
        raise ValueError("target_proportion must be between 0 and 1.")

    # Normal critical value approximation for standard confidence levels
    z_table = {0.90: 1.6449, 0.95: 1.95996, 0.99: 2.5758}
    z_score = z_table.get(confidence_level, 1.95996)

    variance = target_proportion * (1.0 - target_proportion)
    se = math.sqrt(variance / sample_size)
    moe = z_score * se

    ci_lower = max(0.0, target_proportion - moe)
    ci_upper = min(1.0, target_proportion + moe)

    interp = (
        f"With n={sample_size} and expected event proportion p={target_proportion:.2f}, "
        f"the {int(confidence_level*100)}% confidence interval half-width is +/-{moe:.4f} "
        f"([{ci_lower:.4f}, {ci_upper:.4f}]). Precision planning only; not a power claim."
    )

    return PrecisionPlanningResult(
        target_proportion=target_proportion,
        sample_size=sample_size,
        confidence_level=confidence_level,
        z_score=z_score,
        margin_of_error=moe,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        interpretation=interp,
    )


@dataclass
class TestEventAdequacyReport:
    """Post-hoc descriptive test event counts."""
    total_test_claims: int
    observed_hallucinations: int
    observed_supported: int
    observed_unknown: int
    disagreements_count: int
    is_evaluable: bool
    status: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_test_event_adequacy(
    ground_truth_labels: List[str],
    min_event_count: int = 10,
    disagreements_count: int = 0,
) -> TestEventAdequacyReport:
    """
    Descriptive post-hoc audit of test split events.

    STRICT METHODOLOGICAL RULE:
    If observed event counts are low, reports LOW EVENT COUNT / NOT EVALUABLE.
    Under no circumstances are model thresholds, splits, or parameters altered
    in response to test event counts.
    """
    n_hallucinated = 0
    n_supported = 0
    n_unknown = 0

    for lbl in ground_truth_labels:
        clean = lbl.strip().lower()
        if clean == "hallucinated":
            n_hallucinated += 1
        elif clean == "supported":
            n_supported += 1
        elif clean == "unknown":
            n_unknown += 1

    total = len(ground_truth_labels)
    # Check if there are sufficient minority class events for metrics like ROC-AUC
    evaluable = (n_hallucinated >= min_event_count) and (n_supported >= min_event_count)

    if evaluable:
        status = "EVALUABLE"
        rationale = (
            f"Test split contains {n_hallucinated} hallucinations and {n_supported} supported claims. "
            f"Both exceed minimum event threshold ({min_event_count}). Evaluability confirmed."
        )
    else:
        status = "LOW EVENT COUNT / NOT EVALUABLE"
        rationale = (
            f"Test split contains {n_hallucinated} hallucinations and {n_supported} supported claims. "
            f"One or both classes fall below minimum event threshold ({min_event_count}). "
            "Reported as low event count; parameters and splits must remain frozen."
        )

    return TestEventAdequacyReport(
        total_test_claims=total,
        observed_hallucinations=n_hallucinated,
        observed_supported=n_supported,
        observed_unknown=n_unknown,
        disagreements_count=disagreements_count,
        is_evaluable=evaluable,
        status=status,
        rationale=rationale,
    )
