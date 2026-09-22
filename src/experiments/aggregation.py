"""
Statistical aggregation, image-level cluster bootstrap, and case study selection.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.data.schemas import GroundTruthStatus, DecisionStatus
from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics
from src.experiments.interval_metrics import compute_interval_statistics, RobustIntervalStatistics


@dataclass
class BootstrapConfidenceInterval:
    metric_name: str
    point_estimate: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    confidence_level: float = 0.95
    n_resamples: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BootstrapComparisonResult:
    method_a: str
    method_b: str
    metric_name: str
    estimate_a: Optional[float]
    estimate_b: Optional[float]
    diff_estimate: Optional[float]  # a - b
    ci_diff_lower: Optional[float]
    ci_diff_upper: Optional[float]
    p_value_two_sided: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_image_cluster_bootstrap(
    results: List[ClaimEvaluationResult],
    method_name: str = "standard_bp",
    n_resamples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, BootstrapConfidenceInterval]:
    """
    Non-parametric cluster bootstrap resampling at the IMAGE level.
    Resamples image IDs with replacement to respect intra-image claim correlations.
    """
    if not results or len(results) < 3:
        return {}

    rng = np.random.default_rng(seed)

    # Group results by image_id
    results_by_img: Dict[str, List[ClaimEvaluationResult]] = {}
    for r in results:
        if r.image_id not in results_by_img:
            results_by_img[r.image_id] = []
        results_by_img[r.image_id].append(r)

    unique_images = list(results_by_img.keys())
    n_images = len(unique_images)
    if n_images == 0:
        return {}

    # Point estimates on full dataset
    point_metrics = evaluate_claim_results(results, method_name=method_name)
    point_interval = compute_interval_statistics(results)

    boot_metrics: Dict[str, List[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
        "mean_width": [],
        "evidence_sensitive_fraction": [],
    }

    for b in range(n_resamples):
        sampled_imgs = rng.choice(unique_images, size=n_images, replace=True)
        boot_sample: List[ClaimEvaluationResult] = []
        for img in sampled_imgs:
            boot_sample.extend(results_by_img[img])

        m = evaluate_claim_results(boot_sample, method_name=method_name)
        inv = compute_interval_statistics(boot_sample)

        if m.accuracy is not None:
            boot_metrics["accuracy"].append(m.accuracy)
        if m.precision is not None:
            boot_metrics["precision"].append(m.precision)
        if m.recall is not None:
            boot_metrics["recall"].append(m.recall)
        if m.f1 is not None:
            boot_metrics["f1"].append(m.f1)
        if m.roc_auc is not None:
            boot_metrics["roc_auc"].append(m.roc_auc)

        boot_metrics["mean_width"].append(inv.mean_width)
        boot_metrics["evidence_sensitive_fraction"].append(inv.evidence_sensitive_fraction)

    alpha_tail = (1.0 - confidence_level) / 2.0 * 100.0
    out: Dict[str, BootstrapConfidenceInterval] = {}

    point_map = {
        "accuracy": point_metrics.accuracy,
        "precision": point_metrics.precision,
        "recall": point_metrics.recall,
        "f1": point_metrics.f1,
        "roc_auc": point_metrics.roc_auc,
        "mean_width": point_interval.mean_width,
        "evidence_sensitive_fraction": point_interval.evidence_sensitive_fraction,
    }

    for k, vals in boot_metrics.items():
        if len(vals) >= 10:
            low = float(np.percentile(vals, alpha_tail))
            high = float(np.percentile(vals, 100.0 - alpha_tail))
            out[k] = BootstrapConfidenceInterval(
                metric_name=k,
                point_estimate=point_map.get(k),
                ci_lower=low,
                ci_upper=high,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
            )
        else:
            out[k] = BootstrapConfidenceInterval(
                metric_name=k,
                point_estimate=point_map.get(k),
                ci_lower=None,
                ci_upper=None,
                confidence_level=confidence_level,
                n_resamples=n_resamples,
            )

    return out


def run_method_comparison_bootstrap(
    results: List[ClaimEvaluationResult],
    method_a: str = "robust_bp",
    method_b: str = "standard_bp",
    n_resamples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, BootstrapComparisonResult]:
    """
    Compute paired image-level cluster bootstrap difference between two methods.
    """
    if not results or len(results) < 3:
        return {}

    boot_a = run_image_cluster_bootstrap(results, method_name=method_a, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed)
    boot_b = run_image_cluster_bootstrap(results, method_name=method_b, n_resamples=n_resamples, confidence_level=confidence_level, seed=seed)

    out: Dict[str, BootstrapComparisonResult] = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        ci_a = boot_a.get(metric)
        ci_b = boot_b.get(metric)
        est_a = ci_a.point_estimate if ci_a else None
        est_b = ci_b.point_estimate if ci_b else None
        diff = (est_a - est_b) if (est_a is not None and est_b is not None) else None
        
        low_diff = (ci_a.ci_lower - ci_b.ci_upper) if (ci_a and ci_b and ci_a.ci_lower is not None and ci_b.ci_upper is not None) else None
        high_diff = (ci_a.ci_upper - ci_b.ci_lower) if (ci_a and ci_b and ci_a.ci_upper is not None and ci_b.ci_lower is not None) else None

        out[metric] = BootstrapComparisonResult(
            method_a=method_a,
            method_b=method_b,
            metric_name=metric,
            estimate_a=est_a,
            estimate_b=est_b,
            diff_estimate=diff,
            ci_diff_lower=low_diff,
            ci_diff_upper=high_diff,
            p_value_two_sided=None,
        )

    return out


@dataclass
class CaseStudyExample:
    category_tag: str  # "high_conf_correct", "high_conf_wrong", "narrow_interval", "wide_interval", "threshold_crossing"
    image_id: str
    claim_id: str
    object_category: str
    ground_truth: Optional[str]
    detector_score: Optional[float]
    clip_score: Optional[float]
    standard_posterior: float
    standard_prediction: str
    robust_lower: float
    robust_upper: float
    interval_width: float
    robust_prediction: str
    evidence_sensitive: bool
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_case_studies(
    results: List[ClaimEvaluationResult],
    tau: float = 0.5,
) -> List[CaseStudyExample]:
    """
    Select representative case studies using deterministic, transparent rules.
    """
    if not results:
        return []

    examples: List[CaseStudyExample] = []

    # Filter annotated results
    annotated = [r for r in results if r.ground_truth in [GroundTruthStatus.SUPPORTED.value, GroundTruthStatus.HALLUCINATED.value]]

    # A. High-confidence correct
    for r in sorted(annotated, key=lambda x: abs(x.standard_posterior - (1.0 if x.ground_truth == "hallucinated" else 0.0))):
        pred_halluc = (r.standard_posterior >= tau)
        gt_halluc = (r.ground_truth == "hallucinated")
        if pred_halluc == gt_halluc:
            examples.append(CaseStudyExample(
                category_tag="high_confidence_correct",
                image_id=r.image_id,
                claim_id=r.claim_id,
                object_category=r.object_category,
                ground_truth=r.ground_truth,
                detector_score=r.detector_score,
                clip_score=r.clip_score,
                standard_posterior=r.standard_posterior,
                standard_prediction=r.standard_prediction,
                robust_lower=r.robust_lower,
                robust_upper=r.robust_upper,
                interval_width=r.interval_width,
                robust_prediction=r.robust_prediction,
                evidence_sensitive=r.evidence_sensitive,
                description="High agreement between detector evidence and point posterior with correct prediction.",
            ))
            break

    # B. High-confidence wrong (standard BP error)
    for r in sorted(annotated, key=lambda x: abs(x.standard_posterior - (0.0 if x.ground_truth == "hallucinated" else 1.0))):
        pred_halluc = (r.standard_posterior >= tau)
        gt_halluc = (r.ground_truth == "hallucinated")
        if pred_halluc != gt_halluc:
            examples.append(CaseStudyExample(
                category_tag="high_confidence_wrong",
                image_id=r.image_id,
                claim_id=r.claim_id,
                object_category=r.object_category,
                ground_truth=r.ground_truth,
                detector_score=r.detector_score,
                clip_score=r.clip_score,
                standard_posterior=r.standard_posterior,
                standard_prediction=r.standard_prediction,
                robust_lower=r.robust_lower,
                robust_upper=r.robust_upper,
                interval_width=r.interval_width,
                robust_prediction=r.robust_prediction,
                evidence_sensitive=r.evidence_sensitive,
                description="Point posterior was overconfident but standard BP prediction was incorrect.",
            ))
            break

    # C. Narrow robust interval
    sorted_by_width = sorted(results, key=lambda x: x.interval_width)
    if sorted_by_width:
        narrowest = sorted_by_width[0]
        examples.append(CaseStudyExample(
            category_tag="narrow_interval",
            image_id=narrowest.image_id,
            claim_id=narrowest.claim_id,
            object_category=narrowest.object_category,
            ground_truth=narrowest.ground_truth,
            detector_score=narrowest.detector_score,
            clip_score=narrowest.clip_score,
            standard_posterior=narrowest.standard_posterior,
            standard_prediction=narrowest.standard_prediction,
            robust_lower=narrowest.robust_lower,
            robust_upper=narrowest.robust_upper,
            interval_width=narrowest.interval_width,
            robust_prediction=narrowest.robust_prediction,
            evidence_sensitive=narrowest.evidence_sensitive,
            description="Strong unambiguous evidence produces narrow robust uncertainty interval.",
        ))

    # D. Wide robust interval
    if sorted_by_width:
        widest = sorted_by_width[-1]
        examples.append(CaseStudyExample(
            category_tag="wide_interval",
            image_id=widest.image_id,
            claim_id=widest.claim_id,
            object_category=widest.object_category,
            ground_truth=widest.ground_truth,
            detector_score=widest.detector_score,
            clip_score=widest.clip_score,
            standard_posterior=widest.standard_posterior,
            standard_prediction=widest.standard_prediction,
            robust_lower=widest.robust_lower,
            robust_upper=widest.robust_upper,
            interval_width=widest.interval_width,
            robust_prediction=widest.robust_prediction,
            evidence_sensitive=widest.evidence_sensitive,
            description="Ambiguous or conflicting multimodal evidence produces wide robust interval.",
        ))

    # E. Threshold-crossing interval
    crossers = [r for r in results if r.evidence_sensitive]
    if crossers:
        cross = crossers[0]
        examples.append(CaseStudyExample(
            category_tag="threshold_crossing_abstention",
            image_id=cross.image_id,
            claim_id=cross.claim_id,
            object_category=cross.object_category,
            ground_truth=cross.ground_truth,
            detector_score=cross.detector_score,
            clip_score=cross.clip_score,
            standard_posterior=cross.standard_posterior,
            standard_prediction=cross.standard_prediction,
            robust_lower=cross.robust_lower,
            robust_upper=cross.robust_upper,
            interval_width=cross.interval_width,
            robust_prediction=cross.robust_prediction,
            evidence_sensitive=cross.evidence_sensitive,
            description="Robust interval brackets decision threshold, triggering robust abstention / human review.",
        ))

    return examples
