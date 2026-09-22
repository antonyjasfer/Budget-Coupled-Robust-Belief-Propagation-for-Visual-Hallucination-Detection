"""
Statistical comparisons, hypothesis testing, and error categorization for Milestone 9.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
from sklearn import metrics

from src.experiments.inference_runner import ClaimEvaluationResult
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics

logger = logging.getLogger("m9_statistics")


@dataclass
class PairedMetricDifference:
    """Paired bootstrap comparison result for a single metric."""
    metric_name: str
    observed_standard: Optional[float]
    observed_robust: Optional[float]
    observed_difference: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    confidence_level: float
    n_resamples: int
    is_significant: bool  # 0 not in CI

    @property
    def mean(self) -> Optional[float]:
        return self.observed_difference

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean"] = self.observed_difference
        return d


@dataclass
class CorrelationHypothesisResult:
    """Hypothesis test analyzing relationship between interval width and uncertainty signals."""
    signal_name: str
    sample_size: int
    pearson_r: Optional[float]
    pearson_p: Optional[float]
    spearman_rho: Optional[float]
    spearman_p: Optional[float]
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorCategoryRecord:
    """Categorized claim error or uncertainty case."""
    category_id: str
    category_name: str
    claim_id: str
    image_id: str
    object_category: str
    ground_truth: Optional[str]
    standard_posterior: float
    standard_prediction: str
    robust_lower: float
    robust_upper: float
    interval_width: float
    robust_prediction: str
    is_evidence_sensitive: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_m9_paired_bootstrap(
    results_std: List[ClaimEvaluationResult],
    results_rob: Optional[List[ClaimEvaluationResult]] = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    decision_threshold: float = 0.5,
    n_bootstrap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform paired image-level cluster bootstrap comparison between Robust BP and Standard BP.
    Computes difference: Delta = Metric(Robust BP) - Metric(Standard BP).
    """
    if n_bootstrap is not None:
        n_resamples = n_bootstrap
    if results_rob is None:
        results_rob = results_std

    rng = np.random.RandomState(seed)

    # Filter to annotated clean claims
    clean_std = [r for r in results_std if r.ground_truth in ["supported", "hallucinated"]]
    clean_rob = [r for r in results_rob if r.ground_truth in ["supported", "hallucinated"]]

    # Build claim lookup
    rob_map = {r.claim_id: r for r in clean_rob}
    paired_claims = [r for r in clean_std if r.claim_id in rob_map]

    if not paired_claims:
        return {}

    # Observed point estimates
    m_std = evaluate_claim_results(paired_claims, method_name="standard_bp", decision_threshold=decision_threshold)
    m_rob = evaluate_claim_results([rob_map[r.claim_id] for r in paired_claims], method_name="robust_bp", exclude_abstain=True, decision_threshold=decision_threshold)

    # Cluster by image
    unique_images = sorted(list({r.image_id for r in paired_claims}))
    img_to_claims: Dict[str, List[ClaimEvaluationResult]] = {img: [] for img in unique_images}
    for r in paired_claims:
        img_to_claims[r.image_id].append(r)

    n_images = len(unique_images)
    diff_samples: Dict[str, List[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    metric_pairs = [
        ("accuracy", m_std.accuracy, m_rob.accuracy),
        ("precision", m_std.precision, m_rob.precision),
        ("recall", m_std.recall, m_rob.recall),
        ("f1", m_std.f1, m_rob.f1),
        ("roc_auc", m_std.roc_auc, m_rob.roc_auc),
    ]

    for _ in range(n_resamples):
        resampled_images = rng.choice(unique_images, size=n_images, replace=True)
        resampled_std: List[ClaimEvaluationResult] = []
        resampled_rob: List[ClaimEvaluationResult] = []

        for img in resampled_images:
            std_c = img_to_claims[img]
            resampled_std.extend(std_c)
            resampled_rob.extend([rob_map[c.claim_id] for c in std_c])

        b_std = evaluate_claim_results(resampled_std, method_name="standard_bp", decision_threshold=decision_threshold)
        b_rob = evaluate_claim_results(resampled_rob, method_name="robust_bp", exclude_abstain=True, decision_threshold=decision_threshold)

        if b_std.accuracy is not None and b_rob.accuracy is not None:
            diff_samples["accuracy"].append(b_rob.accuracy - b_std.accuracy)
        if b_std.precision is not None and b_rob.precision is not None:
            diff_samples["precision"].append(b_rob.precision - b_std.precision)
        if b_std.recall is not None and b_rob.recall is not None:
            diff_samples["recall"].append(b_rob.recall - b_std.recall)
        if b_std.f1 is not None and b_rob.f1 is not None:
            diff_samples["f1"].append(b_rob.f1 - b_std.f1)
        if b_std.roc_auc is not None and b_rob.roc_auc is not None:
            diff_samples["roc_auc"].append(b_rob.roc_auc - b_std.roc_auc)

    alpha_low = ((1.0 - confidence_level) / 2.0) * 100
    alpha_high = (1.0 - (1.0 - confidence_level) / 2.0) * 100

    out: Dict[str, PairedMetricDifference] = {}
    for name, obs_s, obs_r in metric_pairs:
        samples = diff_samples[name]
        obs_diff = (obs_r - obs_s) if (obs_r is not None and obs_s is not None) else None

        if len(samples) >= 10:
            ci_l = float(np.percentile(samples, alpha_low))
            ci_u = float(np.percentile(samples, alpha_high))
            sig = bool((ci_l > 0) or (ci_u < 0))
        else:
            ci_l = None
            ci_u = None
            sig = False

        diff_item = PairedMetricDifference(
            metric_name=name,
            observed_standard=obs_s,
            observed_robust=obs_r,
            observed_difference=obs_diff,
            ci_lower=ci_l,
            ci_upper=ci_u,
            confidence_level=confidence_level,
            n_resamples=len(samples),
            is_significant=sig,
        )
        out[name] = diff_item.to_dict()
        out[f"diff_{name}"] = diff_item.to_dict()

    return out


def compute_correlation_hypotheses(results: List[ClaimEvaluationResult]) -> List[CorrelationHypothesisResult]:
    """
    Test Phase 6 hypotheses: Does interval width correlate with errors and evidence conflict?
    """
    clean_annotated = [r for r in results if r.ground_truth in ["supported", "hallucinated"] and r.corruption_severity == "clean"]
    all_clean = [r for r in results if r.corruption_severity == "clean"] or results
    all_corrupt = [r for r in results if r.corruption_type != "none" or r.corruption_severity == "clean"]

    hypotheses: List[CorrelationHypothesisResult] = []

    # 1. Width vs Standard BP Error
    if len(clean_annotated) >= 3:
        w = np.array([r.interval_width for r in clean_annotated])
        e = np.array([1.0 if ((r.standard_posterior >= 0.5) != (r.ground_truth == "hallucinated")) else 0.0 for r in clean_annotated])

        r_val, p_val = stats.pearsonr(w, e) if len(np.unique(e)) > 1 and len(np.unique(w)) > 1 else (None, None)
        rho_val, sp_val = stats.spearmanr(w, e) if len(np.unique(e)) > 1 and len(np.unique(w)) > 1 else (None, None)

        interp = "Positive correlation indicates wider intervals reflect higher probability of standard-BP classification error."
        hypotheses.append(
            CorrelationHypothesisResult(
                signal_name="width_vs_standard_bp_error",
                sample_size=len(clean_annotated),
                pearson_r=float(r_val) if r_val is not None and not np.isnan(r_val) else None,
                pearson_p=float(p_val) if p_val is not None and not np.isnan(p_val) else None,
                spearman_rho=float(rho_val) if rho_val is not None and not np.isnan(rho_val) else None,
                spearman_p=float(sp_val) if sp_val is not None and not np.isnan(sp_val) else None,
                interpretation=interp,
            )
        )

    # 2. Width vs Detector-CLIP Discrepancy
    claims_with_both = [r for r in all_clean if r.detector_score is not None and r.clip_score is not None]
    if len(claims_with_both) >= 3:
        w = np.array([r.interval_width for r in claims_with_both])
        # Rescale clip [-1, 1] to [0, 1] for difference
        clip_norm = np.array([float(np.clip((r.clip_score + 1.0) / 2.0, 0.0, 1.0)) for r in claims_with_both])
        det_norm = np.array([float(r.detector_score) for r in claims_with_both])
        disc = np.abs(clip_norm - det_norm)

        r_val, p_val = stats.pearsonr(w, disc) if len(np.unique(w)) > 1 and len(np.unique(disc)) > 1 else (None, None)
        rho_val, sp_val = stats.spearmanr(w, disc) if len(np.unique(w)) > 1 and len(np.unique(disc)) > 1 else (None, None)

        interp = "Tests whether visual-semantic evidence conflicts expand the robust perturbation envelope."
        hypotheses.append(
            CorrelationHypothesisResult(
                signal_name="width_vs_detector_clip_conflict",
                sample_size=len(claims_with_both),
                pearson_r=float(r_val) if r_val is not None and not np.isnan(r_val) else None,
                pearson_p=float(p_val) if p_val is not None and not np.isnan(p_val) else None,
                spearman_rho=float(rho_val) if rho_val is not None and not np.isnan(rho_val) else None,
                spearman_p=float(sp_val) if sp_val is not None and not np.isnan(sp_val) else None,
                interpretation=interp,
            )
        )

    # 3. Width vs Visual Corruption Severity
    sev_map = {"clean": 0, "light": 1, "medium": 2, "heavy": 3}
    valid_corrupt = [r for r in all_corrupt if r.corruption_severity in sev_map]
    if len(valid_corrupt) >= 4 and len({r.corruption_severity for r in valid_corrupt}) >= 2:
        w = np.array([r.interval_width for r in valid_corrupt])
        s_levels = np.array([sev_map[r.corruption_severity] for r in valid_corrupt])

        r_val, p_val = stats.pearsonr(w, s_levels) if len(np.unique(w)) > 1 else (None, None)
        rho_val, sp_val = stats.spearmanr(w, s_levels) if len(np.unique(w)) > 1 else (None, None)

        interp = "Tests primary hypothesis: visual degradation induces wider uncertainty intervals."
        hypotheses.append(
            CorrelationHypothesisResult(
                signal_name="width_vs_corruption_severity",
                sample_size=len(valid_corrupt),
                pearson_r=float(r_val) if r_val is not None and not np.isnan(r_val) else None,
                pearson_p=float(p_val) if p_val is not None and not np.isnan(p_val) else None,
                spearman_rho=float(rho_val) if rho_val is not None and not np.isnan(rho_val) else None,
                spearman_p=float(sp_val) if sp_val is not None and not np.isnan(sp_val) else None,
                interpretation=interp,
            )
        )

    return hypotheses


def categorize_errors(results: List[ClaimEvaluationResult], tau: float = 0.5) -> List[ErrorCategoryRecord]:
    """
    Deterministically categorize claims into the 8 structured error/uncertainty regimes.
    """
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed"]] or results
    records: List[ErrorCategoryRecord] = []

    for r in clean_res:
        gt = r.ground_truth
        p_std = r.standard_posterior
        l = r.robust_lower
        u = r.robust_upper
        w = r.interval_width
        is_err = (gt in ["supported", "hallucinated"]) and ((p_std >= tau) != (gt == "hallucinated"))

        # 1. False Hallucination
        if gt == "supported" and p_std >= tau:
            records.append(
                ErrorCategoryRecord(
                    category_id="ERR_01_FALSE_HALLUCINATION",
                    category_name="False Hallucination",
                    claim_id=r.claim_id,
                    image_id=r.image_id,
                    object_category=r.object_category,
                    ground_truth=gt,
                    standard_posterior=p_std,
                    standard_prediction=r.standard_prediction,
                    robust_lower=l,
                    robust_upper=u,
                    interval_width=w,
                    robust_prediction=r.robust_prediction,
                    is_evidence_sensitive=r.evidence_sensitive,
                    explanation="Standard BP misclassified true supported object as hallucinated.",
                )
            )

        # 2. Missed Hallucination
        elif gt == "hallucinated" and p_std < tau:
            records.append(
                ErrorCategoryRecord(
                    category_id="ERR_02_MISSED_HALLUCINATION",
                    category_name="Missed Hallucination",
                    claim_id=r.claim_id,
                    image_id=r.image_id,
                    object_category=r.object_category,
                    ground_truth=gt,
                    standard_posterior=p_std,
                    standard_prediction=r.standard_prediction,
                    robust_lower=l,
                    robust_upper=u,
                    interval_width=w,
                    robust_prediction=r.robust_prediction,
                    is_evidence_sensitive=r.evidence_sensitive,
                    explanation="Standard BP failed to detect actual hallucinated object.",
                )
            )

        # 3. Evidence Conflict
        if r.detector_score is not None and r.clip_score is not None:
            clip_rescaled = (r.clip_score + 1.0) / 2.0
            if abs(r.detector_score - clip_rescaled) > 0.4:
                records.append(
                    ErrorCategoryRecord(
                        category_id="ERR_03_EVIDENCE_CONFLICT",
                        category_name="Evidence Conflict",
                        claim_id=r.claim_id,
                        image_id=r.image_id,
                        object_category=r.object_category,
                        ground_truth=gt,
                        standard_posterior=p_std,
                        standard_prediction=r.standard_prediction,
                        robust_lower=l,
                        robust_upper=u,
                        interval_width=w,
                        robust_prediction=r.robust_prediction,
                        is_evidence_sensitive=r.evidence_sensitive,
                        explanation=f"Strong disagreement between detector ({r.detector_score:.2f}) and CLIP ({r.clip_score:.2f}).",
                    )
                )

        # 4. Narrow Wrong
        if is_err and w < 0.25:
            records.append(
                ErrorCategoryRecord(
                    category_id="ERR_04_NARROW_WRONG",
                    category_name="Narrow but Incorrect",
                    claim_id=r.claim_id,
                    image_id=r.image_id,
                    object_category=r.object_category,
                    ground_truth=gt,
                    standard_posterior=p_std,
                    standard_prediction=r.standard_prediction,
                    robust_lower=l,
                    robust_upper=u,
                    interval_width=w,
                    robust_prediction=r.robust_prediction,
                    is_evidence_sensitive=r.evidence_sensitive,
                    explanation="Standard BP was wrong, but robust interval was narrow (overconfident failure).",
                )
            )

        # 5. Wide Evidence-Sensitive
        if r.evidence_sensitive:
            records.append(
                ErrorCategoryRecord(
                    category_id="ERR_05_WIDE_EVIDENCE_SENSITIVE",
                    category_name="Wide Evidence-Sensitive",
                    claim_id=r.claim_id,
                    image_id=r.image_id,
                    object_category=r.object_category,
                    ground_truth=gt,
                    standard_posterior=p_std,
                    standard_prediction=r.standard_prediction,
                    robust_lower=l,
                    robust_upper=u,
                    interval_width=w,
                    robust_prediction=r.robust_prediction,
                    is_evidence_sensitive=True,
                    explanation=f"Interval [{l:.3f}, {u:.3f}] straddles decision threshold tau={tau:.2f} (triggers abstain).",
                )
            )

    return records


def select_m9_case_studies(results: List[ClaimEvaluationResult], tau: float = 0.5) -> List[ErrorCategoryRecord]:
    """
    Select representative case studies (A through F) matching Phase 14 specifications.
    """
    clean_res = [r for r in results if r.condition in ["clean", "main_evaluation", "robust_bp_proposed"]] or results
    cases: List[ErrorCategoryRecord] = []

    # Case A: Standard BP correct + narrow robust interval
    case_a = [r for r in clean_res if r.ground_truth in ["supported", "hallucinated"] and ((r.standard_posterior >= tau) == (r.ground_truth == "hallucinated")) and r.interval_width < 0.35]
    if case_a:
        best_a = min(case_a, key=lambda x: x.interval_width)
        cases.append(
            ErrorCategoryRecord(
                category_id="CASE_A",
                category_name="High-Confidence Robustly Verified Prediction",
                claim_id=best_a.claim_id,
                image_id=best_a.image_id,
                object_category=best_a.object_category,
                ground_truth=best_a.ground_truth,
                standard_posterior=best_a.standard_posterior,
                standard_prediction=best_a.standard_prediction,
                robust_lower=best_a.robust_lower,
                robust_upper=best_a.robust_upper,
                interval_width=best_a.interval_width,
                robust_prediction=best_a.robust_prediction,
                is_evidence_sensitive=best_a.evidence_sensitive,
                explanation="Standard BP made correct classification and robust interval is narrow, certifying high confidence.",
            )
        )

    # Case B: Standard BP incorrect + wide robust interval
    case_b = [r for r in clean_res if r.ground_truth in ["supported", "hallucinated"] and ((r.standard_posterior >= tau) != (r.ground_truth == "hallucinated")) and r.interval_width > 0.35]
    if case_b:
        best_b = max(case_b, key=lambda x: x.interval_width)
        cases.append(
            ErrorCategoryRecord(
                category_id="CASE_B",
                category_name="BP Error Exposed by Robust Interval Width",
                claim_id=best_b.claim_id,
                image_id=best_b.image_id,
                object_category=best_b.object_category,
                ground_truth=best_b.ground_truth,
                standard_posterior=best_b.standard_posterior,
                standard_prediction=best_b.standard_prediction,
                robust_lower=best_b.robust_lower,
                robust_upper=best_b.robust_upper,
                interval_width=best_b.interval_width,
                robust_prediction=best_b.robust_prediction,
                is_evidence_sensitive=best_b.evidence_sensitive,
                explanation="Standard BP made an incorrect decision, but robust BP flagged high uncertainty (width > 0.35), rescuing the decision via abstention.",
            )
        )

    # Case C: Evidence-sensitive threshold crosser
    case_c = [r for r in clean_res if r.evidence_sensitive]
    if case_c:
        best_c = max(case_c, key=lambda x: x.interval_width)
        cases.append(
            ErrorCategoryRecord(
                category_id="CASE_C",
                category_name="Evidence-Sensitive Threshold Crossing Claim",
                claim_id=best_c.claim_id,
                image_id=best_c.image_id,
                object_category=best_c.object_category,
                ground_truth=best_c.ground_truth,
                standard_posterior=best_c.standard_posterior,
                standard_prediction=best_c.standard_prediction,
                robust_lower=best_c.robust_lower,
                robust_upper=best_c.robust_upper,
                interval_width=best_c.interval_width,
                robust_prediction=best_c.robust_prediction,
                is_evidence_sensitive=True,
                explanation=f"Interval [{best_c.robust_lower:.3f}, {best_c.robust_upper:.3f}] straddles threshold tau={tau:.2f}, demonstrating sensitivity to evidence perturbations.",
            )
        )

    # Case D: Corruption-induced uncertainty expansion
    corrupt_res = [r for r in results if r.corruption_severity == "heavy"]
    if corrupt_res:
        best_d = max(corrupt_res, key=lambda x: x.interval_width)
        cases.append(
            ErrorCategoryRecord(
                category_id="CASE_D",
                category_name="Visual Degradation Induced Uncertainty Expansion",
                claim_id=best_d.claim_id,
                image_id=best_d.image_id,
                object_category=best_d.object_category,
                ground_truth=best_d.ground_truth,
                standard_posterior=best_d.standard_posterior,
                standard_prediction=best_d.standard_prediction,
                robust_lower=best_d.robust_lower,
                robust_upper=best_d.robust_upper,
                interval_width=best_d.interval_width,
                robust_prediction=best_d.robust_prediction,
                is_evidence_sensitive=best_d.evidence_sensitive,
                explanation="Heavy visual corruption degraded evidence signals, widening the robust interval to reflect visual ambiguity.",
            )
        )

    return cases
