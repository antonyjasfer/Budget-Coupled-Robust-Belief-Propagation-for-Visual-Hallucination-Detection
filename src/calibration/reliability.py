"""Reliability validation, selective prediction, and robust interval utility.

Provides statistical evaluation to determine whether robust posterior intervals [L_i, U_i]
and width W_i = U_i - L_i provide useful reliability information beyond nominal uncertainty
(margin uncertainty, binary entropy, threshold distance, calibrated logistic fusion).

All outputs on development subsets are strictly flagged as DEVELOPMENT ONLY.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


# ============================================================================
# Phase 9D-11: Robust Decision Categories and Enums
# ============================================================================

class RobustDecisionCategory(str, Enum):
    """Model output decision categories based on robust posterior interval [L, U] vs tau."""
    ROBUST_SUPPORTED = "ROBUST_SUPPORTED"      # U < tau
    ROBUST_HALLUCINATED = "ROBUST_HALLUCINATED"  # L > tau
    EVIDENCE_SENSITIVE = "EVIDENCE_SENSITIVE"    # L <= tau <= U


# ============================================================================
# Phase 9D-27: Complete Claim Result Schema (33 Fields)
# ============================================================================

@dataclass
class ReliabilityClaimRecord:
    """Standardized schema containing all 33 claim-level fields for uncertainty evaluation."""

    image_id: str
    claim_id: str
    ground_truth: Optional[int] = None  # 0 = SUPPORTED, 1 = HALLUCINATED, None = UNKNOWN/unlabeled
    annotator_A: Optional[int] = None
    annotator_B: Optional[int] = None
    annotator_disagreement: Optional[bool] = None
    nominal_posterior: float = 0.5
    nominal_prediction: int = 0
    nominal_correct: Optional[bool] = None
    nominal_entropy: float = 1.0
    nominal_margin_uncertainty: float = 1.0
    logistic_uncertainty: Optional[float] = None
    detector_uncertainty: Optional[float] = None
    threshold_uncertainty: float = 0.0
    local_lower: Optional[float] = None
    local_upper: Optional[float] = None
    local_width: Optional[float] = None
    global_lower: Optional[float] = None
    global_upper: Optional[float] = None
    global_width: Optional[float] = None
    local_threshold_crossing: bool = False   # strict: L < tau < U
    global_threshold_crossing: bool = False  # strict: L < tau < U
    local_contains_threshold: bool = False   # inclusive: L <= tau <= U
    global_contains_threshold: bool = False  # inclusive: L <= tau <= U
    robust_margin: float = 0.0               # min(|L-tau|, |U-tau|) when not containing tau
    robust_decision: str = "EVIDENCE_SENSITIVE"
    abstained: bool = False
    corruption_type: str = "none"
    corruption_severity: str = "clean"
    budget: float = 1.0
    budget_ratio: float = 1.0
    epsilon: float = 0.2
    graph_size: int = 1
    topology: str = "mst"
    coupling_strategy: str = "JWEIGHTED"
    split: str = "test"
    run_id: str = "run_0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Phase 9D-1: Nominal Uncertainty Baselines
# ============================================================================

def compute_margin_uncertainty(p: float) -> float:
    """U1: Margin uncertainty: 1 - |2p - 1|. Range [0, 1], higher = more uncertain."""
    p_clamped = float(np.clip(p, 0.0, 1.0))
    return float(1.0 - abs(2.0 * p_clamped - 1.0))


def compute_binary_entropy(p: float, eps: float = 1e-12) -> float:
    """U2: Binary entropy in bits: -p log2 p - (1-p) log2(1-p). Higher = more uncertain."""
    p_clamped = float(np.clip(p, eps, 1.0 - eps))
    return float(-p_clamped * math.log2(p_clamped) - (1.0 - p_clamped) * math.log2(1.0 - p_clamped))


def compute_threshold_distance_uncertainty(p: float, tau: float = 0.5) -> float:
    """U3: Distance to decision threshold: -|p - tau|. Higher (less negative) = closer = more uncertain."""
    p_clamped = float(np.clip(p, 0.0, 1.0))
    return float(-abs(p_clamped - tau))


def compute_normalized_threshold_uncertainty(p: float, tau: float = 0.5) -> float:
    """Normalized threshold uncertainty in [0, 1], where 1.0 is at tau and 0.0 is at extreme."""
    p_clamped = float(np.clip(p, 0.0, 1.0))
    max_dist = max(tau, 1.0 - tau)
    if max_dist <= 0:
        return 0.0
    return float(np.clip(1.0 - abs(p_clamped - tau) / max_dist, 0.0, 1.0))


def compute_detector_uncertainty(detector_prob: float) -> float:
    """U4: Margin uncertainty derived from calibrated detector-only probability."""
    return compute_margin_uncertainty(detector_prob)


def compute_logistic_uncertainty(logistic_prob: float) -> float:
    """U5: Margin uncertainty derived from calibrated logistic fusion probability."""
    return compute_margin_uncertainty(logistic_prob)


# ============================================================================
# Phase 9D-2 & 9D-11: Robust Uncertainty Scores & Classifications
# ============================================================================

def classify_robust_decision(
    lower: float,
    upper: float,
    tau: float = 0.5,
) -> RobustDecisionCategory:
    """Classify robust decision according to strict mathematical definitions:

    ROBUST_SUPPORTED: U < tau
    ROBUST_HALLUCINATED: L > tau
    EVIDENCE_SENSITIVE: L <= tau <= U
    """
    if upper < tau:
        return RobustDecisionCategory.ROBUST_SUPPORTED
    if lower > tau:
        return RobustDecisionCategory.ROBUST_HALLUCINATED
    return RobustDecisionCategory.EVIDENCE_SENSITIVE


def compute_robust_uncertainty_scores(
    lower: float,
    upper: float,
    tau: float = 0.5,
) -> Dict[str, Any]:
    """Compute all Phase 9D-2 robust uncertainty metrics."""
    l_c = float(np.clip(lower, 0.0, 1.0))
    u_c = float(np.clip(upper, l_c, 1.0))
    width = float(u_c - l_c)

    strict_cross = bool(l_c < tau < u_c)
    contains_tau = bool(l_c <= tau <= u_c)

    if contains_tau:
        robust_margin = 0.0
    else:
        robust_margin = float(min(abs(l_c - tau), abs(u_c - tau)))

    # Orientation: higher score = more uncertain (negative of margin)
    u_robust_margin = float(-robust_margin)

    decision = classify_robust_decision(l_c, u_c, tau=tau)

    return {
        "width": width,
        "strict_crossing": strict_cross,
        "contains_threshold": contains_tau,
        "robust_margin": robust_margin,
        "u_robust_margin": u_robust_margin,
        "robust_decision": decision.value,
    }


# ============================================================================
# Phase 9D-3: Error-Prediction Task
# ============================================================================

def compute_error_detection_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    uncertainty_scores: Sequence[float],
    min_samples: int = 4,
) -> Dict[str, Any]:
    """Evaluate an uncertainty score's ability to rank prediction errors e_i = I(y_pred != y_true).

    Returns 'NOT EVALUABLE' when errors are absent, all samples are errors, or sample size is insufficient.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    scores = np.asarray(uncertainty_scores, dtype=np.float64)

    n = len(y_t)
    if n < min_samples:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Sample size {n} < minimum required {min_samples}",
            "error_detection_auroc": None,
            "error_detection_auprc": None,
            "mean_uncertainty_correct": None,
            "mean_uncertainty_incorrect": None,
            "median_uncertainty_correct": None,
            "median_uncertainty_incorrect": None,
            "n_samples": n,
            "n_errors": 0,
        }

    errors = (y_t != y_p).astype(np.int64)
    n_errors = int(np.sum(errors))

    if n_errors == 0 or n_errors == n:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Single error class: {n_errors} errors out of {n} samples",
            "error_detection_auroc": None,
            "error_detection_auprc": None,
            "mean_uncertainty_correct": float(np.mean(scores)) if n_errors == 0 else None,
            "mean_uncertainty_incorrect": float(np.mean(scores)) if n_errors == n else None,
            "median_uncertainty_correct": float(np.median(scores)) if n_errors == 0 else None,
            "median_uncertainty_incorrect": float(np.median(scores)) if n_errors == n else None,
            "n_samples": n,
            "n_errors": n_errors,
        }

    # Valid binary error classification
    try:
        auroc = float(roc_auc_score(errors, scores))
    except Exception as e:
        auroc = None

    try:
        auprc = float(average_precision_score(errors, scores))
    except Exception as e:
        auprc = None

    scores_correct = scores[errors == 0]
    scores_incorrect = scores[errors == 1]

    return {
        "status": "EVALUATED",
        "reason": None,
        "error_detection_auroc": auroc,
        "error_detection_auprc": auprc,
        "mean_uncertainty_correct": float(np.mean(scores_correct)),
        "mean_uncertainty_incorrect": float(np.mean(scores_incorrect)),
        "median_uncertainty_correct": float(np.median(scores_correct)),
        "median_uncertainty_incorrect": float(np.median(scores_incorrect)),
        "n_samples": n,
        "n_errors": n_errors,
    }


# ============================================================================
# Phase 9D-4: Group-Aware Incremental Information Test
# ============================================================================

def compute_group_aware_incremental_utility(
    records: List[ReliabilityClaimRecord],
    nominal_key: str = "nominal_entropy",
    robust_key: str = "global_width",
    n_splits: int = 3,
    min_samples: int = 8,
    min_errors: int = 2,
) -> Dict[str, Any]:
    """Test whether robust width adds incremental predictive information beyond nominal uncertainty.

    STRICTLY ENFORCES:
    1. GroupKFold grouping by image_id (all claims of an image stay in the same fold).
    2. Zero claim-level fallback.
    3. If grouped CV folds cannot be constructed with both error classes in each training split,
       returns 'NOT EVALUABLE' with explicit justification.
    """
    valid_recs = [r for r in records if r.ground_truth is not None and r.nominal_correct is not None]
    n = len(valid_recs)

    if n < min_samples:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Sample size {n} < minimum required {min_samples}",
            "model_A_log_loss": None,
            "model_B_log_loss": None,
            "delta_log_loss": None,
        }

    y_err = np.array([0 if r.nominal_correct else 1 for r in valid_recs], dtype=np.int64)
    n_err = int(np.sum(y_err))
    if n_err < min_errors or n_err > (n - min_errors):
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Insufficient error minority class count: {n_err} errors in {n} claims (requires >= {min_errors})",
            "model_A_log_loss": None,
            "model_B_log_loss": None,
            "delta_log_loss": None,
        }

    groups = np.array([r.image_id for r in valid_recs])
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Number of image groups ({len(unique_groups)}) < CV splits ({n_splits})",
            "model_A_log_loss": None,
            "model_B_log_loss": None,
            "delta_log_loss": None,
        }

    X_nom = np.array([[getattr(r, nominal_key, 0.0)] for r in valid_recs], dtype=np.float64)
    X_both = np.array([[getattr(r, nominal_key, 0.0), getattr(r, robust_key, 0.0)] for r in valid_recs], dtype=np.float64)

    gkf = GroupKFold(n_splits=n_splits)

    oof_preds_A = np.zeros(n, dtype=np.float64)
    oof_preds_B = np.zeros(n, dtype=np.float64)
    valid_fold_count = 0

    for train_idx, val_idx in gkf.split(X_nom, y_err, groups=groups):
        y_train = y_err[train_idx]
        if len(np.unique(y_train)) < 2:
            return {
                "status": "NOT EVALUABLE",
                "reason": "Grouped CV split produced a training fold containing only a single error class",
                "model_A_log_loss": None,
                "model_B_log_loss": None,
                "delta_log_loss": None,
            }

        # Model A: error ~ nominal
        clf_A = LogisticRegression(C=1e9, solver="lbfgs")
        clf_A.fit(X_nom[train_idx], y_train)
        oof_preds_A[val_idx] = clf_A.predict_proba(X_nom[val_idx])[:, 1]

        # Model B: error ~ nominal + robust
        clf_B = LogisticRegression(C=1e9, solver="lbfgs")
        clf_B.fit(X_both[train_idx], y_train)
        oof_preds_B[val_idx] = clf_B.predict_proba(X_both[val_idx])[:, 1]

        valid_fold_count += 1

    loss_A = float(log_loss(y_err, oof_preds_A))
    loss_B = float(log_loss(y_err, oof_preds_B))
    brier_A = float(brier_score_loss(y_err, oof_preds_A))
    brier_B = float(brier_score_loss(y_err, oof_preds_B))
    auroc_A = float(roc_auc_score(y_err, oof_preds_A))
    auroc_B = float(roc_auc_score(y_err, oof_preds_B))

    return {
        "status": "EVALUATED",
        "reason": None,
        "n_samples": n,
        "n_images": len(unique_groups),
        "n_errors": n_err,
        "model_A_log_loss": loss_A,
        "model_B_log_loss": loss_B,
        "delta_log_loss": loss_B - loss_A,  # negative means model B improved
        "model_A_brier": brier_A,
        "model_B_brier": brier_B,
        "delta_brier": brier_B - brier_A,
        "model_A_auroc": auroc_A,
        "model_B_auroc": auroc_B,
        "delta_auroc": auroc_B - auroc_A,
    }


# ============================================================================
# Phase 9D-5: Conditional Width & Residual Analysis
# ============================================================================

def compute_conditional_width_analysis(
    records: List[ReliabilityClaimRecord],
    min_samples: int = 6,
) -> Dict[str, Any]:
    """Compute Pearson/Spearman correlations and residual robust sensitivity vs error."""
    valid_recs = [r for r in records if r.global_width is not None]
    n = len(valid_recs)
    if n < min_samples:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Sample size {n} < minimum required {min_samples}",
            "spearman_global_entropy": None,
            "pearson_global_entropy": None,
            "spearman_global_margin": None,
            "spearman_global_local": None,
            "residual_error_spearman": None,
        }

    w_g = np.array([r.global_width for r in valid_recs], dtype=np.float64)
    u_ent = np.array([r.nominal_entropy for r in valid_recs], dtype=np.float64)
    u_mar = np.array([r.nominal_margin_uncertainty for r in valid_recs], dtype=np.float64)
    w_loc = np.array([r.local_width if r.local_width is not None else r.global_width for r in valid_recs], dtype=np.float64)

    # Correlations
    sp_ent, _ = stats.spearmanr(w_g, u_ent)
    pe_ent, _ = stats.pearsonr(w_g, u_ent)
    sp_mar, _ = stats.spearmanr(w_g, u_mar)
    sp_loc, _ = stats.spearmanr(w_g, w_loc)

    # Residual regression: W_global ~ alpha + beta * nominal_entropy
    res_err_corr = None
    if len(np.unique(u_ent)) > 1:
        slope, intercept, _, _, _ = stats.linregress(u_ent, w_g)
        residuals = w_g - (intercept + slope * u_ent)

        errs = [0 if r.nominal_correct else 1 for r in valid_recs if r.nominal_correct is not None]
        if len(errs) == n and len(np.unique(errs)) > 1:
            res_sp, _ = stats.spearmanr(residuals, errs)
            res_err_corr = float(res_sp) if not np.isnan(res_sp) else 0.0

    return {
        "status": "EVALUATED",
        "reason": None,
        "n_samples": n,
        "spearman_global_entropy": float(sp_ent) if not np.isnan(sp_ent) else 0.0,
        "pearson_global_entropy": float(pe_ent) if not np.isnan(pe_ent) else 0.0,
        "spearman_global_margin": float(sp_mar) if not np.isnan(sp_mar) else 0.0,
        "spearman_global_local": float(sp_loc) if not np.isnan(sp_loc) else 0.0,
        "residual_error_spearman": res_err_corr,
    }


# ============================================================================
# Phase 9D-6, 9D-7, 9D-8, 9D-9: Selective Prediction & Discrete AURC
# ============================================================================

@dataclass
class RiskCoveragePoint:
    """Operating point on the discrete risk-coverage curve."""
    prefix_k: int
    coverage: float
    risk: float
    threshold_value: float
    selective_accuracy: float
    selective_f1: Optional[float] = None


def compute_discrete_risk_coverage_curve(
    records: List[ReliabilityClaimRecord],
    uncertainty_extractor: Callable[[ReliabilityClaimRecord], float],
    risk_targets: Sequence[float] = (0.05, 0.10, 0.20),
    coverage_targets: Sequence[float] = (0.50, 0.70, 0.80, 0.90, 1.00),
    min_samples: int = 4,
) -> Dict[str, Any]:
    """Compute discrete risk-coverage curve and discrete AURC / Excess-AURC.

    STRICT STANDARDS APPLIED:
    1. Sorts predictions from lowest to highest uncertainty (tie-broken by claim_id).
    2. Evaluates empirical risk r_k across discrete prefixes k = 1 ... N.
    3. Discrete AURC = (1/N) * sum_{k=1}^N r_k.
    4. Oracle AURC* sorts all correct claims first, then all error claims.
    5. Excess AURC = AURC - AURC* >= 0.
    6. Coverage@Risk<=r: maximum observed achievable coverage kappa_k where r_k <= r (NO interpolation).
    7. Risk@FixedCoverage: closest discrete prefix coverage >= target.
    """
    valid_recs = [r for r in records if r.ground_truth is not None and r.nominal_correct is not None]
    n = len(valid_recs)

    if n < min_samples:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Sample size {n} < minimum required {min_samples}",
            "points": [],
            "discrete_aurc": None,
            "trapezoidal_aurc": None,
            "oracle_aurc": None,
            "excess_aurc": None,
            "coverage_at_fixed_risk": {r: 0.0 for r in risk_targets},
            "risk_at_fixed_coverage": {c: None for c in coverage_targets},
        }

    # Deterministic tie-breaking: (score, claim_id)
    sorted_recs = sorted(
        valid_recs,
        key=lambda r: (float(uncertainty_extractor(r)), str(r.claim_id)),
    )

    y_true = np.array([r.ground_truth for r in sorted_recs], dtype=np.int64)
    y_pred = np.array([r.nominal_prediction for r in sorted_recs], dtype=np.int64)
    errors = (y_true != y_pred).astype(np.int64)
    scores = np.array([float(uncertainty_extractor(r)) for r in sorted_recs], dtype=np.float64)

    total_errors = int(np.sum(errors))
    total_correct = n - total_errors

    points: List[RiskCoveragePoint] = []
    risks: List[float] = []
    coverages: List[float] = []

    for k in range(1, n + 1):
        err_k = int(np.sum(errors[:k]))
        cov = float(k / n)
        risk = float(err_k / k)
        score_k = float(scores[k - 1])

        # Selective F1 if applicable
        tp = int(np.sum((y_pred[:k] == 1) & (y_true[:k] == 1)))
        fp = int(np.sum((y_pred[:k] == 1) & (y_true[:k] == 0)))
        fn = int(np.sum((y_pred[:k] == 0) & (y_true[:k] == 1)))
        f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        pt = RiskCoveragePoint(
            prefix_k=k,
            coverage=cov,
            risk=risk,
            threshold_value=score_k,
            selective_accuracy=1.0 - risk,
            selective_f1=float(f1),
        )
        points.append(pt)
        risks.append(risk)
        coverages.append(cov)

    # 1. Discrete AURC (Geifman & El-Yaniv standard formulation)
    discrete_aurc = float(np.mean(risks))

    # 2. Trapezoidal AURC (explicitly separated)
    trapezoidal_aurc = float(np.trapezoid(risks, coverages)) if hasattr(np, "trapezoid") else float(np.trapz(risks, coverages))

    # 3. Oracle AURC*
    # Perfect ranker: zero errors until all correct samples accepted, then error rate increases
    oracle_risks = []
    for k in range(1, n + 1):
        err_opt = max(0, k - total_correct)
        oracle_risks.append(float(err_opt / k))
    oracle_aurc = float(np.mean(oracle_risks))
    excess_aurc = float(max(0.0, discrete_aurc - oracle_aurc))

    # 4. Coverage@Risk <= r (maximum observed achievable coverage whose empirical risk <= r)
    cov_at_risk: Dict[float, float] = {}
    for r_target in risk_targets:
        achievable_covs = [pt.coverage for pt in points if pt.risk <= r_target]
        cov_at_risk[r_target] = float(max(achievable_covs)) if achievable_covs else 0.0

    # 5. Risk@FixedCoverage (closest discrete prefix coverage >= target)
    risk_at_cov: Dict[float, Optional[float]] = {}
    for c_target in coverage_targets:
        matching = [pt.risk for pt in points if pt.coverage >= (c_target - 1e-9)]
        risk_at_cov[c_target] = float(matching[0]) if matching else float(risks[-1])

    return {
        "status": "EVALUATED",
        "reason": None,
        "n_samples": n,
        "total_errors": total_errors,
        "points": points,
        "discrete_aurc": discrete_aurc,
        "trapezoidal_aurc": trapezoidal_aurc,
        "oracle_aurc": oracle_aurc,
        "excess_aurc": excess_aurc,
        "coverage_at_fixed_risk": cov_at_risk,
        "risk_at_fixed_coverage": risk_at_cov,
    }


# ============================================================================
# Phase 9D-10: Threshold-Crossing Analysis
# ============================================================================

def evaluate_threshold_crossing_stability(
    records: List[ReliabilityClaimRecord],
    tau: float = 0.5,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Analyze whether threshold-crossing intervals identify unstable decisions.

    Compares crossers (contains_threshold: L <= tau <= U) vs non-crossers.
    Includes image-level bootstrap CI for difference in error rate.
    """
    valid_recs = [r for r in records if r.ground_truth is not None and r.nominal_correct is not None]
    n = len(valid_recs)
    if n == 0:
        return {
            "status": "NOT EVALUABLE",
            "reason": "No labeled records available",
            "n_crossers": 0,
            "n_non_crossers": 0,
        }

    crossers = [r for r in valid_recs if r.global_contains_threshold]
    non_crossers = [r for r in valid_recs if not r.global_contains_threshold]

    n_cr = len(crossers)
    n_nc = len(non_crossers)

    err_cr = float(np.mean([0 if r.nominal_correct else 1 for r in crossers])) if n_cr > 0 else 0.0
    err_nc = float(np.mean([0 if r.nominal_correct else 1 for r in non_crossers])) if n_nc > 0 else 0.0

    relative_risk = (err_cr / err_nc) if err_nc > 0 else (float("inf") if err_cr > 0 else 1.0)
    risk_difference = err_cr - err_nc

    # Image-level bootstrap CI on difference
    img_groups: Dict[str, List[ReliabilityClaimRecord]] = {}
    for r in valid_recs:
        img_groups.setdefault(r.image_id, []).append(r)
    unique_imgs = sorted(list(img_groups.keys()))

    diffs = []
    if len(unique_imgs) >= 3 and n_cr > 0 and n_nc > 0:
        rng = np.random.RandomState(seed)
        for _ in range(n_bootstraps):
            samp_imgs = rng.choice(unique_imgs, size=len(unique_imgs), replace=True)
            samp_recs = []
            for img in samp_imgs:
                samp_recs.extend(img_groups[img])
            b_cr = [r for r in samp_recs if r.global_contains_threshold]
            b_nc = [r for r in samp_recs if not r.global_contains_threshold]
            if b_cr and b_nc:
                b_err_cr = np.mean([0 if r.nominal_correct else 1 for r in b_cr])
                b_err_nc = np.mean([0 if r.nominal_correct else 1 for r in b_nc])
                diffs.append(float(b_err_cr - b_err_nc))

    ci_low = float(np.percentile(diffs, 2.5)) if len(diffs) >= 10 else None
    ci_high = float(np.percentile(diffs, 97.5)) if len(diffs) >= 10 else None

    return {
        "status": "EVALUATED",
        "n_claims": n,
        "n_crossers": n_cr,
        "n_non_crossers": n_nc,
        "crosser_fraction": float(n_cr / n) if n > 0 else 0.0,
        "error_rate_crossers": err_cr,
        "error_rate_non_crossers": err_nc,
        "relative_risk": relative_risk,
        "risk_difference": risk_difference,
        "risk_diff_bootstrap_ci": (ci_low, ci_high) if ci_low is not None else None,
    }


# ============================================================================
# Phase 9D-12: Robust Abstention Policy
# ============================================================================

def evaluate_robust_abstention_policy(
    records: List[ReliabilityClaimRecord],
    tau: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate abstention policy: abstain on EVIDENCE_SENSITIVE (L <= tau <= U).

    Accepted decisions make stable class decision (ROBUST_SUPPORTED or ROBUST_HALLUCINATED).
    Never counts abstentions as correct; never silently counts them as incorrect.
    """
    valid_recs = [r for r in records if r.ground_truth is not None and r.nominal_correct is not None]
    n = len(valid_recs)
    if n == 0:
        return {
            "status": "NOT EVALUABLE",
            "coverage": 0.0,
            "accepted_n": 0,
            "abstained_n": 0,
        }

    accepted = [r for r in valid_recs if not r.global_contains_threshold]
    abstained = [r for r in valid_recs if r.global_contains_threshold]

    n_acc = len(accepted)
    n_abs = len(abstained)
    coverage = float(n_acc / n)

    if n_acc == 0:
        return {
            "status": "EVALUATED",
            "coverage": 0.0,
            "accepted_n": 0,
            "abstained_n": n_abs,
            "selective_accuracy": 0.0,
            "selective_risk": 0.0,
            "selective_precision": 0.0,
            "selective_recall": 0.0,
            "selective_f1": 0.0,
        }

    y_t = np.array([r.ground_truth for r in accepted], dtype=np.int64)
    y_p = np.array([r.nominal_prediction for r in accepted], dtype=np.int64)
    errors = (y_t != y_p).astype(np.int64)

    acc = float(1.0 - np.mean(errors))
    risk = float(np.mean(errors))

    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "status": "EVALUATED",
        "coverage": coverage,
        "accepted_n": n_acc,
        "abstained_n": n_abs,
        "selective_accuracy": acc,
        "selective_risk": risk,
        "selective_precision": prec,
        "selective_recall": rec,
        "selective_f1": f1,
    }


# ============================================================================
# Phase 9D-13: Width-Quantile Analysis
# ============================================================================

def evaluate_width_quantiles(
    records: List[ReliabilityClaimRecord],
    n_quantiles: int = 4,
    min_samples_per_bin: int = 3,
) -> Dict[str, Any]:
    """Partition claims by robust width quantile (Q1-Q4) with sample size protection."""
    valid_recs = [r for r in records if r.global_width is not None and r.nominal_correct is not None]
    n = len(valid_recs)

    if n < n_quantiles * min_samples_per_bin:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Sample size {n} < minimum required {n_quantiles * min_samples_per_bin} for {n_quantiles} quantiles",
            "bins": [],
        }

    widths = np.array([r.global_width for r in valid_recs], dtype=np.float64)
    # Check if widths have sufficient distinct values
    if len(np.unique(widths)) < n_quantiles:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Fewer unique width values ({len(np.unique(widths))}) than quantiles ({n_quantiles})",
            "bins": [],
        }

    bins_data = []
    # Use quantile cuts
    quantiles = np.linspace(0.0, 1.0, n_quantiles + 1)
    bin_edges = np.quantile(widths, quantiles)

    for q in range(n_quantiles):
        low_edge = bin_edges[q]
        high_edge = bin_edges[q + 1]
        if q == n_quantiles - 1:
            bin_recs = [r for r in valid_recs if low_edge <= r.global_width <= high_edge]
        else:
            bin_recs = [r for r in valid_recs if low_edge <= r.global_width < high_edge]

        bin_n = len(bin_recs)
        if bin_n > 0:
            err_rate = float(np.mean([0 if r.nominal_correct else 1 for r in bin_recs]))
            cross_rate = float(np.mean([1 if r.global_contains_threshold else 0 for r in bin_recs]))
            mean_conf = float(np.mean([abs(r.nominal_posterior - 0.5) * 2.0 for r in bin_recs]))
        else:
            err_rate, cross_rate, mean_conf = 0.0, 0.0, 0.0

        bins_data.append({
            "quantile": f"Q{q+1}",
            "range": (float(low_edge), float(high_edge)),
            "n_claims": bin_n,
            "error_rate": err_rate,
            "threshold_crossing_fraction": cross_rate,
            "mean_posterior_confidence": mean_conf,
        })

    return {
        "status": "EVALUATED",
        "reason": None,
        "n_samples": n,
        "bins": bins_data,
    }


# ============================================================================
# Phase 9D-14 & 9D-15: Human Annotation Disagreement and UNKNOWN Analysis
# ============================================================================

def evaluate_annotation_disagreement_uncertainty(
    records: List[ReliabilityClaimRecord],
    min_samples: int = 4,
    n_bootstraps: int = 100,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare robust interval width for annotator agreement vs disagreement with image clustering."""
    recs_with_annotators = [
        r for r in records
        if r.annotator_A is not None and r.annotator_B is not None and r.global_width is not None
    ]
    n = len(recs_with_annotators)

    if n < min_samples:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Only {n} claims with dual A/B annotations (minimum required {min_samples})",
            "n_agree": 0,
            "n_disagree": 0,
            "mean_width_agree": None,
            "mean_width_disagree": None,
            "median_width_agree": None,
            "median_width_disagree": None,
        }

    agree_recs = [r for r in recs_with_annotators if r.annotator_A == r.annotator_B]
    disagree_recs = [r for r in recs_with_annotators if r.annotator_A != r.annotator_B]

    n_ag = len(agree_recs)
    n_dis = len(disagree_recs)

    if n_ag == 0 or n_dis == 0:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Absence of dual classes: {n_ag} agreements, {n_dis} disagreements",
            "n_agree": n_ag,
            "n_disagree": n_dis,
            "mean_width_agree": float(np.mean([r.global_width for r in agree_recs])) if n_ag > 0 else None,
            "mean_width_disagree": float(np.mean([r.global_width for r in disagree_recs])) if n_dis > 0 else None,
            "median_width_agree": float(np.median([r.global_width for r in agree_recs])) if n_ag > 0 else None,
            "median_width_disagree": float(np.median([r.global_width for r in disagree_recs])) if n_dis > 0 else None,
        }

    w_ag = [r.global_width for r in agree_recs]
    w_dis = [r.global_width for r in disagree_recs]

    # Clustered bootstrap for mean difference
    img_groups: Dict[str, List[ReliabilityClaimRecord]] = {}
    for r in recs_with_annotators:
        img_groups.setdefault(r.image_id, []).append(r)
    unique_imgs = sorted(list(img_groups.keys()))

    diffs = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        samp_imgs = rng.choice(unique_imgs, size=len(unique_imgs), replace=True)
        samp_recs = []
        for img in samp_imgs:
            samp_recs.extend(img_groups[img])
        b_ag = [r.global_width for r in samp_recs if r.annotator_A == r.annotator_B and r.global_width is not None]
        b_dis = [r.global_width for r in samp_recs if r.annotator_A != r.annotator_B and r.global_width is not None]
        if b_ag and b_dis:
            diffs.append(float(np.mean(b_dis) - np.mean(b_ag)))

    ci_low = float(np.percentile(diffs, 2.5)) if len(diffs) >= 10 else None
    ci_high = float(np.percentile(diffs, 97.5)) if len(diffs) >= 10 else None

    return {
        "status": "EVALUATED",
        "reason": None,
        "n_agree": n_ag,
        "n_disagree": n_dis,
        "mean_width_agree": float(np.mean(w_ag)),
        "mean_width_disagree": float(np.mean(w_dis)),
        "median_width_agree": float(np.median(w_ag)),
        "median_width_disagree": float(np.median(w_dis)),
        "mean_difference": float(np.mean(w_dis) - np.mean(w_ag)),
        "mean_diff_bootstrap_ci": (ci_low, ci_high) if ci_low is not None else None,
    }


def evaluate_unknown_claim_uncertainty(
    records: List[ReliabilityClaimRecord],
    unknown_indicator_fn: Optional[Callable[[ReliabilityClaimRecord], bool]] = None,
) -> Dict[str, Any]:
    """Compare robust interval width across SUPPORTED, HALLUCINATED, and UNKNOWN claims.

    UNKNOWN claims are strictly excluded from binary classification but analyzed here.
    """
    valid_recs = [r for r in records if r.global_width is not None]

    sup_widths = [r.global_width for r in valid_recs if r.ground_truth == 0]
    hall_widths = [r.global_width for r in valid_recs if r.ground_truth == 1]

    if unknown_indicator_fn is not None:
        unk_widths = [r.global_width for r in valid_recs if unknown_indicator_fn(r)]
    else:
        # Ground truth None or explicit label 2
        unk_widths = [r.global_width for r in valid_recs if r.ground_truth is None]

    return {
        "status": "EVALUATED",
        "n_supported": len(sup_widths),
        "n_hallucinated": len(hall_widths),
        "n_unknown": len(unk_widths),
        "mean_width_supported": float(np.mean(sup_widths)) if sup_widths else None,
        "mean_width_hallucinated": float(np.mean(hall_widths)) if hall_widths else None,
        "mean_width_unknown": float(np.mean(unk_widths)) if unk_widths else None,
        "median_width_supported": float(np.median(sup_widths)) if sup_widths else None,
        "median_width_hallucinated": float(np.median(hall_widths)) if hall_widths else None,
        "median_width_unknown": float(np.median(unk_widths)) if unk_widths else None,
    }


# ============================================================================
# Phase 9D-16 & 9D-17: Corruption Sensitivity Tracking
# ============================================================================

def evaluate_corruption_width_tracking(
    records: List[ReliabilityClaimRecord],
    min_claims: int = 3,
) -> Dict[str, Any]:
    """Track within-claim robust width expansion Delta W_{i,c} = W_{i,c} - W_{i,clean}.

    Uses repeated-measures pairing across severities: clean, light, medium, heavy.
    Evaluates Spearman rank correlation between severity and width.
    """
    # Map severities to ranks
    severity_ranks = {"clean": 0, "none": 0, "light": 1, "medium": 2, "heavy": 3}

    # Group by claim key: (image_id, claim_id)
    claims_map: Dict[Tuple[str, str], Dict[str, ReliabilityClaimRecord]] = {}
    for r in records:
        if r.global_width is not None:
            claims_map.setdefault((r.image_id, r.claim_id), {})[r.corruption_severity.lower()] = r

    # Compute delta width for paired claims with clean condition
    paired_deltas: Dict[str, List[float]] = {"light": [], "medium": [], "heavy": []}
    correlations = []

    for (img_id, claim_id), sev_dict in claims_map.items():
        clean_rec = sev_dict.get("clean") or sev_dict.get("none")
        if clean_rec is not None:
            w_clean = clean_rec.global_width
            for sev in ["light", "medium", "heavy"]:
                if sev in sev_dict:
                    paired_deltas[sev].append(float(sev_dict[sev].global_width - w_clean))

        # Severity vs width correlation per claim if >= 3 severities
        if len(sev_dict) >= 3:
            ranks = [severity_ranks.get(s, 0) for s in sev_dict.keys()]
            w_vals = [sev_dict[s].global_width for s in sev_dict.keys()]
            sp, _ = stats.spearmanr(ranks, w_vals)
            if not np.isnan(sp):
                correlations.append(float(sp))

    n_paired = len([c for c, d in claims_map.items() if ("clean" in d or "none" in d)])
    if n_paired < min_claims:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Only {n_paired} claims with repeated clean/corrupted measurements (minimum required {min_claims})",
            "n_paired_claims": n_paired,
            "mean_delta_light": None,
            "mean_delta_medium": None,
            "mean_delta_heavy": None,
            "mean_spearman_severity_width": None,
        }

    return {
        "status": "EVALUATED",
        "n_paired_claims": n_paired,
        "mean_delta_light": float(np.mean(paired_deltas["light"])) if paired_deltas["light"] else None,
        "mean_delta_medium": float(np.mean(paired_deltas["medium"])) if paired_deltas["medium"] else None,
        "mean_delta_heavy": float(np.mean(paired_deltas["heavy"])) if paired_deltas["heavy"] else None,
        "mean_spearman_severity_width": float(np.mean(correlations)) if correlations else None,
    }


# ============================================================================
# Phase 9D-18: Local vs Global Interval Utility Comparison
# ============================================================================

def compare_local_vs_global_utility(
    records: List[ReliabilityClaimRecord],
) -> Dict[str, Any]:
    """Direct comparison of Local-Box vs Global-Budget interval utility."""
    valid_recs = [r for r in records if r.global_width is not None and r.local_width is not None]
    if not valid_recs:
        return {"status": "NOT EVALUABLE", "reason": "No claims with both local and global widths"}

    y_t = [r.ground_truth for r in valid_recs if r.ground_truth is not None]
    y_p = [r.nominal_prediction for r in valid_recs if r.ground_truth is not None]

    w_g = [r.global_width for r in valid_recs if r.ground_truth is not None]
    w_l = [r.local_width for r in valid_recs if r.ground_truth is not None]

    err_g = compute_error_detection_metrics(y_t, y_p, w_g)
    err_l = compute_error_detection_metrics(y_t, y_p, w_l)

    rc_g = compute_discrete_risk_coverage_curve(valid_recs, lambda r: r.global_width or 0.0)
    rc_l = compute_discrete_risk_coverage_curve(valid_recs, lambda r: r.local_width or 0.0)

    return {
        "status": "EVALUATED",
        "n_claims": len(valid_recs),
        "mean_width_global": float(np.mean(w_g)),
        "mean_width_local": float(np.mean(w_l)),
        "global_error_auroc": err_g.get("error_detection_auroc"),
        "local_error_auroc": err_l.get("error_detection_auroc"),
        "global_error_auprc": err_g.get("error_detection_auprc"),
        "local_error_auprc": err_l.get("error_detection_auprc"),
        "global_discrete_aurc": rc_g.get("discrete_aurc"),
        "local_discrete_aurc": rc_l.get("discrete_aurc"),
        "global_excess_aurc": rc_g.get("excess_aurc"),
        "local_excess_aurc": rc_l.get("excess_aurc"),
    }


# ============================================================================
# Phase 9D-19: Budget Ratio Analysis
# ============================================================================

def evaluate_budget_ratio_utility_sweep(
    records_by_rho: Dict[float, List[ReliabilityClaimRecord]],
) -> Dict[float, Dict[str, Any]]:
    """Evaluate utility metrics across budget ratios rho = B / sum_i epsilon_i."""
    sweep_results: Dict[float, Dict[str, Any]] = {}

    for rho, recs in sorted(records_by_rho.items()):
        valid = [r for r in recs if r.global_width is not None]
        if not valid:
            continue

        widths = [r.global_width for r in valid]
        y_t = [r.ground_truth for r in valid if r.ground_truth is not None]
        y_p = [r.nominal_prediction for r in valid if r.ground_truth is not None]
        w_vals = [r.global_width for r in valid if r.ground_truth is not None]

        err_m = compute_error_detection_metrics(y_t, y_p, w_vals)
        rc_m = compute_discrete_risk_coverage_curve(valid, lambda r: r.global_width or 0.0)

        cross_rate = float(np.mean([1 if r.global_contains_threshold else 0 for r in valid]))

        sweep_results[float(rho)] = {
            "rho": float(rho),
            "n_claims": len(valid),
            "mean_width": float(np.mean(widths)),
            "error_auroc": err_m.get("error_detection_auroc"),
            "error_auprc": err_m.get("error_detection_auprc"),
            "discrete_aurc": rc_m.get("discrete_aurc"),
            "threshold_crossing_rate": cross_rate,
        }

    return sweep_results


# ============================================================================
# Phase 9D-20 & 9D-21: BP Posterior Calibration Audit
# ============================================================================

def audit_bp_posterior_calibration(
    unary_probs: Sequence[float],
    bp_probs: Sequence[float],
    ground_truth: Sequence[int],
    n_bins: int = 5,
    min_samples: int = 6,
) -> Dict[str, Any]:
    """Evaluate calibration of standard BP posterior after graph coupling vs independent unary.

    PRIMARY METRICS:
    - Brier score
    - Log loss

    SECONDARY DIAGNOSTIC:
    - ECE (Expected Calibration Error) with documented bins and sample size
    """
    y = np.asarray(ground_truth, dtype=np.int64)
    p_unary = np.clip(np.asarray(unary_probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    p_bp = np.clip(np.asarray(bp_probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)

    n = len(y)
    if n < min_samples:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Sample size {n} < minimum required {min_samples}",
            "unary_brier": None,
            "bp_brier": None,
            "unary_log_loss": None,
            "bp_log_loss": None,
            "unary_ece": None,
            "bp_ece": None,
        }

    # Primary
    brier_u = float(brier_score_loss(y, p_unary))
    brier_bp = float(brier_score_loss(y, p_bp))
    ll_u = float(log_loss(y, p_unary))
    ll_bp = float(log_loss(y, p_bp))

    # Secondary ECE
    def compute_ece(probs: np.ndarray) -> float:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for b in range(n_bins):
            mask = (probs >= bin_edges[b]) & (probs <= bin_edges[b + 1])
            if np.any(mask):
                bin_acc = np.mean(y[mask])
                bin_conf = np.mean(probs[mask])
                ece += np.sum(mask) / n * abs(bin_acc - bin_conf)
        return float(ece)

    ece_u = compute_ece(p_unary)
    ece_bp = compute_ece(p_bp)

    return {
        "status": "EVALUATED",
        "n_samples": n,
        "n_bins": n_bins,
        "unary_brier": brier_u,
        "bp_brier": brier_bp,
        "delta_brier": brier_bp - brier_u,
        "unary_log_loss": ll_u,
        "bp_log_loss": ll_bp,
        "delta_log_loss": ll_bp - ll_u,
        "unary_ece": ece_u,
        "bp_ece": ece_bp,
        "delta_ece": ece_bp - ece_u,
    }


# ============================================================================
# Phase 9D-23 & 9D-24: Image-Level Bootstrap & Paired Method Comparisons
# ============================================================================

def compute_paired_image_bootstrap_comparisons(
    records: List[ReliabilityClaimRecord],
    n_bootstraps: int = 100,
    seed: int = 42,
    min_images: int = 4,
) -> Dict[str, Any]:
    """Compute paired bootstrap distribution of metric differences across resampled images.

    Resampling unit: IMAGE_ID (all claims belonging to an image resampled together).
    Evaluates:
    - Delta AUROC: AUROC(W_global) - AUROC(nominal_entropy)
    - Delta AURC: AURC(W_global) - AURC(nominal_entropy)
    - Delta AURC_local: AURC(W_global) - AURC(W_local)
    """
    valid_recs = [r for r in records if r.ground_truth is not None and r.nominal_correct is not None]

    img_groups: Dict[str, List[ReliabilityClaimRecord]] = {}
    for r in valid_recs:
        img_groups.setdefault(r.image_id, []).append(r)

    unique_imgs = sorted(list(img_groups.keys()))
    n_imgs = len(unique_imgs)

    if n_imgs < min_images:
        return {
            "status": "NOT EVALUABLE",
            "reason": f"Only {n_imgs} unique images (minimum required {min_images})",
            "delta_auroc_ci": None,
            "delta_aurc_ci": None,
        }

    rng = np.random.RandomState(seed)
    delta_aurocs = []
    delta_aurcs = []
    delta_local_aurcs = []

    for _ in range(n_bootstraps):
        samp_imgs = rng.choice(unique_imgs, size=n_imgs, replace=True)
        samp_recs: List[ReliabilityClaimRecord] = []
        for img in samp_imgs:
            samp_recs.extend(img_groups[img])

        y_t = [r.ground_truth for r in samp_recs]
        y_p = [r.nominal_prediction for r in samp_recs]
        w_g = [r.global_width for r in samp_recs]
        u_ent = [r.nominal_entropy for r in samp_recs]
        w_loc = [r.local_width if r.local_width is not None else r.global_width for r in samp_recs]

        # Check binary error classes
        errs = [0 if r.nominal_correct else 1 for r in samp_recs]
        if len(np.unique(errs)) < 2:
            continue

        try:
            auc_wg = float(roc_auc_score(errs, w_g))
            auc_ent = float(roc_auc_score(errs, u_ent))
            delta_aurocs.append(auc_wg - auc_ent)
        except Exception:
            pass

        rc_g = compute_discrete_risk_coverage_curve(samp_recs, lambda r: r.global_width or 0.0)
        rc_ent = compute_discrete_risk_coverage_curve(samp_recs, lambda r: r.nominal_entropy)
        rc_loc = compute_discrete_risk_coverage_curve(samp_recs, lambda r: r.local_width or 0.0)

        if rc_g["discrete_aurc"] is not None and rc_ent["discrete_aurc"] is not None:
            delta_aurcs.append(rc_g["discrete_aurc"] - rc_ent["discrete_aurc"])
        if rc_g["discrete_aurc"] is not None and rc_loc["discrete_aurc"] is not None:
            delta_local_aurcs.append(rc_g["discrete_aurc"] - rc_loc["discrete_aurc"])

    def get_ci(arr: List[float]) -> Optional[Tuple[float, float, float]]:
        if len(arr) < 10:
            return None
        return (float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "status": "EVALUATED",
        "n_images": n_imgs,
        "n_claims": len(valid_recs),
        "delta_auroc_mean_ci": get_ci(delta_aurocs),
        "delta_aurc_mean_ci": get_ci(delta_aurcs),
        "delta_aurc_local_mean_ci": get_ci(delta_local_aurcs),
    }


# ============================================================================
# Phase 9D-30: Deterministic Case Study Selector (Cases A through H)
# ============================================================================

def select_deterministic_case_studies(
    records: List[ReliabilityClaimRecord],
    tau: float = 0.5,
) -> Dict[str, Optional[ReliabilityClaimRecord]]:
    """Automatically choose deterministic examples according to strict scientific criteria:

    A. nominally confident + correct + narrow robust interval
    B. nominally confident + wrong + wide robust interval
    C. nominally uncertain + wide interval
    D. threshold-crossing interval
    E. human annotator disagreement + wide interval
    F. corruption causes width expansion
    G. nominal uncertainty is high but robust width is low
    H. nominal uncertainty is low but robust width is high
    """
    valid = [r for r in records if r.global_width is not None]
    if not valid:
        return {k: None for k in ["A", "B", "C", "D", "E", "F", "G", "H"]}

    median_w = float(np.median([r.global_width for r in valid]))
    median_ent = float(np.median([r.nominal_entropy for r in valid]))

    case_A, case_B, case_C, case_D = None, None, None, None
    case_E, case_F, case_G, case_H = None, None, None, None

    # Case A: Confident (entropy < median), correct, narrow (width < median)
    cand_A = [r for r in valid if r.nominal_entropy <= median_ent and r.nominal_correct is True and r.global_width <= median_w]
    if cand_A:
        case_A = min(cand_A, key=lambda r: (r.global_width, r.nominal_entropy))

    # Case B: Confident (entropy < median), wrong, wide (width > median)
    cand_B = [r for r in valid if r.nominal_entropy <= median_ent and r.nominal_correct is False and r.global_width >= median_w]
    if cand_B:
        case_B = max(cand_B, key=lambda r: (r.global_width, -r.nominal_entropy))

    # Case C: Uncertain (entropy > median), wide interval
    cand_C = [r for r in valid if r.nominal_entropy >= median_ent and r.global_width >= median_w]
    if cand_C:
        case_C = max(cand_C, key=lambda r: (r.global_width, r.nominal_entropy))

    # Case D: Threshold-crossing (contains_threshold == True)
    cand_D = [r for r in valid if r.global_contains_threshold]
    if cand_D:
        case_D = max(cand_D, key=lambda r: r.global_width)

    # Case E: Annotator disagreement + wide interval
    cand_E = [r for r in valid if r.annotator_disagreement is True and r.global_width >= median_w]
    if cand_E:
        case_E = max(cand_E, key=lambda r: r.global_width)

    # Case F: Corruption condition (severity != clean) with wide interval
    cand_F = [r for r in valid if r.corruption_severity.lower() in ("medium", "heavy") and r.global_width >= median_w]
    if cand_F:
        case_F = max(cand_F, key=lambda r: r.global_width)

    # Case G: High nominal uncertainty (entropy > median) but LOW robust width (width < median)
    cand_G = [r for r in valid if r.nominal_entropy >= median_ent and r.global_width <= median_w]
    if cand_G:
        case_G = min(cand_G, key=lambda r: (r.global_width, -r.nominal_entropy))

    # Case H: Low nominal uncertainty (entropy < median) but HIGH robust width (width > median)
    cand_H = [r for r in valid if r.nominal_entropy <= median_ent and r.global_width >= median_w]
    if cand_H:
        case_H = max(cand_H, key=lambda r: (-r.global_width, r.nominal_entropy))

    return {
        "case_A_confident_correct_narrow": case_A,
        "case_B_confident_wrong_wide": case_B,
        "case_C_uncertain_wide": case_C,
        "case_D_threshold_crossing": case_D,
        "case_E_annotation_disagreement_wide": case_E,
        "case_F_corruption_expansion": case_F,
        "case_G_high_uncertainty_low_width": case_G,
        "case_H_low_uncertainty_high_width": case_H,
    }


# ============================================================================
# Phase 9D-32: Research Decision Gate Evaluator
# ============================================================================

@dataclass
class ReliabilityResearchDecisionGate:
    """Status labels for M9D Central Research Questions."""
    robust_width_error_signal: str
    incremental_value: str
    selective_prediction_value: str
    annotation_alignment: str
    corruption_sensitivity: str
    global_vs_local_utility: str
    final_data_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_reliability_decision_gate(
    num_images: int,
    num_claims: int,
    is_development: bool = True,
) -> ReliabilityResearchDecisionGate:
    """Enforce development safety on research claims."""
    if is_development or num_images < 100 or num_claims < 300:
        return ReliabilityResearchDecisionGate(
            robust_width_error_signal="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            incremental_value="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            selective_prediction_value="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            annotation_alignment="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            corruption_sensitivity="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            global_vs_local_utility="NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)",
            final_data_required=True,
        )

    return ReliabilityResearchDecisionGate(
        robust_width_error_signal="FINAL EVIDENCE",
        incremental_value="FINAL EVIDENCE",
        selective_prediction_value="FINAL EVIDENCE",
        annotation_alignment="FINAL EVIDENCE",
        corruption_sensitivity="FINAL EVIDENCE",
        global_vs_local_utility="FINAL EVIDENCE",
        final_data_required=False,
    )
