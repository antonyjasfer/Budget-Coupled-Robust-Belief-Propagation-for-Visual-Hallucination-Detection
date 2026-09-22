"""
Metrics and statistical analysis for robust posterior uncertainty intervals.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats

from src.data.schemas import GroundTruthStatus
from src.experiments.inference_runner import ClaimEvaluationResult


@dataclass
class RobustIntervalStatistics:
    """
    Descriptive and inferential properties of robust posterior intervals.
    """
    total_claims: int
    mean_width: float
    median_width: float
    std_width: float
    min_width: float
    max_width: float
    quantiles: Dict[str, float]  # "p25", "p50", "p75", "p90"
    
    # Categorical breakdown
    mean_width_supported: Optional[float]
    mean_width_hallucinated: Optional[float]
    mean_width_unknown: Optional[float]
    mean_width_correct_bp: Optional[float]
    mean_width_wrong_bp: Optional[float]

    # Threshold & Decision fractions
    evidence_sensitive_fraction: float  # L_i <= tau <= U_i
    robustly_supported_fraction: float  # U_i < tau
    robustly_hallucinated_fraction: float  # L_i > tau
    decision_threshold: float

    # Key scientific correlation test: interval width vs BP error indicator
    pearson_corr_width_error: Optional[float]
    pearson_p_value: Optional[float]
    spearman_corr_width_error: Optional[float]
    spearman_p_value: Optional[float]

    # Width vs Evidence Conflict correlation: width vs |detector - clip|
    corr_width_evidence_conflict: Optional[float]

    # Uncertainty bin analysis
    binned_error_rates: Dict[str, Optional[float]] = field(default_factory=dict)
    sample_counts: Dict[str, int] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_interval_statistics(
    results: List[ClaimEvaluationResult],
    decision_threshold: float = 0.5,
) -> RobustIntervalStatistics:
    """
    Compute comprehensive interval metrics and error correlation on evaluated claims.
    """
    total = len(results)
    if total == 0:
        return RobustIntervalStatistics(
            total_claims=0,
            mean_width=0.0,
            median_width=0.0,
            std_width=0.0,
            min_width=0.0,
            max_width=0.0,
            quantiles={"p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0},
            mean_width_supported=None,
            mean_width_hallucinated=None,
            mean_width_unknown=None,
            mean_width_correct_bp=None,
            mean_width_wrong_bp=None,
            evidence_sensitive_fraction=0.0,
            robustly_supported_fraction=0.0,
            robustly_hallucinated_fraction=0.0,
            decision_threshold=decision_threshold,
            pearson_corr_width_error=None,
            pearson_p_value=None,
            spearman_corr_width_error=None,
            spearman_p_value=None,
            corr_width_evidence_conflict=None,
        )

    widths = np.array([r.interval_width for r in results], dtype=np.float64)
    tau = decision_threshold

    mean_w = float(np.mean(widths))
    median_w = float(np.median(widths))
    std_w = float(np.std(widths))
    min_w = float(np.min(widths))
    max_w = float(np.max(widths))

    q25 = float(np.percentile(widths, 25))
    q50 = float(np.percentile(widths, 50))
    q75 = float(np.percentile(widths, 75))
    q90 = float(np.percentile(widths, 90))

    # Threshold crossings
    ev_sensitive_cnt = sum(1 for r in results if r.robust_lower <= tau <= r.robust_upper)
    rob_supp_cnt = sum(1 for r in results if r.robust_upper < tau)
    rob_halluc_cnt = sum(1 for r in results if r.robust_lower > tau)

    ev_sens_frac = float(ev_sensitive_cnt / total)
    rob_supp_frac = float(rob_supp_cnt / total)
    rob_halluc_frac = float(rob_halluc_cnt / total)

    # Class-conditional width
    widths_supp = [r.interval_width for r in results if str(r.ground_truth).lower() == GroundTruthStatus.SUPPORTED.value]
    widths_halluc = [r.interval_width for r in results if str(r.ground_truth).lower() == GroundTruthStatus.HALLUCINATED.value]
    widths_unk = [r.interval_width for r in results if str(r.ground_truth).lower() == GroundTruthStatus.UNKNOWN.value]

    mean_w_supp = float(np.mean(widths_supp)) if widths_supp else None
    mean_w_halluc = float(np.mean(widths_halluc)) if widths_halluc else None
    mean_w_unk = float(np.mean(widths_unk)) if widths_unk else None

    # Correct vs Incorrect Standard BP predictions
    widths_correct: List[float] = []
    widths_wrong: List[float] = []
    errors: List[int] = []
    valid_widths_for_error: List[float] = []
    conflicts: List[float] = []
    valid_widths_for_conflict: List[float] = []

    for r in results:
        gt = r.ground_truth
        if gt is not None:
            gt_norm = str(gt).strip().lower()
            if gt_norm in [GroundTruthStatus.SUPPORTED.value, GroundTruthStatus.HALLUCINATED.value]:
                # Prediction is 1 if standard_posterior >= tau else 0
                pred_halluc = (r.standard_posterior >= tau)
                gt_halluc = (gt_norm == GroundTruthStatus.HALLUCINATED.value)
                is_error = int(pred_halluc != gt_halluc)

                errors.append(is_error)
                valid_widths_for_error.append(r.interval_width)

                if is_error:
                    widths_wrong.append(r.interval_width)
                else:
                    widths_correct.append(r.interval_width)

        # Conflict between detector and clip
        if r.detector_score is not None and r.clip_score is not None:
            s_det = float(np.clip(r.detector_score, 0.0, 1.0))
            s_clip = float(np.clip((r.clip_score + 1.0) / 2.0, 0.0, 1.0))
            conflicts.append(abs(s_det - s_clip))
            valid_widths_for_conflict.append(r.interval_width)

    mean_w_correct = float(np.mean(widths_correct)) if widths_correct else None
    mean_w_wrong = float(np.mean(widths_wrong)) if widths_wrong else None

    # Correlations
    pearson_r, pearson_p = None, None
    spearman_rho, spearman_p = None, None
    if len(errors) >= 3 and len(set(errors)) > 1 and np.std(valid_widths_for_error) > 1e-12:
        try:
            pr, pp = stats.pearsonr(valid_widths_for_error, errors)
            pearson_r = float(pr)
            pearson_p = float(pp)
        except Exception:
            pass
        try:
            sr, sp = stats.spearmanr(valid_widths_for_error, errors)
            spearman_rho = float(sr)
            spearman_p = float(sp)
        except Exception:
            pass

    corr_conflict = None
    if len(conflicts) >= 3 and np.std(conflicts) > 1e-12 and np.std(valid_widths_for_conflict) > 1e-12:
        try:
            cr, _ = stats.pearsonr(valid_widths_for_conflict, conflicts)
            corr_conflict = float(cr)
        except Exception:
            pass

    # Binned error rate analysis (Narrow vs Wide intervals)
    binned_errors: Dict[str, Optional[float]] = {}
    if valid_widths_for_error:
        vw_arr = np.array(valid_widths_for_error)
        err_arr = np.array(errors)
        med = float(np.median(vw_arr))
        
        low_mask = (vw_arr <= med)
        high_mask = (vw_arr > med)

        binned_errors["narrow_interval_error_rate"] = float(np.mean(err_arr[low_mask])) if np.any(low_mask) else None
        binned_errors["wide_interval_error_rate"] = float(np.mean(err_arr[high_mask])) if np.any(high_mask) else None

    diagnostics: Dict[str, Any] = {
        "annotated_eval_count": len(errors),
        "is_smoke_sample": total < 30,
    }

    return RobustIntervalStatistics(
        total_claims=total,
        mean_width=mean_w,
        median_width=median_w,
        std_width=std_w,
        min_width=min_w,
        max_width=max_w,
        quantiles={"p25": q25, "p50": q50, "p75": q75, "p90": q90},
        mean_width_supported=mean_w_supp,
        mean_width_hallucinated=mean_w_halluc,
        mean_width_unknown=mean_w_unk,
        mean_width_correct_bp=mean_w_correct,
        mean_width_wrong_bp=mean_w_wrong,
        evidence_sensitive_fraction=ev_sens_frac,
        robustly_supported_fraction=rob_supp_frac,
        robustly_hallucinated_fraction=rob_halluc_frac,
        decision_threshold=tau,
        pearson_corr_width_error=pearson_r,
        pearson_p_value=pearson_p,
        spearman_corr_width_error=spearman_rho,
        spearman_p_value=spearman_p,
        corr_width_evidence_conflict=corr_conflict,
        binned_error_rates=binned_errors,
        sample_counts={
            "supported": len(widths_supp),
            "hallucinated": len(widths_halluc),
            "unknown": len(widths_unk),
            "correct_bp": len(widths_correct),
            "wrong_bp": len(widths_wrong),
        },
        diagnostics=diagnostics,
    )
