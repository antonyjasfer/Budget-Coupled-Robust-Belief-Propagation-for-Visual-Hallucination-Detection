"""
Statistical evaluation and classification metrics for hallucination detection.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from sklearn import metrics

from src.data.schemas import GroundTruthStatus, DecisionStatus
from src.experiments.inference_runner import ClaimEvaluationResult


@dataclass
class ConfusionMatrix:
    tp: int = 0  # True Positives: Ground Truth = HALLUCINATED, Predicted = HALLUCINATED
    tn: int = 0  # True Negatives: Ground Truth = SUPPORTED, Predicted = SUPPORTED
    fp: int = 0  # False Positives: Ground Truth = SUPPORTED, Predicted = HALLUCINATED
    fn: int = 0  # False Negatives: Ground Truth = HALLUCINATED, Predicted = SUPPORTED

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    """
    Standard binary classification and ranking metrics for hallucination detection.
    """
    total_claims: int
    evaluated_claims: int
    supported_gt_count: int
    hallucinated_gt_count: int
    unknown_gt_count: int
    unannotated_gt_count: int
    abstained_count: int
    coverage_rate: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: ConfusionMatrix
    decision_threshold: float
    method_name: str
    condition: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "evaluated_claims": self.evaluated_claims,
            "supported_gt_count": self.supported_gt_count,
            "hallucinated_gt_count": self.hallucinated_gt_count,
            "unknown_gt_count": self.unknown_gt_count,
            "unannotated_gt_count": self.unannotated_gt_count,
            "abstained_count": self.abstained_count,
            "coverage_rate": self.coverage_rate,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "decision_threshold": self.decision_threshold,
            "method_name": self.method_name,
            "condition": self.condition,
            "diagnostics": self.diagnostics,
        }


def compute_classification_metrics(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], ConfusionMatrix]:
    """
    Compute binary classification metrics with zero-division and single-class safeguards.
    Convention: 1 = HALLUCINATED (Positive Class), 0 = SUPPORTED (Negative Class).
    """
    if len(y_true_binary) == 0:
        return None, None, None, None, None, None, ConfusionMatrix()

    # Confusion matrix
    tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))
    tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))
    fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))
    fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))
    cm = ConfusionMatrix(tp=tp, tn=tn, fp=fp, fn=fn)

    # Accuracy
    total_eval = len(y_true_binary)
    acc = float((tp + tn) / total_eval) if total_eval > 0 else None

    # Precision
    denom_prec = tp + fp
    prec = float(tp / denom_prec) if denom_prec > 0 else None

    # Recall
    denom_rec = tp + fn
    rec = float(tp / denom_rec) if denom_rec > 0 else None

    # F1
    if prec is not None and rec is not None and (prec + rec) > 0:
        f1 = float(2.0 * prec * rec / (prec + rec))
    else:
        f1 = None

    # ROC-AUC & PR-AUC
    roc_auc = None
    pr_auc = None

    if y_scores is not None and len(y_scores) == len(y_true_binary):
        unique_classes = np.unique(y_true_binary)
        if len(unique_classes) == 2:
            try:
                roc_auc = float(metrics.roc_auc_score(y_true_binary, y_scores))
                pr_auc = float(metrics.average_precision_score(y_true_binary, y_scores))
            except Exception:
                roc_auc = None
                pr_auc = None
        else:
            # Degenerate one-class subset -> ROC-AUC undefined
            roc_auc = None
            pr_auc = None

    return acc, prec, rec, f1, roc_auc, pr_auc, cm


def evaluate_claim_results(
    results: List[ClaimEvaluationResult],
    method_name: str = "standard_bp",
    decision_threshold: float = 0.5,
    exclude_unknown: bool = True,
    exclude_abstain: bool = False,
) -> ClassificationMetrics:
    """
    Evaluate claim-level predictions against ground truth.

    Ground truth mapping:
    - SUPPORTED -> 0 (Negative class)
    - HALLUCINATED -> 1 (Positive class)
    - UNKNOWN -> Excluded from binary evaluation (never mapped to 0 or 1).
    """
    total_claims = len(results)
    if total_claims == 0:
        return ClassificationMetrics(
            total_claims=0,
            evaluated_claims=0,
            supported_gt_count=0,
            hallucinated_gt_count=0,
            unknown_gt_count=0,
            unannotated_gt_count=0,
            abstained_count=0,
            coverage_rate=0.0,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            roc_auc=None,
            pr_auc=None,
            confusion_matrix=ConfusionMatrix(),
            decision_threshold=decision_threshold,
            method_name=method_name,
            condition="empty",
        )

    condition = results[0].condition if results else "clean"

    supported_count = 0
    hallucinated_count = 0
    unknown_count = 0
    unannotated_count = 0
    abstained_count = 0

    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    y_scores_list: List[float] = []

    for r in results:
        gt = r.ground_truth
        if gt is None:
            unannotated_count += 1
            continue

        gt_norm = str(gt).strip().lower()

        if gt_norm == GroundTruthStatus.UNKNOWN.value:
            unknown_count += 1
            if exclude_unknown:
                continue

        if gt_norm == GroundTruthStatus.SUPPORTED.value:
            supported_count += 1
            gt_bin = 0
        elif gt_norm == GroundTruthStatus.HALLUCINATED.value:
            hallucinated_count += 1
            gt_bin = 1
        else:
            unannotated_count += 1
            continue

        # Extract prediction and continuous score
        if method_name == "robust_bp":
            is_abstain = (r.robust_prediction == DecisionStatus.ABSTAIN.value) or r.abstained
            if is_abstain:
                abstained_count += 1
                if exclude_abstain:
                    continue
                # If evaluating abstain without exclusion, fallback to midpoint vs threshold
                pred_bin = 1 if r.robust_midpoint >= decision_threshold else 0
            else:
                pred_bin = 1 if r.robust_prediction == DecisionStatus.HALLUCINATED.value else 0
            score = float(r.robust_midpoint)
        elif method_name == "evidence_point":
            score = float(1.0 - (r.detector_score if r.detector_score is not None else 0.5))
            pred_bin = 1 if score >= decision_threshold else 0
        else:
            # Default standard BP
            score = float(r.standard_posterior)
            pred_bin = 1 if score >= decision_threshold else 0

        y_true_list.append(gt_bin)
        y_pred_list.append(pred_bin)
        y_scores_list.append(score)

    evaluated_claims = len(y_true_list)
    cov_rate = float(evaluated_claims / total_claims) if total_claims > 0 else 0.0

    y_true_arr = np.array(y_true_list, dtype=np.int32)
    y_pred_arr = np.array(y_pred_list, dtype=np.int32)
    y_scores_arr = np.array(y_scores_list, dtype=np.float64)

    acc, prec, rec, f1, roc_auc, pr_auc, cm = compute_classification_metrics(
        y_true_binary=y_true_arr,
        y_pred_binary=y_pred_arr,
        y_scores=y_scores_arr,
    )

    diagnostics: Dict[str, Any] = {}
    if evaluated_claims > 0:
        unique_classes = np.unique(y_true_arr).tolist()
        if len(unique_classes) < 2:
            diagnostics["single_class_eval"] = True
            diagnostics["unique_classes"] = unique_classes
            diagnostics["roc_auc_note"] = "ROC-AUC undefined because test split contains only 1 ground-truth class."

    return ClassificationMetrics(
        total_claims=total_claims,
        evaluated_claims=evaluated_claims,
        supported_gt_count=supported_count,
        hallucinated_gt_count=hallucinated_count,
        unknown_gt_count=unknown_count,
        unannotated_gt_count=unannotated_count,
        abstained_count=abstained_count,
        coverage_rate=cov_rate,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        decision_threshold=decision_threshold,
        method_name=method_name,
        condition=condition,
        diagnostics=diagnostics,
    )
