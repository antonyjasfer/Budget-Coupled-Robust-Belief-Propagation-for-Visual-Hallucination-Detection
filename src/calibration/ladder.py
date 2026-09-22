"""Controlled baseline ladder for evaluating hallucination detection methods.

Implements the single controlled ladder where every successive method adds
exactly ONE conceptual component:
- M0: Detector only (d_i -> hallucination score)
- M1: CLIP only (g_i -> hallucination score)
- M2: Detector + CLIP logistic fusion ([d_i, g_i] -> calibrated p_i)
- M3: Independent calibrated Ising unaries (theta_i = 0.5 * logit(p_i), J = 0)
- M4: Standard BP with graph coupling (theta_i, J_ij >= 0, nominal posterior)
- M5: Local-box robust inference (|delta_i| <= epsilon_i, unconstrained budget)
- M6: Budget-coupled robust inference (|delta_i| <= epsilon_i, sum_i |delta_i| <= B)
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import expit

from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
    compute_calibration_diagnostics,
)
from src.calibration.theta_mapping import probability_to_theta, theta_to_probability
from src.pgm.standard_bp import run_standard_bp
from src.pgm.tree_model import TreeModel
from src.robust_bp.solver import solve_robust_bp


class BaselineMethod(str, Enum):
    M0_DETECTOR = "M0_detector_only"
    M1_CLIP = "M1_clip_only"
    M2_FUSION = "M2_logistic_fusion"
    M3_UNARY_ISOLATED = "M3_unary_isolated"
    M4_STANDARD_BP = "M4_standard_bp"
    M5_LOCAL_ROBUST = "M5_local_box_robust"
    M6_GLOBAL_ROBUST = "M6_budget_coupled_robust"


@dataclass
class ClaimPredictionResult:
    """Standardized result schema for evaluating any method on a single claim."""

    run_id: str
    image_id: str
    claim_id: str
    split: str
    graph_size: int
    evidence_variant: str
    baseline_name: str
    topology: str
    coupling_strategy: str
    lambda_param: float
    theta: float
    epsilon: float
    budget: float
    budget_ratio: float
    nominal_posterior: float
    robust_lower: Optional[float] = None
    robust_upper: Optional[float] = None
    robust_width: Optional[float] = None
    local_lower: Optional[float] = None
    local_upper: Optional[float] = None
    local_width: Optional[float] = None
    ground_truth: Optional[int] = None  # 0 = SUPPORTED, 1 = HALLUCINATED
    prediction: int = 0
    correct: Optional[bool] = None
    threshold_crossing: bool = False
    runtime_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MethodEvaluationSummary:
    """Summary metrics for a specific baseline method."""

    method_name: str
    evidence_variant: str
    has_graph: bool
    uncertainty_type: str  # "NONE", "LOCAL", "GLOBAL"
    has_global_budget: bool
    num_claims: int
    num_images: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    log_loss: float
    mean_runtime_ms: float
    mean_width: Optional[float] = None
    threshold_crossing_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_binary_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard classification and ranking metrics."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    n = len(y)
    if n == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "roc_auc": 0.5, "pr_auc": 0.0, "brier": 0.0, "log_loss": 0.0,
        }

    preds = (p >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))

    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ROC-AUC via rank calculation
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    if pos_count > 0 and neg_count > 0:
        order = np.argsort(p)
        rank = np.empty_like(order)
        rank[order] = np.arange(n)
        pos_ranks = rank[y == 1]
        roc_auc = float((np.sum(pos_ranks) - pos_count * (pos_count - 1) / 2.0) / (pos_count * neg_count))
    else:
        roc_auc = 0.5

    # PR-AUC via trapezoidal integration over thresholds
    thresh_grid = np.linspace(0.0, 1.0, 50)
    prec_list = []
    rec_list = []
    for t in thresh_grid:
        pr = (p >= t).astype(np.int64)
        t_p = int(np.sum((pr == 1) & (y == 1)))
        f_p = int(np.sum((pr == 1) & (y == 0)))
        f_n = int(np.sum((pr == 0) & (y == 1)))
        prc = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 1.0
        rcl = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0.0
        prec_list.append(prc)
        rec_list.append(rcl)

    # Sort by recall
    sort_idx = np.argsort(rec_list)
    rec_sorted = np.array(rec_list)[sort_idx]
    prec_sorted = np.array(prec_list)[sort_idx]
    pr_auc = float(np.trapz(prec_sorted, rec_sorted)) if pos_count > 0 else 0.0

    diag = compute_calibration_diagnostics(y, p)

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(np.clip(roc_auc, 0.0, 1.0)),
        "pr_auc": float(np.clip(pr_auc, 0.0, 1.0)),
        "brier": float(diag.brier_score),
        "log_loss": float(diag.log_loss),
    }


class BaselineLadderRunner:
    """Executes the controlled baseline ladder (M0 to M6) on identical claim sets."""

    def __init__(
        self,
        logistic_model: LogisticEvidenceModel,
        prob_calibrator: ProbabilityCalibrator,
        epsilon_val: float = 0.20,
        budget_val: float = 0.50,
        coupling_lambda: float = 0.30,
        run_id: str = "run_ladder_m9c",
    ) -> None:
        self.logistic_model = logistic_model
        self.prob_calibrator = prob_calibrator
        self.epsilon_val = float(epsilon_val)
        self.budget_val = float(budget_val)
        self.coupling_lambda = float(coupling_lambda)
        self.run_id = run_id

    def run_image(
        self,
        image_records: List[Dict[str, Any]],
        candidate_edges: Optional[List[Tuple[int, int, float]]] = None,
        method: BaselineMethod = BaselineMethod.M6_GLOBAL_ROBUST,
        threshold: float = 0.5,
    ) -> List[ClaimPredictionResult]:
        """Execute inference for a specific method on all claims of a single image/tree."""
        t0 = time.perf_counter()
        n = len(image_records)
        if n == 0:
            return []

        image_id = str(image_records[0].get("image_id", "img_unknown"))
        split = str(image_records[0].get("split", "test"))

        # Extract features and base predictions
        p_raw = self.logistic_model.predict_proba(image_records)
        p_cal = self.prob_calibrator.calibrate(p_raw)
        thetas = probability_to_theta(p_cal)
        if np.isscalar(thetas):
            thetas = np.array([thetas])

        results: List[ClaimPredictionResult] = []

        if method == BaselineMethod.M0_DETECTOR:
            # Detector only: higher detector score -> lower hallucination probability
            for i, r in enumerate(image_records):
                t_i0 = time.perf_counter()
                d_score = float(r.get("detector_score", 0.5))
                # Heuristic mapping: p = 1.0 - d
                p_h = float(np.clip(1.0 - d_score, 0.001, 0.999))
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_h >= threshold else 0
                correct = (pred == y) if y is not None else None
                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="detector_only",
                    baseline_name=method.value,
                    topology="none",
                    coupling_strategy="none",
                    lambda_param=0.0,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=self.budget_val,
                    budget_ratio=self.budget_val / (n * self.epsilon_val) if n > 0 else 1.0,
                    nominal_posterior=p_h,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    runtime_ms=(time.perf_counter() - t_i0) * 1000.0,
                ))
            return results

        elif method == BaselineMethod.M1_CLIP:
            # CLIP only: higher clip similarity -> lower hallucination probability
            for i, r in enumerate(image_records):
                t_i0 = time.perf_counter()
                g_score = float(r.get("clip_score", 0.0))
                # Rescale [-1, 1] -> [0, 1] then invert
                g_norm = float(np.clip((g_score + 1.0) / 2.0, 0.0, 1.0))
                p_h = float(np.clip(1.0 - g_norm, 0.001, 0.999))
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_h >= threshold else 0
                correct = (pred == y) if y is not None else None
                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="clip_only",
                    baseline_name=method.value,
                    topology="none",
                    coupling_strategy="none",
                    lambda_param=0.0,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=self.budget_val,
                    budget_ratio=self.budget_val / (n * self.epsilon_val) if n > 0 else 1.0,
                    nominal_posterior=p_h,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    runtime_ms=(time.perf_counter() - t_i0) * 1000.0,
                ))
            return results

        elif method == BaselineMethod.M2_FUSION:
            # Calibrated logistic fusion without Ising field
            for i, r in enumerate(image_records):
                t_i0 = time.perf_counter()
                p_h = float(p_cal[i])
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_h >= threshold else 0
                correct = (pred == y) if y is not None else None
                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="combined",
                    baseline_name=method.value,
                    topology="none",
                    coupling_strategy="none",
                    lambda_param=0.0,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=self.budget_val,
                    budget_ratio=self.budget_val / (n * self.epsilon_val) if n > 0 else 1.0,
                    nominal_posterior=p_h,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    runtime_ms=(time.perf_counter() - t_i0) * 1000.0,
                ))
            return results

        elif method == BaselineMethod.M3_UNARY_ISOLATED:
            # Independent Ising unaries (J = 0)
            for i, r in enumerate(image_records):
                t_i0 = time.perf_counter()
                p_h = float(theta_to_probability(thetas[i]))
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_h >= threshold else 0
                correct = (pred == y) if y is not None else None
                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="combined",
                    baseline_name=method.value,
                    topology="independent",
                    coupling_strategy="J0",
                    lambda_param=0.0,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=self.budget_val,
                    budget_ratio=self.budget_val / (n * self.epsilon_val) if n > 0 else 1.0,
                    nominal_posterior=p_h,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    runtime_ms=(time.perf_counter() - t_i0) * 1000.0,
                ))
            return results

        # Graph-based methods: build TreeModel
        if candidate_edges is None or (n > 1 and len(candidate_edges) != n - 1):
            from src.calibration.coupling_calibrator import build_candidate_tree_edges
            candidate_edges = build_candidate_tree_edges(n, topology="chain")

        edges_list = [(int(u), int(v)) for u, v, _ in candidate_edges]
        j_dict = {(min(u, v), max(u, v)): self.coupling_lambda * float(s) for u, v, s in candidate_edges}
        eps_arr = np.full(n, self.epsilon_val, dtype=np.float64)

        model = TreeModel(
            num_nodes=n,
            theta=thetas.tolist(),
            edges=edges_list,
            coupling=j_dict,
            epsilon=eps_arr,
        )

        if method == BaselineMethod.M4_STANDARD_BP:
            res_bp = run_standard_bp(model)
            marginals = res_bp.marginals
            for i, r in enumerate(image_records):
                p_h = float(marginals[i])
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_h >= threshold else 0
                correct = (pred == y) if y is not None else None
                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="combined",
                    baseline_name=method.value,
                    topology="mst",
                    coupling_strategy="JWEIGHTED",
                    lambda_param=self.coupling_lambda,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=self.budget_val,
                    budget_ratio=self.budget_val / (n * self.epsilon_val) if n > 0 else 1.0,
                    nominal_posterior=p_h,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    runtime_ms=(time.perf_counter() - t0) * 1000.0 / n,
                ))
            return results

        elif method == BaselineMethod.M5_LOCAL_ROBUST:
            # Local box: budget is unconstrained B = sum(epsilons)
            step = 0.01
            b_box = float(np.sum(eps_arr))
            k_box = max(1, int(round(b_box / step)))
            b_box_grid = k_box * step

            for i, r in enumerate(image_records):
                t_i0 = time.perf_counter()
                res_box = solve_robust_bp(model, target_node=i, budget=b_box_grid, num_grid_steps=k_box)
                p_nom = float(res_box.nominal_marginal)
                l_val = float(res_box.lower_grid)
                u_val = float(res_box.upper_grid)
                w_val = float(u_val - l_val)
                crossing = bool(l_val < threshold < u_val)
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_nom >= threshold else 0
                correct = (pred == y) if y is not None else None
                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="combined",
                    baseline_name=method.value,
                    topology="mst",
                    coupling_strategy="JWEIGHTED",
                    lambda_param=self.coupling_lambda,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=b_box,
                    budget_ratio=1.0,
                    nominal_posterior=p_nom,
                    robust_lower=l_val,
                    robust_upper=u_val,
                    robust_width=w_val,
                    local_lower=l_val,
                    local_upper=u_val,
                    local_width=w_val,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    threshold_crossing=crossing,
                    runtime_ms=(time.perf_counter() - t_i0) * 1000.0,
                ))
            return results

        elif method == BaselineMethod.M6_GLOBAL_ROBUST:
            # Full method: shared budget B
            step = 0.01
            b_box = float(np.sum(eps_arr))
            if self.budget_val <= 1e-12:
                k_glob = 0
                b_glob = 0.0
            else:
                k_glob = max(1, int(round(self.budget_val / step)))
                b_glob = k_glob * step

            k_box = max(k_glob, int(round(b_box / step)))
            b_box_grid = k_box * step

            for i, r in enumerate(image_records):
                t_i0 = time.perf_counter()
                # 1. Global budget
                res_g = solve_robust_bp(model, target_node=i, budget=b_glob, num_grid_steps=k_glob)
                # 2. Local box benchmark on same node with aligned grid
                res_l = solve_robust_bp(model, target_node=i, budget=b_box_grid, num_grid_steps=k_box)

                p_nom = float(res_g.nominal_marginal)
                l_g = float(res_g.lower_grid)
                u_g = float(res_g.upper_grid)
                w_g = float(u_g - l_g)

                l_l = float(res_l.lower_grid)
                u_l = float(res_l.upper_grid)
                w_l = float(u_l - l_l)

                crossing = bool(l_g < threshold < u_g)
                y = int(r["label"]) if "label" in r and r["label"] is not None else None
                pred = 1 if p_nom >= threshold else 0
                correct = (pred == y) if y is not None else None

                results.append(ClaimPredictionResult(
                    run_id=self.run_id,
                    image_id=image_id,
                    claim_id=r["claim_id"],
                    split=split,
                    graph_size=n,
                    evidence_variant="combined",
                    baseline_name=method.value,
                    topology="mst",
                    coupling_strategy="JWEIGHTED",
                    lambda_param=self.coupling_lambda,
                    theta=float(thetas[i]),
                    epsilon=self.epsilon_val,
                    budget=self.budget_val,
                    budget_ratio=self.budget_val / b_box if b_box > 0 else 1.0,
                    nominal_posterior=p_nom,
                    robust_lower=l_g,
                    robust_upper=u_g,
                    robust_width=w_g,
                    local_lower=l_l,
                    local_upper=u_l,
                    local_width=w_l,
                    ground_truth=y,
                    prediction=pred,
                    correct=correct,
                    threshold_crossing=crossing,
                    runtime_ms=(time.perf_counter() - t_i0) * 1000.0,
                ))
            return results

        return results

    def run_all_methods_on_records(
        self,
        records: List[Dict[str, Any]],
        tree_edges_by_image: Optional[Dict[str, List[Tuple[int, int, float]]]] = None,
        methods: Optional[List[BaselineMethod]] = None,
        threshold: float = 0.5,
    ) -> Dict[str, List[ClaimPredictionResult]]:
        """Run each baseline method over all images in records and return results per method."""
        eval_methods = methods or list(BaselineMethod)
        tree_edges = tree_edges_by_image or {}

        # Group records by image_id
        img_groups: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            img_id = str(r.get("image_id", "default_img"))
            img_groups.setdefault(img_id, []).append(r)

        output: Dict[str, List[ClaimPredictionResult]] = {m.value: [] for m in eval_methods}

        for m in eval_methods:
            for img_id, img_recs in img_groups.items():
                edges = tree_edges.get(img_id, [])
                res_list = self.run_image(img_recs, candidate_edges=edges, method=m, threshold=threshold)
                output[m.value].extend(res_list)

        return output

    def evaluate_method_summary(
        self,
        results: List[ClaimPredictionResult],
    ) -> MethodEvaluationSummary:
        """Compute aggregated evaluation summary for a method's results."""
        if not results:
            return MethodEvaluationSummary(
                method_name="empty", evidence_variant="none", has_graph=False,
                uncertainty_type="NONE", has_global_budget=False, num_claims=0,
                num_images=0, accuracy=0.0, precision=0.0, recall=0.0, f1=0.0,
                roc_auc=0.5, pr_auc=0.0, brier_score=0.0, log_loss=0.0, mean_runtime_ms=0.0,
            )

        first = results[0]
        y_true = []
        y_prob = []
        widths = []
        crossings = 0
        runtimes = []
        distinct_images = set()

        for r in results:
            distinct_images.add(r.image_id)
            runtimes.append(r.runtime_ms)
            if r.ground_truth is not None:
                y_true.append(r.ground_truth)
                y_prob.append(r.nominal_posterior)
            if r.robust_width is not None:
                widths.append(r.robust_width)
            if r.threshold_crossing:
                crossings += 1

        y_arr = np.array(y_true, dtype=np.int64) if y_true else np.array([0, 1])
        p_arr = np.array(y_prob, dtype=np.float64) if y_prob else np.array([0.5, 0.5])
        class_metrics = compute_binary_classification_metrics(y_arr, p_arr)

        has_graph = first.baseline_name in (
            BaselineMethod.M4_STANDARD_BP.value,
            BaselineMethod.M5_LOCAL_ROBUST.value,
            BaselineMethod.M6_GLOBAL_ROBUST.value,
        )
        has_b = first.baseline_name == BaselineMethod.M6_GLOBAL_ROBUST.value
        unc_type = "GLOBAL" if has_b else ("LOCAL" if first.baseline_name == BaselineMethod.M5_LOCAL_ROBUST.value else "NONE")

        return MethodEvaluationSummary(
            method_name=first.baseline_name,
            evidence_variant=first.evidence_variant,
            has_graph=has_graph,
            uncertainty_type=unc_type,
            has_global_budget=has_b,
            num_claims=len(results),
            num_images=len(distinct_images),
            accuracy=class_metrics["accuracy"],
            precision=class_metrics["precision"],
            recall=class_metrics["recall"],
            f1=class_metrics["f1"],
            roc_auc=class_metrics["roc_auc"],
            pr_auc=class_metrics["pr_auc"],
            brier_score=class_metrics["brier"],
            log_loss=class_metrics["log_loss"],
            mean_runtime_ms=float(np.mean(runtimes)) if runtimes else 0.0,
            mean_width=float(np.mean(widths)) if widths else None,
            threshold_crossing_rate=float(crossings / len(results)) if results else 0.0,
        )


def build_baseline_comparison_table(
    summaries: List[MethodEvaluationSummary],
    mode: str = "DEVELOPMENT",
) -> str:
    """Generate Markdown comparison table across baseline ladder M0 through M6."""
    lines = [
        f"### Baseline Ladder Comparison Table ({mode})",
        "",
        "| Method | Evidence | Graph | Uncertainty | Global Budget | Acc | Prec | Rec | F1 | ROC-AUC | PR-AUC | Brier | LogLoss | Mean Width | Runtime (ms) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for s in summaries:
        g_str = "YES" if s.has_graph else "NO"
        b_str = "YES" if s.has_global_budget else "NO"
        w_str = f"{s.mean_width:.3f}" if s.mean_width is not None else "N/A"
        lines.append(
            f"| `{s.method_name}` | {s.evidence_variant} | {g_str} | {s.uncertainty_type} | {b_str} | "
            f"{s.accuracy:.3f} | {s.precision:.3f} | {s.recall:.3f} | {s.f1:.3f} | {s.roc_auc:.3f} | "
            f"{s.pr_auc:.3f} | {s.brier_score:.3f} | {s.log_loss:.3f} | {w_str} | {s.mean_runtime_ms:.1f} |"
        )
    return "\n".join(lines)


def build_component_contribution_table(
    summaries_by_method: Dict[str, MethodEvaluationSummary],
) -> str:
    """Generate component contribution transition table (M0->M2, M2->M3, M3->M4, M4->M5, M5->M6)."""
    transitions = [
        ("M0 -> M2", "Evidence Fusion (Detector + CLIP)", BaselineMethod.M0_DETECTOR.value, BaselineMethod.M2_FUSION.value),
        ("M2 -> M3", "Ising Field Representation (J=0)", BaselineMethod.M2_FUSION.value, BaselineMethod.M3_UNARY_ISOLATED.value),
        ("M3 -> M4", "Graph Coupling Effect (Standard BP)", BaselineMethod.M3_UNARY_ISOLATED.value, BaselineMethod.M4_STANDARD_BP.value),
        ("M4 -> M5", "Local Robustness Addition (Box Bounds)", BaselineMethod.M4_STANDARD_BP.value, BaselineMethod.M5_LOCAL_ROBUST.value),
        ("M5 -> M6", "Global Budget Coupling (L1 Shared Energy)", BaselineMethod.M5_LOCAL_ROBUST.value, BaselineMethod.M6_GLOBAL_ROBUST.value),
    ]

    lines = [
        "### Component Contribution Step-by-Step Table",
        "",
        "| Transition | Conceptual Component | Delta F1 | Delta Brier | Delta LogLoss | Delta ROC-AUC | Interval Width Change |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for label, desc, m_src, m_dst in transitions:
        s1 = summaries_by_method.get(m_src)
        s2 = summaries_by_method.get(m_dst)
        if s1 and s2:
            d_f1 = s2.f1 - s1.f1
            d_brier = s2.brier_score - s1.brier_score
            d_nll = s2.log_loss - s1.log_loss
            d_auc = s2.roc_auc - s1.roc_auc
            w1 = s1.mean_width or 0.0
            w2 = s2.mean_width or 0.0
            d_w_str = f"{(w2 - w1):+.3f}" if (s1.mean_width is not None and s2.mean_width is not None) else ("Added Width" if s2.mean_width is not None else "None")
            lines.append(
                f"| `{label}` | {desc} | {d_f1:+.3f} | {d_brier:+.3f} | {d_nll:+.3f} | {d_auc:+.3f} | {d_w_str} |"
            )
        else:
            lines.append(f"| `{label}` | {desc} | N/A | N/A | N/A | N/A | N/A |")

    return "\n".join(lines)
