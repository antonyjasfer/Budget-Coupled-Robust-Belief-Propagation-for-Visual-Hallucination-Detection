"""Principled calibration of the global perturbation budget B and uncertainty-set inclusion diagnostics.

The global budget B is calibrated from empirical image-level aggregate perturbation sums:
    S = sum_{j in image} |theta_{j,corrupt} - theta_{j,clean}|
on the CALIBRATION split.

Also computes uncertainty-set inclusion diagnostics:
- local-bound inclusion rate: fraction of perturbation vectors with |delta_i| <= eps_i for all i
- global-budget inclusion rate: fraction of perturbation vectors with sum_i |delta_i| <= B
- joint perturbation-set inclusion rate: fraction satisfying both simultaneously
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class BudgetCalibrationResult:
    """Calibrated budget parameters and distribution provenance."""

    quantile: float
    chosen_budget: float
    sample_size: int
    image_sums_summary: Dict[str, float] = field(default_factory=dict)
    normalized_ratio_rho: Optional[float] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerturbationSetDiagnostics:
    """Diagnostics evaluating what fraction of observed visual evidence shifts

    fall inside the uncertainty set U(B, epsilon).
    """

    local_bound_inclusion_rate: float
    global_budget_inclusion_rate: float
    joint_inclusion_rate: float
    n_samples: int
    budget_tested: float = 0.0
    epsilon_tested_mean: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward compatibility alias
UncertaintySetInclusionMetrics = PerturbationSetDiagnostics


class BudgetCalibrator:
    """Fits empirical global budget B from image-level aggregate shift distributions."""

    def __init__(
        self,
        default_quantile: float = 0.90,
        min_budget: float = 0.10,
        quantile: Optional[float] = None,
    ) -> None:
        q = quantile if quantile is not None else default_quantile
        self.default_quantile = float(np.clip(q, 0.0, 1.0))
        self.min_budget = float(min_budget)
        self.observed_sums: List[float] = []
        self.cal_image_ids: List[str] = []
        self.provenance: Dict[str, Any] = {}

    def fit_residuals(
        self,
        residuals: List[Union[Dict[str, Any], float]],
        cal_image_ids: Optional[List[str]] = None,
    ) -> "BudgetCalibrator":
        """Fit empirical budget distribution from a list of residuals.

        If list of dicts: aggregates shift residuals by (image_id, condition).
        If list of floats: treats directly as image aggregate sums S.
        """
        if not residuals:
            self.observed_sums = [0.5493]
            self.provenance = {"status": "fallback_empty_residuals"}
            return self

        if isinstance(residuals[0], dict):
            # Group by (image_id, condition)
            grouped: Dict[Tuple[str, str], float] = {}
            for r in residuals:
                img_id = str(r.get("image_id", "default_img"))
                cond = str(r.get("condition", "default_cond"))
                shift = float(r.get("residual", 0.0))
                key = (img_id, cond)
                grouped[key] = grouped.get(key, 0.0) + shift
            self.observed_sums = list(grouped.values())
        else:
            self.observed_sums = [float(s) for s in residuals]

        self.cal_image_ids = cal_image_ids or []
        self.provenance = {
            "num_observed_sums": len(self.observed_sums),
            "mean_sum": float(np.mean(self.observed_sums)),
            "median_sum": float(np.median(self.observed_sums)),
        }
        return self

    def fit_from_image_sums(
        self,
        image_sums: List[float],
        cal_image_ids: Optional[List[str]] = None,
        total_epsilons: Optional[List[float]] = None,
    ) -> BudgetCalibrationResult:
        """Calibrate budget B as a chosen quantile of image aggregate perturbation sums S."""
        self.fit_residuals(image_sums, cal_image_ids=cal_image_ids)
        chosen_b = self.get_budget(self.default_quantile)
        rho = self.get_normalized_budget(total_epsilons, budget=chosen_b) if total_epsilons else None

        summary = {
            "mean": float(np.mean(self.observed_sums)),
            "median": float(np.median(self.observed_sums)),
            "std": float(np.std(self.observed_sums)) if len(self.observed_sums) > 1 else 0.0,
            "max": float(np.max(self.observed_sums)),
        }
        return BudgetCalibrationResult(
            quantile=self.default_quantile,
            chosen_budget=chosen_b,
            sample_size=len(self.observed_sums),
            image_sums_summary=summary,
            normalized_ratio_rho=rho,
            provenance=self.provenance,
        )

    def get_budget(self, quantile: Optional[float] = None) -> float:
        """Get calibrated budget for a specific quantile (defaults to default_quantile)."""
        if not self.observed_sums:
            return 0.5493061443340549  # 0.5 * ln(3)
        q = self.default_quantile if quantile is None else float(np.clip(quantile, 0.0, 1.0))
        b = float(np.quantile(self.observed_sums, q))
        return float(max(self.min_budget, b))

    def get_normalized_budget(
        self,
        epsilons: Union[List[float], np.ndarray],
        budget: Optional[float] = None,
    ) -> float:
        """Compute rho = B / sum(epsilon_i)."""
        b = self.get_budget() if budget is None else float(budget)
        tot_eps = float(np.sum(epsilons))
        if tot_eps < 1e-9:
            return 1.0
        return float(b / tot_eps)

    def compute_inclusion_rate(
        self,
        observed_perturbations: List[np.ndarray],
        epsilon_vector: np.ndarray,
        budget: float,
    ) -> PerturbationSetDiagnostics:
        """Evaluate empirical perturbation-set inclusion rates:

        - local bound: |delta_i| <= eps_i for all i
        - global budget: sum_i |delta_i| <= B
        - joint: both hold simultaneously
        """
        N = len(observed_perturbations)
        if N == 0:
            return PerturbationSetDiagnostics(
                local_bound_inclusion_rate=1.0,
                global_budget_inclusion_rate=1.0,
                joint_inclusion_rate=1.0,
                n_samples=0,
                budget_tested=budget,
                epsilon_tested_mean=float(np.mean(epsilon_vector)) if len(epsilon_vector) > 0 else 0.0,
            )

        local_passes = 0
        budget_passes = 0
        joint_passes = 0

        for d_vec in observed_perturbations:
            abs_d = np.abs(np.asarray(d_vec, dtype=np.float64))
            local_ok = bool(np.all(abs_d <= epsilon_vector + 1e-7))
            budget_ok = bool(np.sum(abs_d) <= budget + 1e-7)

            if local_ok:
                local_passes += 1
            if budget_ok:
                budget_passes += 1
            if local_ok and budget_ok:
                joint_passes += 1

        return PerturbationSetDiagnostics(
            local_bound_inclusion_rate=float(local_passes / N),
            global_budget_inclusion_rate=float(budget_passes / N),
            joint_inclusion_rate=float(joint_passes / N),
            n_samples=N,
            budget_tested=float(budget),
            epsilon_tested_mean=float(np.mean(epsilon_vector)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_quantile": self.default_quantile,
            "min_budget": self.min_budget,
            "chosen_budget": self.get_budget(),
            "sample_size": len(self.observed_sums),
            "provenance": self.provenance,
        }


def evaluate_uncertainty_set_inclusion(
    observed_deltas: List[np.ndarray],
    epsilons: List[np.ndarray],
    budget: float,
) -> PerturbationSetDiagnostics:
    """Helper evaluating inclusion rates across per-image perturbation lists."""
    cal = BudgetCalibrator()
    if not observed_deltas:
        return cal.compute_inclusion_rate([], np.array([]), budget)
    # Average epsilons across list for representative diagnostic
    mean_eps = np.mean(np.array(epsilons), axis=0) if epsilons else np.array([0.5])
    return cal.compute_inclusion_rate(observed_deltas, mean_eps, budget)
