"""Principled estimation and calibration of unary perturbation limits epsilon_i.

Grounded in empirical visual evidence instability:
Measures the distribution of unary field shifts under standardized corruptions:
    r_ic = |theta_{i,c} - theta_{i,clean}|
on the CALIBRATION data split.

Supports:
- Method A: Global quantile (e.g. q = 0.80, 0.90, 0.95).
- Method B: Evidence-conditioned uncertainty (e.g. higher uncertainty when clean_p is near 0.5).
- Method C: Category-conditioned quantile with robust global fallbacks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np


class EpsilonMethod(str, Enum):
    GLOBAL_QUANTILE = "global_quantile"
    EVIDENCE_CONDITIONED = "evidence_conditioned"
    CATEGORY_CONDITIONED = "category_conditioned"


@dataclass
class EpsilonCalibrationResult:
    """Summary and provenance of calibrated epsilon parameters."""

    method: str
    quantile: float
    global_epsilon: float
    category_epsilons: Dict[str, float] = field(default_factory=dict)
    min_epsilon: float = 0.01
    sample_size: int = 0
    residuals_summary: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpsilonCalibrator:
    """Fits epsilon estimators from calibration split residual distributions."""

    def __init__(
        self,
        method: Union[str, EpsilonMethod] = EpsilonMethod.GLOBAL_QUANTILE,
        default_quantile: float = 0.90,
        min_epsilon: float = 0.01,
        min_category_samples: int = 5,
        quantile: Optional[float] = None,
        min_samples_per_category: Optional[int] = None,
    ) -> None:
        if isinstance(method, EpsilonMethod):
            self.method = method.value
        else:
            self.method = str(method).lower()

        q = quantile if quantile is not None else default_quantile
        self.default_quantile = float(np.clip(q, 0.0, 1.0))
        self.min_epsilon = float(min_epsilon)
        min_cat = min_samples_per_category if min_samples_per_category is not None else min_category_samples
        self.min_category_samples = int(min_cat)

        self.residuals_: List[float] = []
        self.categories_: List[str] = []
        self.clean_p_: List[float] = []
        self.category_residuals_: Dict[str, List[float]] = {}
        self.provenance: Dict[str, Any] = {}

    def fit_residuals(
        self,
        residuals: List[Union[Dict[str, Any], float]],
        categories: Optional[List[str]] = None,
        cal_ids: Optional[List[str]] = None,
    ) -> "EpsilonCalibrator":
        """Fit epsilon estimators from residuals list.

        Accepts either list of dicts or list of floats.
        """
        self.residuals_ = []
        self.categories_ = []
        self.clean_p_ = []
        self.category_residuals_ = {}

        if not residuals:
            self.residuals_ = [0.5493]
            self.provenance = {"status": "fallback_empty_residuals"}
            return self

        if isinstance(residuals[0], dict):
            for r in residuals:
                res_val = float(r.get("residual", 0.0))
                cat_val = str(r.get("category", "object")).lower()
                p_val = float(r.get("clean_p", 0.5))
                self.residuals_.append(res_val)
                self.categories_.append(cat_val)
                self.clean_p_.append(p_val)
                self.category_residuals_.setdefault(cat_val, []).append(res_val)
        else:
            self.residuals_ = [float(r) for r in residuals]
            if categories and len(categories) == len(residuals):
                self.categories_ = [str(c).lower() for c in categories]
                for r_val, c_val in zip(self.residuals_, self.categories_):
                    self.category_residuals_.setdefault(c_val, []).append(r_val)

        self.provenance = {
            "method": self.method,
            "sample_size": len(self.residuals_),
            "mean_residual": float(np.mean(self.residuals_)),
            "num_categories": len(self.category_residuals_),
        }
        return self

    def fit_from_residuals(
        self,
        residuals: List[float],
        categories: Optional[List[str]] = None,
        cal_ids: Optional[List[str]] = None,
    ) -> EpsilonCalibrationResult:
        """Compatibility method returning EpsilonCalibrationResult."""
        self.fit_residuals(residuals, categories=categories, cal_ids=cal_ids)
        r_arr = np.asarray(self.residuals_, dtype=np.float64)
        global_eps = self.get_epsilon(quantile=self.default_quantile)

        cat_eps = {}
        for cat in self.category_residuals_:
            cat_eps[cat] = self.get_epsilon(category=cat, quantile=self.default_quantile)

        summary = {
            "mean": float(np.mean(r_arr)),
            "median": float(np.median(r_arr)),
            "std": float(np.std(r_arr)) if len(r_arr) > 1 else 0.0,
            "max": float(np.max(r_arr)),
        }
        return EpsilonCalibrationResult(
            method=self.method,
            quantile=self.default_quantile,
            global_epsilon=global_eps,
            category_epsilons=cat_eps,
            min_epsilon=self.min_epsilon,
            sample_size=len(self.residuals_),
            residuals_summary=summary,
            provenance=self.provenance,
        )

    def get_epsilon(
        self,
        quantile: Optional[float] = None,
        category: Optional[str] = None,
        clean_p: Optional[float] = None,
        detector_score: Optional[float] = None,
        clip_score: Optional[float] = None,
    ) -> float:
        """Get calibrated epsilon for claim/category/evidence condition."""
        if not self.residuals_:
            return 0.5493061443340549  # 0.5 * ln(3)

        q = self.default_quantile if quantile is None else float(np.clip(quantile, 0.0, 1.0))
        global_eps = float(np.quantile(self.residuals_, q))
        global_eps = float(max(self.min_epsilon, global_eps))

        if self.method == "category_conditioned" and category:
            cat_clean = str(category).lower()
            cat_list = self.category_residuals_.get(cat_clean, [])
            if len(cat_list) >= self.min_category_samples:
                cat_val = float(np.quantile(cat_list, q))
                return float(max(self.min_epsilon, cat_val))
            # Fallback to global
            return global_eps

        elif self.method == "evidence_conditioned":
            # Higher uncertainty when clean_p is close to 0.5 (maximum entropy)
            p = 0.5 if clean_p is None else float(np.clip(clean_p, 0.0, 1.0))
            entropy_factor = 1.0 - 2.0 * abs(p - 0.5)  # 1.0 at p=0.5, 0.0 at p=0 or 1
            cond_eps = global_eps * (0.5 + 0.5 * entropy_factor)
            return float(max(self.min_epsilon, cond_eps))

        return global_eps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "default_quantile": self.default_quantile,
            "min_epsilon": self.min_epsilon,
            "min_category_samples": self.min_category_samples,
            "global_epsilon": self.get_epsilon(),
            "sample_size": len(self.residuals_),
            "provenance": self.provenance,
        }
