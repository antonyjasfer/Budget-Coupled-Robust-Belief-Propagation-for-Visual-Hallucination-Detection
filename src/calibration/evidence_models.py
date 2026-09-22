"""Evidence feature extraction, logistic evidence fusion baseline, and probability calibration.

Supports:
1. Feature interfaces: Detector-only, CLIP-only, Combined, Combined with interaction.
2. L2-regularized logistic regression fit on TRAIN and tuned on VALIDATION.
3. Probability calibration (Platt scaling / Isotonic regression).
4. Full calibration diagnostics (Brier score, NLL / Log loss, ECE).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class EvidenceFeatureType(str, Enum):
    DETECTOR_ONLY = "detector_only"
    CLIP_ONLY = "clip_only"
    COMBINED = "combined"
    COMBINED_INTERACTION = "combined_interaction"


class CalibrationMethod(str, Enum):
    NONE = "none"
    UNWEIGHTED = "uncalibrated"
    PLATT = "platt"
    ISOTONIC = "isotonic"


@dataclass
class CalibrationMetrics:
    """Diagnostic metrics for probability calibration quality."""

    brier_score: float
    log_loss: float
    ece: float
    sample_size: int
    reliability_bins: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_evidence_features(
    data_or_detector: Union[List[Dict[str, Any]], np.ndarray],
    clip_scores: Optional[np.ndarray] = None,
    feature_type: Union[str, EvidenceFeatureType] = EvidenceFeatureType.COMBINED,
) -> Union[Tuple[np.ndarray, List[str]], np.ndarray]:
    """Extract numeric features from either records list or arrays.

    If records list is passed: returns (X, feature_names).
    If detector & clip arrays passed: returns X.
    """
    ft_str = str(feature_type.value if isinstance(feature_type, EvidenceFeatureType) else feature_type).lower()

    if isinstance(data_or_detector, list):
        records = data_or_detector
        d_list = []
        c_list = []
        for r in records:
            d_val = r.get("detector_score", r.get("raw_detector_score", 0.5))
            c_val = r.get("clip_score", r.get("raw_clip_score", 0.0))
            d_list.append(float(d_val))
            c_list.append(float(c_val))
        d = np.array(d_list, dtype=np.float64)
        c = np.array(c_list, dtype=np.float64)
        return_tuple = True
    else:
        d = np.asarray(data_or_detector, dtype=np.float64)
        c = np.asarray(clip_scores, dtype=np.float64) if clip_scores is not None else np.zeros_like(d)
        return_tuple = False

    N = len(d)
    if ft_str == "detector_only":
        X = d.reshape(N, 1)
        names = ["detector_score"]
    elif ft_str == "clip_only":
        X = c.reshape(N, 1)
        names = ["clip_score"]
    elif ft_str == "combined_interaction":
        inter = d * c
        X = np.column_stack([d, c, inter])
        names = ["detector_score", "clip_score", "detector_x_clip"]
    else:  # combined
        X = np.column_stack([d, c])
        names = ["detector_score", "clip_score"]

    if return_tuple:
        return X, names
    return X


def compute_calibration_diagnostics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Compute Brier score, log loss, and Expected Calibration Error (ECE)."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    N = len(y)

    if N == 0:
        return CalibrationMetrics(brier_score=0.0, log_loss=0.0, ece=0.0, sample_size=0)

    # 1. Brier score
    brier = float(np.mean((p - y) ** 2))

    # 2. Log loss / Binary Cross Entropy
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))

    # 3. Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability_bins = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        mask = (p >= bin_lower) & (p <= bin_upper) if i == n_bins - 1 else (p >= bin_lower) & (p < bin_upper)
        bin_count = int(np.sum(mask))

        if bin_count > 0:
            avg_conf = float(np.mean(p[mask]))
            avg_acc = float(np.mean(y[mask]))
            bin_err = abs(avg_acc - avg_conf)
            ece += (bin_count / N) * bin_err
            reliability_bins.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": bin_count,
                "avg_confidence": avg_conf,
                "empirical_accuracy": avg_acc,
                "calibration_gap": bin_err,
            })

    return CalibrationMetrics(
        brier_score=brier,
        log_loss=nll,
        ece=float(ece),
        sample_size=N,
        reliability_bins=reliability_bins,
    )


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Helper returning calibration metrics as a dictionary."""
    diag = compute_calibration_diagnostics(y_true, y_prob, n_bins=n_bins)
    return diag.to_dict()


class LogisticEvidenceModel:
    """Logistic evidence fusion baseline.

    Predicts p_i = P(H_i = HALLUCINATED | evidence) using TRAIN data.
    """

    def __init__(
        self,
        feature_type: Union[str, EvidenceFeatureType] = EvidenceFeatureType.COMBINED,
        c_grid: Optional[List[float]] = None,
    ) -> None:
        if isinstance(feature_type, EvidenceFeatureType):
            self.feature_type = feature_type.value
        else:
            self.feature_type = str(feature_type).lower()
        self.c_grid = c_grid or [0.01, 0.1, 1.0, 10.0, 100.0]
        self.best_c: float = 1.0
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.scaler_mean_: Optional[np.ndarray] = None
        self.scaler_scale_: Optional[np.ndarray] = None
        self.is_fitted: bool = False
        self.feature_names_: List[str] = []
        self.provenance: Dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self.is_fitted

    @property
    def coefficients(self) -> Optional[List[float]]:
        if self.coef_ is None:
            return None
        return self.coef_.ravel().tolist()

    @property
    def intercept(self) -> float:
        return float(self.intercept_)

    @property
    def model_hash(self) -> str:
        d = {
            "feature_type": self.feature_type,
            "best_c": self.best_c,
            "coef": self.coefficients,
            "intercept": self.intercept,
        }
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def fit(
        self,
        train_data: Union[List[Dict[str, Any]], np.ndarray],
        y_train: Optional[np.ndarray] = None,
        val_records: Optional[List[Dict[str, Any]]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        train_ids: Optional[List[str]] = None,
        val_ids: Optional[List[str]] = None,
    ) -> "LogisticEvidenceModel":
        """Fit model on TRAIN data with regularization selection on VALIDATION."""
        if isinstance(train_data, list):
            X_tr, names = extract_evidence_features(train_data, feature_type=self.feature_type)
            self.feature_names_ = names
            y_tr = np.array([float(r.get("label", 0)) for r in train_data], dtype=np.float64)
            tr_ids = [r.get("claim_id", "") for r in train_data]
        else:
            X_tr = np.asarray(train_data, dtype=np.float64)
            y_tr = np.asarray(y_train, dtype=np.float64)
            self.feature_names_ = [f"f_{i}" for i in range(X_tr.shape[1])]
            tr_ids = train_ids or []

        if val_records is not None:
            X_v, _ = extract_evidence_features(val_records, feature_type=self.feature_type)
            y_v = np.array([float(r.get("label", 0)) for r in val_records], dtype=np.float64)
            v_ids = [r.get("claim_id", "") for r in val_records]
        elif X_val is not None and y_val is not None:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64)
            v_ids = val_ids or []
        else:
            X_v = None
            y_v = None
            v_ids = []

        N, D = X_tr.shape
        unique_classes = np.unique(y_tr)

        # Standardize features
        self.scaler_mean_ = np.mean(X_tr, axis=0)
        self.scaler_scale_ = np.std(X_tr, axis=0)
        self.scaler_scale_[self.scaler_scale_ < 1e-6] = 1.0

        X_tr_scaled = (X_tr - self.scaler_mean_) / self.scaler_scale_

        # Fallback for degenerate single-class training (e.g. development fixtures)
        if len(unique_classes) < 2 or N < 4:
            self.coef_ = -1.5 * np.ones((1, D))
            self.intercept_ = 0.0
            self.best_c = 1.0
            self.is_fitted = True
            self.provenance = {
                "fit_mode": "fallback_deterministic",
                "reason": "insufficient_train_class_diversity",
                "sample_size": N,
                "classes": unique_classes.tolist(),
            }
            return self

        best_score = np.inf
        best_model = None
        chosen_c = self.c_grid[0]

        X_v_scaled = (X_v - self.scaler_mean_) / self.scaler_scale_ if X_v is not None and len(X_v) > 0 else None

        for c_val in self.c_grid:
            clf = LogisticRegression(C=c_val, penalty="l2", solver="lbfgs", max_iter=500, random_state=42)
            clf.fit(X_tr_scaled, y_tr)

            if X_v_scaled is not None and y_v is not None and len(y_v) > 0:
                p_v = clf.predict_proba(X_v_scaled)[:, 1]
                diag = compute_calibration_diagnostics(y_v, p_v)
                eval_score = diag.brier_score
            else:
                p_tr = clf.predict_proba(X_tr_scaled)[:, 1]
                diag = compute_calibration_diagnostics(y_tr, p_tr)
                eval_score = diag.brier_score

            if eval_score < best_score:
                best_score = eval_score
                best_model = clf
                chosen_c = c_val

        self.best_c = chosen_c
        self.coef_ = best_model.coef_.copy()
        self.intercept_ = float(best_model.intercept_[0])
        self.is_fitted = True

        train_hash = hashlib.sha256(json.dumps(sorted(tr_ids)).encode("utf-8")).hexdigest()
        val_hash = hashlib.sha256(json.dumps(sorted(v_ids)).encode("utf-8")).hexdigest()

        self.provenance = {
            "fit_mode": "supervised_l2_logistic",
            "best_c": self.best_c,
            "best_val_score": float(best_score),
            "train_sample_size": N,
            "train_ids_hash": train_hash,
            "val_ids_hash": val_hash,
            "feature_type": self.feature_type,
            "model_hash": self.model_hash,
        }
        return self

    def predict_proba(self, data: Union[List[Dict[str, Any]], np.ndarray]) -> np.ndarray:
        """Predict probability of HALLUCINATED state."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        if isinstance(data, list):
            X, _ = extract_evidence_features(data, feature_type=self.feature_type)
        else:
            X = np.asarray(data, dtype=np.float64)

        X_scaled = (X - self.scaler_mean_) / self.scaler_scale_
        logits = X_scaled @ self.coef_.T + self.intercept_
        return expit(logits.ravel())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_type": self.feature_type,
            "best_c": self.best_c,
            "coef": self.coefficients,
            "intercept": self.intercept,
            "scaler_mean": self.scaler_mean_.tolist() if self.scaler_mean_ is not None else None,
            "scaler_scale": self.scaler_scale_.tolist() if self.scaler_scale_ is not None else None,
            "feature_names": self.feature_names_,
            "model_hash": self.model_hash,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogisticEvidenceModel":
        model = cls(feature_type=data["feature_type"], c_grid=[data.get("best_c", 1.0)])
        model.best_c = float(data.get("best_c", 1.0))
        if data.get("coef") is not None:
            model.coef_ = np.array(data["coef"]).reshape(1, -1)
        model.intercept_ = float(data.get("intercept", 0.0))
        if data.get("scaler_mean") is not None:
            model.scaler_mean_ = np.array(data["scaler_mean"])
        if data.get("scaler_scale") is not None:
            model.scaler_scale_ = np.array(data["scaler_scale"])
        model.feature_names_ = data.get("feature_names", [])
        model.provenance = data.get("provenance", {})
        model.is_fitted = True
        return model


class ProbabilityCalibrator:
    """Probability calibration layer wrapping logistic output.

    Fit strictly on CALIBRATION split.
    """

    def __init__(self, method: Union[str, CalibrationMethod] = CalibrationMethod.PLATT) -> None:
        m_str = str(method.value if isinstance(method, CalibrationMethod) else method).lower()
        if m_str in ("none", "uncalibrated"):
            self.method = "none"
        elif m_str == "isotonic":
            self.method = "isotonic"
        else:
            self.method = "platt"

        self.is_fitted: bool = False
        self.platt_model: Optional[LogisticRegression] = None
        self.isotonic_model: Optional[IsotonicRegression] = None
        self.provenance: Dict[str, Any] = {}

    @property
    def fitted(self) -> bool:
        return self.is_fitted

    def fit(
        self,
        uncalibrated_probs: np.ndarray,
        y_cal: np.ndarray,
        cal_ids: Optional[List[str]] = None,
    ) -> "ProbabilityCalibrator":
        """Fit calibration transform on CALIBRATION split data."""
        p = np.clip(np.asarray(uncalibrated_probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        y = np.asarray(y_cal, dtype=np.float64)
        N = len(y)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2 or N < 4:
            self.method = "none"
            self.is_fitted = True
            self.provenance = {
                "method": "none",
                "reason": "insufficient_calibration_samples_or_classes",
                "sample_size": N,
            }
            return self

        if self.method == "isotonic" and N < 20:
            self.method = "platt"

        if self.method == "platt":
            logits = (np.log(p) - np.log1p(-p)).reshape(N, 1)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            clf.fit(logits, y)
            self.platt_model = clf
            self.is_fitted = True
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.isotonic_model = iso
            self.is_fitted = True
        else:
            self.is_fitted = True

        cal_hash = hashlib.sha256(json.dumps(sorted(cal_ids or [])).encode("utf-8")).hexdigest()
        self.provenance = {
            "method": self.method,
            "sample_size": N,
            "calibration_ids_hash": cal_hash,
        }
        return self

    def calibrate(self, uncalibrated_probs: np.ndarray) -> np.ndarray:
        """Apply calibration transform."""
        p = np.clip(np.asarray(uncalibrated_probs, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        if not self.is_fitted or self.method == "none":
            return p

        if self.method == "platt" and self.platt_model is not None:
            logits = (np.log(p) - np.log1p(-p)).reshape(-1, 1)
            return self.platt_model.predict_proba(logits)[:, 1]
        elif self.method == "isotonic" and self.isotonic_model is not None:
            return self.isotonic_model.predict(p)
        return p

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "platt_coef": self.platt_model.coef_.tolist() if self.platt_model is not None else None,
            "platt_intercept": float(self.platt_model.intercept_[0]) if self.platt_model is not None else None,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProbabilityCalibrator":
        cal = cls(method=data.get("method", "platt"))
        cal.is_fitted = data.get("is_fitted", False)
        cal.provenance = data.get("provenance", {})
        if data.get("platt_coef") is not None and data.get("platt_intercept") is not None:
            clf = LogisticRegression()
            clf.coef_ = np.array(data["platt_coef"])
            clf.intercept_ = np.array([data["platt_intercept"]])
            clf.classes_ = np.array([0, 1])
            cal.platt_model = clf
        return cal
