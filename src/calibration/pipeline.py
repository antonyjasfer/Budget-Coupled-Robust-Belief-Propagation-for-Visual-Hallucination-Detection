"""End-to-end calibration and parameterization pipeline for robust Ising inference.

Orchestrates:
1. Strict split enforcement (TRAIN, VAL, CALIBRATION, TEST).
2. Evidence feature extraction (detector-only, clip-only, combined).
3. Non-PGM logistic baseline fitting on TRAIN with L2 hyperparameter selection on VAL.
4. Probability calibration (Platt/Isotonic) on CALIBRATION.
5. Evidence-to-theta mapping (theta = 0.5 * ln(p / (1-p))).
6. Epsilon calibration under image corruption residuals on CALIBRATION.
7. Budget B calibration under tree-level perturbation sums on CALIBRATION.
8. Empirical perturbation-set inclusion rate diagnostics (local, global, joint).
9. Attractive coupling lambda >= 0 selection on VAL.
10. ParameterBundle export with SHA-256 provenance tracking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.calibration.budget_calibrator import BudgetCalibrator, PerturbationSetDiagnostics
from src.calibration.bundle import ParameterBundle
from src.calibration.coupling_calibrator import (
    CouplingCalibrator,
    build_candidate_tree_edges,
)
from src.calibration.epsilon_calibrator import EpsilonCalibrator
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
    compute_calibration_metrics,
    extract_evidence_features,
)
from src.calibration.splits import SplitContract
from src.calibration.theta_mapping import probability_to_theta

logger = logging.getLogger(__name__)


class CalibrationPipeline:
    """Full parameterization and calibration pipeline for robust Ising hallucination detection."""

    def __init__(
        self,
        split_contract: SplitContract,
        feature_type: str = "combined",
        calibration_method: str = "platt",
        epsilon_quantile: float = 0.90,
        budget_quantile: float = 0.90,
        code_sha: str = "d3b7f57",
        dataset_hash: str = "coco_train2017_m6_canonical",
        notes: str = "Phase 9B principled parameterization",
    ) -> None:
        self.split_contract = split_contract
        self.feature_type = feature_type
        self.calibration_method = calibration_method
        self.epsilon_quantile = epsilon_quantile
        self.budget_quantile = budget_quantile
        self.code_sha = code_sha
        self.dataset_hash = dataset_hash
        self.notes = notes

        # Trained / calibrated components
        self.logistic_model: Optional[LogisticEvidenceModel] = None
        self.prob_calibrator: Optional[ProbabilityCalibrator] = None
        self.epsilon_calibrator: Optional[EpsilonCalibrator] = None
        self.budget_calibrator: Optional[BudgetCalibrator] = None
        self.coupling_calibrator: Optional[CouplingCalibrator] = None

        # Baselines storage
        self.detector_baseline: Optional[LogisticEvidenceModel] = None
        self.clip_baseline: Optional[LogisticEvidenceModel] = None

    def run_baselines(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fit detector-only, CLIP-only, and combined logistic baselines on TRAIN, selecting C on VAL."""
        # 1. Detector-only baseline
        m_det = LogisticEvidenceModel(feature_type="detector_only")
        m_det.fit(train_records, val_records=val_records)
        self.detector_baseline = m_det

        # 2. CLIP-only baseline
        m_clip = LogisticEvidenceModel(feature_type="clip_only")
        m_clip.fit(train_records, val_records=val_records)
        self.clip_baseline = m_clip

        # 3. Combined model
        m_comb = LogisticEvidenceModel(feature_type=self.feature_type)
        m_comb.fit(train_records, val_records=val_records)
        self.logistic_model = m_comb

        return {
            "detector_only": m_det.to_dict(),
            "clip_only": m_clip.to_dict(),
            "combined": m_comb.to_dict(),
        }

    def calibrate_probabilities(
        self,
        cal_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fit Platt/Isotonic probability calibration using CALIBRATION records only."""
        if self.logistic_model is None:
            raise RuntimeError("Logistic model must be fitted before probability calibration.")

        uncal_probs = self.logistic_model.predict_proba(cal_records)
        y_cal = np.array([float(r.get("label", 0)) for r in cal_records])

        calibrator = ProbabilityCalibrator(method=self.calibration_method)
        calibrator.fit(uncal_probs, y_cal)
        self.prob_calibrator = calibrator

        # Evaluate calibration metrics
        cal_probs = calibrator.calibrate(uncal_probs)
        metrics = compute_calibration_metrics(y_cal, cal_probs)

        return {
            "calibrator": calibrator.to_dict(),
            "calibration_metrics": metrics,
        }

    def calibrate_uncertainty_set(
        self,
        clean_cal_records: List[Dict[str, Any]],
        corrupt_cal_records_by_condition: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], PerturbationSetDiagnostics]:
        """Calibrate local bounds epsilon_i and global budget B from observed corruption shifts."""
        if self.logistic_model is None or self.prob_calibrator is None:
            raise RuntimeError("Evidence and probability calibrator must be fitted first.")

        # Compute clean thetas
        p_clean = self.prob_calibrator.calibrate(
            self.logistic_model.predict_proba(clean_cal_records)
        )
        theta_clean = probability_to_theta(p_clean)

        claim_ids = [r["claim_id"] for r in clean_cal_records]
        claim_categories = {r["claim_id"]: r.get("category", "object") for r in clean_cal_records}
        claim_image_ids = {r["claim_id"]: r.get("image_id", "default") for r in clean_cal_records}

        # Collect residuals
        residuals_list: List[Dict[str, Any]] = []
        observed_vectors: List[np.ndarray] = []
        observed_image_vectors: Dict[str, List[np.ndarray]] = {}

        for cond, c_records in corrupt_cal_records_by_condition.items():
            if len(c_records) != len(clean_cal_records):
                continue
            p_corrupt = self.prob_calibrator.calibrate(
                self.logistic_model.predict_proba(c_records)
            )
            theta_corrupt = probability_to_theta(p_corrupt)
            shifts = np.abs(theta_corrupt - theta_clean)
            observed_vectors.append(shifts)

            for cid, shift, clean_r in zip(claim_ids, shifts, clean_cal_records):
                residuals_list.append({
                    "claim_id": cid,
                    "image_id": clean_r.get("image_id", "default"),
                    "category": clean_r.get("category", "object"),
                    "condition": cond,
                    "residual": float(shift),
                })

        # Fit epsilon
        eps_cal = EpsilonCalibrator(
            method="global_quantile",
            default_quantile=self.epsilon_quantile,
        )
        eps_cal.fit_residuals(residuals_list)
        self.epsilon_calibrator = eps_cal

        # Fit budget B
        b_cal = BudgetCalibrator(default_quantile=self.budget_quantile)
        b_cal.fit_residuals(residuals_list)
        self.budget_calibrator = b_cal

        # Diagnostics: inclusion rates
        chosen_eps = eps_cal.get_epsilon()
        chosen_b = b_cal.get_budget()

        # Vector of epsilons
        eps_vector = np.full(len(clean_cal_records), chosen_eps)
        diagnostics = b_cal.compute_inclusion_rate(
            observed_perturbations=observed_vectors,
            epsilon_vector=eps_vector,
            budget=chosen_b,
        )

        return eps_cal.to_dict(), b_cal.to_dict(), diagnostics

    def calibrate_coupling(
        self,
        val_records: List[Dict[str, Any]],
        candidate_edges: List[Tuple[int, int, float]],
        root_index: int = 0,
    ) -> Dict[str, Any]:
        """Calibrate non-negative attractive coupling lambda >= 0 using VALIDATION split."""
        if self.logistic_model is None or self.prob_calibrator is None:
            raise RuntimeError("Unary evidence model must be fitted before coupling calibration.")

        val_probs = self.prob_calibrator.calibrate(
            self.logistic_model.predict_proba(val_records)
        )
        val_theta = probability_to_theta(val_probs)
        val_labels = np.array([float(r.get("label", 0)) for r in val_records])

        # Map binary 0/1 labels to Ising {-1, +1}
        # 0 (SUPPORTED) -> -1, 1 (HALLUCINATED) -> +1
        ising_labels = np.where(val_labels > 0.5, 1.0, -1.0)

        c_cal = CouplingCalibrator(objective="nll")
        best_lambda = c_cal.fit(
            val_thetas=val_theta,
            val_labels=ising_labels,
            candidate_edges=candidate_edges,
            root_index=root_index,
        )
        self.coupling_calibrator = c_cal
        return c_cal.to_dict()

    def build_bundle(
        self,
        mode: Optional[str] = None,
        test_records: Optional[List[Dict[str, Any]]] = None,
    ) -> ParameterBundle:
        """Assemble all fitted artifacts into a ParameterBundle with provenance hashes."""
        # Check sample sizes to determine mode
        n_train = len(self.split_contract.train_ids)
        is_final = n_train >= 100 and mode == "FINAL"
        chosen_mode = "FINAL" if is_final else "DEVELOPMENT"

        test_ids_list = [r["claim_id"] for r in (test_records or [])]
        test_hash = ParameterBundle().compute_checksum()[:16] if not test_ids_list else self.split_contract.get_hash(test_ids_list)

        bundle = ParameterBundle(
            version="1.0.0",
            mode=chosen_mode,
            theta_model=self.logistic_model.to_dict() if self.logistic_model else {},
            probability_calibration=self.prob_calibrator.to_dict() if self.prob_calibrator else {},
            epsilon_model=self.epsilon_calibrator.to_dict() if self.epsilon_calibrator else {},
            budget_model=self.budget_calibrator.to_dict() if self.budget_calibrator else {},
            coupling_model=self.coupling_calibrator.to_dict() if self.coupling_calibrator else {},
            topology_model={"default_topology": "minimum_spanning_tree"},
            train_ids_hash=self.split_contract.train_hash,
            validation_ids_hash=self.split_contract.validation_hash,
            calibration_ids_hash=self.split_contract.calibration_hash,
            test_ids_hash=test_hash,
            dataset_hash=self.dataset_hash,
            code_sha=self.code_sha,
            notes=self.notes if chosen_mode == "FINAL" else f"{self.notes} (DEVELOPMENT PARAMETERIZATION - NOT FINAL CALIBRATION)",
        )
        return bundle
