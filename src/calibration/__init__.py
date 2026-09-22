"""Principled evidence calibration and robust parameterization module for hallucination detection."""

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
from src.calibration.pipeline import CalibrationPipeline
from src.calibration.splits import (
    SplitContract,
    SplitLeakageError,
    SplitRole,
    create_deterministic_splits,
)
from src.calibration.theta_mapping import (
    probability_to_theta,
    theta_to_probability,
    verify_roundtrip,
)

__all__ = [
    "probability_to_theta",
    "theta_to_probability",
    "verify_roundtrip",
    "SplitContract",
    "SplitRole",
    "SplitLeakageError",
    "create_deterministic_splits",
    "extract_evidence_features",
    "LogisticEvidenceModel",
    "ProbabilityCalibrator",
    "compute_calibration_metrics",
    "EpsilonCalibrator",
    "BudgetCalibrator",
    "PerturbationSetDiagnostics",
    "CouplingCalibrator",
    "build_candidate_tree_edges",
    "ParameterBundle",
    "CalibrationPipeline",
]
