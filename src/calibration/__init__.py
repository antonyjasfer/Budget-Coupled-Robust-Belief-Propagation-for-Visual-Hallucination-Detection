"""Principled evidence calibration, robust parameterization, and ablation framework."""

from src.calibration.ablation_study import (
    FactorialConditionResult,
    GraphCoverageAudit,
    ResearchDecisionGate,
    audit_dataset_graph_coverage,
    compute_nominal_vs_robust_width_correlation,
    evaluate_decision_gate,
    generate_baseline_graph_audit_report,
    perform_image_level_bootstrap,
    run_graph_necessity_test,
    run_graph_uncertainty_factorial_ablation,
    run_local_vs_global_budget_ablation,
    run_star_root_sensitivity_test,
    run_topology_ablation,
)
from src.calibration.budget_calibrator import BudgetCalibrator, PerturbationSetDiagnostics
from src.calibration.bundle import ParameterBundle
from src.calibration.coupling_calibrator import (
    CouplingCalibrator,
    build_candidate_tree_edges,
)
from src.calibration.edge_audit import (
    EdgeValidityAuditReport,
    PairwiseDependenceMetrics,
    audit_edge_validity,
    compute_pairwise_state_dependence,
    generate_edge_validity_report,
)
from src.calibration.epsilon_calibrator import EpsilonCalibrator
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
    compute_calibration_metrics,
    extract_evidence_features,
)
from src.calibration.ladder import (
    BaselineLadderRunner,
    BaselineMethod,
    ClaimPredictionResult,
    MethodEvaluationSummary,
    build_baseline_comparison_table,
    build_component_contribution_table,
    compute_binary_classification_metrics,
)
from src.calibration.pipeline import CalibrationPipeline
from src.calibration.splits import (
    SplitContract,
    SplitLeakageError,
    SplitRole,
    create_deterministic_splits,
    create_image_level_deterministic_splits,
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
    "create_image_level_deterministic_splits",
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
    "BaselineMethod",
    "ClaimPredictionResult",
    "MethodEvaluationSummary",
    "BaselineLadderRunner",
    "compute_binary_classification_metrics",
    "build_baseline_comparison_table",
    "build_component_contribution_table",
    "PairwiseDependenceMetrics",
    "EdgeValidityAuditReport",
    "compute_pairwise_state_dependence",
    "audit_edge_validity",
    "generate_edge_validity_report",
    "GraphCoverageAudit",
    "FactorialConditionResult",
    "ResearchDecisionGate",
    "audit_dataset_graph_coverage",
    "run_graph_necessity_test",
    "run_topology_ablation",
    "run_star_root_sensitivity_test",
    "run_local_vs_global_budget_ablation",
    "run_graph_uncertainty_factorial_ablation",
    "compute_nominal_vs_robust_width_correlation",
    "perform_image_level_bootstrap",
    "evaluate_decision_gate",
    "generate_baseline_graph_audit_report",
]
