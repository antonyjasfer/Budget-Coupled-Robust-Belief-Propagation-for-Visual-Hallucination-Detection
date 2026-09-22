"""Tests for Calibration Package Import Structure.

Verifies that src.calibration is a proper Python package with __init__.py,
does not have a bogus init.py, and cleanly exports all required modules.
"""

from pathlib import Path
import pytest


def test_calibration_init_file_structure():
    """Verify src/calibration/__init__.py exists and src/calibration/init.py does not."""
    calib_dir = Path("src/calibration")
    assert calib_dir.is_dir(), "src/calibration must be a directory"

    pkg_init = calib_dir / "__init__.py"
    assert pkg_init.exists(), "src/calibration/__init__.py must exist"

    bogus_init = calib_dir / "init.py"
    assert not bogus_init.exists(), "src/calibration/init.py must not exist"


def test_calibration_package_imports():
    """Verify all intended exports can be cleanly imported from src.calibration."""
    from src.calibration import (
        probability_to_theta,
        theta_to_probability,
        verify_roundtrip,
        SplitContract,
        SplitRole,
        SplitLeakageError,
        create_deterministic_splits,
        create_image_level_deterministic_splits,
        extract_evidence_features,
        LogisticEvidenceModel,
        ProbabilityCalibrator,
        compute_calibration_metrics,
        EpsilonCalibrator,
        BudgetCalibrator,
        PerturbationSetDiagnostics,
        CouplingCalibrator,
        build_candidate_tree_edges,
        ParameterBundle,
        CalibrationPipeline,
        BaselineMethod,
        ClaimPredictionResult,
        MethodEvaluationSummary,
        BaselineLadderRunner,
        compute_binary_classification_metrics,
        build_baseline_comparison_table,
        build_component_contribution_table,
        PairwiseDependenceMetrics,
        EdgeValidityAuditReport,
        compute_pairwise_state_dependence,
        audit_edge_validity,
        generate_edge_validity_report,
        GraphCoverageAudit,
        FactorialConditionResult,
        ResearchDecisionGate,
        audit_dataset_graph_coverage,
        run_graph_necessity_test,
        run_topology_ablation,
        run_star_root_sensitivity_test,
        run_local_vs_global_budget_ablation,
        run_graph_uncertainty_factorial_ablation,
        compute_nominal_vs_robust_width_correlation,
        perform_image_level_bootstrap,
        evaluate_decision_gate,
        generate_baseline_graph_audit_report,
    )

    # Sanity assertion
    assert callable(probability_to_theta)
    assert callable(perform_image_level_bootstrap)
