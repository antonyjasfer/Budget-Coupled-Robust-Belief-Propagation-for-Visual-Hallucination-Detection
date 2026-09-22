"""Tests for Milestone 8 Ablation Matrix."""

import pytest

from src.experiments.configs import M8ExperimentConfig
from src.experiments.loaders import create_synthetic_smoke_bundle
from src.experiments.ablations import AblationMatrixRunner


def test_ablation_matrix_execution():
    cfg = M8ExperimentConfig()
    cfg.ablations.budget_sweep_multipliers = [0.0, 1.0]
    cfg.ablations.epsilon_scale_values = [0.5, 1.0]
    cfg.ablations.grid_step_values = [10]

    bundle = create_synthetic_smoke_bundle(num_images=2, seed=42)
    runner = AblationMatrixRunner(cfg)
    results = runner.run_all_ablations(bundle)

    assert "robust_bp_proposed" in results
    assert "standard_bp" in results
    assert "robust_bp_b0" in results
    assert "robust_bp_no_coupling" in results
    assert "budget_sweep_b_0.00" in results
    assert "budget_sweep_b_1.00" in results
    assert "epsilon_sweep_alpha_0.50" in results
    assert "grid_sweep_k_10" in results

    assert len(results["standard_bp"]) > 0
    assert len(results["robust_bp_no_coupling"]) > 0
