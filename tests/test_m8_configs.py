"""Tests for Milestone 8 Configuration and Schemas."""

import pytest
import tempfile
from pathlib import Path

from src.experiments.configs import (
    M8ExperimentConfig,
    PGMParameterConfig,
    AblationConfig,
    CorruptionConfig,
    BootstrapConfig,
    OutputConfig,
    ExecutionMode,
)


def test_default_config_initialization():
    cfg = M8ExperimentConfig()
    assert cfg.mode == ExecutionMode.AUTO
    assert cfg.pgm.budget_ratio == 1.0
    assert cfg.pgm.decision_threshold == 0.5
    assert cfg.pgm.grid_steps == 100
    assert cfg.bootstrap.n_resamples == 500


def test_config_yaml_roundtrip():
    cfg = M8ExperimentConfig(
        experiment_id="test_experiment",
        mode=ExecutionMode.SMOKE,
        seed=123,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = Path(tmpdir) / "config.yaml"
        cfg.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded_cfg = M8ExperimentConfig.from_yaml(yaml_path)
        assert loaded_cfg.experiment_id == "test_experiment"
        assert loaded_cfg.mode == ExecutionMode.SMOKE
        assert loaded_cfg.seed == 123
        assert loaded_cfg.pgm.budget_ratio == 1.0


def test_config_validation_errors():
    with pytest.raises(ValueError, match="grid_steps must be >= 2"):
        PGMParameterConfig(grid_steps=1).validate()

    with pytest.raises(ValueError, match="decision_threshold must be in"):
        PGMParameterConfig(decision_threshold=1.5).validate()
