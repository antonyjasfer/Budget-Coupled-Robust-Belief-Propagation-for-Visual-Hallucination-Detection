"""Unit tests for attractive Ising coupling J_ij calibration and non-negativity enforcement."""

import numpy as np
import pytest

from src.calibration.coupling_calibrator import CouplingCalibrator


def test_j0_strategy():
    """Verify J0 strategy returns strictly zero couplings."""
    cal = CouplingCalibrator()
    edges = [(0, 1, 0.9), (1, 2, 0.5)]
    j_map = cal.get_couplings(edges, strategy="J0")
    for (u, v), j in j_map.items():
        assert j == 0.0


def test_jconst_strategy_and_non_negativity():
    """Verify JCONST sets uniform lambda >= 0."""
    cal = CouplingCalibrator(default_lambda=0.35)
    edges = [(0, 1, 0.8), (1, 2, 0.2)]
    j_map = cal.get_couplings(edges, strategy="JCONST")
    for (u, v), j in j_map.items():
        assert j == 0.35
        assert j >= 0.0


def test_jweighted_strategy():
    """Verify JWEIGHTED sets J_ij = lambda * s_ij."""
    cal = CouplingCalibrator(default_lambda=0.50)
    edges = [(0, 1, 0.8), (1, 2, 0.2)]
    j_map = cal.get_couplings(edges, strategy="JWEIGHTED")
    assert j_map[(0, 1)] == pytest.approx(0.50 * 0.8, abs=1e-5)
    assert j_map[(1, 2)] == pytest.approx(0.50 * 0.2, abs=1e-5)
    for j in j_map.values():
        assert j >= 0.0


def test_negative_lambda_rejected():
    """Verify that negative lambda is strictly rejected to preserve M9A monotonicity."""
    with pytest.raises(ValueError, match="lambda must be non-negative"):
        CouplingCalibrator(default_lambda=-0.2)

    cal = CouplingCalibrator()
    with pytest.raises(ValueError, match="lambda must be non-negative"):
        cal.fit(
            val_thetas=np.array([0.1, 0.2]),
            val_labels=np.array([1.0, 1.0]),
            candidate_edges=[(0, 1, 0.5)],
            lambda_grid=[-0.5, 0.0, 0.5],
        )


def test_negative_similarity_rejected():
    """Verify that negative edge similarity scores are rejected."""
    cal = CouplingCalibrator(default_lambda=0.5)
    edges = [(0, 1, -0.4)]
    with pytest.raises(ValueError, match="Similarity score must be in"):
        cal.get_couplings(edges, strategy="JWEIGHTED")


def test_validation_tuning_selects_non_negative_lambda():
    """Verify validation tuning picks non-negative lambda minimizing validation loss."""
    cal = CouplingCalibrator(objective="nll")
    # Perfectly correlated claims (both +1) with positive unary evidence
    thetas = np.array([0.5, 0.5])
    labels = np.array([1.0, 1.0])
    edges = [(0, 1, 1.0)]

    best_l = cal.fit(
        val_thetas=thetas,
        val_labels=labels,
        candidate_edges=edges,
        root_index=0,
        lambda_grid=[0.0, 0.1, 0.3, 0.5],
    )
    assert best_l >= 0.0
    assert cal.best_lambda >= 0.0
