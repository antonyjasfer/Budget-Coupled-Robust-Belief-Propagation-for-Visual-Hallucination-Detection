"""Unit tests for Phase 9C coupling strategies (J0, JC, JS, JE).

Verifies:
1. J0 produces J_ij = 0 everywhere.
2. JC produces constant J_ij = lambda.
3. JS scales J_ij by semantic similarity s_ij in [0, 1].
4. Crucial invariant: J_ij is strictly non-negative (J_ij >= 0) to preserve attractive Ising monotonicity.
"""

import pytest
import numpy as np
from src.calibration.coupling_calibrator import (
    CouplingStrategy,
    CouplingCalibrator,
)


def test_coupling_strategy_non_negative():
    """All coupling values must be >= 0."""
    calibrator = CouplingCalibrator(default_lambda=0.5, strategy="JWEIGHTED")
    edges = [(0, 1, 0.8), (1, 2, 0.0)]
    couplings = calibrator.get_couplings(edges)

    for (u, v), j in couplings.items():
        assert j >= 0.0, f"Coupling between {u} and {v} is negative: {j}"


def test_j0_constant_zero():
    """J0 strategy produces exact 0.0 coupling for all edges."""
    calibrator = CouplingCalibrator(default_lambda=1.2, strategy="J0")
    edges = [(0, 1, 0.9), (1, 2, 0.7)]
    couplings = calibrator.get_couplings(edges)

    for (u, v), j in couplings.items():
        assert j == 0.0


def test_jc_constant_lambda():
    """JC / JCONST strategy produces lambda regardless of semantic similarity."""
    lam = 0.65
    calibrator = CouplingCalibrator(default_lambda=lam, strategy="JCONST")
    edges = [(0, 1, 0.2), (1, 2, 0.9)]
    couplings = calibrator.get_couplings(edges)

    for (u, v), j in couplings.items():
        assert pytest.approx(j, 1e-6) == lam


def test_js_semantic_scaling():
    """JS / JWEIGHTED strategy scales lambda by semantic similarity."""
    lam = 0.5
    calibrator = CouplingCalibrator(default_lambda=lam, strategy="JWEIGHTED")
    edges = [(0, 1, 0.8), (1, 2, 0.4)]
    couplings = calibrator.get_couplings(edges)

    assert pytest.approx(couplings[(0, 1)], 1e-6) == 0.4
    assert pytest.approx(couplings[(1, 2)], 1e-6) == 0.2
