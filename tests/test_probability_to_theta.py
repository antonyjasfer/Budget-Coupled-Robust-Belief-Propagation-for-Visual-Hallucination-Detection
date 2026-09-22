"""Unit tests for evidence probability to Ising unary field theta mapping."""

import numpy as np
import pytest

from src.calibration.theta_mapping import (
    probability_to_theta,
    theta_to_probability,
    verify_roundtrip,
)


def test_sign_convention():
    """Verify that p > 0.5 maps to positive theta (hallucinated) and p < 0.5 to negative theta."""
    # Isolated node: P(H = +1) = sigma(2 * theta)
    assert probability_to_theta(0.5) == pytest.approx(0.0, abs=1e-9)
    assert probability_to_theta(0.8) > 0.0
    assert probability_to_theta(0.2) < 0.0
    assert probability_to_theta(0.99) > probability_to_theta(0.8)


def test_inverse_mapping_roundtrip():
    """Verify that sigma(2 * theta) recovers original p within machine precision."""
    probs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for p in probs:
        theta = probability_to_theta(p)
        recovered_p = theta_to_probability(theta)
        assert recovered_p == pytest.approx(p, abs=1e-6)


def test_vectorized_roundtrip():
    """Verify vectorized array conversion and verify_roundtrip helper."""
    p_grid = np.linspace(0.01, 0.99, 50)
    thetas = probability_to_theta(p_grid)
    p_rec = theta_to_probability(thetas)
    assert np.allclose(p_grid, p_rec, atol=1e-6)
    assert verify_roundtrip(p_grid, tolerance=1e-6)


def test_clipping_and_extreme_values():
    """Verify that extreme probabilities (0, 1, out-of-bounds) do not produce NaN or Inf."""
    extremes = [-0.5, 0.0, 1.0, 1.5]
    for p in extremes:
        theta = probability_to_theta(p, p_min=1e-4)
        assert np.isfinite(theta)
        # For p <= 0, should clip to p_min, yielding negative theta
        if p <= 0.0:
            assert theta < 0.0
        # For p >= 1, should clip to 1 - p_min, yielding positive theta
        if p >= 1.0:
            assert theta > 0.0


def test_custom_p_min():
    """Verify custom clipping bounds."""
    p_min = 0.1
    # 0.05 clipped to 0.1
    th1 = probability_to_theta(0.05, p_min=p_min)
    th2 = probability_to_theta(0.10, p_min=p_min)
    assert th1 == pytest.approx(th2, abs=1e-9)
