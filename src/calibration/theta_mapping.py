"""Mathematically rigorous mapping between calibrated probabilities and Ising unary fields.

For binary spin h_i in {-1, +1}:
    -1 = SUPPORTED
    +1 = HALLUCINATED

For an isolated node:
    P(H_i = +1) = sigmoid(2 * theta_i)
Inverting this gives:
    theta_i = 1/2 * ln(p_i / (1 - p_i))
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.special import expit


def probability_to_theta(
    p: Union[float, np.ndarray, list],
    p_min: float = 1e-5,
    theta_max: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Map calibrated hallucination probability p in [0, 1] to Ising unary field theta.

    Args:
        p: Calibrated probability that claim is HALLUCINATED (scalar or array).
        p_min: Numerical safety clipping floor and ceiling (default 1e-5).
        theta_max: Optional maximum bound on |theta|.

    Returns:
        theta: Ising unary field.
    """
    is_scalar = np.isscalar(p)
    p_arr = np.asarray(p, dtype=np.float64)
    p_clipped = np.clip(p_arr, p_min, 1.0 - p_min)

    theta = 0.5 * (np.log(p_clipped) - np.log1p(-p_clipped))
    if theta_max is not None:
        th_max = float(abs(theta_max))
        theta = np.clip(theta, -th_max, th_max)

    if is_scalar:
        return float(theta)
    return theta


def theta_to_probability(
    theta: Union[float, np.ndarray, list],
) -> Union[float, np.ndarray]:
    """Map Ising unary field theta to nominal hallucination probability for an isolated node.

    P(H = +1) = sigmoid(2 * theta)
    """
    is_scalar = np.isscalar(theta)
    th_arr = np.asarray(theta, dtype=np.float64)
    prob = expit(2.0 * th_arr)
    if is_scalar:
        return float(prob)
    return prob


def probability_to_theta_arr(
    p_arr: np.ndarray,
    p_min: float = 1e-5,
    theta_max: Optional[float] = None,
) -> np.ndarray:
    """Vectorized conversion of hallucination probabilities to unary fields."""
    return probability_to_theta(p_arr, p_min=p_min, theta_max=theta_max)  # type: ignore


def theta_to_probability_arr(theta_arr: np.ndarray) -> np.ndarray:
    """Vectorized conversion of unary fields to nominal probabilities."""
    return theta_to_probability(theta_arr)  # type: ignore


def verify_roundtrip(
    p: Union[float, np.ndarray, list],
    tolerance: float = 1e-6,
    p_min: float = 1e-5,
) -> bool:
    """Verify analytical inverse identity: sigma(2 * theta(p)) == p within numerical tolerance."""
    p_arr = np.asarray(p, dtype=np.float64)
    p_clipped = np.clip(p_arr, p_min, 1.0 - p_min)
    thetas = probability_to_theta(p_clipped, p_min=p_min)
    recovered = theta_to_probability(thetas)
    return bool(np.allclose(p_clipped, recovered, atol=tolerance))


# Alias for backward compatibility
verify_roundtrip_mapping = verify_roundtrip
