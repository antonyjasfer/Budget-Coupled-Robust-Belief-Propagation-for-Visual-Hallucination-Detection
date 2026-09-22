"""Unit tests for empirical epsilon calibration under corruption residuals."""

import numpy as np
import pytest

from src.calibration.epsilon_calibrator import EpsilonCalibrator


def test_global_quantile_calibration():
    """Verify global quantile estimation across corruption residuals."""
    residuals = [
        {"claim_id": f"c_{i}", "category": "object", "residual": float(i) * 0.05}
        for i in range(21)  # 0.0, 0.05, ..., 1.0
    ]

    cal = EpsilonCalibrator(method="global_quantile", default_quantile=0.90)
    cal.fit_residuals(residuals)

    eps_90 = cal.get_epsilon(quantile=0.90)
    eps_50 = cal.get_epsilon(quantile=0.50)
    eps_95 = cal.get_epsilon(quantile=0.95)

    assert eps_50 <= eps_90 <= eps_95
    assert eps_90 == pytest.approx(0.90, abs=1e-2)
    assert eps_50 == pytest.approx(0.50, abs=1e-2)


def test_category_conditioned_fallback():
    """Verify category conditioning uses category data when sufficient and falls back when small."""
    residuals = [
        # Large category: "attribute" (15 samples, residuals ~ 0.2)
        *[{"claim_id": f"a_{i}", "category": "attribute", "residual": 0.20} for i in range(15)],
        # Tiny category: "relation" (2 samples, residuals ~ 0.9)
        *[{"claim_id": f"r_{i}", "category": "relation", "residual": 0.90} for i in range(2)],
    ]

    cal = EpsilonCalibrator(method="category_conditioned", min_samples_per_category=5)
    cal.fit_residuals(residuals)

    # Attribute category should use its own quantile (~ 0.20)
    eps_attr = cal.get_epsilon(category="attribute")
    assert eps_attr == pytest.approx(0.20, abs=1e-3)

    # Relation category has only 2 samples (< 5), so it falls back to global
    eps_rel = cal.get_epsilon(category="relation")
    assert eps_rel != 0.90  # Fallback to global, which is blended with attribute


def test_evidence_conditioned_epsilon():
    """Verify evidence-conditioned uncertainty: lower confidence evidence yields larger epsilon."""
    cal = EpsilonCalibrator(method="evidence_conditioned")
    # Residuals with clean_p
    residuals = [
        {"claim_id": "c1", "clean_p": 0.50, "residual": 0.40},
        {"claim_id": "c2", "clean_p": 0.95, "residual": 0.05},
        {"claim_id": "c3", "clean_p": 0.05, "residual": 0.05},
    ]
    cal.fit_residuals(residuals)

    # Near 0.5 (maximum entropy/uncertainty) should have larger epsilon than near 0.95
    eps_uncertain = cal.get_epsilon(clean_p=0.50)
    eps_certain = cal.get_epsilon(clean_p=0.95)
    assert eps_uncertain >= eps_certain


def test_empty_or_zero_residuals():
    """Verify edge case of zero or empty residuals handled safely."""
    cal = EpsilonCalibrator(method="global_quantile")
    cal.fit_residuals([])
    # Must return fallback epsilon without error
    assert cal.get_epsilon() >= 0.0
