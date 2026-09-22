"""Tests for Robust Decision Categories.

Validates:
- ROBUST_SUPPORTED: U < tau
- ROBUST_HALLUCINATED: L > tau
- EVIDENCE_SENSITIVE: L <= tau <= U (consistent inclusivity at equality)
"""

import pytest

from src.calibration.reliability import (
    RobustDecisionCategory,
    classify_robust_decision,
)


def test_robust_decision_strict_supported():
    # U < 0.5
    cat = classify_robust_decision(lower=0.1, upper=0.49, tau=0.5)
    assert cat == RobustDecisionCategory.ROBUST_SUPPORTED


def test_robust_decision_strict_hallucinated():
    # L > 0.5
    cat = classify_robust_decision(lower=0.51, upper=0.85, tau=0.5)
    assert cat == RobustDecisionCategory.ROBUST_HALLUCINATED


def test_robust_decision_evidence_sensitive_straddling():
    # L < 0.5 < U
    cat = classify_robust_decision(lower=0.3, upper=0.7, tau=0.5)
    assert cat == RobustDecisionCategory.EVIDENCE_SENSITIVE


def test_robust_decision_boundary_equalities():
    # Lower boundary exactly at tau -> EVIDENCE_SENSITIVE
    cat_low = classify_robust_decision(lower=0.5, upper=0.8, tau=0.5)
    assert cat_low == RobustDecisionCategory.EVIDENCE_SENSITIVE

    # Upper boundary exactly at tau -> EVIDENCE_SENSITIVE
    cat_high = classify_robust_decision(lower=0.2, upper=0.5, tau=0.5)
    assert cat_high == RobustDecisionCategory.EVIDENCE_SENSITIVE

    # Point interval exactly at tau -> EVIDENCE_SENSITIVE
    cat_point = classify_robust_decision(lower=0.5, upper=0.5, tau=0.5)
    assert cat_point == RobustDecisionCategory.EVIDENCE_SENSITIVE
