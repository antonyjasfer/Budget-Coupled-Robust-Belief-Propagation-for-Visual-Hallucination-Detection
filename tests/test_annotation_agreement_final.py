"""
Unit tests for inter-annotator agreement computation and Cohen's Kappa.
"""

import pytest
from src.data.schemas import GroundTruthStatus
from src.annotation.agreement import compute_cohens_kappa


def test_agreement_computation_perfect():
    """Verify agreement calculation when annotators agree perfectly."""
    statuses_a = [GroundTruthStatus.SUPPORTED, GroundTruthStatus.HALLUCINATED]
    statuses_b = [GroundTruthStatus.SUPPORTED, GroundTruthStatus.HALLUCINATED]

    res = compute_cohens_kappa(statuses_a, statuses_b)
    assert res.joint_count == 2
    assert res.agreement_count == 2
    assert res.observed_agreement == 1.0
    assert res.cohens_kappa == 1.0


def test_agreement_computation_with_disagreement():
    """Verify disagreement detection."""
    statuses_a = [GroundTruthStatus.SUPPORTED, GroundTruthStatus.HALLUCINATED]
    statuses_b = [GroundTruthStatus.SUPPORTED, GroundTruthStatus.SUPPORTED]  # disagreement

    res = compute_cohens_kappa(statuses_a, statuses_b)
    assert res.joint_count == 2
    assert res.agreement_count == 1
    assert res.observed_agreement == 0.5
    assert res.cohens_kappa is not None
    assert res.cohens_kappa < 1.0

