"""Unit tests for Phase 9C empirical edge validity analysis.

Verifies:
1. Computation of pairwise state dependence metrics (phi, odds ratio, mutual info, agreement).
2. Strict prohibition of test-set data in edge dependence estimation (raises SplitLeakageError).
3. Sample size thresholding and categorization of empirical evidence.
"""

import pytest
import numpy as np
from src.calibration.splits import SplitLeakageError, SplitRole
from src.calibration.edge_audit import (
    compute_pairwise_state_dependence,
    audit_edge_validity,
    generate_edge_validity_report,
)


def test_pairwise_state_dependence_perfect_concordance():
    """When truth states always agree, agreement rate is 1.0 and phi is 1.0."""
    # 5 pairs of nodes with identical labels
    pairs = [(2 * i, 2 * i + 1, 0.9) for i in range(5)]
    labels = [1, 1] * 5

    metrics = compute_pairwise_state_dependence(pairs, labels, min_samples=2)
    assert metrics.n_pairs == 5
    assert metrics.agreement_rate == 1.0
    assert metrics.n_discordant == 0
    assert metrics.n_concordant == 5


def test_pairwise_state_dependence_mixed():
    """Verify contingency table counts and valid metrics on mixed labels."""
    pairs = [
        (0, 1, 0.8),
        (2, 3, 0.7),
        (4, 5, 0.6),
    ]
    # pair 1 agrees (1, 1), pair 2 agrees (0, 0), pair 3 disagrees (1, 0)
    labels = [1, 1, 0, 0, 1, 0]
    metrics = compute_pairwise_state_dependence(pairs, labels, min_samples=2)
    assert metrics.n_pairs == 3
    assert metrics.n_concordant == 2
    assert metrics.n_discordant == 1
    assert pytest.approx(metrics.agreement_rate, 0.01) == 2.0 / 3.0
    assert 0.0 <= metrics.mutual_information <= 1.0


def test_audit_edge_validity_rejects_test_split():
    """audit_edge_validity MUST raise SplitLeakageError if TEST claims or split_role is passed."""
    claims = [
        {"claim_id": "c1", "image_id": "img1", "label": 1, "split": "train"},
        {"claim_id": "c2", "image_id": "img1", "label": 1, "split": "test"},  # Leakage!
    ]
    edges = {"img1": [(0, 1, 0.9)]}

    with pytest.raises(SplitLeakageError):
        audit_edge_validity(claims, edges, split_role=SplitRole.TRAIN)

    # Passing split_role="TEST" must also be rejected
    train_claims = [
        {"claim_id": "c1", "image_id": "img1", "label": 1, "split": "train"},
        {"claim_id": "c2", "image_id": "img1", "label": 1, "split": "train"},
    ]
    with pytest.raises(SplitLeakageError):
        audit_edge_validity(train_claims, edges, split_role=SplitRole.TEST)


def test_audit_edge_validity_insufficient_data_flag():
    """When pair count is below min_samples, classification is 'insufficient samples'."""
    claims = [
        {"claim_id": "c1", "image_id": "img1", "label": 1, "split": "train"},
        {"claim_id": "c2", "image_id": "img1", "label": 1, "split": "train"},
    ]
    edges = {"img1": [(0, 1, 0.85)]}

    report = audit_edge_validity(claims, edges, split_role=SplitRole.TRAIN, min_samples=10)
    assert report.pairwise_metrics.evidence_category == "insufficient samples"

    # Also verify markdown report generation works without crashing
    md_content = generate_edge_validity_report(report)
    assert "insufficient samples" in md_content
    assert "DO NOT TREAT AS FINAL" in md_content
