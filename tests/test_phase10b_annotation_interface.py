"""
Unit tests for scripts/phase10b_annotation_interface.py and canonical ground truth schema.

Validates:
1. Accepted canonical labels: supported, hallucinated, unknown
2. "refuted" and non-canonical labels rejected
3. UNKNOWN accepted in completed annotations
4. Null rejected in completed annotations
5. 3x3 multiclass confusion matrix
6. Multiclass Cohen's kappa computation with known example
7. Degenerate Pe == 1 case (kappa is None, kappa_defined is False)
8. Adjudication preserves original A/B labels
9. UNKNOWN adjudication allowed in adjudication queue
10. UNKNOWN cannot map to binary Ising spin PGM label (raises ValueError)
"""

import json
from pathlib import Path
import pytest
from src.data.schemas import GroundTruthStatus, ground_truth_to_pgm_label
from scripts.phase10b_annotation_interface import (
    validate_imported_labels,
    validate_claims_match,
    compute_inter_annotator_agreement,
    create_adjudication_queue,
)


def test_validate_imported_labels_canonical_success():
    """Verify supported, hallucinated, and unknown are accepted."""
    data = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "hallucinated"},
            {"claim_id": "c3", "label": "unknown"},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is True
    assert len(issues) == 0


def test_validate_imported_labels_refuted_rejected():
    """Verify 'refuted' is rejected in completed annotation imports."""
    data = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "refuted"},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is False
    assert any("invalid label 'refuted'" in iss for iss in issues)


def test_validate_imported_labels_unknown_accepted():
    """Verify 'unknown' is an accepted ground truth status."""
    data = {
        "tasks": [
            {"claim_id": "c1", "label": "unknown"},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is True
    assert len(issues) == 0


def test_validate_imported_labels_null_rejected():
    """Verify null label is rejected in completed annotation import."""
    data = {
        "tasks": [
            {"claim_id": "c1", "label": None},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is False
    assert any("null label" in iss for iss in issues)


def test_validate_imported_labels_abstain_rejected():
    """Verify model decision 'abstain' is rejected as a human annotation label."""
    data = {
        "tasks": [
            {"claim_id": "c1", "label": "abstain"},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is False
    assert any("invalid label 'abstain'" in iss for iss in issues)


def test_validate_claims_match():
    """Verify identical claim sets pass matching validation."""
    da = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "hallucinated"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "unknown"},
        ]
    }
    valid, issues = validate_claims_match(da, db)
    assert valid is True
    assert len(issues) == 0


def test_validate_claims_mismatch():
    """Verify mismatched claim sets fail with specific error."""
    da = {"tasks": [{"claim_id": "c1"}, {"claim_id": "c2"}]}
    db = {"tasks": [{"claim_id": "c1"}, {"claim_id": "c3"}]}
    valid, issues = validate_claims_match(da, db)
    assert valid is False
    assert any("present in A but missing in B" in iss for iss in issues)


def test_multiclass_confusion_matrix_3x3():
    """Verify multiclass agreement produces a full 3x3 confusion matrix."""
    da = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "hallucinated"},
            {"claim_id": "c3", "label": "unknown"},
            {"claim_id": "c4", "label": "supported"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "hallucinated"},
            {"claim_id": "c3", "label": "supported"},
            {"claim_id": "c4", "label": "unknown"},
        ]
    }
    res = compute_inter_annotator_agreement(da, db)
    assert res["total_claims"] == 4
    assert res["agreement_count"] == 2
    assert res["disagreement_count"] == 2
    assert res["raw_agreement"] == 0.5

    matrix = res["confusion_matrix"]
    assert "supported" in matrix
    assert "hallucinated" in matrix
    assert "unknown" in matrix
    assert len(matrix["supported"]) == 3
    assert len(matrix["hallucinated"]) == 3
    assert len(matrix["unknown"]) == 3
    # Check specific cell counts
    assert matrix["supported"]["supported"] == 1
    assert matrix["hallucinated"]["hallucinated"] == 1
    assert matrix["unknown"]["supported"] == 1
    assert matrix["supported"]["unknown"] == 1


def test_known_multiclass_cohens_kappa():
    """Verify multiclass kappa matches analytical calculation."""
    # Annotations across 3 classes: S, H, U
    # 6 items:
    # A: [S, S, H, H, U, U]
    # B: [S, H, H, U, U, S]
    # Agreements: item 1 (S,S), item 3 (H,H), item 5 (U,U) -> 3/6 = 0.50
    # Marginals A: S:2/6, H:2/6, U:2/6
    # Marginals B: S:2/6, H:2/6, U:2/6
    # Pe = (2/6)*(2/6) + (2/6)*(2/6) + (2/6)*(2/6) = 3 * (4/36) = 12/36 = 1/3
    # Kappa = (0.5 - 1/3) / (1 - 1/3) = (1/6) / (2/3) = 1/4 = 0.25
    da = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "supported"},
            {"claim_id": "c3", "label": "hallucinated"},
            {"claim_id": "c4", "label": "hallucinated"},
            {"claim_id": "c5", "label": "unknown"},
            {"claim_id": "c6", "label": "unknown"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "hallucinated"},
            {"claim_id": "c3", "label": "hallucinated"},
            {"claim_id": "c4", "label": "unknown"},
            {"claim_id": "c5", "label": "unknown"},
            {"claim_id": "c6", "label": "supported"},
        ]
    }
    res = compute_inter_annotator_agreement(da, db)
    assert res["total_claims"] == 6
    assert res["agreement_count"] == 3
    assert res["raw_agreement"] == 0.5
    assert res["chance_agreement"] == round(1 / 3, 4)
    assert res["cohens_kappa"] == 0.25
    assert res["kappa_defined"] is True


def test_kappa_degenerate_pe_equals_one():
    """Verify when Pe == 1.0 (zero denominator), kappa is None and kappa_defined is False."""
    # Both annotators label 100% of claims identically as 'supported'
    da = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "supported"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "supported"},
        ]
    }
    res = compute_inter_annotator_agreement(da, db)
    assert res["total_claims"] == 2
    assert res["raw_agreement"] == 1.0
    assert res["chance_agreement"] == 1.0
    assert res["cohens_kappa"] is None
    assert res["kappa_defined"] is False
    assert "Pe == 1.0" in res["status_message"]


def test_create_adjudication_queue_preserves_original_labels(tmp_path):
    """Verify original A and B labels are preserved without auto-consensus."""
    da = {
        "tasks": [
            {"claim_id": "c1", "image_id": "img1", "claim_surface": "dog", "image_path": "p1", "label": "supported"},
            {"claim_id": "c2", "image_id": "img2", "claim_surface": "cat", "image_path": "p2", "label": "hallucinated"},
            {"claim_id": "c3", "image_id": "img3", "claim_surface": "car", "image_path": "p3", "label": "unknown"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "image_id": "img1", "claim_surface": "dog", "image_path": "p1", "label": "hallucinated"},
            {"claim_id": "c2", "image_id": "img2", "claim_surface": "cat", "image_path": "p2", "label": "hallucinated"},
            {"claim_id": "c3", "image_id": "img3", "claim_surface": "car", "image_path": "p3", "label": "supported"},
        ]
    }
    out_file = tmp_path / "adjudication_queue.json"
    queue = create_adjudication_queue(da, db, out_file)
    assert len(queue) == 2  # c1 and c3 disagreed

    # c1: A=supported, B=hallucinated
    q0 = [item for item in queue if item["claim_id"] == "c1"][0]
    assert q0["annotator_A_label"] == "supported"
    assert q0["annotator_B_label"] == "hallucinated"
    assert q0["adjudicated_label"] is None
    assert q0["adjudicator_notes"] is None

    # c3: A=unknown, B=supported
    q1 = [item for item in queue if item["claim_id"] == "c3"][0]
    assert q1["annotator_A_label"] == "unknown"
    assert q1["annotator_B_label"] == "supported"

    # Verify JSON file has allowed labels
    with open(out_file, "r", encoding="utf-8") as f:
        saved = json.load(f)
    assert set(saved["allowed_adjudicated_labels"]) == {"supported", "hallucinated", "unknown"}


def test_unknown_ground_truth_raises_for_pgm_mapping():
    """Verify UNKNOWN raises ValueError when mapping to binary Ising spin (-1, +1)."""
    assert ground_truth_to_pgm_label(GroundTruthStatus.SUPPORTED) == -1
    assert ground_truth_to_pgm_label("supported") == -1
    assert ground_truth_to_pgm_label(GroundTruthStatus.HALLUCINATED) == +1
    assert ground_truth_to_pgm_label("hallucinated") == +1

    # UNKNOWN must NOT map to binary spin
    with pytest.raises(ValueError, match="UNKNOWN cannot be converted to a binary Ising spin"):
        ground_truth_to_pgm_label(GroundTruthStatus.UNKNOWN)

    with pytest.raises(ValueError, match="UNKNOWN cannot be converted to a binary Ising spin"):
        ground_truth_to_pgm_label("unknown")
