"""
Unit tests for scripts/phase10b_annotation_interface.py.
"""

import json
from pathlib import Path
import pytest
from scripts.phase10b_annotation_interface import (
    validate_imported_labels,
    validate_claims_match,
    compute_inter_annotator_agreement,
    create_adjudication_queue,
)


def test_validate_imported_labels_success():
    data = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "refuted"},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is True
    assert len(issues) == 0


def test_validate_imported_labels_null_rejected():
    data = {
        "tasks": [
            {"claim_id": "c1", "label": None},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is False
    assert any("null label" in iss for iss in issues)


def test_validate_imported_labels_invalid_value():
    data = {
        "tasks": [
            {"claim_id": "c1", "label": "hallucinated"},
        ]
    }
    valid, issues = validate_imported_labels(data)
    assert valid is False
    assert any("invalid label" in iss for iss in issues)


def test_validate_claims_match():
    da = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "refuted"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "supported"},
        ]
    }
    valid, issues = validate_claims_match(da, db)
    assert valid is True
    assert len(issues) == 0


def test_validate_claims_mismatch():
    da = {"tasks": [{"claim_id": "c1"}, {"claim_id": "c2"}]}
    db = {"tasks": [{"claim_id": "c1"}, {"claim_id": "c3"}]}
    valid, issues = validate_claims_match(da, db)
    assert valid is False
    assert any("present in A but missing in B" in iss for iss in issues)


def test_compute_inter_annotator_agreement():
    da = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "refuted"},
            {"claim_id": "c3", "label": "supported"},
            {"claim_id": "c4", "label": "refuted"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "label": "supported"},
            {"claim_id": "c2", "label": "refuted"},
            {"claim_id": "c3", "label": "refuted"},
            {"claim_id": "c4", "label": "refuted"},
        ]
    }
    res = compute_inter_annotator_agreement(da, db)
    assert res["total_shared_claims"] == 4
    assert res["agreements"] == 3
    assert res["disagreements"] == 1
    assert res["raw_agreement_rate"] == 0.75
    assert "cohens_kappa" in res


def test_create_adjudication_queue(tmp_path):
    da = {
        "tasks": [
            {"claim_id": "c1", "image_id": "img1", "claim_surface": "dog", "image_path": "p1", "label": "supported"},
            {"claim_id": "c2", "image_id": "img2", "claim_surface": "cat", "image_path": "p2", "label": "supported"},
        ]
    }
    db = {
        "tasks": [
            {"claim_id": "c1", "image_id": "img1", "claim_surface": "dog", "image_path": "p1", "label": "refuted"},
            {"claim_id": "c2", "image_id": "img2", "claim_surface": "cat", "image_path": "p2", "label": "supported"},
        ]
    }
    out_file = tmp_path / "adjudication_queue.json"
    queue = create_adjudication_queue(da, db, out_file)
    assert len(queue) == 1
    assert queue[0]["claim_id"] == "c1"
    assert queue[0]["annotator_A_label"] == "supported"
    assert queue[0]["annotator_B_label"] == "refuted"
    assert out_file.exists()
