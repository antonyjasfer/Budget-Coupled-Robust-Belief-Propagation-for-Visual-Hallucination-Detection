"""
Unit tests for predeclared evidence failure policy and zero-substitution prohibition (Correction 7).
"""

import pytest
from src.data.provenance import (
    EvidenceFailureState,
    EvidenceFailureRecord,
    validate_evidence_status,
)


def test_missing_evidence_never_substituted_with_zero():
    """
    Verify that when detector or CLIP evidence is unavailable,
    substituting 0.0 is flagged as an invalid ZeroSubstitutionViolation.
    """
    # Violated record: detector unavailable but score is 0.0
    bad_record = {
        "claim_id": "c_bad_01",
        "detector_score": 0.0,
        "detector_available": False,
        "clip_score": 0.5,
        "similarity_available": True,
    }
    is_valid, issues = validate_evidence_status(bad_record)
    assert is_valid is False
    assert any("ZeroSubstitutionViolation: detector_score is 0.0" in iss for iss in issues)

    # Valid record: unavailable score is None
    good_record = {
        "claim_id": "c_good_01",
        "detector_score": None,
        "detector_available": False,
        "clip_score": 0.5,
        "similarity_available": True,
    }
    is_valid_good, issues_good = validate_evidence_status(good_record, required_providers=["clip"])
    assert is_valid_good is True
    assert len(issues_good) == 0


def test_required_provider_enforcement():
    """
    Verify that when a required provider is missing, it is reported accurately.
    """
    record = {
        "claim_id": "c_missing",
        "detector_score": 0.8,
        "detector_available": True,
        "clip_score": None,
        "similarity_available": False,
    }
    is_valid, issues = validate_evidence_status(record, required_providers=["owl_vit", "clip"])
    assert is_valid is False
    assert any("required evidence provider 'clip' is unavailable" in iss for iss in issues)
