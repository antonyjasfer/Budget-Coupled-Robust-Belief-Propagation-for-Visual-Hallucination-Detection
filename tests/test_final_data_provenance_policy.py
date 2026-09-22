"""
Unit tests for data provenance policy, taxonomy, and validation isolation.
"""

import pytest
from src.data.provenance import (
    DataProvenanceState,
    EvidenceFailureState,
    ValidationMode,
    EvidenceFailureRecord,
    parse_provenance_state,
    validate_provenance_isolation,
    validate_evidence_status,
    FORBIDDEN_FINAL_PROVENANCES,
)


def test_seven_level_provenance_states():
    """Verify all 7 canonical provenance states exist."""
    expected = {
        "real_unlabeled",
        "real_human_annotated",
        "real_adjudicated",
        "synthetic_fixture",
        "mock_annotation",
        "pseudo_label",
        "development_only",
    }
    actual = {s.value for s in DataProvenanceState}
    assert actual == expected


def test_parse_provenance_state():
    """Test case-insensitive and whitespace-tolerant parsing."""
    assert parse_provenance_state("REAL_ADJUDICATED") == DataProvenanceState.REAL_ADJUDICATED
    assert parse_provenance_state("  mock_annotation  ") == DataProvenanceState.MOCK_ANNOTATION
    assert parse_provenance_state(DataProvenanceState.SYNTHETIC_FIXTURE) == DataProvenanceState.SYNTHETIC_FIXTURE

    with pytest.raises(ValueError, match="Unknown data provenance state"):
        parse_provenance_state("invalid_state_123")


def test_forbidden_final_provenances():
    """Verify forbidden states set for final mode."""
    assert DataProvenanceState.SYNTHETIC_FIXTURE in FORBIDDEN_FINAL_PROVENANCES
    assert DataProvenanceState.MOCK_ANNOTATION in FORBIDDEN_FINAL_PROVENANCES
    assert DataProvenanceState.PSEUDO_LABEL in FORBIDDEN_FINAL_PROVENANCES
    assert DataProvenanceState.DEVELOPMENT_ONLY in FORBIDDEN_FINAL_PROVENANCES


def test_provenance_isolation_development_mode():
    """Development mode allows mock/synthetic records with tracking."""
    records = [
        {"claim_id": "c1", "provenance": "synthetic_fixture"},
        {"claim_id": "c2", "provenance": "mock_annotation"},
        {"claim_id": "c3", "provenance": "real_unlabeled"},
    ]
    res = validate_provenance_isolation(records, mode=ValidationMode.DEVELOPMENT)
    assert res.is_valid is True
    assert res.counts_by_provenance["synthetic_fixture"] == 1
    assert res.counts_by_provenance["mock_annotation"] == 1
    assert res.counts_by_provenance["real_unlabeled"] == 1
    assert len(res.violations) == 0


def test_provenance_isolation_final_mode_valid():
    """Final mode accepts real annotated and adjudicated records."""
    records = [
        {"claim_id": "c1", "provenance": "real_human_annotated"},
        {"claim_id": "c2", "provenance": "real_adjudicated"},
    ]
    res = validate_provenance_isolation(records, mode=ValidationMode.FINAL)
    assert res.is_valid is True
    assert len(res.violations) == 0


def test_evidence_failure_record_serialization():
    """Test typed evidence failure record creation and dictionary round-trip."""
    rec = EvidenceFailureRecord(
        claim_id="claim_001",
        provider="owl_vit",
        failure_state=EvidenceFailureState.FAILED,
        reason_code="TIMEOUT_ERROR",
        attempt_count=3,
        last_error_class="TimeoutException",
        provenance_status="failed",
    )
    d = rec.to_dict()
    assert d["provider"] == "owl_vit"
    assert d["failure_state"] == "failed"
    assert d["attempt_count"] == 3

    recovered = EvidenceFailureRecord.from_dict(d)
    assert recovered.claim_id == rec.claim_id
    assert recovered.failure_state == rec.failure_state
    assert recovered.reason_code == rec.reason_code
