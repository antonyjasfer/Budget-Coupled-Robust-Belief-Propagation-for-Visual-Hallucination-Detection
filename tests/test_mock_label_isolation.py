"""
Adversarial Integration Tests for Mock/Pseudo Label Isolation (Correction 19).

Constructs an otherwise valid candidate FINAL dataset and verifies that injecting
even a single record of forbidden provenance:
- SYNTHETIC_FIXTURE
- MOCK_ANNOTATION
- PSEUDO_LABEL
- DEVELOPMENT_ONLY
- REAL_UNLABELED
causes FINAL validation to reject EACH case independently.
"""

import pytest
from src.data.provenance import (
    DataProvenanceState,
    ValidationMode,
    validate_provenance_isolation,
)


@pytest.fixture
def clean_final_candidate_records():
    """Valid final dataset records consisting exclusively of real annotated/adjudicated claims."""
    return [
        {"claim_id": f"claim_{i:04d}", "provenance": "real_human_annotated"}
        for i in range(50)
    ] + [
        {"claim_id": f"claim_{i:04d}", "provenance": "real_adjudicated"}
        for i in range(50, 100)
    ]


def test_clean_final_candidate_passes(clean_final_candidate_records):
    """Verify that pristine candidate dataset passes final isolation validation."""
    audit = validate_provenance_isolation(clean_final_candidate_records, mode=ValidationMode.FINAL)
    assert audit.is_valid is True
    assert len(audit.violations) == 0


@pytest.mark.parametrize(
    "forbidden_state",
    [
        DataProvenanceState.SYNTHETIC_FIXTURE,
        DataProvenanceState.MOCK_ANNOTATION,
        DataProvenanceState.PSEUDO_LABEL,
        DataProvenanceState.DEVELOPMENT_ONLY,
    ],
)
def test_adversarial_forbidden_provenance_rejection(clean_final_candidate_records, forbidden_state):
    """
    Inject exactly one record of each forbidden provenance into an otherwise valid dataset.
    Verify FINAL validation rejects EACH case independently.
    """
    contaminated = list(clean_final_candidate_records)
    contaminated.append({
        "claim_id": "adversarial_injected_claim",
        "provenance": forbidden_state.value,
    })

    audit = validate_provenance_isolation(contaminated, mode=ValidationMode.FINAL)
    assert audit.is_valid is False
    assert len(audit.violations) == 1
    assert forbidden_state.value in audit.violations[0]
    assert "FINAL MODE VIOLATION" in audit.violations[0]


def test_adversarial_real_unlabeled_rejection(clean_final_candidate_records):
    """
    Verify that an unresolved REAL_UNLABELED claim is rejected in FINAL mode
    (preventing incomplete datasets from passing final certification).
    """
    contaminated = list(clean_final_candidate_records)
    contaminated.append({
        "claim_id": "unlabeled_claim_001",
        "provenance": DataProvenanceState.REAL_UNLABELED.value,
    })

    audit = validate_provenance_isolation(contaminated, mode=ValidationMode.FINAL, allow_unlabeled_in_final=False)
    assert audit.is_valid is False
    assert len(audit.violations) == 1
    assert "real_unlabeled" in audit.violations[0]


def test_development_mode_permits_all_states(clean_final_candidate_records):
    """
    Verify that DEVELOPMENT mode permits all states and tracks counts accurately.
    """
    dev_records = list(clean_final_candidate_records)
    dev_records.append({"claim_id": "m1", "provenance": "synthetic_fixture"})
    dev_records.append({"claim_id": "m2", "provenance": "mock_annotation"})
    dev_records.append({"claim_id": "m3", "provenance": "pseudo_label"})
    dev_records.append({"claim_id": "m4", "provenance": "development_only"})
    dev_records.append({"claim_id": "m5", "provenance": "real_unlabeled"})

    audit = validate_provenance_isolation(dev_records, mode=ValidationMode.DEVELOPMENT)
    assert audit.is_valid is True
    assert audit.counts_by_provenance["synthetic_fixture"] == 1
    assert audit.counts_by_provenance["mock_annotation"] == 1
    assert audit.counts_by_provenance["pseudo_label"] == 1
    assert audit.counts_by_provenance["development_only"] == 1
    assert audit.counts_by_provenance["real_unlabeled"] == 1
