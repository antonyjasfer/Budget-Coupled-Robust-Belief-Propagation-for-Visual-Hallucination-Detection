"""
Unit tests verifying strict evidence masking for human annotators.
"""

import pytest
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import M7ClaimRecord, HumanAnnotationTask
from src.annotation.masking import (
    create_masked_annotation_task,
    audit_masked_task,
    extract_minimal_context,
    FORBIDDEN_EVIDENCE_FIELDS,
)


def test_extract_minimal_context():
    """Verify context extraction limits verbosity and extracts local clause."""
    long_caption = (
        "In the background there is a tall mountain covered in snow. In the foreground a red car "
        "is parked beside an antique wooden bench. Above them flies a flock of birds in a bright blue sky."
    )
    context = extract_minimal_context(long_caption, "red car")
    assert "red car" in context.lower()
    # Ensure it didn't return the entire multi-sentence paragraph
    assert "mountain" not in context.lower() or "birds" not in context.lower()


def test_evidence_masking_strictly_excludes_all_forbidden_fields():
    """Machine-checkable test proving no model scores, posteriors, or parameters exist in task."""
    ev = ClaimLevelEvidenceRecord(
        claim_id="claim_mask_01",
        image_id="coco_mask_01",
        object_category="car",
        text_span="red car",
        caption="A shiny red car on the street.",
        image_hash="sha256_mock_hash",
        split="test",
        detector_score=0.912,
        detector_available=True,
        clip_score=0.456,
        similarity_available=True,
        is_synthetic=True,
    )
    m7_claim = M7ClaimRecord.from_m6_evidence(ev, assigned_split="test")

    task = create_masked_annotation_task(m7_claim, annotator_tag="A")
    task_dict = task.to_dict()

    # Assert none of the forbidden fields exist in the dictionary
    for forbidden in FORBIDDEN_EVIDENCE_FIELDS:
        assert forbidden not in task_dict, f"Forbidden field '{forbidden}' found in masked task dictionary!"
        assert forbidden not in task_dict.get("metadata", {}), f"Forbidden field '{forbidden}' found in task metadata!"

    # Also assert split is absent from human-facing task
    assert "split" not in task_dict

    # Assert audit_masked_task passes
    audit_masked_task(task)


def test_audit_masked_task_detects_leakage():
    """Verify that audit_masked_task raises ValueError if a forbidden field is injected."""
    task = HumanAnnotationTask(
        task_id="task_leak",
        claim_id="c1",
        image_id="img1",
        image_path="img1.jpg",
        object_category="car",
        surface_form="car",
        metadata={"detector_score": 0.9},
    )
    with pytest.raises(ValueError, match="Forbidden field 'detector_score' found"):
        audit_masked_task(task)
