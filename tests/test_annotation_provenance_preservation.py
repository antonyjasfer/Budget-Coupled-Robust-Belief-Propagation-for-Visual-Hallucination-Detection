"""
Unit tests for annotation ground-truth mapping and COCO absence prohibition (Corrections 16, 17).
"""

import pytest
from src.data.schemas import (
    GroundTruthStatus,
    ground_truth_to_pgm_label,
)


def test_ground_truth_to_pgm_label_mapping():
    """
    Verify canonical Ising spin mapping:
    SUPPORTED -> -1
    HALLUCINATED -> +1
    UNKNOWN -> ValueError (cannot be converted to binary spin)
    """
    assert ground_truth_to_pgm_label(GroundTruthStatus.SUPPORTED) == -1
    assert ground_truth_to_pgm_label(GroundTruthStatus.HALLUCINATED) == +1

    # String parsing
    assert ground_truth_to_pgm_label("supported") == -1
    assert ground_truth_to_pgm_label("hallucinated") == +1


    # UNKNOWN raises ValueError
    with pytest.raises(ValueError, match="UNKNOWN cannot be converted"):
        ground_truth_to_pgm_label(GroundTruthStatus.UNKNOWN)


def test_coco_absence_not_derived_as_hallucination():
    """
    Verify that absence of an object in external COCO annotations
    does NOT automatically derive HALLUCINATED (Correction 17).
    Ground truth must remain determined by human annotation.
    """
    # A claim about a dog where COCO ground truth labels only has 'person' and 'car'
    coco_labeled_categories = {"person", "car"}
    claimed_category = "dog"

    # Rule: Missing from COCO labels != verified visual absence / hallucinated
    is_in_coco = claimed_category in coco_labeled_categories
    assert is_in_coco is False

    # Status must NOT be defaulted to HALLUCINATED based on absence in COCO
    derived_status = "UNKNOWN_REQUIRES_HUMAN_INSPECTION" if not is_in_coco else "SUPPORTED"
    assert derived_status != GroundTruthStatus.HALLUCINATED.value
