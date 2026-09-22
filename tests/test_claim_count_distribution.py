"""
Unit tests for natural claim count distribution and preservation of one-claim images (Correction 1).
"""

import pytest
from src.data.dataset_adequacy import stratify_graph_adequacy


def test_one_claim_images_retained_in_primary_cohort():
    """
    Verify that 1-claim images are never discarded or replaced merely
    to obtain more graph structure in the primary representative cohort.
    """
    claims_by_image = {
        "coco_img_001": ["c1"],  # single claim
        "coco_img_002": ["c2", "c3"],
        "coco_img_003": ["c4"],  # single claim
        "coco_img_004": ["c5", "c6", "c7"],
    }
    
    report = stratify_graph_adequacy(claims_by_image)
    
    # Both 1-claim images must remain part of total cohort
    assert report.counts.total_images == 4
    assert report.counts.g1_images == 2
    # Graph-evaluable subgroup is 2
    assert report.counts.graph_evaluable_images == 2
    # Membership of primary cohort is unaffected by graph testability
    assert len(claims_by_image) == report.counts.total_images
