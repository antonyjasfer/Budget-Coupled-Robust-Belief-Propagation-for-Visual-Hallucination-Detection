"""
Unit tests for pre-label graph adequacy stratification and rating rationale (Correction 11).
"""

import pytest
from src.data.dataset_adequacy import (
    stratify_graph_adequacy,
    GraphAdequacyRating,
)


def test_stratify_graph_adequacy_distribution():
    """Verify exact count of G1, G2, G3, G4 within primary cohort."""
    claims_by_image = {
        "img_unary_1": ["c1"],
        "img_unary_2": ["c2"],
        "img_pair_1": ["c3", "c4"],
        "img_tree_1": ["c5", "c6", "c7"],
        "img_rich_1": ["c8", "c9", "c10", "c11"],
    }
    split_assignments = {
        "img_unary_1": "train",
        "img_unary_2": "test",
        "img_pair_1": "test",
        "img_tree_1": "validation",
        "img_rich_1": "test",
    }

    report = stratify_graph_adequacy(claims_by_image, split_assignments)
    counts = report.counts

    assert counts.g1_images == 2
    assert counts.g2_images == 1
    assert counts.g3_images == 1
    assert counts.g4_images == 1
    assert counts.total_images == 5
    assert counts.graph_evaluable_images == 3  # G >= 2
    assert counts.total_claims == 11
    assert counts.graph_evaluable_claims == 9
    assert counts.test_images_count == 3
    assert counts.test_multi_claim_images == 2  # img_pair_1 and img_rich_1


def test_graph_adequacy_ratings_and_stored_rationale():
    """
    Verify ratings SUFFICIENT, MARGINAL, INSUFFICIENT include explicit scientific
    rationale referencing downstream analyses (Correction 11).
    """
    # Rich cohort: 200 multi-claim images, 40 in test
    rich_claims = {}
    rich_splits = {}
    for i in range(250):
        img_id = f"img_{i:03d}"
        if i < 200:
            rich_claims[img_id] = [f"{img_id}_c1", f"{img_id}_c2"]
        else:
            rich_claims[img_id] = [f"{img_id}_c1"]
        rich_splits[img_id] = "test" if i < 50 else "train"

    report_sufficient = stratify_graph_adequacy(rich_claims, rich_splits)
    assert report_sufficient.rating == GraphAdequacyRating.SUFFICIENT
    assert "pairwise coupling" in report_sufficient.rationale.lower()
    assert len(report_sufficient.downstream_analyses_evaluated) > 0


    # Sparse cohort: only 5 multi-claim images
    sparse_claims = {f"img_{i}": [f"c_{i}"] for i in range(100)}
    sparse_claims["img_multi"] = ["c_m1", "c_m2"]
    report_insufficient = stratify_graph_adequacy(sparse_claims)
    assert report_insufficient.rating == GraphAdequacyRating.INSUFFICIENT
    assert "graph-stress supplement recommended" in report_insufficient.rationale.lower()
