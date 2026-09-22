"""Unit tests for Phase 9C claim-graph size stratification and dataset coverage audit.

Verifies:
1. Counts of single-claim vs multi-claim images.
2. Mean, median, and graph size distribution.
3. Proper identification of the `GRAPH-EVALUATION DATA INSUFFICIENT` flag when
   multi-claim images are inadequate for scientific conclusions.
"""

import pytest
from src.calibration.ablation_study import audit_dataset_graph_coverage


def test_graph_coverage_audit_sparse():
    """Sparse dataset with mostly single-claim images triggers insufficiency flag."""
    # 4 images: 3 with 1 claim, 1 with 2 claims
    claims = [
        {"claim_id": "c1", "image_id": "img1"},
        {"claim_id": "c2", "image_id": "img2"},
        {"claim_id": "c3", "image_id": "img3"},
        {"claim_id": "c4a", "image_id": "img4"},
        {"claim_id": "c4b", "image_id": "img4"},
    ]

    report = audit_dataset_graph_coverage(claims)

    assert report.total_images == 4
    assert report.total_claims == 5
    assert report.images_with_1_claim == 3
    assert report.images_with_ge_2_claims == 1
    assert report.images_with_ge_3_claims == 0
    assert pytest.approx(report.mean_claims_per_image, 0.01) == 1.25
    assert report.testability_status == "GRAPH-EVALUATION DATA INSUFFICIENT"


def test_graph_coverage_audit_rich():
    """Adequate multi-claim dataset clears the minimum insufficiency threshold."""
    claims = []
    for i in range(15):
        # 15 images with 3 claims each -> 45 claims total
        for j in range(3):
            claims.append({"claim_id": f"c_{i}_{j}", "image_id": f"img_{i}"})

    report = audit_dataset_graph_coverage(claims)

    assert report.total_images == 15
    assert report.total_claims == 45
    assert report.images_with_1_claim == 0
    assert report.images_with_ge_2_claims == 15
    assert report.images_with_ge_3_claims == 15
    assert report.mean_claims_per_image == 3.0
    assert report.testability_status == "SUFFICIENT"
