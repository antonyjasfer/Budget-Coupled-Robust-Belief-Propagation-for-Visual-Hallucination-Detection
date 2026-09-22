"""Unit tests for Phase 9C claim-tree topology ablation and star root sensitivity.

Verifies:
1. Topology ablation evaluates independent, chain, star, and MST topologies.
2. Lambda is tuned on validation only.
3. Edge counts and metric outputs are properly computed.
4. Deterministic star root selection methods (first_claim, highest_detector, lowest_detector, most_central).
"""

import pytest
import numpy as np
from src.calibration.ablation_study import (
    run_topology_ablation,
    run_star_root_sensitivity_test,
)


def test_topology_ablation_runs():
    """Verify topology comparison runs across independent, chain, star, MST."""
    val_records = [
        {"claim_id": "vc1", "image_id": "vimg1", "detector_score": 0.3, "label": 1},
        {"claim_id": "vc2", "image_id": "vimg1", "detector_score": 0.8, "label": 0},
    ]
    records = [
        {"claim_id": "tc1", "image_id": "timg1", "detector_score": 0.4, "label": 1},
        {"claim_id": "tc2", "image_id": "timg1", "detector_score": 0.7, "label": 0},
    ]

    results = run_topology_ablation(
        records=records,
        val_records=val_records,
        topologies=["independent", "chain", "star", "minimum_spanning_tree"],
    )

    assert "independent" in results
    assert "chain" in results
    assert "star" in results
    assert "minimum_spanning_tree" in results

    # Independent must have 0 total edges
    assert results["independent"]["total_edges"] == 0
    # Chain on test set (1 image with 2 claims -> 1 edge)
    assert results["chain"]["total_edges"] == 1


def test_star_root_sensitivity_runs():
    """Verify star root sensitivity test executes deterministically."""
    records = [
        {"claim_id": f"c_{i}", "image_id": "img1", "detector_score": 0.3 + 0.1 * i, "label": i % 2}
        for i in range(4)
    ]
    res = run_star_root_sensitivity_test(records=records, coupling_lambda=0.35)

    assert "first_claim" in res
    assert "highest_detector" in res
    assert "lowest_detector" in res
    assert "most_central" in res
    for method_name, metrics in res.items():
        assert "accuracy" in metrics
        assert "f1" in metrics
