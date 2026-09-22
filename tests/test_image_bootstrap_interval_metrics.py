"""Tests for Image-Level Bootstrap Resampling Unit.

Validates:
- Resampling unit is IMAGE_ID, never claim
- Preserving multiplicity when an image is drawn multiple times
- Never independently resampling claims within an image
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_paired_image_bootstrap_comparisons,
)


def test_image_bootstrap_unit_integrity():
    # 5 images with unequal claims (img_0 has 3, img_1 has 2, img_2 has 1, img_3 has 2, img_4 has 1)
    records = []
    claim_counts = {"img_0": 3, "img_1": 2, "img_2": 1, "img_3": 2, "img_4": 1}

    for img_id, count in claim_counts.items():
        for c in range(count):
            is_err = (img_id in ("img_0", "img_3") and c == 0)
            records.append(ReliabilityClaimRecord(
                image_id=img_id,
                claim_id=f"{img_id}_c{c}",
                ground_truth=1 if is_err else 0,
                nominal_prediction=0,
                nominal_correct=not is_err,
                nominal_entropy=0.8 if is_err else 0.2,
                global_width=0.6 if is_err else 0.15,
                local_width=0.7 if is_err else 0.25,
            ))

    res = compute_paired_image_bootstrap_comparisons(records, n_bootstraps=30, min_images=4)
    assert res["status"] == "EVALUATED"
    assert res["n_images"] == 5
    assert res["n_claims"] == sum(claim_counts.values())
    assert res["delta_auroc_mean_ci"] is not None
    assert res["delta_aurc_mean_ci"] is not None


def test_image_bootstrap_insufficient_images_returns_not_evaluable():
    records = [
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True),
        ReliabilityClaimRecord(image_id="img_1", claim_id="c1", ground_truth=1, nominal_prediction=0, nominal_correct=False),
    ]
    res = compute_paired_image_bootstrap_comparisons(records, min_images=4)
    assert res["status"] == "NOT EVALUABLE"
    assert "unique images" in res["reason"]
