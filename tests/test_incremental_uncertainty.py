"""Tests for Group-Aware Incremental Information Test.

Verifies:
- All cross-validation groups by image_id.
- No claim-level fallback.
- Validates failure mode when grouped CV cannot produce valid folds containing both classes.
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    compute_group_aware_incremental_utility,
)


def test_grouped_incremental_cv_success():
    """Verify group-aware CV runs cleanly when sufficient multi-image data exists."""
    records = []
    # 6 images, each with 2 claims, diverse errors
    for img_idx in range(6):
        for c_idx in range(2):
            is_err = (img_idx % 2 == 1 and c_idx == 0)
            rec = ReliabilityClaimRecord(
                image_id=f"img_{img_idx}",
                claim_id=f"claim_{img_idx}_{c_idx}",
                ground_truth=1 if is_err else 0,
                nominal_prediction=0,
                nominal_correct=not is_err,
                nominal_entropy=0.8 if is_err else 0.2,
                global_width=0.6 if is_err else 0.15,
            )
            records.append(rec)

    res = compute_group_aware_incremental_utility(records, n_splits=3, min_samples=8)
    assert res["status"] == "EVALUATED"
    assert res["n_images"] == 6
    assert res["n_samples"] == 12
    assert res["model_A_log_loss"] is not None
    assert res["model_B_log_loss"] is not None
    assert res["delta_log_loss"] is not None


def test_grouped_incremental_cv_single_class_fold_fails_safely():
    """When an image clustering concentrates all errors in one group, grouped CV returns NOT EVALUABLE."""
    records = []
    # 3 images: img_0 has all errors, img_1 and img_2 have no errors
    for c_idx in range(4):
        records.append(ReliabilityClaimRecord(
            image_id="img_0",
            claim_id=f"c0_{c_idx}",
            ground_truth=1,
            nominal_prediction=0,
            nominal_correct=False,
            nominal_entropy=0.9,
            global_width=0.5,
        ))
    for c_idx in range(4):
        records.append(ReliabilityClaimRecord(
            image_id="img_1",
            claim_id=f"c1_{c_idx}",
            ground_truth=0,
            nominal_prediction=0,
            nominal_correct=True,
            nominal_entropy=0.1,
            global_width=0.1,
        ))
    for c_idx in range(4):
        records.append(ReliabilityClaimRecord(
            image_id="img_2",
            claim_id=f"c2_{c_idx}",
            ground_truth=0,
            nominal_prediction=0,
            nominal_correct=True,
            nominal_entropy=0.1,
            global_width=0.1,
        ))

    res = compute_group_aware_incremental_utility(records, n_splits=3, min_samples=6)
    # When img_0 is held out, the training fold contains only non-errors -> must return NOT EVALUABLE
    assert res["status"] == "NOT EVALUABLE"
    assert "single error class" in res["reason"]
