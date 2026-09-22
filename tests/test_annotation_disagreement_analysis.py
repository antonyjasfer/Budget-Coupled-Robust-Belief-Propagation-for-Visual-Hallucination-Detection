"""Tests for Human Annotation Disagreement Analysis.

Validates:
- Comparison of robust width W_global between agreement (A_i == B_i) and disagreement (A_i != B_i)
- Image-level clustered bootstrap for difference in means
- Small sample protection when disagreements are absent
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_annotation_disagreement_uncertainty,
)


def test_annotation_disagreement_analysis():
    records = []
    # 4 images, each with 2 claims:
    # img_0, img_1: agreement (width 0.2, 0.25)
    # img_2, img_3: disagreement (width 0.6, 0.7)
    for i in range(4):
        is_disagree = (i >= 2)
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"c_{i}_0",
            annotator_A=0 if not is_disagree else 0,
            annotator_B=0 if not is_disagree else 1,
            global_width=0.65 if is_disagree else 0.22,
        ))
        records.append(ReliabilityClaimRecord(
            image_id=f"img_{i}",
            claim_id=f"c_{i}_1",
            annotator_A=1 if not is_disagree else 1,
            annotator_B=1 if not is_disagree else 0,
            global_width=0.60 if is_disagree else 0.20,
        ))

    res = evaluate_annotation_disagreement_uncertainty(records, min_samples=4, n_bootstraps=50)
    assert res["status"] == "EVALUATED"
    assert res["n_agree"] == 4
    assert res["n_disagree"] == 4
    assert res["mean_width_disagree"] > res["mean_width_agree"]
    assert res["mean_diff_bootstrap_ci"] is not None


def test_annotation_disagreement_no_disagreements():
    records = [
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", annotator_A=0, annotator_B=0, global_width=0.2),
        ReliabilityClaimRecord(image_id="img_1", claim_id="c1", annotator_A=1, annotator_B=1, global_width=0.3),
    ]
    res = evaluate_annotation_disagreement_uncertainty(records, min_samples=2)
    assert res["status"] == "NOT EVALUABLE"
    assert "Absence of dual classes" in res["reason"]
