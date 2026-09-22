"""Tests for UNKNOWN Ground Truth Uncertainty Analysis.

Validates:
- Comparison of robust width across SUPPORTED, HALLUCINATED, and UNKNOWN
- UNKNOWN claims preserved as separate human ground truth
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_unknown_claim_uncertainty,
)


def test_unknown_claim_uncertainty_comparison():
    records = [
        # Supported claims (ground_truth=0)
        ReliabilityClaimRecord(image_id="img_0", claim_id="c0", ground_truth=0, global_width=0.15),
        ReliabilityClaimRecord(image_id="img_0", claim_id="c1", ground_truth=0, global_width=0.20),
        # Hallucinated claims (ground_truth=1)
        ReliabilityClaimRecord(image_id="img_1", claim_id="c2", ground_truth=1, global_width=0.25),
        ReliabilityClaimRecord(image_id="img_1", claim_id="c3", ground_truth=1, global_width=0.30),
        # UNKNOWN claims (ground_truth=None)
        ReliabilityClaimRecord(image_id="img_2", claim_id="c4", ground_truth=None, global_width=0.75),
        ReliabilityClaimRecord(image_id="img_2", claim_id="c5", ground_truth=None, global_width=0.80),
    ]

    res = evaluate_unknown_claim_uncertainty(records)
    assert res["status"] == "EVALUATED"
    assert res["n_supported"] == 2
    assert res["n_hallucinated"] == 2
    assert res["n_unknown"] == 2
    assert np.isclose(res["mean_width_supported"], 0.175)
    assert np.isclose(res["mean_width_hallucinated"], 0.275)
    assert np.isclose(res["mean_width_unknown"], 0.775)
    assert res["mean_width_unknown"] > res["mean_width_supported"]
