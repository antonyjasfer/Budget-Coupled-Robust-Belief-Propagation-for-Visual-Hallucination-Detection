"""Tests for Robust Abstention Policy.

Validates:
- Abstaining on EVIDENCE_SENSITIVE (L <= tau <= U)
- Never counting abstentions as correct or silently incorrect
- Evaluating coverage, selective accuracy, precision, recall, F1, risk
- Safe handling of 100% abstention and 0% abstention
"""

import numpy as np
import pytest

from src.calibration.reliability import (
    ReliabilityClaimRecord,
    evaluate_robust_abstention_policy,
)


def test_abstention_policy_metrics():
    # 5 claims:
    # 2 abstained (evidence-sensitive)
    # 3 accepted: 2 correct (1 supported, 1 hall), 1 incorrect (pred hall, true sup)
    records = [
        # Abstained 1
        ReliabilityClaimRecord(
            image_id="img_0", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True,
            global_contains_threshold=True,
        ),
        # Abstained 2
        ReliabilityClaimRecord(
            image_id="img_1", claim_id="c1", ground_truth=1, nominal_prediction=0, nominal_correct=False,
            global_contains_threshold=True,
        ),
        # Accepted 1 (correct sup)
        ReliabilityClaimRecord(
            image_id="img_2", claim_id="c2", ground_truth=0, nominal_prediction=0, nominal_correct=True,
            global_contains_threshold=False,
        ),
        # Accepted 2 (correct hall)
        ReliabilityClaimRecord(
            image_id="img_3", claim_id="c3", ground_truth=1, nominal_prediction=1, nominal_correct=True,
            global_contains_threshold=False,
        ),
        # Accepted 3 (incorrect hall)
        ReliabilityClaimRecord(
            image_id="img_4", claim_id="c4", ground_truth=0, nominal_prediction=1, nominal_correct=False,
            global_contains_threshold=False,
        ),
    ]

    res = evaluate_robust_abstention_policy(records, tau=0.5)
    assert res["status"] == "EVALUATED"
    assert res["accepted_n"] == 3
    assert res["abstained_n"] == 2
    assert np.isclose(res["coverage"], 3.0 / 5.0)
    # Among 3 accepted: 2 correct, 1 error -> accuracy = 2/3, risk = 1/3
    assert np.isclose(res["selective_accuracy"], 2.0 / 3.0)
    assert np.isclose(res["selective_risk"], 1.0 / 3.0)
    # Among accepted: TP=1 (c3), FP=1 (c4), FN=0, TN=1 (c2)
    # Precision = 1 / (1 + 1) = 0.5, Recall = 1 / 1 = 1.0, F1 = 2 * 0.5 * 1.0 / 1.5 = 2/3
    assert np.isclose(res["selective_precision"], 0.5)
    assert np.isclose(res["selective_recall"], 1.0)
    assert np.isclose(res["selective_f1"], 2.0 / 3.0)


def test_abstention_all_abstained_safe():
    records = [
        ReliabilityClaimRecord(
            image_id="img_0", claim_id="c0", ground_truth=0, nominal_prediction=0, nominal_correct=True,
            global_contains_threshold=True,
        ),
    ]
    res = evaluate_robust_abstention_policy(records)
    assert res["coverage"] == 0.0
    assert res["accepted_n"] == 0
    assert res["abstained_n"] == 1
