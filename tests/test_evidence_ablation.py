"""Unit tests for Phase 9C evidence source ablation (detector only, CLIP only, combined).

Verifies:
1. M0 (detector only) evaluates solitary visual detection scores.
2. M1 (CLIP only) evaluates solitary multimodal alignment scores.
3. M2 (detector + CLIP logistic fusion) properly trains and predicts on the joint feature vector.
4. Each method uses identical evaluation claims and records valid metrics.
"""

import pytest
import numpy as np
from src.calibration.evidence_models import (
    LogisticEvidenceModel,
    ProbabilityCalibrator,
)
from src.calibration.ladder import (
    BaselineLadderRunner,
    BaselineMethod,
)


def test_evidence_source_ablation_metrics():
    """Verify M0, M1, M2 run on the same data and produce valid predictions."""
    model = LogisticEvidenceModel(feature_type="combined")
    rng = np.random.RandomState(42)
    # Generate synthetic training set with 20 claims
    train_recs = [
        {
            "claim_id": f"c_{i}",
            "image_id": f"img_{i // 2}",
            "detector_score": float(rng.uniform(0.1, 0.9)),
            "clip_score": float(rng.uniform(0.1, 0.9)),
            "label": int(rng.choice([0, 1])),
            "split": "train",
        }
        for i in range(20)
    ]
    model.fit(train_recs)

    prob_cal = ProbabilityCalibrator(method="platt")
    prob_cal.fit(np.array([0.2, 0.8, 0.3, 0.7]), np.array([0, 1, 0, 1]))

    runner = BaselineLadderRunner(
        logistic_model=model,
        prob_calibrator=prob_cal,
        epsilon_val=0.25,
        budget_val=0.5,
        coupling_lambda=0.4,
    )

    test_recs = [
        {"claim_id": "t1", "image_id": "img_test1", "detector_score": 0.9, "clip_score": 0.2, "label": 0, "split": "test"},
        {"claim_id": "t2", "image_id": "img_test1", "detector_score": 0.1, "clip_score": 0.8, "label": 1, "split": "test"},
    ]

    res_m0 = runner.run_image(test_recs, method=BaselineMethod.M0_DETECTOR)
    res_m1 = runner.run_image(test_recs, method=BaselineMethod.M1_CLIP)
    res_m2 = runner.run_image(test_recs, method=BaselineMethod.M2_FUSION)

    assert len(res_m0) == 2
    assert len(res_m1) == 2
    assert len(res_m2) == 2

    for r in res_m0:
        assert r.evidence_variant == "detector_only"
        assert 0.0 <= r.nominal_posterior <= 1.0

    for r in res_m1:
        assert r.evidence_variant == "clip_only"
        assert 0.0 <= r.nominal_posterior <= 1.0

    for r in res_m2:
        assert r.evidence_variant == "combined"
        assert 0.0 <= r.nominal_posterior <= 1.0
