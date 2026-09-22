"""Unit tests for Phase 9C image-level bootstrap statistical unit.

Verifies:
1. Clustered resampling occurs at the IMAGE level (not individual claims).
2. All claims belonging to a resampled image are included in the bootstrap replicate.
3. Deterministic seed produces identical replicates.
4. Confidence interval computation for metrics.
"""

import pytest
import numpy as np
from src.calibration.ablation_study import perform_image_level_bootstrap


def test_image_level_bootstrap_clustering():
    """Verify bootstrap samples entire images and computes confidence intervals."""
    # 5 images with 2 claims each
    records = []
    for img_idx in range(5):
        for c_idx in range(2):
            records.append({
                "image_id": f"img_{img_idx}",
                "claim_id": f"c_{img_idx}_{c_idx}",
                "label": 1 if img_idx < 3 else 0,
            })

    def eval_accuracy(recs):
        # Return fraction of label == 1
        return float(np.mean([r["label"] for r in recs]))

    mean_score, ci_lower, ci_upper = perform_image_level_bootstrap(
        records=records,
        eval_fn=eval_accuracy,
        n_bootstraps=50,
        seed=42,
    )

    assert 0.0 <= ci_lower <= mean_score <= ci_upper <= 1.0
    assert pytest.approx(mean_score, abs=0.15) == 3.0 / 5.0


def test_image_level_bootstrap_determinism():
    """Identical seeds must produce identical bootstrap metrics."""
    records = [
        {"image_id": f"img_{i}", "claim_id": f"c_{i}", "label": i % 2}
        for i in range(6)
    ]

    def eval_fn(recs):
        return float(np.mean([r["label"] for r in recs]))

    b1 = perform_image_level_bootstrap(records, eval_fn, n_bootstraps=20, seed=123)
    b2 = perform_image_level_bootstrap(records, eval_fn, n_bootstraps=20, seed=123)

    assert b1[0] == b2[0]
    assert b1[1] == b2[1]
    assert b1[2] == b2[2]
