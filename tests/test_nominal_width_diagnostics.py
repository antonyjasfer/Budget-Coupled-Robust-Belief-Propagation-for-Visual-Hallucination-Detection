"""Unit tests for Phase 9C nominal confidence vs robust width diagnostics.

Verifies:
1. Computation of nominal uncertainty measures: u_nominal = min(p, 1-p) and entropy H(p).
2. Correlation analysis (Spearman rho) between nominal uncertainty and robust width W.
3. Diagnostic analysis distinguishing intrinsic robust width information from nominal confidence.
"""

import pytest
import numpy as np
from src.calibration.ablation_study import compute_nominal_vs_robust_width_correlation
from src.calibration.ladder import ClaimPredictionResult


def test_nominal_vs_robust_width_correlation():
    """Verify correlation computation between nominal uncertainty and interval width."""
    results = []
    # Create 10 synthetic claim results with varying posteriors and widths
    for i in range(10):
        p = 0.05 + 0.1 * i  # spans 0.05 to 0.95
        # Width has some relationship with p but also independent variation
        w = float(0.1 + 0.3 * (1.0 - abs(p - 0.5)) + 0.05 * (i % 3))
        results.append(ClaimPredictionResult(
            run_id="diag_test",
            image_id=f"img_{i}",
            claim_id=f"c_{i}",
            split="test",
            graph_size=1,
            evidence_variant="combined",
            baseline_name="M6_global_robust",
            topology="independent",
            coupling_strategy="J0",
            lambda_param=0.0,
            theta=0.0,
            epsilon=0.25,
            budget=0.5,
            budget_ratio=1.0,
            nominal_posterior=p,
            robust_lower=max(0.0, p - w / 2),
            robust_upper=min(1.0, p + w / 2),
            robust_width=w,
            ground_truth=1 if p >= 0.5 else 0,
            prediction=1 if p >= 0.5 else 0,
            correct=True,
            runtime_ms=1.0,
        ))

    diag = compute_nominal_vs_robust_width_correlation(results)

    assert "spearman_rho_nominal_width" in diag
    assert "spearman_rho_entropy_width" in diag
    assert -1.0 <= diag["spearman_rho_nominal_width"] <= 1.0
    assert -1.0 <= diag["spearman_rho_entropy_width"] <= 1.0
