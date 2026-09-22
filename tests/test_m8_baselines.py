"""Tests for Milestone 8 Baselines and Inference Runners."""

import pytest

from src.experiments.loaders import create_synthetic_smoke_bundle
from src.experiments.configs import PGMParameterConfig
from src.experiments.parameterization import build_image_pgm_context
from src.data.schemas import DecisionStatus
from src.experiments.baselines import (
    run_evidence_point_baseline,
    run_standard_bp_baseline,
    run_robust_bp_proposed,
    run_robust_bp_no_coupling,
    run_robust_bp_independent_box,
)


def test_baselines_execution():
    bundle = create_synthetic_smoke_bundle(num_images=1, seed=42)
    claims = bundle.claims
    cfg = PGMParameterConfig(base_coupling_j=0.4, budget_ratio=1.0, decision_threshold=0.5)

    ctx = build_image_pgm_context(
        image_id=claims[0].image_id,
        claim_records=claims,
        config=cfg,
    )

    cid = claims[0].claim_id

    # 1. Point baseline
    pt_res = run_evidence_point_baseline(claims[0], threshold=0.5)
    assert 0.0 <= pt_res.hallucination_score <= 1.0
    assert pt_res.predicted_decision in [DecisionStatus.SUPPORTED, DecisionStatus.HALLUCINATED]

    # 2. Standard BP
    std_map = run_standard_bp_baseline(ctx)
    assert cid in std_map
    std_res = std_map[cid]
    assert 0.0 <= std_res.hallucination_score <= 1.0
    assert std_res.predicted_decision in [DecisionStatus.SUPPORTED, DecisionStatus.HALLUCINATED]

    # 3. Robust BP Proposed
    rob_map = run_robust_bp_proposed(ctx)
    assert cid in rob_map
    rob_res = rob_map[cid]
    assert 0.0 <= rob_res.interval_lower <= rob_res.interval_upper <= 1.0
    assert rob_res.interval_width == pytest.approx(rob_res.interval_upper - rob_res.interval_lower, abs=1e-5)
    assert isinstance(rob_res.predicted_decision, DecisionStatus)

    # 4. Robust BP No Coupling
    nc_map = run_robust_bp_no_coupling(ctx)
    assert cid in nc_map
    nc_res = nc_map[cid]
    assert 0.0 <= nc_res.interval_lower <= nc_res.interval_upper <= 1.0

    # 5. Independent Box
    box_map = run_robust_bp_independent_box(ctx)
    assert cid in box_map
    box_res = box_map[cid]
    assert 0.0 <= box_res.interval_lower <= box_res.interval_upper <= 1.0
    # Independent box has no global budget constraint, so its interval should be >= proposed interval
    assert box_res.interval_width >= rob_res.interval_width - 1e-4
