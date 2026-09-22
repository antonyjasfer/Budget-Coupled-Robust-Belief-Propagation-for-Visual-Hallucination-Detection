"""Tests for Milestone 8 Parameterization & Tree Construction."""

import pytest
import numpy as np

from src.experiments.loaders import create_synthetic_smoke_bundle
from src.experiments.configs import PGMParameterConfig, TreeTopology
from src.experiments.parameterization import (
    build_image_pgm_context,
    compute_unary_theta,
    compute_uncertainty_epsilon,
)


def test_unary_theta_computation():
    # detector = 0.5 -> neutral / theta = 0.0
    th = compute_unary_theta(0.5, None)
    assert abs(th) < 1e-4

    # High visual support (det = 0.95) -> SUPPORTED (h_i = -1) -> negative theta
    th_supp = compute_unary_theta(0.95, None)
    assert th_supp < 0.0

    # Low visual support (det = 0.05) -> HALLUCINATED (h_i = +1) -> positive theta
    th_halluc = compute_unary_theta(0.05, None)
    assert th_halluc > 0.0


def test_uncertainty_epsilon_computation():
    eps = compute_uncertainty_epsilon(
        detector_score=0.5,
        clip_score=0.2,
        base_epsilon=0.5,
        epsilon_scale=1.0,
    )
    assert eps > 0.0
    assert eps <= 2.0  # bounded


def test_build_image_pgm_context_chain():
    bundle = create_synthetic_smoke_bundle(num_images=1, seed=42)
    claims = bundle.claims
    cfg = PGMParameterConfig(
        topology=TreeTopology.CHAIN,
        fixed_budget=1.5,
        base_coupling_j=0.6,
    )

    ctx = build_image_pgm_context(
        image_id=claims[0].image_id,
        claim_records=claims,
        config=cfg,
    )

    n = len(claims)
    assert ctx.model.num_nodes == n
    assert len(ctx.model.theta) == n
    assert len(ctx.model.epsilon) == n
    assert ctx.budget == 1.5
    assert len(ctx.parameter_hash) == 64  # SHA256
