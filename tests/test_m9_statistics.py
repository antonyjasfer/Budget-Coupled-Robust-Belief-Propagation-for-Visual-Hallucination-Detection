"""Tests for Milestone 9 Statistical Analysis and Hypothesis Testing."""

import pytest
from src.experiments.inference_runner import ClaimEvaluationResult
from src.scientific.statistics import (
    compute_m9_paired_bootstrap,
    compute_correlation_hypotheses,
    categorize_errors,
    select_m9_case_studies,
)


def _make_statistical_claims():
    recs = []
    for i in range(20):
        is_hall = (i % 2 == 0)
        c = ClaimEvaluationResult(
            run_id="run_stat",
            experiment_name="test_stat",
            condition="clean",
            image_id=f"img_{i // 4}",
            claim_id=f"stat_c_{i}",
            object_category="object",
            text_span=f"Claim text {i}",
            split="test",
            ground_truth="hallucinated" if is_hall else "supported",
            detector_score=0.85 if is_hall else 0.15,
            clip_score=0.80 if is_hall else 0.20,
            theta_i=0.6 if is_hall else -0.6,
            epsilon_i=0.2,
            coupling_j=0.5,
            budget=1.0,
            epsilon_scale=1.0,
            decision_threshold=0.5,
            grid_steps=21,
            standard_posterior=0.90 if is_hall else 0.10,
            standard_prediction="hallucinated" if is_hall else "supported",
            robust_lower=0.70 if is_hall else 0.05,
            robust_upper=0.98 if is_hall else 0.30,
            robust_midpoint=0.84 if is_hall else 0.175,
            interval_width=0.28 if is_hall else 0.25,
            certified_lower=0.70 if is_hall else 0.05,
            certified_upper=0.98 if is_hall else 0.30,
            field_cert_gap=0.0,
            robust_prediction="hallucinated" if is_hall else "supported",
            evidence_sensitive=False,
            abstained=False,
            corruption_type="none",
            corruption_severity="clean",
            runtime_ms=3.0,
        )
        recs.append(c)
    return recs


def test_paired_bootstrap_reproducibility():
    claims = _make_statistical_claims()
    b1 = compute_m9_paired_bootstrap(claims, n_bootstrap=100, seed=42)
    b2 = compute_m9_paired_bootstrap(claims, n_bootstrap=100, seed=42)

    assert b1["diff_accuracy"]["mean"] == pytest.approx(b2["diff_accuracy"]["mean"])
    assert b1["diff_accuracy"]["ci_lower"] == pytest.approx(b2["diff_accuracy"]["ci_lower"])
    assert b1["diff_accuracy"]["ci_upper"] == pytest.approx(b2["diff_accuracy"]["ci_upper"])


def test_paired_bootstrap_bounds():
    claims = _make_statistical_claims()
    b = compute_m9_paired_bootstrap(claims, n_bootstrap=200, seed=123)

    for metric_name in ["diff_accuracy", "diff_f1", "diff_roc_auc"]:
        m_dict = b[metric_name]
        assert m_dict["ci_lower"] <= m_dict["ci_upper"]
        assert -1.0 <= m_dict["mean"] <= 1.0


def test_correlation_hypotheses():
    claims = _make_statistical_claims()
    corrs = compute_correlation_hypotheses(claims)
    sig_names = [c.signal_name for c in corrs]

    assert "width_vs_standard_bp_error" in sig_names or "width_vs_error" in sig_names
    for c in corrs:
        assert c.sample_size >= 0


def test_error_categorization():
    claims = _make_statistical_claims()
    err_c = ClaimEvaluationResult(
        run_id="run_stat",
        experiment_name="test_stat",
        condition="clean",
        image_id="img_0",
        claim_id="err_c_1",
        object_category="object",
        text_span="Error claim",
        split="test",
        ground_truth="supported",
        detector_score=0.9,
        clip_score=0.9,
        theta_i=0.8,
        epsilon_i=0.2,
        coupling_j=0.5,
        budget=1.0,
        epsilon_scale=1.0,
        decision_threshold=0.5,
        grid_steps=21,
        standard_posterior=0.9,
        standard_prediction="hallucinated",
        robust_lower=0.8,
        robust_upper=0.95,
        robust_midpoint=0.875,
        interval_width=0.15,
        certified_lower=0.8,
        certified_upper=0.95,
        field_cert_gap=0.0,
        robust_prediction="hallucinated",
        evidence_sensitive=False,
        abstained=False,
        corruption_type="none",
        corruption_severity="clean",
        runtime_ms=3.0,
    )
    claims.append(err_c)

    cats = categorize_errors(claims)
    cat_ids = [c.category_id for c in cats]
    assert "ERR_01_FALSE_HALLUCINATION" in cat_ids or "false_hallucination" in cat_ids


def test_case_studies_selection():
    claims = _make_statistical_claims()
    cases = select_m9_case_studies(claims)
    assert len(cases) >= 1
    case_ids = [c.category_id for c in cases]
    assert any("CASE" in cid for cid in case_ids)
