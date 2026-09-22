"""Tests for Milestone 9 Paper-Ready Results Package and Reporting."""

from pathlib import Path
import pytest

from src.experiments.inference_runner import ClaimEvaluationResult
from src.scientific.consolidator import M9ResultsConsolidator


def _make_dummy_claims():
    recs = []
    for i in range(6):
        is_hall = (i % 2 == 0)
        c = ClaimEvaluationResult(
            run_id="run_rep",
            experiment_name="test_rep",
            condition="clean",
            image_id=f"rep_img_{i // 2}",
            claim_id=f"rep_c_{i}",
            object_category="object",
            text_span=f"Reporting claim {i}",
            split="test",
            ground_truth="hallucinated" if is_hall else "supported",
            detector_score=0.8 if is_hall else 0.2,
            clip_score=0.75 if is_hall else 0.25,
            theta_i=0.5 if is_hall else -0.5,
            epsilon_i=0.2,
            coupling_j=0.5,
            budget=1.0,
            epsilon_scale=1.0,
            decision_threshold=0.5,
            grid_steps=21,
            standard_posterior=0.85 if is_hall else 0.15,
            standard_prediction="hallucinated" if is_hall else "supported",
            robust_lower=0.6 if is_hall else 0.05,
            robust_upper=0.95 if is_hall else 0.35,
            robust_midpoint=0.775 if is_hall else 0.20,
            interval_width=0.35 if is_hall else 0.30,
            certified_lower=0.6 if is_hall else 0.05,
            certified_upper=0.95 if is_hall else 0.35,
            field_cert_gap=0.0,
            robust_prediction="hallucinated" if is_hall else "supported",
            evidence_sensitive=False,
            abstained=False,
            corruption_type="none",
            corruption_severity="clean",
            runtime_ms=2.0,
        )
        recs.append(c)
    return recs


def test_paper_package_structure(tmp_path):
    out_dir = tmp_path / "paper_pkg"
    consolidator = M9ResultsConsolidator(output_dir=out_dir, allow_overwrite=True)
    claims = _make_dummy_claims()
    consolidator.consolidate(claims, execution_mode="DEVELOPMENT")

    paper_dir = out_dir / "paper_results"

    # 1. Discussion section must distinguish FACT, INTERPRETATION, LIMITATION
    disc_text = (paper_dir / "discussion_section.md").read_text(encoding="utf-8")
    assert "FACT" in disc_text
    assert "INTERPRETATION" in disc_text
    assert "LIMITATION" in disc_text

    # 2. Results section must document methods and intervals
    res_text = (paper_dir / "results_section.md").read_text(encoding="utf-8")
    assert "Primary Classification Performance" in res_text
    assert "Robust Posterior Interval Characterization" in res_text

    # 3. Contribution statement must avoid "first ever" claims
    contrib_text = (out_dir / "contribution_statement.md").read_text(encoding="utf-8")
    assert "first ever" not in contrib_text.lower()
    assert "budget" in contrib_text.lower()

    # 4. Colab runbook must exist
    runbook_path = out_dir.parent / "final_colab_runbook.md"
    assert runbook_path.exists()
    runbook_text = runbook_path.read_text(encoding="utf-8")
    assert "Google Colab" in runbook_text
