"""Tests for Milestone 9 Results Consolidation and Formatting."""

import json
from pathlib import Path
import pytest

from src.experiments.inference_runner import ClaimEvaluationResult
from src.scientific.consolidator import M9ResultsConsolidator


def _make_dummy_claim_results():
    recs = []
    for i in range(10):
        is_hall = (i % 2 == 0)
        c = ClaimEvaluationResult(
            run_id="run_test",
            experiment_name="test_exp",
            condition="clean",
            image_id=f"img_{i % 3}",
            claim_id=f"c_{i}",
            object_category="object",
            text_span=f"A dog {i}",
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


def test_consolidator_creates_all_artifacts(tmp_path):
    out_dir = tmp_path / "m9_final"
    consolidator = M9ResultsConsolidator(output_dir=out_dir, allow_overwrite=True)
    claims = _make_dummy_claim_results()

    consolidated = consolidator.consolidate(claims, execution_mode="DEVELOPMENT")

    # Check top-level files
    assert (out_dir / "final_claim_results.jsonl").exists()
    assert (out_dir / "final_image_results.jsonl").exists()
    assert (out_dir / "final_metrics.json").exists()
    assert (out_dir / "final_intervals.json").exists()
    assert (out_dir / "final_statistics.json").exists()
    assert (out_dir / "final_manifest.json").exists()
    assert (out_dir / "research_summary.md").exists()
    assert (out_dir / "contribution_statement.md").exists()

    # Check tables 1 to 10
    tables_dir = out_dir / "tables"
    for i in range(1, 11):
        assert list(tables_dir.glob(f"table{i}_*.json")), f"Table {i} json missing"
        assert list(tables_dir.glob(f"table{i}_*.csv")), f"Table {i} csv missing"
        assert list(tables_dir.glob(f"table{i}_*.md")), f"Table {i} md missing"
        assert list(tables_dir.glob(f"table{i}_*.tex")), f"Table {i} tex missing"

    # Check figures 1 to 14
    fig_dir = out_dir / "figures"
    for i in range(1, 15):
        assert list(fig_dir.glob(f"fig{i}_*.png")), f"Figure {i} png missing"
        assert list(fig_dir.glob(f"fig{i}_*.pdf")), f"Figure {i} pdf missing"


def test_consolidator_overwrite_protection(tmp_path):
    out_dir = tmp_path / "protected"
    out_dir.mkdir(parents=True)
    (out_dir / "final_manifest.json").write_text("{}", encoding="utf-8")

    consolidator = M9ResultsConsolidator(output_dir=out_dir, allow_overwrite=False)
    with pytest.raises(FileExistsError):
        consolidator.initialize_directory()
