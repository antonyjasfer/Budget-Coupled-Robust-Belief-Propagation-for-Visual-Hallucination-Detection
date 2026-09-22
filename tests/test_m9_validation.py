"""Tests for Milestone 9 Artifact Validator."""

import json
from pathlib import Path
import pytest

from scripts.validate_m9_final import validate_m9_artifacts
from src.experiments.inference_runner import ClaimEvaluationResult
from src.scientific.consolidator import M9ResultsConsolidator


def _make_dummy_claims():
    recs = []
    for i in range(6):
        is_hall = (i % 2 == 0)
        c = ClaimEvaluationResult(
            run_id="run_val",
            experiment_name="test_val",
            condition="clean",
            image_id=f"val_img_{i // 2}",
            claim_id=f"val_c_{i}",
            object_category="object",
            text_span=f"Validation claim {i}",
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
            runtime_ms=2.5,
        )
        recs.append(c)
    return recs


def test_validator_fails_on_empty_dir(tmp_path):
    passed, errors = validate_m9_artifacts(tmp_path)
    assert not passed
    assert len(errors) > 0


def test_validator_passes_on_complete_package(tmp_path):
    out_dir = tmp_path / "valid_pkg"
    consolidator = M9ResultsConsolidator(output_dir=out_dir, allow_overwrite=True)
    claims = _make_dummy_claims()
    consolidator.consolidate(claims, execution_mode="DEVELOPMENT")

    passed, errors = validate_m9_artifacts(out_dir)
    assert passed, f"Validation failed with errors: {errors}"
    assert len(errors) == 0


def test_validator_detects_inverted_bounds(tmp_path):
    out_dir = tmp_path / "inverted_pkg"
    consolidator = M9ResultsConsolidator(output_dir=out_dir, allow_overwrite=True)
    claims = _make_dummy_claims()
    consolidator.consolidate(claims, execution_mode="DEVELOPMENT")

    claim_file = out_dir / "final_claim_results.jsonl"
    lines = claim_file.read_text(encoding="utf-8").strip().split("\n")
    bad_rec = json.loads(lines[0])
    bad_rec["robust_lower"] = 0.95
    bad_rec["robust_upper"] = 0.10  # L > U
    lines[0] = json.dumps(bad_rec)
    claim_file.write_text("\n".join(lines), encoding="utf-8")

    passed, errors = validate_m9_artifacts(out_dir)
    assert not passed
    assert any("inverted bounds" in e.lower() for e in errors)
