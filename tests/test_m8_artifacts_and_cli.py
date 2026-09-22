"""End-to-end and CLI integration tests for Milestone 8."""

import pytest
import tempfile
import subprocess
import sys
from pathlib import Path

from scripts.run_m8_experiments import run_experiments
from scripts.validate_m8_artifacts import validate_m8_artifacts


def test_smoke_experiment_end_to_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "m8_smoke_test"
        
        # Run smoke experiment programmatically
        run_dir = run_experiments(
            smoke=True,
            output_dir=str(out_dir),
            no_plots=False,
            no_corruption=False,
            seed=42,
        )

        assert run_dir.exists()
        assert (run_dir / "experiment_manifest.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "summary.md").exists()
        assert (run_dir / "case_studies.md").exists()
        assert (run_dir / "raw" / "claim_evaluation_results.jsonl").exists()
        assert (run_dir / "raw" / "claim_evaluation_results.csv").exists()

        # Check tables
        tables_dir = run_dir / "tables"
        assert (tables_dir / "table1_dataset_summary.json").exists()
        assert (tables_dir / "table2_method_comparison.json").exists()
        assert (tables_dir / "table3_interval_statistics.json").exists()
        assert (tables_dir / "table4_ablation_study.json").exists()
        assert (tables_dir / "table5_budget_sensitivity.json").exists()
        assert (tables_dir / "table6_corruption_robustness.json").exists()
        assert (tables_dir / "table7_computational_cost.json").exists()

        # Check figures
        figures_dir = run_dir / "figures"
        assert (figures_dir / "fig1_confusion_standard_bp.png").exists()
        assert (figures_dir / "fig2_confusion_robust_bp.png").exists()
        assert (figures_dir / "fig3_roc_curves.png").exists()
        assert (figures_dir / "fig4_interval_width_distribution.png").exists()
        assert (figures_dir / "fig5_posterior_vs_midpoint.png").exists()
        assert (figures_dir / "fig6_representative_intervals.png").exists()

        # Validate with artifact validator
        is_valid = validate_m8_artifacts(run_dir, require_plots=True)
        assert is_valid is True


def test_m8_cli_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "m8_cli_smoke"
        cmd = [
            sys.executable,
            "scripts/run_m8_experiments.py",
            "--smoke",
            "--output-dir",
            str(out_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"CLI failed: {res.stderr}\n{res.stdout}"
        assert (out_dir / "experiment_manifest.json").exists()
