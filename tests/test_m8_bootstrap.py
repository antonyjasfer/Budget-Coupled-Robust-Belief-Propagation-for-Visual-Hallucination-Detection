"""Tests for Milestone 8 Image Cluster Bootstrap & Case Studies."""

import pytest

from src.experiments.configs import M8ExperimentConfig
from src.experiments.loaders import create_synthetic_smoke_bundle
from src.experiments.inference_runner import InferenceRunner
from src.experiments.aggregation import (
    run_image_cluster_bootstrap,
    run_method_comparison_bootstrap,
    select_case_studies,
)


def test_image_cluster_bootstrap():
    cfg = M8ExperimentConfig()
    bundle = create_synthetic_smoke_bundle(num_images=4, seed=42)
    runner = InferenceRunner(cfg, bundle=bundle)
    results = runner.run_all(splits=["all"])

    boot_stats = run_image_cluster_bootstrap(
        results,
        method_name="standard_bp",
        n_resamples=50,
        confidence_level=0.95,
        seed=42,
    )

    assert "accuracy" in boot_stats
    assert "f1" in boot_stats
    assert boot_stats["accuracy"].ci_lower is not None
    assert boot_stats["accuracy"].ci_upper is not None
    assert boot_stats["accuracy"].ci_lower <= boot_stats["accuracy"].ci_upper


def test_method_comparison_bootstrap():
    cfg = M8ExperimentConfig()
    bundle = create_synthetic_smoke_bundle(num_images=4, seed=42)
    runner = InferenceRunner(cfg, bundle=bundle)
    results = runner.run_all(splits=["all"])

    comp_stats = run_method_comparison_bootstrap(
        results,
        method_a="robust_bp",
        method_b="standard_bp",
        n_resamples=50,
        seed=42,
    )

    assert "accuracy" in comp_stats
    assert comp_stats["accuracy"].diff_estimate is not None


def test_case_studies_selection():
    cfg = M8ExperimentConfig()
    bundle = create_synthetic_smoke_bundle(num_images=4, seed=42)
    runner = InferenceRunner(cfg, bundle=bundle)
    results = runner.run_all(splits=["all"])

    cases = select_case_studies(results)
    assert len(cases) > 0
    categories = {c.category_tag for c in cases}
    assert "high_confidence_correct" in categories or "narrow_interval" in categories or "wide_interval" in categories
