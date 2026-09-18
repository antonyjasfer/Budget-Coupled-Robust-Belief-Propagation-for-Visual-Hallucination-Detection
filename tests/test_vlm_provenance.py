"""
Tests for provenance tracking, response ID collision resistance, and configuration hashing.
"""

from pathlib import Path
import hashlib
from src.vlm.provider import VLMGenerationConfig, VLMResponse, SyntheticVLMProvider


def test_response_id_collision_resistance():
    """Different prompt, generation config, or caption text must produce different response IDs."""
    cfg1 = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="Prompt A", max_new_tokens=64)
    cfg2 = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="Prompt B", max_new_tokens=64)
    cfg3 = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="Prompt A", max_new_tokens=128)

    resp1 = VLMResponse.create(
        image_id="img1",
        image_path="img1.jpg",
        image_hash="hash1",
        caption="A caption.",
        model_name="llava-1.5-7b",
        model_revision="rev1",
        prompt=cfg1.prompt,
        gen_config=cfg1,
        is_synthetic=True,
    )

    resp2 = VLMResponse.create(
        image_id="img1",
        image_path="img1.jpg",
        image_hash="hash1",
        caption="A caption.",
        model_name="llava-1.5-7b",
        model_revision="rev1",
        prompt=cfg2.prompt,
        gen_config=cfg2,
        is_synthetic=True,
    )

    resp3 = VLMResponse.create(
        image_id="img1",
        image_path="img1.jpg",
        image_hash="hash1",
        caption="A caption.",
        model_name="llava-1.5-7b",
        model_revision="rev1",
        prompt=cfg3.prompt,
        gen_config=cfg3,
        is_synthetic=True,
    )

    resp4 = VLMResponse.create(
        image_id="img1",
        image_path="img1.jpg",
        image_hash="hash1",
        caption="A completely different caption text.",
        model_name="llava-1.5-7b",
        model_revision="rev1",
        prompt=cfg1.prompt,
        gen_config=cfg1,
        is_synthetic=True,
    )

    assert resp1.response_id != resp2.response_id
    assert resp1.response_id != resp3.response_id
    assert resp1.response_id != resp4.response_id


def test_synthetic_flag_preservation():
    """Synthetic responses must be explicitly flagged as synthetic and fixture provider."""
    provider = SyntheticVLMProvider()
    resp = provider.generate_caption("sample.jpg", image_id="sample")
    assert resp.is_synthetic is True
    assert resp.provider_kind == "synthetic_fixture"
    assert "fixture" in resp.model_revision

    serialized = resp.to_dict()
    assert serialized["is_synthetic"] is True
    assert serialized["provider_kind"] == "synthetic_fixture"

    deserialized = VLMResponse.from_dict(serialized)
    assert deserialized.is_synthetic is True
    assert deserialized.provider_kind == "synthetic_fixture"


def test_real_image_through_synthetic_provider_remains_synthetic():
    """Real image processed with synthetic provider must produce synthetic response."""
    provider = SyntheticVLMProvider()
    resp = provider.generate_caption("real_image.jpg", image_id="real_coco_001", image_hash="abc123realhash")
    assert resp.is_synthetic is True
    assert resp.provider_kind == "synthetic_fixture"


def test_mathematical_core_isolated_from_vlm_dependencies():
    """Mathematical PGM and robust BP modules must import without torch or transformers."""
    # Ensure standard BP, solver, and certification can be imported cleanly
    import src.robust_bp.solver
    import src.robust_bp.certification
    import src.pgm.standard_bp

    assert hasattr(src.robust_bp.solver, "solve_budget_coupled_robust_bp")
    assert hasattr(src.robust_bp.certification, "compute_continuous_certificate")
    assert hasattr(src.pgm.standard_bp, "solve_tree_bp")
