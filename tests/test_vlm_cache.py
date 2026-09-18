"""
Tests for disk caching of VLM responses (VLMCache).
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.vlm.provider import VLMGenerationConfig, VLMResponse, SyntheticVLMProvider
from src.vlm.cache import VLMCache


def test_cache_key_determinism_and_sensitivity():
    """Cache key must be deterministic and sensitive to image, prompt, model, provider_kind, and config changes."""
    cfg1 = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="A cat on couch.", max_new_tokens=64)
    cfg2 = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="A cat on couch.", max_new_tokens=64)
    cfg_diff_prompt = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="A dog on couch.", max_new_tokens=64)
    cfg_diff_tokens = VLMGenerationConfig(model_name="llava-1.5-7b", prompt="A cat on couch.", max_new_tokens=128)

    key1 = VLMCache.compute_cache_key("img_hash_1", "llava-1.5-7b", "rev1", "A cat on couch.", cfg1, provider_kind="synthetic_fixture", is_synthetic=True)
    key2 = VLMCache.compute_cache_key("img_hash_1", "llava-1.5-7b", "rev1", "A cat on couch.", cfg2, provider_kind="synthetic_fixture", is_synthetic=True)
    key_diff_img = VLMCache.compute_cache_key("img_hash_2", "llava-1.5-7b", "rev1", "A cat on couch.", cfg1, provider_kind="synthetic_fixture", is_synthetic=True)
    key_diff_rev = VLMCache.compute_cache_key("img_hash_1", "llava-1.5-7b", "rev2", "A cat on couch.", cfg1, provider_kind="synthetic_fixture", is_synthetic=True)
    key_diff_prompt = VLMCache.compute_cache_key("img_hash_1", "llava-1.5-7b", "rev1", "A dog on couch.", cfg_diff_prompt, provider_kind="synthetic_fixture", is_synthetic=True)
    key_diff_tokens = VLMCache.compute_cache_key("img_hash_1", "llava-1.5-7b", "rev1", "A cat on couch.", cfg_diff_tokens, provider_kind="synthetic_fixture", is_synthetic=True)
    key_real_provider = VLMCache.compute_cache_key("img_hash_1", "llava-1.5-7b", "rev1", "A cat on couch.", cfg1, provider_kind="llava_15_hf", is_synthetic=False)

    # Determinism
    assert key1 == key2
    # Sensitivity
    assert key1 != key_diff_img
    assert key1 != key_diff_rev
    assert key1 != key_diff_prompt
    assert key1 != key_diff_tokens
    assert key1 != key_real_provider


def test_cache_put_and_get():
    """Cache put followed by get must return the identical response record."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig()
        provider = SyntheticVLMProvider()

        response = provider.generate_caption(
            image_path="synthetic_img_1.jpg",
            prompt=cfg.prompt,
            gen_config=cfg,
            image_id="img_101",
        )

        # Cache miss initially
        assert cache.get(
            image_hash=response.image_hash,
            model_name=response.model_name,
            model_revision=response.model_revision,
            prompt=response.prompt,
            gen_config=cfg,
            provider_kind=provider.provider_kind,
            is_synthetic=provider.is_synthetic,
        ) is None
        assert cache.stats.misses == 1

        # Write to cache
        cache_path = cache.put(response, gen_config=cfg)
        assert cache_path.exists()
        assert cache.stats.writes == 1

        # Cache hit
        cached = cache.get(
            image_hash=response.image_hash,
            model_name=response.model_name,
            model_revision=response.model_revision,
            prompt=response.prompt,
            gen_config=cfg,
            provider_kind=provider.provider_kind,
            is_synthetic=provider.is_synthetic,
        )
        assert cached is not None
        assert cached.response_id == response.response_id
        assert cached.caption == response.caption
        assert cached.image_hash == response.image_hash
        assert cached.provider_kind == provider.provider_kind
        assert cached.is_synthetic == provider.is_synthetic
        assert cache.stats.hits == 1


def test_cache_synthetic_and_real_isolation():
    """Synthetic and real providers cannot share cache entries."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig(model_name="llava-1.5-7b")

        synth_resp = VLMResponse.create(
            image_id="img_test",
            image_path="img.jpg",
            image_hash="hash_same",
            caption="Synthetic caption.",
            model_name="llava-1.5-7b",
            model_revision="rev_same",
            prompt=cfg.prompt,
            gen_config=cfg,
            is_synthetic=True,
            provider_kind="synthetic_fixture",
        )
        cache.put(synth_resp, gen_config=cfg)

        # Real query for the same image/model/prompt must MISS
        real_query = cache.get(
            image_hash="hash_same",
            model_name="llava-1.5-7b",
            model_revision="rev_same",
            prompt=cfg.prompt,
            gen_config=cfg,
            provider_kind="llava_15_hf",
            is_synthetic=False,
        )
        assert real_query is None


def test_cache_corrupt_entry_handling():
    """Corrupt or incomplete cache files must be ignored gracefully and recorded in stats."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig()
        cache_key = VLMCache.compute_cache_key(
            "hash_1", "model_1", "rev_1", cfg.prompt, cfg,
            provider_kind="synthetic_fixture", is_synthetic=True
        )
        corrupt_file = Path(tmp_dir) / f"{cache_key}.json"

        # Write invalid JSON
        with open(corrupt_file, "w") as f:
            f.write("{invalid_json: true, ...")

        res = cache.get("hash_1", "model_1", "rev_1", cfg.prompt, cfg, provider_kind="synthetic_fixture", is_synthetic=True)
        assert res is None
        assert cache.stats.corrupt_entries == 1
        assert cache.stats.misses == 1


def test_refuse_to_cache_empty_response():
    """Refuse to cache empty captions or missing response IDs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig()

        invalid_resp = VLMResponse(
            response_id="",
            image_id="img_1",
            image_path="img.jpg",
            image_hash="hash_1",
            caption="",
            model_name="model_1",
            model_revision="rev_1",
            prompt="prompt",
            prompt_hash="phash",
            gen_config={},
            device="cpu",
            dtype="float32",
            is_synthetic=True,
            created_at="now",
            provider_kind="synthetic_fixture",
        )

        with pytest.raises(ValueError, match="Refusing to cache empty"):
            cache.put(invalid_resp, gen_config=cfg)
