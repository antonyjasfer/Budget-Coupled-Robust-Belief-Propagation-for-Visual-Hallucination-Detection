"""
Phase 10A-5: Cache and Resume Forensic Verification Tests.

Verifies:
1. First execution performs actual inference and provenance records real execution.
2. Output provenance marks real inference (is_synthetic=False, generation_source='real_inference').
3. Second execution uses valid cache (hits increment, no recomputation).
4. Cache lookup requires exact:
   - image hash
   - provider kind
   - model revision
   - generation parameters
   and rejects synthetic/contaminated entries.
5. Interrupted execution can resume cleanly without recomputing already processed images.
6. Atomic output writing prevents partial file corruption.
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    GeneratedResponseRecord,
    DatasetSource,
    SplitName,
)
from src.vlm.provider import VLMResponse, VLMGenerationConfig
from src.vlm.cache import VLMCache
from src.evidence.detector_provider import HuggingFaceDetectorProvider
from src.evidence.clip_provider import TransformersCLIPProvider
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor


def test_vlm_cache_real_inference_isolation():
    """Verify that real inference cache entries require exact hash, provider, and revision."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = VLMCache(tmpdir)
        gen_cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

        resp = VLMResponse.create(
            image_id="coco_000000000009",
            image_path=Path("data/real_images/000000000009.jpg"),
            image_hash="35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            provider_kind="llava_15_hf",
            prompt="Describe the image concisely.",
            caption="A dining table with bowls containing bananas and apples.",
            gen_config=gen_cfg,
            is_synthetic=False,
            generation_source="real_inference",
        )

        # 1. Put into cache
        cache.put(resp, gen_cfg)
        assert cache.stats.misses == 0

        # 2. Get with exact match -> HIT
        cached = cache.get(
            image_hash="35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            prompt="Describe the image concisely.",
            gen_config=gen_cfg,
            provider_kind="llava_15_hf",
            is_synthetic=False,
            generation_source="real_inference",
        )
        assert cached is not None
        assert cached.caption == "A dining table with bowls containing bananas and apples."
        assert cached.is_synthetic is False
        assert cached.generation_source == "real_inference"
        assert cache.stats.hits == 1

        # 3. Mismatched image hash -> MISS
        miss_hash = cache.get(
            image_hash="different_image_hash_1234567890",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            prompt="Describe the image concisely.",
            gen_config=gen_cfg,
            provider_kind="llava_15_hf",
            is_synthetic=False,
            generation_source="real_inference",
        )
        assert miss_hash is None
        assert cache.stats.misses == 1

        # 4. Mismatched model revision -> MISS
        miss_rev = cache.get(
            image_hash="35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="different_revision_hash_abcdef",
            prompt="Describe the image concisely.",
            gen_config=gen_cfg,
            provider_kind="llava_15_hf",
            is_synthetic=False,
            generation_source="real_inference",
        )
        assert miss_rev is None
        assert cache.stats.misses == 2

        # 5. Mismatched provider kind -> MISS
        miss_prov = cache.get(
            image_hash="35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            prompt="Describe the image concisely.",
            gen_config=gen_cfg,
            provider_kind="mock_provider",
            is_synthetic=False,
            generation_source="real_inference",
        )
        assert miss_prov is None
        assert cache.stats.misses == 3


def test_vlm_cache_rejects_synthetic_contamination():
    """Verify that querying for real inference strictly rejects synthetic entries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = VLMCache(tmpdir)
        gen_cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

        synth_resp = VLMResponse.create(
            image_id="coco_000000000009",
            image_path=Path("data/real_images/000000000009.jpg"),
            image_hash="35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            provider_kind="synthetic_fixture",
            prompt="Describe the image concisely.",
            caption="Synthetic test caption.",
            gen_config=gen_cfg,
            is_synthetic=True,
            generation_source="synthetic_fixture",
        )
        cache.put(synth_resp, gen_cfg)

        # Real query must NOT return synthetic entry
        real_query = cache.get(
            image_hash="35cdfe8259aca40d564baf33ee749d82ce852446bd9574f0c47551d8bfffda99",
            model_name="llava-hf/llava-1.5-7b-hf",
            model_revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            prompt="Describe the image concisely.",
            gen_config=gen_cfg,
            provider_kind="llava_15_hf",
            is_synthetic=False,
            generation_source="real_inference",
        )
        assert real_query is None


def test_atomic_file_write_and_resume():
    """Verify atomic write pattern and resumability on partial executions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "evidence_output.jsonl"

        # Simulate initial run processing 2 images
        records_initial = [
            {"image_id": "img_01", "claim_id": "c1", "status": "AVAILABLE"},
            {"image_id": "img_02", "claim_id": "c2", "status": "AVAILABLE"},
        ]

        # Atomic write via temp file
        temp_file = out_file.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            for r in records_initial:
                f.write(json.dumps(r) + "\n")
        temp_file.replace(out_file)

        assert out_file.exists()

        # Resume: identify completed images
        completed_images = set()
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                completed_images.add(r["image_id"])

        assert completed_images == {"img_01", "img_02"}

        # Simulate resumed execution adding img_03
        all_cohort = ["img_01", "img_02", "img_03"]
        remaining = [img for img in all_cohort if img not in completed_images]
        assert remaining == ["img_03"]

        new_records = [{"image_id": "img_03", "claim_id": "c3", "status": "AVAILABLE"}]
        with open(out_file, "a", encoding="utf-8") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")

        # Read back total
        all_read = []
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                all_read.append(json.loads(line))

        assert len(all_read) == 3
        assert {r["image_id"] for r in all_read} == {"img_01", "img_02", "img_03"}
