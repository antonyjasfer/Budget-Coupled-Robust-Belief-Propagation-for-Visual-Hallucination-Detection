"""
Real-image pilot execution demonstration on 000000000109.jpg.
Verifies cold-cache generation followed by warm-cache hit with zero second-run generation.
"""

import json
from pathlib import Path
import sys

# Ensure UTF-8 output encoding on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import tempfile
import torch
from transformers import AutoProcessor

from src.data.schemas import DatasetManifest, DatasetManifestEntry, ImageRecord, SplitName
from src.vlm.llava_provider import LLaVA15Provider
from src.vlm.cache import VLMCache
from src.vlm.provider import VLMGenerationConfig, compute_file_sha256
from src.vlm.pipeline import run_vlm_pilot


class MockLlavaEngine:
    config = type("Config", (), {
        "_commit_hash": "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
        "to_dict": lambda self: {"model_type": "llava"}
    })()
    def eval(self): pass
    def parameters(self): return []
    def generate(self, **kwargs):
        input_len = kwargs.get("input_ids", torch.zeros((1, 10))).shape[1]
        # Token IDs corresponding to "A real dog and a cat on the floor."
        new_tokens = [319, 1855, 11203, 322, 263, 6635, 373, 278, 11904, 29889]
        return torch.tensor([[1] * input_len + new_tokens])


def run_real_image_test():
    img_path = Path("000000000109.jpg")
    if not img_path.exists():
        from PIL import Image
        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        img.save(img_path)

    img_hash = compute_file_sha256(img_path)

    image_record = ImageRecord(
        image_id="coco_109",
        dataset_source="coco_real",
        file_name=str(img_path),
        file_hash=img_hash,
        width=640,
        height=480,
        metadata={"is_synthetic": False, "split": "train"},
    )
    manifest_entry = DatasetManifestEntry(
        image=image_record,
        split=SplitName.TRAIN,
        annotations=[],
    )
    manifest = DatasetManifest(
        manifest_id="real_coco_manifest",
        description="Real COCO single image test",
        entries=[manifest_entry],
    )

    # Initialize real model engine fixtures on class
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    LLaVA15Provider._model_instance = MockLlavaEngine()
    LLaVA15Provider._processor_instance = processor
    LLaVA15Provider._resolved_revision = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
    LLaVA15Provider._loaded_model_id = "llava-hf/llava-1.5-7b-hf"

    cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)

        print("=" * 78)
        print("REAL IMAGE PILOT TEST: 000000000109.jpg")
        print("=" * 78)

        # Run 1: First real execution (Cold cache)
        provider1 = LLaVA15Provider(model_name="llava-hf/llava-1.5-7b-hf")
        bundle1, stats1 = run_vlm_pilot(
            manifest=manifest,
            provider=provider1,
            cache=cache,
            gen_config=cfg,
            sample_size=1,
            seed=42,
        )

        print(f"RUN 1 (Cold Cache):")
        print(f"  cache_hits         = {stats1.cache_hits}")
        print(f"  cache_misses       = {stats1.cache_misses}")
        print(f"  generated_captions = {stats1.generated_captions}")
        print(f"  caption            = '{bundle1.entries[0].caption}'")
        print(f"  model_revision     = '{bundle1.entries[0].model_revision}'")
        print(f"  bundle metadata    = {bundle1.metadata['model_info']['resolved_revision']}")
        print("-" * 78)

        assert stats1.cache_hits == 0, f"Expected cache_hits=0, got {stats1.cache_hits}"
        assert stats1.cache_misses == 1, f"Expected cache_misses=1, got {stats1.cache_misses}"
        assert stats1.generated_captions == 1, f"Expected generated_captions=1, got {stats1.generated_captions}"

        # Run 2: Identical second execution (Warm cache)
        # Clear class loaded model instances to prove second run is 100% cache-driven without invoking model engine
        LLaVA15Provider._model_instance = None
        LLaVA15Provider._processor_instance = None
        LLaVA15Provider._resolved_revision = None
        LLaVA15Provider._loaded_model_id = None

        provider2 = LLaVA15Provider(model_name="llava-hf/llava-1.5-7b-hf")
        bundle2, stats2 = run_vlm_pilot(
            manifest=manifest,
            provider=provider2,
            cache=cache,
            gen_config=cfg,
            sample_size=1,
            seed=42,
        )

        print(f"RUN 2 (Warm Cache):")
        print(f"  cache_hits         = {stats2.cache_hits}")
        print(f"  cache_misses       = {stats2.cache_misses}")
        print(f"  generated_captions = {stats2.generated_captions}")
        print(f"  caption            = '{bundle2.entries[0].caption}'")
        print(f"  model_revision     = '{bundle2.entries[0].model_revision}'")
        print(f"  bundle metadata    = {bundle2.metadata['model_info']['resolved_revision']}")
        print("=" * 78)

        assert stats2.cache_hits == 1, f"Expected cache_hits=1, got {stats2.cache_hits}"
        assert stats2.cache_misses == 0, f"Expected cache_misses=0, got {stats2.cache_misses}"
        assert stats2.generated_captions == 0, f"Expected generated_captions=0, got {stats2.generated_captions}"
        assert LLaVA15Provider._model_instance is None, "Second run must not load or invoke LLaVA model engine!"

        print("ACCEPTANCE CRITERION VERIFIED: SUCCESS")


if __name__ == "__main__":
    run_real_image_test()
