"""
Tests for development-only VLM pilot pipeline and annotation bundle generator.
"""

import json
from pathlib import Path
import tempfile
import pytest

from src.data.coco import load_coco_instances
from src.data.splits import create_dataset_splits
from src.data.manifests import create_manifest
from src.data.schemas import SplitName
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.vlm.provider import VLMGenerationConfig, SyntheticVLMProvider
from src.vlm.cache import VLMCache
from src.vlm.pipeline import (
    select_pilot_images,
    run_vlm_pilot,
    export_annotation_bundle,
    AnnotationBundle,
    AnnotationReviewStatus,
)


@pytest.fixture
def synthetic_manifest_and_splits():
    fixture_dir = Path(__file__).parent / "fixtures"
    coco_path = fixture_dir / "coco_instances_synthetic.json"
    reserved_path = fixture_dir / "reserved_external_ids.json"

    with open(reserved_path, "r") as f:
        reserved_ids = set(json.load(f))

    adapter = load_coco_instances(coco_path)
    entries = adapter.to_manifest_entries()
    manifest = create_manifest("test_coco_manifest", entries=entries)

    split_reg = create_dataset_splits(
        manifest=manifest,
        reserved_external_image_ids=reserved_ids,
        train_ratio=0.4,
        val_ratio=0.1,
        calib_ratio=0.1,
        test_ratio=0.4,
        seed=42,
    )
    return manifest, split_reg


def test_select_pilot_images_strictly_train_split(synthetic_manifest_and_splits):
    """Pilot image selection must strictly draw from TRAIN split and respect sample_size."""
    manifest, split_reg = synthetic_manifest_and_splits
    selected = select_pilot_images(manifest, split_registry=split_reg.registry, sample_size=3, seed=42)

    assert len(selected) <= 3
    assert len(selected) > 0
    for entry in selected:
        img_id = entry.image.image_id
        assert split_reg.registry[img_id] == SplitName.TRAIN.value


def test_select_pilot_images_rejects_non_train():
    """If no training images are available, pilot selection raises ValueError."""
    fixture_dir = Path(__file__).parent / "fixtures"
    coco_path = fixture_dir / "coco_instances_synthetic.json"
    adapter = load_coco_instances(coco_path)
    entries = adapter.to_manifest_entries()
    manifest = create_manifest("test_manifest", entries=entries)

    # Set all to test split
    all_test_registry = {e.image.image_id: SplitName.TEST.value for e in entries}

    with pytest.raises(ValueError, match="No images found in TRAIN split"):
        select_pilot_images(manifest, split_registry=all_test_registry, sample_size=5)


def test_run_vlm_pilot_offline_synthetic_pipeline(synthetic_manifest_and_splits):
    """End-to-end pilot execution with synthetic provider and claim extraction."""
    manifest, split_reg = synthetic_manifest_and_splits
    fixtures_m4 = Path(__file__).parent / "fixtures" / "milestone4" / "mock_vlm_responses.json"

    with open(fixtures_m4, "r") as f:
        mock_captions = json.load(f)

    provider = SyntheticVLMProvider(mock_captions=mock_captions)
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(category_registry=registry)
    cfg = VLMGenerationConfig()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)

        # First run (all misses)
        bundle, stats1 = run_vlm_pilot(
            manifest=manifest,
            provider=provider,
            cache=cache,
            extractor=extractor,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=3,
            seed=42,
        )

        assert stats1.selected_image_count == 3
        assert stats1.cache_misses == 3
        assert stats1.cache_hits == 0
        assert stats1.generated_captions == 3
        assert bundle.total_responses == 3
        assert len(bundle.entries) == 3

        # Verify all entries are initially unreviewed
        for entry in bundle.entries:
            assert entry.review_status == AnnotationReviewStatus.NOT_YET_REVIEWED.value
            assert entry.split == "train"

        # Second run (all hits, idempotent)
        bundle2, stats2 = run_vlm_pilot(
            manifest=manifest,
            provider=provider,
            cache=cache,
            extractor=extractor,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=3,
            seed=42,
        )

        assert stats2.selected_image_count == 3
        assert stats2.cache_misses == 0
        assert stats2.cache_hits == 3
        assert stats2.generated_captions == 0
        assert bundle2.total_accepted_claims == bundle.total_accepted_claims



def test_export_and_load_annotation_bundle():
    """Verify bundle serialization and deserialization."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test_bundle.json"
        bundle = AnnotationBundle(
            bundle_id="bundle_test_001",
            created_at="2026-09-18T00:00:00Z",
            total_responses=1,
            total_accepted_claims=2,
            total_rejected_mentions=1,
            entries=[],
            metadata={"test": True},
        )

        saved = export_annotation_bundle(bundle, out_file)
        assert saved.exists()

        with open(saved, "r") as f:
            data = json.load(f)

        loaded = AnnotationBundle.from_dict(data)
        assert loaded.bundle_id == "bundle_test_001"
        assert loaded.total_responses == 1
        assert loaded.total_accepted_claims == 2


def test_production_pilot_rejects_synthetic_fallback(synthetic_manifest_and_splits):
    """Production provider mode must reject any synthetic fallback responses."""
    manifest, split_reg = synthetic_manifest_and_splits
    
    class FakeRealProviderThatReturnsSynthetic:
        is_synthetic: bool = False
        provider_kind: str = "llava_15_hf"
        
        def get_model_info(self):
            return {"model_name": "llava-fake", "resolved_revision": "rev1", "is_synthetic": False, "provider_kind": "llava_15_hf"}
            
        def generate_caption(self, image_path, prompt=None, gen_config=None, image_id=None, image_hash=None):
            from src.vlm.provider import VLMResponse
            cfg = gen_config or VLMGenerationConfig()
            # Returns a response with is_synthetic=True
            return VLMResponse.create(
                image_id=image_id or "img1",
                image_path=image_path,
                image_hash=image_hash or "hash1",
                caption="Fake caption.",
                model_name="llava-fake",
                model_revision="rev1",
                prompt=prompt or cfg.prompt,
                gen_config=cfg,
                is_synthetic=True,
                provider_kind="synthetic_fixture",
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        with pytest.raises(RuntimeError, match="Silent synthetic fallback is forbidden"):
            run_vlm_pilot(
                manifest=manifest,
                provider=FakeRealProviderThatReturnsSynthetic(),
                cache=cache,
                split_registry=split_reg.registry,
                sample_size=2,
            )


def test_real_provider_revision_lifecycle_and_cache_hit(synthetic_manifest_and_splits):
    """
    Regression test: Real provider resolves actual revision, caches on 1st run, hits cache on 2nd run,
    and never serializes 'not_loaded' into response provenance or bundle metadata.
    """
    manifest, split_reg = synthetic_manifest_and_splits

    class MockRealLLaVAProvider:
        is_synthetic: bool = False
        provider_kind: str = "llava_15_hf"
        generation_count: int = 0

        def __init__(self, model_revision: str = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"):
            self.model_name = "llava-hf/llava-1.5-7b-hf"
            self.model_revision = model_revision
            self.device = "cpu"
            self.dtype = "float32"
            self.local_files_only = True
            self.provider_kind = "llava_15_hf"

        def resolve_revision(self) -> str:
            return self.model_revision

        def get_model_info(self):
            rev = self.resolve_revision()
            return {
                "model_name": self.model_name,
                "model_revision": rev,
                "resolved_revision": rev,
                "is_synthetic": False,
                "provider_kind": self.provider_kind,
                "device": self.device,
                "dtype": self.dtype,
                "local_files_only": self.local_files_only,
                "provider_type": "LLaVA15Provider",
            }

        def generate_caption(self, image_path, prompt=None, gen_config=None, image_id=None, image_hash=None):
            self.generation_count += 1
            from src.vlm.provider import VLMResponse
            cfg = gen_config or VLMGenerationConfig()
            return VLMResponse.create(
                image_id=image_id or "img1",
                image_path=image_path,
                image_hash=image_hash or "hash_real_1",
                caption="A real cat and a dog on the floor.",
                model_name=self.model_name,
                model_revision=self.model_revision,
                prompt=prompt or cfg.prompt,
                gen_config=cfg,
                is_synthetic=False,
                provider_kind=self.provider_kind,
                execution_time_seconds=0.05,
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

        # Run 1: First request with fresh provider (cache miss, 1 generation)
        provider1 = MockRealLLaVAProvider()
        bundle1, stats1 = run_vlm_pilot(
            manifest=manifest,
            provider=provider1,
            cache=cache,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=1,
            seed=42,
        )

        assert stats1.cache_hits == 0
        assert stats1.cache_misses == 1
        assert stats1.generated_captions == 1
        assert provider1.generation_count == 1

        # Check provenance and bundle metadata
        assert bundle1.entries[0].model_revision == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        assert bundle1.entries[0].model_revision != "not_loaded"
        assert bundle1.metadata["model_info"]["resolved_revision"] == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        assert bundle1.metadata["model_info"]["model_revision"] == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"

        # Run 2: Identical second request with a new fresh provider instance (cache hit, 0 generation)
        provider2 = MockRealLLaVAProvider()
        bundle2, stats2 = run_vlm_pilot(
            manifest=manifest,
            provider=provider2,
            cache=cache,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=1,
            seed=42,
        )

        assert stats2.cache_hits == 1
        assert stats2.cache_misses == 0
        assert stats2.generated_captions == 0
        assert provider2.generation_count == 0  # Crucial: Must not invoke generation again

        assert bundle2.entries[0].model_revision == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        assert bundle2.metadata["model_info"]["resolved_revision"] == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
        assert bundle2.metadata["model_info"]["model_revision"] == "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"


def test_real_provider_changed_revision_creates_cache_miss(synthetic_manifest_and_splits):
    """Changed model revision must cause a cache miss."""
    manifest, split_reg = synthetic_manifest_and_splits

    class MockRealProvider:
        is_synthetic: bool = False
        provider_kind: str = "llava_15_hf"

        def __init__(self, rev: str):
            self.model_name = "llava-hf/llava-1.5-7b-hf"
            self.model_revision = rev
            self.provider_kind = "llava_15_hf"

        def resolve_revision(self) -> str:
            return self.model_revision

        def get_model_info(self):
            return {
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "resolved_revision": self.model_revision,
                "is_synthetic": False,
                "provider_kind": self.provider_kind,
                "provider_type": "LLaVA15Provider",
            }

        def generate_caption(self, image_path, prompt=None, gen_config=None, image_id=None, image_hash=None):
            from src.vlm.provider import VLMResponse
            cfg = gen_config or VLMGenerationConfig()
            return VLMResponse.create(
                image_id=image_id or "img1",
                image_path=image_path,
                image_hash=image_hash or "hash1",
                caption="A chair and a table.",
                model_name=self.model_name,
                model_revision=self.model_revision,
                prompt=prompt or cfg.prompt,
                gen_config=cfg,
                is_synthetic=False,
                provider_kind=self.provider_kind,
                execution_time_seconds=0.01,
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")

        # Run with revision 1
        bundle1, stats1 = run_vlm_pilot(
            manifest=manifest,
            provider=MockRealProvider(rev="rev_alpha"),
            cache=cache,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=1,
            seed=42,
        )
        assert stats1.cache_misses == 1
        assert stats1.cache_hits == 0

        # Run with revision 2 (different revision -> cache miss)
        bundle2, stats2 = run_vlm_pilot(
            manifest=manifest,
            provider=MockRealProvider(rev="rev_beta"),
            cache=cache,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=1,
            seed=42,
        )
        assert stats2.cache_misses == 1
        assert stats2.cache_hits == 0


def test_real_and_synthetic_provider_isolation_in_pipeline(synthetic_manifest_and_splits):
    """Synthetic and real providers remain strictly isolated in cache during pipeline runs."""
    manifest, split_reg = synthetic_manifest_and_splits

    class MockRealProvider:
        is_synthetic: bool = False
        provider_kind: str = "llava_15_hf"
        model_revision: str = "rev_shared_name"

        def __init__(self):
            self.model_name = "test-model"

        def resolve_revision(self) -> str:
            return self.model_revision

        def get_model_info(self):
            return {
                "model_name": self.model_name,
                "model_revision": self.model_revision,
                "resolved_revision": self.model_revision,
                "is_synthetic": False,
                "provider_kind": self.provider_kind,
                "provider_type": "LLaVA15Provider",
            }

        def generate_caption(self, image_path, prompt=None, gen_config=None, image_id=None, image_hash=None):
            from src.vlm.provider import VLMResponse
            cfg = gen_config or VLMGenerationConfig()
            return VLMResponse.create(
                image_id=image_id or "img1",
                image_path=image_path,
                image_hash=image_hash or "hash1",
                caption="A real car in the driveway.",
                model_name=self.model_name,
                model_revision=self.model_revision,
                prompt=prompt or cfg.prompt,
                gen_config=cfg,
                is_synthetic=False,
                provider_kind=self.provider_kind,
                execution_time_seconds=0.01,
            )

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = VLMCache(tmp_dir)
        cfg = VLMGenerationConfig(model_name="test-model")

        # 1. Run with synthetic provider
        synth_provider = SyntheticVLMProvider(
            model_name="test-model",
            model_revision="rev_shared_name",
        )
        bundle_s, stats_s = run_vlm_pilot(
            manifest=manifest,
            provider=synth_provider,
            cache=cache,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=1,
            seed=42,
        )
        assert stats_s.generated_captions == 1
        assert bundle_s.entries[0].is_synthetic is True

        # 2. Run with real provider (must MISS and not consume synthetic cache)
        real_provider = MockRealProvider()
        bundle_r, stats_r = run_vlm_pilot(
            manifest=manifest,
            provider=real_provider,
            cache=cache,
            gen_config=cfg,
            split_registry=split_reg.registry,
            sample_size=1,
            seed=42,
        )
        assert stats_r.cache_misses == 1
        assert stats_r.cache_hits == 0
        assert stats_r.generated_captions == 1
        assert bundle_r.entries[0].is_synthetic is False

