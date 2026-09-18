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
