"""
Unit and integration tests for the visual evidence pipeline (Milestone 6).
"""

from pathlib import Path
import json
import pytest
import tempfile
import math

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    GeneratedResponseRecord,
    DatasetSource,
    SplitName,
)
from src.data.manifests import create_manifest
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.vlm.provider import VLMGenerationConfig, SyntheticVLMProvider, compute_file_sha256
from src.vlm.cache import VLMCache
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.evidence.detector_provider import MockDetectorProvider, HuggingFaceDetectorProvider
from src.evidence.clip_provider import MockCLIPProvider, TransformersCLIPProvider
from src.evidence.pipeline import VisualEvidencePipeline


def create_dummy_image(path: Path, color=(100, 150, 200)) -> Path:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path)
    return path


def test_claim_level_evidence_record_validation():
    """Verify validation of score ranges, finiteness, and explicit availability flags."""
    # 1. Valid record
    rec = ClaimLevelEvidenceRecord(
        claim_id="claim_001_dog",
        image_id="coco_101",
        object_category="dog",
        text_span="dog",
        caption="A dog is in the room.",
        image_hash="hash123",
        split="train",
        detector_score=0.85,
        detector_available=True,
        detector_model="google/owlvit-base-patch32",
        detector_revision="rev123",
        clip_score=0.42,
        similarity_available=True,
        clip_model="openai/clip-vit-base-patch32",
        clip_revision="rev456",
    )
    rec.validate()
    assert rec.detector_score == 0.85
    assert rec.clip_score == 0.42

    # 2. Out-of-bounds detector score
    with pytest.raises(ValueError, match="detector_score must be in"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            detector_score=1.2, detector_available=True,
        )

    with pytest.raises(ValueError, match="detector_score must be in"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            detector_score=-0.1, detector_available=True,
        )

    # 3. Out-of-bounds CLIP cosine similarity score
    with pytest.raises(ValueError, match="clip_score must be in"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            clip_score=1.05, similarity_available=True,
        )

    with pytest.raises(ValueError, match="clip_score must be in"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            clip_score=-1.05, similarity_available=True,
        )

    # 4. Inconsistent availability flags
    with pytest.raises(ValueError, match="detector_score cannot be None when detector_available=True"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            detector_score=None, detector_available=True,
        )

    with pytest.raises(ValueError, match="detector_score must be None when detector_available=False"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            detector_score=0.5, detector_available=False,
        )

    with pytest.raises(ValueError, match="clip_score cannot be None when similarity_available=True"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            clip_score=None, similarity_available=True,
        )

    with pytest.raises(ValueError, match="clip_score must be None when similarity_available=False"):
        ClaimLevelEvidenceRecord(
            claim_id="c1", image_id="i1", object_category="dog",
            clip_score=0.3, similarity_available=False,
        )


def test_claim_level_evidence_record_round_trip():
    """Verify dictionary serialization and deserialization."""
    rec = ClaimLevelEvidenceRecord(
        claim_id="claim_001_cat",
        image_id="coco_102",
        object_category="cat",
        text_span="cat",
        caption="A sleeping cat on the couch.",
        image_hash="hash999",
        split="train",
        detector_score=0.91,
        detector_available=True,
        detector_model="google/owlvit-base-patch32",
        detector_revision="rev_owl",
        clip_score=0.68,
        similarity_available=True,
        clip_model="openai/clip-vit-base-patch32",
        clip_revision="rev_clip",
        metadata={"note": "test_note"},
    )
    d = rec.to_dict()
    rec2 = ClaimLevelEvidenceRecord.from_dict(d)
    assert rec == rec2


def test_mock_detector_and_clip_providers(tmp_path):
    """Verify Mock providers produce valid bounded values and simulate failures cleanly."""
    img_file = create_dummy_image(tmp_path / "test.jpg")

    # 1. Normal execution
    det = MockDetectorProvider(fixed_scores={"dog": 0.88})
    clip = MockCLIPProvider(fixed_scores={"dog": 0.35})

    d_res = det.detect_category(img_file, "dog")
    assert d_res.available is True
    assert d_res.score == 0.88
    assert d_res.error is None

    c_res = clip.compute_similarity(img_file, "dog")
    assert c_res.available is True
    assert c_res.score == 0.35
    assert c_res.error is None

    # 2. Failure execution: must not silently return 0.0
    det_fail = MockDetectorProvider(simulate_failure=True, failure_error_message="Detector crashed")
    clip_fail = MockCLIPProvider(simulate_failure=True, failure_error_message="CLIP crashed")

    d_fail_res = det_fail.detect_category(img_file, "dog")
    assert d_fail_res.available is False
    assert d_fail_res.score is None
    assert d_fail_res.error == "Detector crashed"

    c_fail_res = clip_fail.compute_similarity(img_file, "dog")
    assert c_fail_res.available is False
    assert c_fail_res.score is None
    assert c_fail_res.error == "CLIP crashed"


def test_visual_evidence_pipeline_train_split_isolation(tmp_path):
    """Verify pipeline rejects non-TRAIN split images to preserve split isolation."""
    img_file = create_dummy_image(tmp_path / "val_img.jpg")
    img_rec = ImageRecord(
        image_id="coco_val_01",
        dataset_source=DatasetSource.COCO,
        file_name=str(img_file),
    )
    entry = DatasetManifestEntry(image=img_rec, split=SplitName.VALIDATION)
    manifest = create_manifest("val_manifest", entries=[entry])

    cache = VLMCache(tmp_path / "cache")
    vlm = SyntheticVLMProvider()
    pipe = VisualEvidencePipeline(
        vlm_provider=vlm,
        vlm_cache=cache,
        detector_provider=MockDetectorProvider(),
        clip_provider=MockCLIPProvider(),
        enforce_train_split=True,
    )

    with pytest.raises(ValueError, match="strictly mandates TRAIN split images"):
        pipe.run(manifest)


def test_visual_evidence_pipeline_end_to_end(tmp_path):
    """
    Test complete pipeline:
    Image -> VLM generation/cache -> claim extraction -> detector -> CLIP -> JSONL export.
    """
    img1 = create_dummy_image(tmp_path / "img1.jpg", color=(100, 150, 200))
    img2 = create_dummy_image(tmp_path / "img2.jpg", color=(200, 150, 100))

    hash1 = compute_file_sha256(img1)
    hash2 = compute_file_sha256(img2)
    assert hash1 != hash2

    entry1 = DatasetManifestEntry(
        image=ImageRecord(image_id="coco_01", dataset_source=DatasetSource.COCO, file_name=str(img1), file_hash=hash1),
        split=SplitName.TRAIN,
    )
    entry2 = DatasetManifestEntry(
        image=ImageRecord(image_id="coco_02", dataset_source=DatasetSource.COCO, file_name=str(img2), file_hash=hash2),
        split=SplitName.TRAIN,
    )
    manifest = create_manifest("test_manifest", entries=[entry1, entry2])

    cache = VLMCache(tmp_path / "cache")

    # Use SyntheticVLMProvider with explicit mock captions
    mock_caps = {
        "coco_01": "A red car and a golden retriever dog are outside.",
        "coco_02": "A person holding an umbrella in the rain.",
    }
    vlm = SyntheticVLMProvider(mock_captions=mock_caps)
    cfg = VLMGenerationConfig(model_name="synthetic-vlm")


    det = MockDetectorProvider(fixed_scores={"car": 0.85, "dog": 0.92, "person": 0.78, "umbrella": 0.65})
    clip = MockCLIPProvider(fixed_scores={"car": 0.45, "dog": 0.55, "person": 0.38, "umbrella": 0.42})

    pipe = VisualEvidencePipeline(
        vlm_provider=vlm,
        vlm_cache=cache,
        claim_extractor=ConservativeClaimExtractor(create_coco_category_registry()),
        detector_provider=det,
        clip_provider=clip,
        gen_config=cfg,
    )

    out_jsonl = tmp_path / "claim_level_evidence.jsonl"
    records, stats = pipe.run(manifest, output_jsonl_path=out_jsonl)

    assert stats.images_processed == 2
    assert stats.total_claims_extracted == 4  # car, dog, person, umbrella
    assert stats.detector_available_count == 4
    assert stats.clip_available_count == 4
    assert stats.unavailable_or_failed_count == 0
    assert stats.uncalibrated_raw_evidence_guaranteed is True
    assert stats.no_pgm_parameters_inferred_guaranteed is True

    # Check JSONL file on disk
    assert out_jsonl.exists()
    with open(out_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    assert len(lines) == 4
    categories = {l["object_category"] for l in lines}
    assert categories == {"car", "dog", "person", "umbrella"}

    for l in lines:
        assert 0.0 <= l["detector_score"] <= 1.0
        assert -1.0 <= l["clip_score"] <= 1.0
        assert l["detector_available"] is True
        assert l["similarity_available"] is True
        assert l["split"] == "train"
        # Confirm PGM fields are NOT present
        assert "theta" not in l
        assert "epsilon" not in l
        assert "J_ij" not in l
        assert "posterior" not in l
