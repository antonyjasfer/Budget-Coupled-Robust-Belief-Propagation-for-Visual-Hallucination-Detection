"""
Unit and integration tests for the visual evidence pipeline (Milestone 6).
"""

from pathlib import Path
import json
import pytest
import tempfile
import math
from unittest.mock import patch, MagicMock
import torch

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
        assert l["vlm_generation_source"] in ("real_inference", "cache", "synthetic_fixture")
        # Confirm PGM fields are NOT present
        assert "theta" not in l
        assert "epsilon" not in l
        assert "J_ij" not in l
        assert "posterior" not in l


def test_vlm_cache_contamination_prevention(tmp_path):
    """
    Ensure the cache prevents contamination:
    A cache record should NOT be treated as a valid real LLaVA response merely because
    is_synthetic == False.
    If generation_source is manual or synthetic, cache retrieval for real inference MUST reject it.
    """
    cache = VLMCache(tmp_path / "cache")
    cfg = VLMGenerationConfig(model_name="llava-hf/llava-1.5-7b-hf")
    img_file = create_dummy_image(tmp_path / "test_img.jpg")
    img_hash = compute_file_sha256(img_file)

    from src.vlm.provider import VLMResponse

    # 1. Store a contaminated record where is_synthetic=False, but generation_source="manual_prepopulated"
    manual_resp = VLMResponse.create(
        image_id="coco_contam_1",
        image_path=img_file,
        image_hash=img_hash,
        caption="A manually fabricated caption pretending to be real.",
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="test_rev",
        prompt="Describe the image.",
        gen_config=cfg,
        is_synthetic=False,
        provider_kind="manual",
        generation_source="manual_prepopulated",
    )
    cache.put(response=manual_resp, gen_config=cfg)

    # 2. Querying cache for real inference (is_synthetic=False, generation_source="real_inference")
    # MUST return None because the key or content does not match genuine real inference
    cached_entry = cache.get(
        image_hash=img_hash,
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="test_rev",
        prompt="Describe the image.",
        gen_config=cfg,
        provider_kind="llava_15_hf",
        is_synthetic=False,
        generation_source="real_inference",
    )
    assert cached_entry is None, "Cache failed to reject contaminated manual response!"

    # 3. Store a genuine real inference record
    real_resp = VLMResponse.create(
        image_id="coco_real_1",
        image_path=img_file,
        image_hash=img_hash,
        caption="A real LLaVA caption produced by neural weights.",
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="test_rev",
        prompt="Describe the image.",
        gen_config=cfg,
        is_synthetic=False,
        provider_kind="llava_15_hf",
        generation_source="real_inference",
    )
    cache.put(response=real_resp, gen_config=cfg)

    # 4. Querying cache for real inference must successfully return the genuine real inference record
    cached_real = cache.get(
        image_hash=img_hash,
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="test_rev",
        prompt="Describe the image.",
        gen_config=cfg,
        provider_kind="llava_15_hf",
        is_synthetic=False,
        generation_source="real_inference",
    )
    assert cached_real is not None
    assert cached_real.caption == "A real LLaVA caption produced by neural weights."
    assert cached_real.generation_source == "real_inference"


def test_pipeline_provenance_and_source_propagation(tmp_path):
    """
    Test that VisualEvidencePipeline sets vlm_generation_source to:
    - 'synthetic_fixture' on fresh generation when using SyntheticVLMProvider
    - 'cache' when loaded from a valid cached entry
    """
    img = create_dummy_image(tmp_path / "img_prov.jpg")
    img_hash = compute_file_sha256(img)

    entry = DatasetManifestEntry(
        image=ImageRecord(image_id="coco_prov_01", dataset_source=DatasetSource.COCO, file_name=str(img), file_hash=img_hash),
        split=SplitName.TRAIN,
    )
    manifest = create_manifest("prov_manifest", entries=[entry])

    cache = VLMCache(tmp_path / "cache")
    vlm = SyntheticVLMProvider(mock_captions={"coco_prov_01": "A bird sitting on a tree branch."})
    cfg = VLMGenerationConfig(model_name="synthetic-vlm")

    pipe = VisualEvidencePipeline(
        vlm_provider=vlm,
        vlm_cache=cache,
        claim_extractor=ConservativeClaimExtractor(create_coco_category_registry()),
        detector_provider=MockDetectorProvider(fixed_scores={"bird": 0.82}),
        clip_provider=MockCLIPProvider(fixed_scores={"bird": 0.51}),
        gen_config=cfg,
    )

    # Run 1: Fresh inference from synthetic provider
    records_1, _ = pipe.run(manifest)
    assert len(records_1) == 1
    assert records_1[0].vlm_generation_source == "synthetic_fixture"

    # Run 2: Re-run with the same cache -> provenance must indicate "cache"
    records_2, _ = pipe.run(manifest)
    assert len(records_2) == 1
    assert records_2[0].vlm_generation_source == "cache"


def test_independent_device_configuration_argparse():
    """Verify that --device (VLM) and --evidence-device can be configured independently."""
    from experiments.run_full_evidence_pipeline import parse_args

    # 1. Defaults: --evidence-device defaults to 'cpu', --device follows torch.cuda.is_available()
    with patch("sys.argv", ["run_full_evidence_pipeline.py"]):
        args = parse_args()
        assert args.evidence_device == "cpu"
        expected_vlm_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        assert args.device == expected_vlm_device

    # 2. Independent configuration: VLM on cuda:0, evidence on cpu
    with patch("sys.argv", ["run_full_evidence_pipeline.py", "--device", "cuda:0", "--evidence-device", "cpu"]):
        args = parse_args()
        assert args.device == "cuda:0"
        assert args.evidence_device == "cpu"

    # 3. Independent configuration: VLM on cuda:1, evidence on cuda:0
    with patch("sys.argv", ["run_full_evidence_pipeline.py", "--device", "cuda:1", "--evidence-device", "cuda:0"]):
        args = parse_args()
        assert args.device == "cuda:1"
        assert args.evidence_device == "cuda:0"


def test_evidence_providers_and_claim_records_device_provenance(tmp_path):
    """
    Verify HuggingFaceDetectorProvider and TransformersCLIPProvider preserve
    independent device assignments in their configuration/provenance payloads,
    and that ClaimLevelEvidenceRecord records them faithfully.
    """
    # 1. Detector device provenance
    det_cpu = HuggingFaceDetectorProvider(model_name="google/owlvit-base-patch32", model_revision="test_rev", device="cpu")
    assert det_cpu.device == "cpu"
    res_det_cpu = det_cpu.detect_category(tmp_path / "nonexistent.jpg", "dog")
    assert res_det_cpu.configuration["device"] == "cpu"

    det_cuda = HuggingFaceDetectorProvider(model_name="google/owlvit-base-patch32", model_revision="test_rev", device="cuda:0")
    assert det_cuda.device == "cuda:0"
    res_det_cuda = det_cuda.detect_category(tmp_path / "nonexistent.jpg", "dog")
    assert res_det_cuda.configuration["device"] == "cuda:0"

    # 2. CLIP device provenance
    clip_cpu = TransformersCLIPProvider(model_name="openai/clip-vit-base-patch32", model_revision="test_rev", device="cpu")
    assert clip_cpu.device == "cpu"
    res_clip_cpu = clip_cpu.compute_similarity(tmp_path / "nonexistent.jpg", "dog")
    assert res_clip_cpu.preprocessing_configuration["device"] == "cpu"

    clip_cuda = TransformersCLIPProvider(model_name="openai/clip-vit-base-patch32", model_revision="test_rev", device="cuda:0")
    assert clip_cuda.device == "cuda:0"
    res_clip_cuda = clip_cuda.compute_similarity(tmp_path / "nonexistent.jpg", "dog")
    assert res_clip_cuda.preprocessing_configuration["device"] == "cuda:0"

    # 3. ClaimLevelEvidenceRecord captures independent device provenance
    rec = ClaimLevelEvidenceRecord(
        claim_id="claim_test_dev",
        image_id="coco_test_01",
        object_category="dog",
        detector_score=0.75,
        detector_available=True,
        detector_model="google/owlvit-base-patch32",
        detector_configuration={"device": "cpu", "prompt_template": "a photo of a {category}"},
        clip_score=0.45,
        similarity_available=True,
        clip_model="openai/clip-vit-base-patch32",
        preprocessing_configuration={"device": "cpu", "normalization": "l2"},
        metadata={"vlm_device": "cuda:0", "vlm_provider_kind": "llava_15_hf"},
    )
    rec.validate()
    d = rec.to_dict()
    assert d["detector_configuration"]["device"] == "cpu"
    assert d["preprocessing_configuration"]["device"] == "cpu"
    assert d["metadata"]["vlm_device"] == "cuda:0"


def test_allow_download_propagation_to_detector_and_clip():
    """Verify that --allow-download sets local_files_only=False on detector and CLIP."""
    from experiments.run_full_evidence_pipeline import parse_args

    # 1. When --allow-download is passed
    with patch("sys.argv", ["run_full_evidence_pipeline.py", "--allow-download"]):
        args = parse_args()
        assert args.allow_download is True
        detector = HuggingFaceDetectorProvider(
            model_name="google/owlvit-base-patch32",
            local_files_only=not args.allow_download,
        )
        assert detector.local_files_only is False

        clip = TransformersCLIPProvider(
            model_name="openai/clip-vit-base-patch32",
            local_files_only=not args.allow_download,
        )
        assert clip.local_files_only is False

    # 2. When --allow-download is omitted
    with patch("sys.argv", ["run_full_evidence_pipeline.py"]):
        args = parse_args()
        assert args.allow_download is False
        detector = HuggingFaceDetectorProvider(
            model_name="google/owlvit-base-patch32",
            local_files_only=not args.allow_download,
        )
        assert detector.local_files_only is True

        clip = TransformersCLIPProvider(
            model_name="openai/clip-vit-base-patch32",
            local_files_only=not args.allow_download,
        )
        assert clip.local_files_only is True


def test_safe_none_reporting():
    """Verify that format_evidence_score and reporting loop handle None scores safely."""
    from experiments.run_full_evidence_pipeline import format_evidence_score

    assert format_evidence_score(None) == "UNAVAILABLE"
    assert format_evidence_score(0.85432) == "0.8543"
    assert format_evidence_score(-0.12345) == "-0.1235"

    rec = ClaimLevelEvidenceRecord(
        claim_id="c_none",
        image_id="img_none",
        object_category="cat",
        detector_score=None,
        detector_available=False,
        clip_score=None,
        similarity_available=False,
    )
    rec.validate()

    det_str = format_evidence_score(rec.detector_score)
    clip_str = format_evidence_score(rec.clip_score)
    formatted = f"Detector: {det_str} | CLIP: {clip_str}"
    assert formatted == "Detector: UNAVAILABLE | CLIP: UNAVAILABLE"


def test_unavailable_evidence_produces_controlled_failure(capsys):
    """Verify that unavailable evidence outputs diagnostics and exits with code 1 without TypeError."""
    from experiments.run_full_evidence_pipeline import validate_evidence_results

    rec_fail = ClaimLevelEvidenceRecord(
        claim_id="c_fail_01",
        image_id="img_01",
        object_category="person",
        detector_score=None,
        detector_available=False,
        clip_score=None,
        similarity_available=False,
        metadata={
            "detector_error": "Cannot find model weights locally",
            "clip_error": "Connection timed out",
        },
    )
    rec_fail.validate()

    mock_stats = MagicMock()
    mock_stats.unavailable_or_failed_count = 1

    with pytest.raises(SystemExit) as exc_info:
        validate_evidence_results([rec_fail], mock_stats)

    assert exc_info.value.code == 1
    captured = capsys.readouterr().out
    assert "CRITICAL EXPERIMENT FAILURE: UNAVAILABLE OR FAILED EVIDENCE DETECTED" in captured
    assert "c_fail_01" in captured
    assert "person" in captured
    assert "Cannot find model weights locally" in captured
    assert "Connection timed out" in captured


def test_valid_evidence_produces_normal_output(capsys):
    """Verify that complete valid evidence passes validation cleanly."""
    from experiments.run_full_evidence_pipeline import validate_evidence_results

    rec_ok = ClaimLevelEvidenceRecord(
        claim_id="c_ok_01",
        image_id="img_01",
        object_category="dog",
        detector_score=0.88,
        detector_available=True,
        clip_score=0.62,
        similarity_available=True,
    )
    rec_ok.validate()

    mock_stats = MagicMock()
    mock_stats.unavailable_or_failed_count = 0

    # Should not raise SystemExit
    validate_evidence_results([rec_ok], mock_stats)
    captured = capsys.readouterr().out
    assert "CRITICAL EXPERIMENT FAILURE" not in captured


