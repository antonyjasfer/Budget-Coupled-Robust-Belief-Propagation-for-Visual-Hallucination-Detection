"""
Integration tests for end-to-end M7 workflow and real M6 export compatibility.
"""

from pathlib import Path
import json
import pytest

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    DatasetSource,
    SplitName,
    GroundTruthStatus,
)
from src.data.manifests import create_manifest
from src.data.splits import split_manifest_by_image_groups
from src.evidence.schemas import ClaimLevelEvidenceRecord
from src.annotation.schemas import (
    M7ClaimRecord,
    AnnotationRecord,
    AdjudicationRecord,
)
from src.annotation.workflow import (
    load_m6_evidence_file,
    ingest_m6_evidence_to_m7_claims,
    export_masked_templates,
    build_final_ground_truth_dataset,
    generate_quality_report,
)


def test_real_m6_export_compatibility(tmp_path):
    """
    CRITICAL REAL-DATA COMPATIBILITY TEST:
    Verifies that the actual M6 export (data/exports/claim_level_evidence.jsonl)
    can be ingested, processed, and audited honestly without errors or fabricated annotations.
    """
    real_m6_path = Path("data/exports/claim_level_evidence.jsonl")
    assert real_m6_path.exists(), "M6 export file missing from data/exports/"

    # 1. Ingest real M6 records
    evidence_records = load_m6_evidence_file(real_m6_path)
    assert len(evidence_records) == 15, f"Expected 15 M6 records, got {len(evidence_records)}"

    # 2. Convert to M7 claims
    m7_claims = ingest_m6_evidence_to_m7_claims(evidence_records, manifest=None)
    assert len(m7_claims) == 15
    assert all(not c.is_synthetic for c in m7_claims)

    # 3. Export masked templates
    template_dir = tmp_path / "templates"
    tpl_a, tpl_b = export_masked_templates(m7_claims, output_dir=template_dir)
    assert tpl_a.exists() and tpl_b.exists()

    # Verify template contains 15 lines and no model scores
    with open(tpl_a, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    assert len(lines) == 15
    for item in lines:
        assert "detector_score" not in item
        assert "clip_score" not in item
        assert "split" not in item

    # 4. Build final dataset with ZERO human annotations
    final_records = build_final_ground_truth_dataset(
        claims=m7_claims,
        annotations_a={},
        annotations_b={},
        adjudications={},
    )
    assert len(final_records) == 15

    # 5. Generate quality report and verify HONEST zero-coverage reporting
    report = generate_quality_report(
        manifest=None,
        claims=m7_claims,
        final_records=final_records,
        target_reserved_count=600,
    )
    assert report.total_claims == 15
    assert report.evidence_covered_images == 10
    assert report.evidence_covered_claims == 15
    assert report.total_annotations == 0
    assert report.annotation_coverage_pct == 0.0
    assert report.agreement_metrics["joint_count"] == 0
    assert report.agreement_metrics["cohens_kappa"] is None


def test_synthetic_end_to_end_pipeline_with_annotations(tmp_path):
    """
    Synthetic fixture testing end-to-end flow with dual annotations,
    adjudication, and split manifest integration.
    """
    # 1. Create a synthetic manifest of 20 images
    entries = []
    for i in range(20):
        img = ImageRecord(
            image_id=f"syn_img_{i:03d}",
            dataset_source=DatasetSource.SYNTHETIC,
            file_name=f"syn_{i}.jpg",
            file_hash=f"hash_{i:03d}",
        )
        entries.append(DatasetManifestEntry(image=img))
    manifest = create_manifest(manifest_id="m7_syn_manifest", entries=entries)

    # Split deterministically
    split_res = split_manifest_by_image_groups(manifest, seed=42)
    unified_manifest = split_res.unified_manifest

    # 2. Create synthetic evidence records for 5 images
    evidence_records = []
    for i in range(5):
        ev = ClaimLevelEvidenceRecord(
            claim_id=f"syn_claim_{i}",
            image_id=f"syn_img_{i:03d}",
            object_category="cat",
            caption="A cat sits.",
            detector_score=0.8,
            detector_available=True,
            clip_score=0.2,
            similarity_available=True,
            is_synthetic=True,
        )
        evidence_records.append(ev)

    # Ingest with manifest split linkage
    m7_claims = ingest_m6_evidence_to_m7_claims(evidence_records, manifest=unified_manifest)
    assert len(m7_claims) == 5

    # 3. Simulate Annotations
    # Claim 0: Agree SUPPORTED
    # Claim 1: Disagree (A=SUPPORTED, B=HALLUCINATED) -> Adjudicated to SUPPORTED
    # Claim 2: Agree HALLUCINATED
    # Claim 3: Agree UNKNOWN
    # Claim 4: Only Annotator A labeled -> Unresolved
    anns_a = {
        ("syn_img_000", "syn_claim_0"): AnnotationRecord("syn_claim_0", "syn_img_000", "A", GroundTruthStatus.SUPPORTED),
        ("syn_img_001", "syn_claim_1"): AnnotationRecord("syn_claim_1", "syn_img_001", "A", GroundTruthStatus.SUPPORTED),
        ("syn_img_002", "syn_claim_2"): AnnotationRecord("syn_claim_2", "syn_img_002", "A", GroundTruthStatus.HALLUCINATED),
        ("syn_img_003", "syn_claim_3"): AnnotationRecord("syn_claim_3", "syn_img_003", "A", GroundTruthStatus.UNKNOWN),
        ("syn_img_004", "syn_claim_4"): AnnotationRecord("syn_claim_4", "syn_img_004", "A", GroundTruthStatus.SUPPORTED),
    }
    anns_b = {
        ("syn_img_000", "syn_claim_0"): AnnotationRecord("syn_claim_0", "syn_img_000", "B", GroundTruthStatus.SUPPORTED),
        ("syn_img_001", "syn_claim_1"): AnnotationRecord("syn_claim_1", "syn_img_001", "B", GroundTruthStatus.HALLUCINATED),
        ("syn_img_002", "syn_claim_2"): AnnotationRecord("syn_claim_2", "syn_img_002", "B", GroundTruthStatus.HALLUCINATED),
        ("syn_img_003", "syn_claim_3"): AnnotationRecord("syn_claim_3", "syn_img_003", "B", GroundTruthStatus.UNKNOWN),
    }
    adjudications = {
        ("syn_img_001", "syn_claim_1"): AdjudicationRecord("syn_claim_1", "syn_img_001", "expert", GroundTruthStatus.SUPPORTED),
    }

    final_records = build_final_ground_truth_dataset(
        claims=m7_claims,
        annotations_a=anns_a,
        annotations_b=anns_b,
        adjudications=adjudications,
    )
    assert len(final_records) == 5

    # Check resolutions
    rec0 = next(r for r in final_records if r.claim_id == "syn_claim_0")
    assert rec0.final_ground_truth == GroundTruthStatus.SUPPORTED
    assert not rec0.has_disagreement

    rec1 = next(r for r in final_records if r.claim_id == "syn_claim_1")
    assert rec1.has_disagreement is True
    assert rec1.final_ground_truth == GroundTruthStatus.SUPPORTED  # From adjudication

    rec4 = next(r for r in final_records if r.claim_id == "syn_claim_4")
    assert rec4.final_ground_truth is None  # Missing Annotator B

    # 4. Generate report
    report = generate_quality_report(
        manifest=unified_manifest,
        claims=m7_claims,
        final_records=final_records,
        target_reserved_count=20,
    )
    assert report.total_claims == 5
    assert report.actual_images_in_manifest == 20
    assert report.evidence_covered_images == 5
    assert report.claims_with_annotator_a == 5
    assert report.claims_with_annotator_b == 4
    assert report.jointly_annotated_claims == 4
    assert report.adjudicated_claims == 1
    assert report.supported_count == 2
    assert report.hallucinated_count == 1
    assert report.unknown_count == 1
    # 4 resolved out of 5 total claims = 80.0%
    assert report.annotation_coverage_pct == pytest.approx(80.0)


def test_generate_dataset_lock_and_checksums(tmp_path):
    """Verify dataset lock generation, SHA-256 calculation, and status determination."""
    from src.annotation.workflow import generate_dataset_lock

    manifest_p = tmp_path / "m7_manifest.json"
    claims_p = tmp_path / "m7_claims.jsonl"
    gt_p = tmp_path / "m7_ground_truth.jsonl"
    lock_p = tmp_path / "m7_dataset_lock.json"

    # 1. Incomplete/Pending test
    manifest_p.write_text(json.dumps({"entries": [{"image": {"image_id": "img1"}, "split": "train"}]}))
    claims_p.write_text(json.dumps({"claim_id": "c1", "is_synthetic": False}) + "\n")
    gt_p.write_text(json.dumps({"claim_id": "c1", "final_ground_truth": None}) + "\n")

    lock_info = generate_dataset_lock(
        manifest_path=manifest_p,
        claims_path=claims_p,
        ground_truth_path=gt_p,
        target_count=600,
        lock_output_path=lock_p,
    )

    assert lock_info["status"] == "PENDING_ANNOTATION"
    assert lock_info["actual_manifest_image_count"] == 1
    assert lock_info["target_image_count"] == 600
    assert lock_info["total_claims_count"] == 1
    assert lock_info["resolved_ground_truth_count"] == 0
    assert lock_p.exists()
    assert lock_info["checksums"]["m7_manifest_sha256"] is not None
    assert lock_info["checksums"]["m7_claims_sha256"] is not None
    assert lock_info["checksums"]["m7_ground_truth_sha256"] is not None

    # 2. Complete/Locked test (satisfying all conditions)
    # 2 images, target count 2
    man_data = {
        "entries": [
            {"image": {"image_id": "img1"}, "split": "train"},
            {"image": {"image_id": "img2"}, "split": "test"},
        ]
    }
    manifest_p.write_text(json.dumps(man_data))
    claims_p.write_text(
        json.dumps({"claim_id": "c1", "is_synthetic": False}) + "\n" +
        json.dumps({"claim_id": "c2", "is_synthetic": False}) + "\n"
    )
    gt_p.write_text(
        json.dumps({"claim_id": "c1", "final_ground_truth": "supported", "disagreement": False}) + "\n" +
        json.dumps({"claim_id": "c2", "final_ground_truth": "hallucinated", "disagreement": False}) + "\n"
    )

    lock_info_full = generate_dataset_lock(
        manifest_path=manifest_p,
        claims_path=claims_p,
        ground_truth_path=gt_p,
        target_count=2,
        lock_output_path=lock_p,
    )

    assert lock_info_full["status"] == "LOCKED"
    assert lock_info_full["actual_manifest_image_count"] == 2
    assert lock_info_full["resolved_ground_truth_count"] == 2
    assert lock_info_full["unresolved_disputes_count"] == 0
    assert lock_info_full["all_real_evidence"] is True


def test_scalable_evidence_acquisition_mocked(tmp_path):
    """Verify scalable evidence acquisition runner with mock providers and checkpointing."""
    from experiments.run_m7_evidence_acquisition import run_evidence_acquisition
    from src.evidence.detector_provider import MockDetectorProvider
    from src.evidence.clip_provider import MockCLIPProvider
    from src.vlm.provider import SyntheticVLMProvider, VLMGenerationConfig
    from src.vlm.cache import VLMCache
    from PIL import Image

    # Create dummy images
    img1 = tmp_path / "img1.jpg"
    img2 = tmp_path / "img2.jpg"
    Image.new("RGB", (32, 32), color="red").save(img1)
    Image.new("RGB", (32, 32), color="blue").save(img2)

    manifest_entries = [
        DatasetManifestEntry(
            image=ImageRecord(image_id="coco_1", dataset_source=DatasetSource.COCO, file_name=str(img1)),
            split=SplitName.TRAIN,
        ),
        DatasetManifestEntry(
            image=ImageRecord(image_id="coco_2", dataset_source=DatasetSource.COCO, file_name=str(img2)),
            split=SplitName.VALIDATION,
        ),
    ]
    manifest = create_manifest("m7_test_manifest", entries=manifest_entries)
    manifest_p = tmp_path / "m7_manifest.json"
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    out_claims_p = tmp_path / "m7_claims.jsonl"
    cache = VLMCache(tmp_path / "vlm_cache")

    mock_captions = {
        "coco_1": "A cat on a rug.",
        "coco_2": "A dog on grass.",
    }
    vlm = SyntheticVLMProvider(mock_captions=mock_captions)
    det = MockDetectorProvider(fixed_scores={"cat": 0.85, "dog": 0.90})
    clip = MockCLIPProvider(fixed_scores={"cat": 0.40, "dog": 0.50})

    # Pass 1: max_images=1
    claims_pass1 = run_evidence_acquisition(
        manifest_path=str(manifest_p),
        output_claims_path=str(out_claims_p),
        vlm_provider=vlm,
        vlm_cache=cache,
        detector_provider=det,
        clip_provider=clip,
        image_base_dir=str(tmp_path),
        max_images=1,
        resume=True,
    )
    assert len(claims_pass1) == 1
    assert claims_pass1[0].image_id == "coco_1"

    # Pass 2: resume and process remaining image
    claims_pass2 = run_evidence_acquisition(
        manifest_path=str(manifest_p),
        output_claims_path=str(out_claims_p),
        vlm_provider=vlm,
        vlm_cache=cache,
        detector_provider=det,
        clip_provider=clip,
        image_base_dir=str(tmp_path),
        max_images=None,
        resume=True,
    )
    assert len(claims_pass2) == 2
    assert set(c.image_id for c in claims_pass2) == {"coco_1", "coco_2"}
    assert claims_pass2[0].split == "train"
    assert claims_pass2[1].split == "validation"

