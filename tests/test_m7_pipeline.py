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
