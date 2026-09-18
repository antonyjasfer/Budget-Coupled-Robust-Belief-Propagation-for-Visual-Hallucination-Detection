"""
Unit tests for data schemas, contracts, and PGM spin mappings.
"""

import pytest
import json

from src.data.schemas import (
    GroundTruthStatus,
    DecisionStatus,
    AnnotationSource,
    DatasetSource,
    SplitName,
    ImageRecord,
    GeneratedResponseRecord,
    AtomicObjectExistenceClaim,
    AnnotationRecord,
    PredictionDecisionRecord,
    DatasetManifestEntry,
    DatasetManifest,
    ground_truth_to_pgm_label,
    SCHEMA_VERSION,
)


def test_ground_truth_to_pgm_label_mapping():
    """Verify exact binary Ising spin mappings and strict rejection of UNKNOWN."""
    # SUPPORTED maps to -1 (non-hallucinated)
    assert ground_truth_to_pgm_label(GroundTruthStatus.SUPPORTED) == -1
    assert ground_truth_to_pgm_label("supported") == -1

    # HALLUCINATED maps to +1 (hallucinated)
    assert ground_truth_to_pgm_label(GroundTruthStatus.HALLUCINATED) == +1
    assert ground_truth_to_pgm_label("hallucinated") == +1

    # UNKNOWN MUST raise ValueError (never converted to 0, negative label, or 3rd spin state)
    with pytest.raises(ValueError, match="UNKNOWN cannot be converted to a binary Ising spin label"):
        ground_truth_to_pgm_label(GroundTruthStatus.UNKNOWN)

    with pytest.raises(ValueError, match="UNKNOWN cannot be converted to a binary Ising spin label"):
        ground_truth_to_pgm_label("unknown")

    # Invalid input raises ValueError
    with pytest.raises(ValueError, match="Unknown ground truth status"):
        ground_truth_to_pgm_label("invalid_status")


def test_atomic_object_existence_claim_normalization():
    """Verify canonical category normalization and empty check."""
    claim = AtomicObjectExistenceClaim(
        claim_id="c_001",
        image_id="img_001",
        object_category="  Dog  ",
    )
    assert claim.object_category == "dog"

    with pytest.raises(ValueError, match="object_category must be a non-empty string"):
        AtomicObjectExistenceClaim(
            claim_id="c_002",
            image_id="img_001",
            object_category="   ",
        )


def test_schema_serialization_round_trip():
    """Verify lossless serialization and deserialization of all schema dataclasses."""
    img = ImageRecord(
        image_id="coco_101",
        dataset_source=DatasetSource.COCO,
        file_name="000000000101.jpg",
        file_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        width=640,
        height=480,
        coco_id=101,
        metadata={"license": 1},
    )
    claim = AtomicObjectExistenceClaim(
        claim_id="claim_101_dog",
        image_id="coco_101",
        object_category="dog",
        text_span="a brown dog",
        raw_claim_text="There is a brown dog playing.",
    )
    ann = AnnotationRecord(
        annotation_id="ann_101_dog",
        claim_id="claim_101_dog",
        ground_truth=GroundTruthStatus.SUPPORTED,
        source=AnnotationSource.HUMAN_ADJUDICATED,
        annotator_notes="Clear dog in foreground.",
    )
    resp = GeneratedResponseRecord(
        response_id="resp_101_1",
        image_id="coco_101",
        model_name="llava-1.5-7b",
        response_text="The photo shows a brown dog playing.",
    )
    pred = PredictionDecisionRecord(
        decision_id="dec_101_dog",
        claim_id="claim_101_dog",
        decision=DecisionStatus.SUPPORTED,
        lower_bound=0.08,
        upper_bound=0.22,
        nominal_marginal=0.14,
    )

    entry = DatasetManifestEntry(
        image=img,
        claims=[claim],
        annotations=[ann],
        responses=[resp],
        split=SplitName.TRAIN,
    )
    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        manifest_id="test_manifest_v1",
        created_at="2026-09-18T10:00:00Z",
        description="Test Manifest",
        entries=[entry],
    )

    # JSON round trip
    json_str = manifest.to_json()
    reconstructed = DatasetManifest.from_json(json_str)

    assert reconstructed.schema_version == SCHEMA_VERSION
    assert reconstructed.manifest_id == "test_manifest_v1"
    assert len(reconstructed.entries) == 1

    rec_entry = reconstructed.entries[0]
    assert rec_entry.image.image_id == "coco_101"
    assert rec_entry.image.dataset_source == DatasetSource.COCO
    assert rec_entry.split == SplitName.TRAIN

    assert len(rec_entry.claims) == 1
    assert rec_entry.claims[0].claim_id == "claim_101_dog"
    assert rec_entry.claims[0].object_category == "dog"

    assert len(rec_entry.annotations) == 1
    assert rec_entry.annotations[0].ground_truth == GroundTruthStatus.SUPPORTED
    assert rec_entry.annotations[0].source == AnnotationSource.HUMAN_ADJUDICATED

    assert len(rec_entry.responses) == 1
    assert rec_entry.responses[0].response_id == "resp_101_1"

    # PredictionDecision round trip
    pred_dict = pred.to_dict()
    pred_rec = PredictionDecisionRecord.from_dict(pred_dict)
    assert pred_rec.decision == DecisionStatus.SUPPORTED
    assert pred_rec.lower_bound == 0.08
