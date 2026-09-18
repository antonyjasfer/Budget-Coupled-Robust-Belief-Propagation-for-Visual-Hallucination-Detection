"""
Integration tests for the complete response -> claim extraction -> fixture evidence pipeline.
"""

from pathlib import Path
import json
import pytest

from src.data.schemas import (
    DatasetManifest,
    DatasetManifestEntry,
    ImageRecord,
    GeneratedResponseRecord,
    DatasetSource,
    SplitName,
)
from src.data.manifests import create_manifest, validate_manifest
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor
from src.evidence.fixture_provider import FixtureEvidenceProvider
from src.claims.cli import main as claims_cli_main

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "milestone3"
EVIDENCE_FIXTURE_PATH = FIXTURES_DIR / "fixture_evidence.json"


def test_claim_evidence_end_to_end_pipeline(tmp_path):
    """
    Verify full offline pipeline:
    Stored response -> conservative claim extraction -> diagnostics -> evidence attachment -> manifest validation.
    """
    # 1. Create input manifest with stored response
    img = ImageRecord(
        image_id="coco_101",
        dataset_source=DatasetSource.COCO,
        file_name="000000000101.jpg",
    )
    resp = GeneratedResponseRecord(
        response_id="resp_101",
        image_id="coco_101",
        model_name="llava-1.5-7b",
        response_text="The photo shows a dog and a red car. There is no cat.",
    )
    entry = DatasetManifestEntry(image=img, responses=[resp], split=SplitName.TRAIN)
    manifest = create_manifest(manifest_id="pipeline_test_manifest", entries=[entry])

    in_manifest_path = tmp_path / "input_manifest.json"
    extracted_manifest_path = tmp_path / "extracted_manifest.json"
    diag_path = tmp_path / "diagnostics.json"

    with open(in_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # 2. Run CLI extract-claims
    ret_ext = claims_cli_main([
        "extract-claims",
        str(in_manifest_path),
        "--output-manifest", str(extracted_manifest_path),
        "--diagnostics-out", str(diag_path),
    ])
    assert ret_ext == 0
    assert extracted_manifest_path.exists()
    assert diag_path.exists()

    with open(extracted_manifest_path, "r", encoding="utf-8") as f:
        ext_data = json.load(f)

    # Should have 2 accepted claims: dog, car. Cat rejected as negated.
    extracted_entry = ext_data["entries"][0]
    claim_cats = {c["object_category"] for c in extracted_entry["claims"]}
    assert claim_cats == {"car", "dog"}
    assert "cat" not in claim_cats

    # Verify split assignment preserved
    assert extracted_entry["split"] == "train"

    # 3. Run CLI attach-evidence
    evidence_manifest_path = tmp_path / "final_evidence_manifest.json"
    ret_ev = claims_cli_main([
        "attach-evidence",
        str(extracted_manifest_path),
        "--evidence-fixture", str(EVIDENCE_FIXTURE_PATH),
        "--output-manifest", str(evidence_manifest_path),
        "--view-id", "original",
    ])
    assert ret_ev == 0
    assert evidence_manifest_path.exists()

    with open(evidence_manifest_path, "r", encoding="utf-8") as f:
        ev_manifest_data = json.load(f)

    final_entry = ev_manifest_data["entries"][0]
    ev_list = final_entry["image"]["metadata"].get("evidence_original", [])
    assert len(ev_list) == 2  # dog and car
    assert all(e["view_id"] == "original" for e in ev_list)
    assert any(e["object_category"] == "dog" and e["detector_score"] == 0.88 for e in ev_list)
