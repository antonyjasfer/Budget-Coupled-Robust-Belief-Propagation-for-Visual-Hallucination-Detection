"""
Regression test for Phase 10A-R:
Verifies that claim extraction operates strictly on VLM-generated descriptions
without requiring any MS COCO human reference captions or annotations.
"""

from pathlib import Path
import tempfile
import pytest
from PIL import Image

import src.data.schemas as schemas
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import ConservativeClaimExtractor

def test_claim_generation_without_coco_reference_caption(tmp_path):
    """
    Ensure the primary claim extraction pipeline takes a raw image and VLM output
    directly, with zero dependency on COCO human ground truth captions.
    """
    # 1. Create a raw image on disk
    img_path = tmp_path / "test_raw_image_001.jpg"
    img = Image.new("RGB", (640, 480), color=(128, 128, 128))
    img.save(img_path)

    # 2. Simulate VLM generating a description for the image (no COCO caption exists)
    vlm_generated_text = "There is a dog sitting on a bench next to a bicycle."
    vlm_model_id = "llava-hf/llava-1.5-7b-hf"
    vlm_revision = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"

    resp_record = schemas.GeneratedResponseRecord(
        response_id="resp_vlm_direct_001",
        image_id="coco_test_direct_001",
        model_name=vlm_model_id,
        response_text=vlm_generated_text,
        metadata={"model_revision": vlm_revision},
    )

    # 3. Extract atomic claims using the frozen conservative extractor
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)
    report = extractor.extract_from_response(resp_record)

    # 4. Verify claims are produced strictly from the VLM output
    assert len(report.accepted_claims) > 0, "Should extract claims from VLM output"
    categories = [c.object_category for c in report.accepted_claims]
    assert "dog" in categories
    assert "bench" in categories
    assert "bicycle" in categories

    # 5. Verify provenance points to VLM and not to COCO annotations
    for claim in report.accepted_claims:
        assert claim.response_id == "resp_vlm_direct_001"
        assert claim.image_id == "coco_test_direct_001"
        assert claim.raw_claim_text in vlm_generated_text
