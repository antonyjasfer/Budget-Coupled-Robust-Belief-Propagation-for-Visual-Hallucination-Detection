"""
Unit tests for FixtureEvidenceProvider.
"""

from pathlib import Path
import pytest

from src.data.schemas import AtomicObjectExistenceClaim
from src.evidence.fixture_provider import FixtureEvidenceProvider
from src.evidence.schemas import RawEvidenceRecord

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "milestone3"
EVIDENCE_FIXTURE_PATH = FIXTURES_DIR / "fixture_evidence.json"


def test_fixture_evidence_provider_loads_and_queries():
    """Verify loading from disk fixture and querying by (image_id, claim_id, view_id)."""
    provider = FixtureEvidenceProvider(fixture_path=EVIDENCE_FIXTURE_PATH)

    claim_dog = AtomicObjectExistenceClaim(
        claim_id="claim_resp_101_dog",
        image_id="coco_101",
        object_category="dog",
    )

    # 1. Original view query
    ev_orig = provider.get_evidence("coco_101", claim_dog, view_id="original")
    assert ev_orig.image_id == "coco_101"
    assert ev_orig.claim_id == "claim_resp_101_dog"
    assert ev_orig.detector_score == 0.88
    assert ev_orig.detector_available is True
    assert ev_orig.similarity_score == 0.42

    # 2. Perturbation view query (e.g. gaussian_blur_s1)
    ev_blur = provider.get_evidence("coco_101", claim_dog, view_id="gaussian_blur_s1")
    assert ev_blur.detector_score == 0.76
    assert ev_blur.view_id == "gaussian_blur_s1"


def test_fixture_evidence_provider_rejects_missing_claims():
    """Verify that provider raises KeyError rather than hallucinating fake scores."""
    provider = FixtureEvidenceProvider(fixture_path=EVIDENCE_FIXTURE_PATH)

    unknown_claim = AtomicObjectExistenceClaim(
        claim_id="claim_unknown_999",
        image_id="coco_999",
        object_category="elephant",
    )

    with pytest.raises(KeyError, match="No explicit fixture evidence found"):
        provider.get_evidence("coco_999", unknown_claim, view_id="original")
