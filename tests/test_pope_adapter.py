"""
Unit tests for the local POPE benchmark adapter.
"""

from pathlib import Path
import pytest

from src.data.pope import POPEAdapter, parse_canonical_coco_id_from_image_ref

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
POPE_FIXTURE_PATH = FIXTURES_DIR / "pope_synthetic.jsonl"


def test_parse_canonical_coco_id():
    """Verify robust extraction of canonical COCO IDs from various filename styles."""
    assert parse_canonical_coco_id_from_image_ref("COCO_val2014_000000123456.jpg") == "coco_123456"
    assert parse_canonical_coco_id_from_image_ref("000000000101.jpg") == "coco_101"
    assert parse_canonical_coco_id_from_image_ref("coco_456") == "coco_456"
    assert parse_canonical_coco_id_from_image_ref("789") == "coco_789"

    with pytest.raises(ValueError, match="Unable to resolve canonical image ID"):
        parse_canonical_coco_id_from_image_ref("invalid_filename_without_digits.jpg")


def test_pope_adapter_loads_fixture():
    """Verify loading and indexing of POPE records."""
    adapter = POPEAdapter(POPE_FIXTURE_PATH)

    assert len(adapter.records) == 10
    assert adapter.provenance["num_records"] == 10
    assert adapter.provenance["num_unique_images"] == 5

    # Check records for image 101
    records_101 = adapter.get_records_for_image("coco_101")
    assert len(records_101) == 3

    # Check properties
    r1 = records_101[0]
    assert r1.canonical_image_id == "coco_101"
    assert r1.category_name == "person"
    assert r1.benchmark_label == "yes"
    assert r1.variant == "popular"

    r3 = records_101[2]
    assert r3.category_name == "frisbee"
    assert r3.benchmark_label == "no"
    assert r3.variant == "adversarial"


def test_pope_adapter_rejects_corrupt_entries(tmp_path):
    """Verify rejection of invalid labels or missing fields."""
    bad_pope = tmp_path / "bad_pope.jsonl"
    bad_pope.write_text('{"question_id": 1, "image": "000000000101.jpg", "text": "Is there a dog?", "label": "maybe"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid POPE benchmark label 'maybe'"):
        POPEAdapter(bad_pope)
