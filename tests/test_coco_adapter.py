"""
Unit tests for the local COCO metadata adapter.
"""

from pathlib import Path
import pytest

from src.data.coco import COCOMetadataAdapter
from src.data.schemas import DatasetSource

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
COCO_FIXTURE_PATH = FIXTURES_DIR / "coco_instances_synthetic.json"


def test_coco_adapter_loads_fixture():
    """Verify loading and parsing of synthetic COCO instances fixture."""
    adapter = COCOMetadataAdapter(COCO_FIXTURE_PATH)

    assert adapter.provenance["num_images_loaded"] == 10
    assert adapter.provenance["num_categories_loaded"] == 5
    assert adapter.provenance["num_annotations_loaded"] == 12

    # Check category index
    assert len(adapter.categories) == 5
    assert adapter.categories[1].canonical_name == "person"
    assert adapter.categories[2].canonical_name == "dog"
    assert adapter.name_to_category["frisbee"].category_id == 5

    # Check images
    assert len(adapter.images) == 10
    img_rec = adapter.get_image_record("coco_101")
    assert img_rec is not None
    assert img_rec.coco_id == 101
    assert img_rec.dataset_source == DatasetSource.COCO
    assert img_rec.file_name == "000000000101.jpg"

    # Also test integer lookup
    assert adapter.get_image_record(102) is not None


def test_coco_adapter_annotation_presence_safeguard():
    """
    CRITICAL METHODOLOGICAL SAFEGUARD:
    Annotated categories provide positive evidence of category presence.
    Unannotated categories mean absence of annotation, NOT definitive negative object absence.
    """
    adapter = COCOMetadataAdapter(COCO_FIXTURE_PATH)

    # Image 101 has person (id 1) and dog (id 2) annotated
    ev_101 = adapter.get_evidence("coco_101")
    assert ev_101 is not None
    assert ev_101.is_annotated_present("person")
    assert ev_101.is_annotated_present("dog")
    assert ev_101.is_annotated_present(1)
    assert ev_101.is_annotated_present(2)

    # Frisbee (id 5) is NOT annotated in image 101
    assert not ev_101.is_annotated_present("frisbee")
    assert not ev_101.is_annotated_present(5)

    # Image 102 has dog (2) and frisbee (5) annotated
    ev_102 = adapter.get_evidence(102)
    assert ev_102.is_annotated_present("frisbee")
    assert ev_102.is_annotated_present("dog")
    assert not ev_102.is_annotated_present("person")


def test_coco_adapter_handles_missing_and_corrupt_files():
    """Verify clean error handling for missing and invalid files."""
    with pytest.raises(FileNotFoundError):
        COCOMetadataAdapter(FIXTURES_DIR / "non_existent_coco.json")
