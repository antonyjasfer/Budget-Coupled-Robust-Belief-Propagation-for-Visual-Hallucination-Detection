"""
Unit tests verifying strict evidence masking in human annotation tasks.
"""

import pytest
from src.annotation.schemas import HumanAnnotationTask
from src.annotation.masking import (
    FORBIDDEN_EVIDENCE_FIELDS,
    extract_minimal_context,
)


def test_human_annotation_task_masking_strictness():
    """
    Verify that HumanAnnotationTask contains zero model evidence,
    zero splits, zero predictions, and zero parameters.
    """
    task = HumanAnnotationTask(
        task_id="task_001",
        claim_id="claim_001",
        image_id="coco_000000000001",
        image_path="data/coco/val2017/000000000001.jpg",
        object_category="dog",
        surface_form="a fluffy dog",
        minimal_context="There is a fluffy dog in the park.",
    )
    d = task.to_dict()

    for forbidden in FORBIDDEN_EVIDENCE_FIELDS:
        assert forbidden not in d
        assert forbidden not in d.get("metadata", {})

    assert "detector_score" not in d
    assert "clip_score" not in d
    assert "split" not in d
    assert "theta" not in d
    assert "epsilon" not in d
    assert "J" not in d
    assert "prediction" not in d


def test_extract_minimal_context_truncates_long_captions():
    """
    Verify that full linguistic context is minimized to prevent confirmation bias.
    """
    long_caption = (
        "In this vibrant sunny morning photograph, an exquisite golden retriever is playing "
        "joyfully with a bright red frisbee on the lush green grass while several people watch."
    )
    ctx = extract_minimal_context(long_caption, surface_form="golden retriever", max_words=10)
    assert ctx is not None
    # Must be reasonably short
    assert len(ctx.split()) <= 15
