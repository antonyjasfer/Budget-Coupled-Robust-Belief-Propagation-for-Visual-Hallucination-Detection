"""
Unit tests for precision planning, post-hoc test events, external benchmark adapters, and VLM interfaces (Corrections 3, 4, 12, 13).
"""

import pytest
from src.data.dataset_adequacy import (
    compute_binomial_precision,
    evaluate_test_event_adequacy,
)
from src.data.external_benchmarks import (
    POPEAdapter,
    AMBERAdapter,
    BenchmarkTaskType,
    BenchmarkAdapterStatus,
)
from src.data.vlm_interface import (
    BaseVLMProvider,
    LLaVAProviderWrapper,
    SecondVLMProviderInterface,
)
from src.vlm.provider import VLMResponse, VLMGenerationConfig


def test_precision_planning_calculation():
    """
    Verify precision planning calculates margin of error and CI width
    without making unsubstantiated power claims (Correction 12).
    """
    res = compute_binomial_precision(sample_size=200, target_proportion=0.20, confidence_level=0.95)
    # Expected MoE ~ 1.95996 * sqrt(0.20 * 0.80 / 200) = 1.95996 * 0.028284 = ~0.0554
    assert 0.055 <= res.margin_of_error <= 0.056
    assert res.ci_lower < 0.20 < res.ci_upper
    assert "Precision planning only; not a power claim" in res.interpretation


def test_test_event_adequacy_post_hoc_descriptive():
    """
    Verify post-hoc descriptive test event adequacy reporting (Correction 13).
    """
    # Adequate events
    labels_ok = ["hallucinated"] * 15 + ["supported"] * 85
    rep_ok = evaluate_test_event_adequacy(labels_ok, min_event_count=10)
    assert rep_ok.is_evaluable is True
    assert rep_ok.status == "EVALUABLE"

    # Inadequate events
    labels_sparse = ["hallucinated"] * 3 + ["supported"] * 97
    rep_sparse = evaluate_test_event_adequacy(labels_sparse, min_event_count=10)
    assert rep_sparse.is_evaluable is False
    assert rep_sparse.status == "LOW EVENT COUNT / NOT EVALUABLE"
    assert "parameters and splits must remain frozen" in rep_sparse.rationale


def test_pope_adapter_mapping():
    """Verify lightweight POPE adapter maps binary existence questions."""
    adapter = POPEAdapter()
    record = {
        "question_id": 101,
        "image_id": "coco_001",
        "text": "Is there a dog in the image?",
        "category": "dog",
        "label": "yes",
    }
    res = adapter.map_record(record)
    assert res.is_mappable is True
    assert res.task_type == BenchmarkTaskType.OBJECT_EXISTENCE
    assert res.claim is not None
    assert res.claim.object_category == "dog"
    assert res.claim.provenance["ground_truth"] == "supported"



def test_amber_adapter_unsupported_dimension_rejection():
    """
    Verify AMBER non-existence dimensions (attribute, relation)
    are explicitly flagged as UNSUPPORTED_TASK_TYPE (Correction 3).
    """
    adapter = AMBERAdapter()
    
    # Attribute hallucination record
    attrib_record = {
        "id": "amber_attr_01",
        "task_type": "attribute",
        "object": "car",
        "attribute": "red",
    }
    res = adapter.map_record(attrib_record)
    assert res.is_mappable is False
    assert res.task_type == BenchmarkTaskType.ATTRIBUTE_BINDING
    assert res.adapter_status == BenchmarkAdapterStatus.UNSUPPORTED
    assert "outside the object-existence scope" in res.reason


def test_second_vlm_interface_contract():
    """
    Verify SecondVLMProviderInterface raises NotImplementedError
    preventing unauthorized downloads or executions during Phase 9E (Correction 4).
    """
    second_vlm = SecondVLMProviderInterface(model_name="qwen-vl-chat")
    assert second_vlm.is_available is False
    with pytest.raises(NotImplementedError, match="interface contract only in Phase 9E"):
        second_vlm.generate_caption("path/to/image.jpg")
