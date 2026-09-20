"""
Offline unit tests for LLaVA-1.5 4-bit bitsandbytes quantization and provenance.
Guarantees:
1. Exact checkpoint llava-hf/llava-1.5-7b-hf is maintained.
2. 4-bit quantization configuration (NF4, double quant, float16 compute, device_map="auto").
3. FP16 / standard loading path remains fully functional.
4. Model and response provenance serialization captures all 4 required fields:
   - quantization_enabled
   - quantization_type
   - compute_dtype
   - device_map
5. Offline compatibility without requiring an actual CUDA GPU.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch

from src.vlm.provider import VLMGenerationConfig, VLMResponse
from src.vlm.llava_provider import LLaVA15Provider
from src.vlm.cache import VLMCache


def test_vlm_generation_config_default_fields():
    """Default generation config defaults to FP32/CPU without quantization."""
    cfg = VLMGenerationConfig()
    assert cfg.load_in_4bit is False
    assert cfg.quantization_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.compute_dtype == "float16"
    assert cfg.device_map is None
    assert cfg.model_name == "llava-hf/llava-1.5-7b-hf"


def test_vlm_generation_config_4bit_fields():
    """VLMGenerationConfig explicitly stores 4-bit quantization parameters."""
    cfg = VLMGenerationConfig(
        load_in_4bit=True,
        quantization_type="nf4",
        bnb_4bit_use_double_quant=True,
        compute_dtype="float16",
        device_map="auto",
    )
    assert cfg.load_in_4bit is True
    assert cfg.quantization_type == "nf4"
    assert cfg.bnb_4bit_use_double_quant is True
    assert cfg.compute_dtype == "float16"
    assert cfg.device_map == "auto"

    # Serialization round-trip
    d = cfg.to_dict()
    assert d["load_in_4bit"] is True
    assert d["quantization_type"] == "nf4"
    assert d["bnb_4bit_use_double_quant"] is True
    assert d["compute_dtype"] == "float16"
    assert d["device_map"] == "auto"

    recovered = VLMGenerationConfig.from_dict(d)
    assert recovered.load_in_4bit is True
    assert recovered.quantization_type == "nf4"
    assert recovered.bnb_4bit_use_double_quant is True
    assert recovered.compute_dtype == "float16"
    assert recovered.device_map == "auto"


def test_param_hash_separates_4bit_and_fp16():
    """4-bit and FP16 configs produce different parameter hashes and cache keys."""
    cfg_fp16 = VLMGenerationConfig(
        dtype="float16",
        load_in_4bit=False,
    )
    cfg_4bit = VLMGenerationConfig(
        dtype="float16",
        load_in_4bit=True,
        quantization_type="nf4",
        device_map="auto",
    )

    hash_fp16 = cfg_fp16.get_param_hash()
    hash_4bit = cfg_4bit.get_param_hash()
    assert hash_fp16 != hash_4bit

    # VLMCache key separation
    cache_key_fp16 = VLMCache.compute_cache_key(
        image_hash="abc123hash",
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="rev001",
        prompt="Describe the image.",
        gen_config=cfg_fp16,
        provider_kind="llava_15_hf",
        is_synthetic=False,
        generation_source="real_inference",
    )
    cache_key_4bit = VLMCache.compute_cache_key(
        image_hash="abc123hash",
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="rev001",
        prompt="Describe the image.",
        gen_config=cfg_4bit,
        provider_kind="llava_15_hf",
        is_synthetic=False,
        generation_source="real_inference",
    )
    assert cache_key_fp16 != cache_key_4bit


def test_vlm_response_quantization_provenance_roundtrip():
    """VLMResponse records and serializes all 4 quantization provenance fields."""
    cfg_4bit = VLMGenerationConfig(
        load_in_4bit=True,
        quantization_type="nf4",
        bnb_4bit_use_double_quant=True,
        compute_dtype="float16",
        device_map="auto",
    )

    resp = VLMResponse.create(
        image_id="coco_000000000042",
        image_path="/path/to/000000000042.jpg",
        image_hash="img_sha256_mock",
        caption="A dog sits beside a park bench.",
        model_name="llava-hf/llava-1.5-7b-hf",
        model_revision="rev_snapshot_123",
        prompt="Describe the image.",
        gen_config=cfg_4bit,
        is_synthetic=False,
        provider_kind="llava_15_hf",
        generation_source="real_inference",
    )

    assert resp.quantization_enabled is True
    assert resp.quantization_type == "nf4"
    assert resp.compute_dtype == "float16"
    assert resp.device_map == "auto"
    assert resp.is_synthetic is False
    assert resp.generation_source == "real_inference"

    serialized = resp.to_dict()
    assert serialized["quantization_enabled"] is True
    assert serialized["quantization_type"] == "nf4"
    assert serialized["compute_dtype"] == "float16"
    assert serialized["device_map"] == "auto"

    deserialized = VLMResponse.from_dict(serialized)
    assert deserialized.quantization_enabled is True
    assert deserialized.quantization_type == "nf4"
    assert deserialized.compute_dtype == "float16"
    assert deserialized.device_map == "auto"
    assert deserialized.response_id == resp.response_id


def test_llava_provider_model_info_fp16():
    """FP16 LLaVA15Provider reports quantization_enabled=False in model info."""
    provider = LLaVA15Provider(
        model_name="llava-hf/llava-1.5-7b-hf",
        device="cuda:0",
        dtype="float16",
        load_in_4bit=False,
    )
    info = provider.get_model_info()
    assert info["model_name"] == "llava-hf/llava-1.5-7b-hf"
    assert info["quantization_enabled"] is False
    assert info["quantization_type"] is None
    assert info["compute_dtype"] == "float16"
    assert info["device"] == "cuda:0"
    assert info["dtype"] == "float16"
    assert info["is_synthetic"] is False
    assert info["provider_kind"] == "llava_15_hf"


def test_llava_provider_model_info_4bit():
    """4-bit LLaVA15Provider reports quantization_enabled=True and NF4 provenance."""
    provider = LLaVA15Provider(
        model_name="llava-hf/llava-1.5-7b-hf",
        device="cuda:0",
        dtype="float16",
        load_in_4bit=True,
        quantization_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    info = provider.get_model_info()
    assert info["model_name"] == "llava-hf/llava-1.5-7b-hf"
    assert info["quantization_enabled"] is True
    assert info["quantization_type"] == "nf4"
    assert info["compute_dtype"] == "float16"
    assert info["device_map"] == "auto"
    assert info["is_synthetic"] is False
    assert info["provider_kind"] == "llava_15_hf"


def test_4bit_requires_cuda_when_loaded():
    """Calling _ensure_loaded with load_in_4bit=True on non-CUDA system raises informative error."""
    provider = LLaVA15Provider(
        model_name="llava-hf/llava-1.5-7b-hf",
        load_in_4bit=True,
    )
    with patch("torch.cuda.is_available", return_value=False):
        # Reset cached instances to force loading attempt
        LLaVA15Provider._model_instance = None
        LLaVA15Provider._loaded_model_id = None
        LLaVA15Provider._loaded_quantization = None
        with pytest.raises(RuntimeError, match="4-bit inference using bitsandbytes requires an active CUDA GPU"):
            provider._ensure_loaded()


def test_4bit_loading_path_configures_bitsandbytes():
    """Verify BitsAndBytesConfig creation and device_map='auto' without forcing cuda:0."""
    provider = LLaVA15Provider(
        model_name="llava-hf/llava-1.5-7b-hf",
        device="cuda:0",
        load_in_4bit=True,
        quantization_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    mock_model = MagicMock()
    mock_model.eval.return_value = None
    mock_model.parameters.return_value = [MagicMock()]
    mock_processor = MagicMock()

    mock_bnb_config_cls = MagicMock()

    # Reset cached instances
    LLaVA15Provider._model_instance = None
    LLaVA15Provider._loaded_model_id = None
    LLaVA15Provider._loaded_quantization = None

    with patch("torch.cuda.is_available", return_value=True), \
         patch("transformers.AutoProcessor.from_pretrained", return_value=mock_processor) as mock_proc_load, \
         patch("transformers.LlavaForConditionalGeneration.from_pretrained", return_value=mock_model) as mock_model_load, \
         patch("transformers.BitsAndBytesConfig", mock_bnb_config_cls):

        provider._ensure_loaded()

        # Verify BitsAndBytesConfig arguments
        mock_bnb_config_cls.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Verify LlavaForConditionalGeneration.from_pretrained called with device_map="auto"
        call_kwargs = mock_model_load.call_args[1]
        assert call_kwargs["device_map"] == "auto"
        assert "quantization_config" in call_kwargs
        assert call_kwargs["low_cpu_mem_usage"] is True

        # Verify model.to("cuda:0") was NOT called (preserving device_map="auto")
        mock_model.to.assert_not_called()

        # Verify eval mode and frozen weights
        mock_model.eval.assert_called_once()
        for p in mock_model.parameters():
            assert p.requires_grad is False
