"""
VLM Provider Interface and LLaVA Pipeline Delegation Wrapper.

Strictly enforces:
1. BaseVLMProvider interface contract.
2. LLaVAProviderWrapper delegates directly to existing, tested M6 pipeline
   (LLaVA15Provider and ConservativeClaimExtractor) without duplicating inference
   or model-loading code.
3. Second-VLM provider is an interface contract only in Phase 9E; external execution
   and model downloads are strictly prohibited.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from src.vlm.provider import VLMGenerationConfig, VLMResponse, VLMProvider
from src.data.schemas import AtomicObjectExistenceClaim


class BaseVLMProvider(ABC):
    """Abstract interface contract for vision-language models."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier for the VLM provider."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """HuggingFace model ID or identifier."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the model weights or runtime are available."""
        pass

    @abstractmethod
    def generate_caption(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        config: Optional[VLMGenerationConfig] = None,
    ) -> VLMResponse:
        """Generate caption response for an image."""
        pass


class LLaVAProviderWrapper(BaseVLMProvider):
    """
    Wrapper that delegates directly to existing M6 LLaVA-1.5 implementation.
    Reuses frozen revisions, cache semantics, and provenance tags without code duplication.
    """

    def __init__(
        self,
        config: Optional[VLMGenerationConfig] = None,
        underlying_provider: Optional[VLMProvider] = None,
    ):
        self._config = config or VLMGenerationConfig()
        if underlying_provider is not None:
            self._provider = underlying_provider
        else:
            # Delegate lazily to existing LLaVA15Provider
            from src.vlm.llava_provider import LLaVA15Provider
            self._provider = LLaVA15Provider(
                model_name=self._config.model_name,
                device=self._config.device,
                dtype=self._config.dtype,
                load_in_4bit=self._config.load_in_4bit,
                local_files_only=self._config.local_files_only,
            )

    @property
    def provider_name(self) -> str:
        return "LLaVA15ProviderWrapper"

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def is_available(self) -> bool:
        if hasattr(self._provider, "is_available"):
            return self._provider.is_available()
        return True

    def generate_caption(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        config: Optional[VLMGenerationConfig] = None,
    ) -> VLMResponse:
        cfg = config or self._config
        active_prompt = prompt or cfg.prompt
        return self._provider.generate_caption(
            image_path=image_path,
            prompt=active_prompt,
            gen_config=cfg,
        )


class SecondVLMProviderInterface(BaseVLMProvider):
    """
    Interface contract only for potential future second VLM (e.g. Qwen-VL, InstructBLIP).
    Strictly forbids model downloading or execution during Phase 9E.
    """

    def __init__(self, model_name: str = "future-second-vlm"):
        self._model_name = model_name

    @property
    def provider_name(self) -> str:
        return "SecondVLMInterfaceContract"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_available(self) -> bool:
        return False

    def generate_caption(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        config: Optional[VLMGenerationConfig] = None,
    ) -> VLMResponse:
        raise NotImplementedError(
            "Second-VLM provider is an interface contract only in Phase 9E. "
            "Execution, model downloads, and experiments for secondary VLMs "
            "are strictly planned for future milestones."
        )
