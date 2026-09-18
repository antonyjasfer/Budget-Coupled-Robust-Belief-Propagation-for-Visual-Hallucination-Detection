"""
Raw evidence contracts, provider interfaces, and multimodal pipeline.
"""

from src.evidence.schemas import (
    RawEvidenceRecord,
    ClaimLevelEvidenceRecord,
    EVIDENCE_SCHEMA_VERSION,
)
from src.evidence.provider import EvidenceProvider
from src.evidence.fixture_provider import FixtureEvidenceProvider
from src.evidence.detector_provider import (
    BaseDetectorProvider,
    HuggingFaceDetectorProvider,
    MockDetectorProvider,
    DetectorResult,
)
from src.evidence.clip_provider import (
    BaseCLIPProvider,
    TransformersCLIPProvider,
    MockCLIPProvider,
    CLIPResult,
)
from src.evidence.pipeline import (
    VisualEvidencePipeline,
    PipelineStatistics,
)

__all__ = [
    "RawEvidenceRecord",
    "ClaimLevelEvidenceRecord",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceProvider",
    "FixtureEvidenceProvider",
    "BaseDetectorProvider",
    "HuggingFaceDetectorProvider",
    "MockDetectorProvider",
    "DetectorResult",
    "BaseCLIPProvider",
    "TransformersCLIPProvider",
    "MockCLIPProvider",
    "CLIPResult",
    "VisualEvidencePipeline",
    "PipelineStatistics",
]
