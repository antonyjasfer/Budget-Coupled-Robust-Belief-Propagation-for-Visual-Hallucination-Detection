"""
Raw evidence contracts and provider interfaces.
"""

from src.evidence.schemas import (
    RawEvidenceRecord,
    EVIDENCE_SCHEMA_VERSION,
)
from src.evidence.provider import EvidenceProvider
from src.evidence.fixture_provider import FixtureEvidenceProvider

__all__ = [
    "RawEvidenceRecord",
    "EVIDENCE_SCHEMA_VERSION",
    "EvidenceProvider",
    "FixtureEvidenceProvider",
]
