"""
Object-existence claim extraction package.
Provides category vocabulary management, conservative rule-based claim extraction, and extraction diagnostics.
"""

from src.claims.vocabulary import (
    CategoryInfo,
    AliasRule,
    CategoryRegistry,
    create_coco_category_registry,
    ALIAS_RULES_VERSION,
)
from src.claims.extraction import (
    MentionSpan,
    MentionRejectionReason,
    RejectedMention,
    ExtractedClaim,
    ExtractionReport,
    ConservativeClaimExtractor,
)

__all__ = [
    "CategoryInfo",
    "AliasRule",
    "CategoryRegistry",
    "create_coco_category_registry",
    "ALIAS_RULES_VERSION",
    "MentionSpan",
    "MentionRejectionReason",
    "RejectedMention",
    "ExtractedClaim",
    "ExtractionReport",
    "ConservativeClaimExtractor",
]
