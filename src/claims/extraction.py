"""
Conservative, reproducible object-existence claim extraction.

Extracts atomic object-existence claims from VLM text responses.
Applies conservative contextual filters for negation, uncertainty, questions,
hypotheticals, non-visual references, depictions, and part-of-speech ambiguity.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple
import re

from src.data.schemas import GeneratedResponseRecord, AtomicObjectExistenceClaim
from src.claims.vocabulary import CategoryRegistry


class MentionRejectionReason(str, Enum):
    """Reason codes for rejecting candidate object mentions."""
    NEGATED = "negated"
    UNCERTAIN = "uncertain"
    QUESTION = "question"
    HYPOTHETICAL = "hypothetical"
    NON_VISUAL = "non_visual"
    DEPICTION = "depiction"
    AMBIGUOUS_SENSE = "ambiguous_sense"
    UNMATCHED_SUBSTRING = "unmatched_substring"
    UNRESOLVED_ALIAS = "unresolved_alias"


@dataclass
class MentionSpan:
    """
    Exact text span and character offsets of a detected object mention in original text.

    Attributes:
        start_char: 0-indexed start character offset in full response text.
        end_char: 0-indexed end character offset in full response text.
        matched_text: Exact surface text substring.
        sentence_idx: Index of containing sentence.
        sentence_text: Full text of containing sentence.
    """
    start_char: int
    end_char: int
    matched_text: str
    sentence_idx: int
    sentence_text: str


@dataclass
class RejectedMention:
    """
    Detailed diagnostic record for a rejected or ambiguous candidate mention.

    Attributes:
        mention_span: MentionSpan record.
        candidate_category: Identified canonical category name.
        reason: Rejection reason code.
        explanation: Human-readable diagnostic description.
    """
    mention_span: MentionSpan
    candidate_category: str
    reason: MentionRejectionReason
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_char": self.mention_span.start_char,
            "end_char": self.mention_span.end_char,
            "matched_text": self.mention_span.matched_text,
            "candidate_category": self.candidate_category,
            "reason": self.reason.value,
            "explanation": self.explanation,
            "sentence_text": self.mention_span.sentence_text,
        }


@dataclass
class ExtractedClaim:
    """
    Extracted affirmative atomic object-existence claim.

    Attributes:
        claim_id: Deterministic claim ID.
        image_id: Associated image ID.
        object_category: Canonical object category name.
        response_id: Associated response ID.
        spans: List of supporting MentionSpans within the response.
        raw_claim_text: Primary surface phrase representation.
    """
    claim_id: str
    image_id: str
    object_category: str
    response_id: str
    spans: List[MentionSpan]
    raw_claim_text: str

    def to_atomic_claim(self) -> AtomicObjectExistenceClaim:
        """Convert to standard AtomicObjectExistenceClaim contract."""
        spans_summary = "; ".join(f"[{s.start_char}:{s.end_char}] '{s.matched_text}'" for s in self.spans)
        return AtomicObjectExistenceClaim(
            claim_id=self.claim_id,
            image_id=self.image_id,
            object_category=self.object_category,
            response_id=self.response_id,
            text_span=spans_summary,
            raw_claim_text=self.raw_claim_text,
            provenance={
                "num_mentions": len(self.spans),
                "extractor": "conservative_rule_based_v1",
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ExtractedClaim to dictionary representation."""
        return {
            "claim_id": self.claim_id,
            "image_id": self.image_id,
            "object_category": self.object_category,
            "response_id": self.response_id,
            "spans": [
                {
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "matched_text": s.matched_text,
                    "sentence_idx": s.sentence_idx,
                    "sentence_text": s.sentence_text,
                }
                for s in self.spans
            ],
            "raw_claim_text": self.raw_claim_text,
        }



@dataclass
class ExtractionReport:
    """
    Comprehensive diagnostics and output of the extraction pipeline on a single response.

    Attributes:
        response_id: Response ID.
        image_id: Image ID.
        num_accepted_claims: Number of distinct canonical categories accepted.
        num_rejected_mentions: Number of rejected candidate mentions.
        accepted_claims: List of ExtractedClaim instances.
        rejected_mentions: List of RejectedMention instances.
        raw_text: Original input response text.
        provenance: Extractor metadata.
    """
    response_id: str
    image_id: str
    num_accepted_claims: int
    num_rejected_mentions: int
    accepted_claims: List[ExtractedClaim]
    rejected_mentions: List[RejectedMention]
    raw_text: str
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id": self.response_id,
            "image_id": self.image_id,
            "num_accepted_claims": self.num_accepted_claims,
            "num_rejected_mentions": self.num_rejected_mentions,
            "accepted_claims": [
                {
                    "claim_id": c.claim_id,
                    "object_category": c.object_category,
                    "num_spans": len(c.spans),
                    "raw_claim_text": c.raw_claim_text,
                }
                for c in self.accepted_claims
            ],
            "rejected_mentions": [r.to_dict() for r in self.rejected_mentions],
            "provenance": self.provenance,
        }


# Contextual regex patterns for conservative filtering
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnone\b",
    r"\bwithout\b",
    r"\blacks?\b",
    r"\bnever\b",
    r"\bzero\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bcan(?:not|'t)\s+see\b",
    r"\bdo(?:es)?\s+not\s+(?:contain|see|have|show)\b",
    r"\bdo(?:es)n't\s+(?:contain|see|have|show)\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bmight\b",
    r"\bmaybe\b",
    r"\bpossibly\b",
    r"\bperhaps\b",
    r"\bcould\b",
    r"\bmay\b",
    r"\bappears?\s+to\b",
    r"\bseems?\s+to\b",
    r"\bunclear\b",
    r"\bnot\s+sure\b",
    r"\bhard\s+to\s+tell\b",
]

HYPOTHETICAL_PATTERNS = [
    r"\bif\b",
    r"\bwould\b",
    r"\bsuppose\b",
    r"\bassume\b",
    r"\bhypothetically\b",
]

DEPICTION_PATTERNS = [
    r"\bpainting\s+of\b",
    r"\bdrawing\s+of\b",
    r"\bsketch\s+of\b",
    r"\bstatue\s+of\b",
    r"\bsculpture\s+of\b",
    r"\bposter\s+of\b",
    r"\billustration\s+of\b",
    r"\bphoto\s+of\s+a\s+picture\s+of\b",
    r"\bt-?shirt\s+with\b",
]

NON_VISUAL_PATTERNS = [
    r"\bat\s+home\b",
    r"\bin\s+the\s+future\b",
    r"\bin\s+my\s+mind\b",
    r"\breminds\s+me\s+of\b",
    r"\bthought\s+of\b",
]


class ConservativeClaimExtractor:
    """
    Deterministic rule-based extractor for object-existence claims.
    """

    def __init__(
        self,
        registry: Optional[CategoryRegistry] = None,
        category_registry: Optional[CategoryRegistry] = None,
    ):
        active_reg = registry or category_registry
        if active_reg is None:
            raise ValueError("Must provide CategoryRegistry instance (as 'registry' or 'category_registry')")
        self.registry = active_reg

    def _split_sentences(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Split text into sentences with precise character offsets.
        Returns: List of (start_char, end_char, sentence_text).
        """
        sentences = []
        # Pattern split by punctuation (. ! ?) followed by whitespace or end of string
        pattern = re.compile(r'([^.!?]+[.!?]*)', re.UNICODE)
        for m in pattern.finditer(text):
            s_text = m.group(1)
            if s_text.strip():
                # Strip leading whitespace from start offset
                l_space = len(s_text) - len(s_text.lstrip())
                r_space = len(s_text) - len(s_text.rstrip())
                start = m.start() + l_space
                end = m.end() - r_space
                sentences.append((start, end, text[start:end]))
        if not sentences and text.strip():
            sentences.append((0, len(text), text))
        return sentences

    def extract_from_response(self, response: GeneratedResponseRecord) -> ExtractionReport:
        """
        Extract atomic object-existence claims from a single GeneratedResponseRecord.
        """
        text = response.response_text
        sentences = self._split_sentences(text)

        accepted_by_category: Dict[str, List[MentionSpan]] = {}
        first_mention_text: Dict[str, str] = {}
        rejected_mentions: List[RejectedMention] = []

        # All registered phrases sorted longest first
        sorted_phrases = self.registry.sorted_phrases

        for s_idx, (s_start, s_end, s_text) in enumerate(sentences):
            s_lower = s_text.lower()
            is_question = s_text.strip().endswith("?") or re.search(r"^(?:is|are|do|can|could|what|where)\s+(?:there|you|we)", s_lower.strip())

            # Find all phrase matches within the sentence
            # Track matched character intervals in sentence to avoid sub-phrase collisions (e.g. 'hot dog' vs 'dog')
            matched_spans: List[Tuple[int, int, str, str]] = []  # (start_in_s, end_in_s, matched_surface, canonical)

            for phrase in sorted_phrases:
                # Word-boundary regex for the phrase
                pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                for m in pattern.finditer(s_lower):
                    p_start = m.start()
                    p_end = m.end()

                    # Check if already covered by an earlier (longer) matched phrase
                    is_covered = any(
                        (existing_start <= p_start and p_end <= existing_end)
                        for existing_start, existing_end, _, _ in matched_spans
                    )
                    if is_covered:
                        continue

                    canonical = self.registry.lookup_canonical(phrase)
                    if canonical:
                        matched_spans.append((p_start, p_end, s_text[p_start:p_end], canonical))

            # Sort matched spans by sentence position
            matched_spans.sort(key=lambda item: item[0])

            # Analyze each match in its sentence context
            for p_start, p_end, surface_text, canonical in matched_spans:
                global_start = s_start + p_start
                global_end = s_start + p_end
                span = MentionSpan(
                    start_char=global_start,
                    end_char=global_end,
                    matched_text=surface_text,
                    sentence_idx=s_idx,
                    sentence_text=s_text,
                )

                # Context before the mention (clause up to mention)
                pre_context = s_lower[:p_start]
                post_context = s_lower[p_end:]

                # Determine local scope window (checking if an intervening preposition establishes a new prepositional phrase)
                # Prepositions: on, in, under, behind, near, beside, inside, at, with, by, from, over, into
                prep_match = list(re.finditer(r'\b(on|in|under|behind|near|beside|inside|at|with|by|from|over|into)\b', pre_context))
                if prep_match:
                    last_prep = prep_match[-1]
                    # If the preposition itself is not a negation cue like 'without'
                    # and there was an earlier object mention between the sentence start and this preposition
                    local_pre_context = pre_context[last_prep.start():]
                else:
                    local_pre_context = pre_context

                # 1. Check Question
                if is_question:
                    rejected_mentions.append(
                        RejectedMention(
                            mention_span=span,
                            candidate_category=canonical,
                            reason=MentionRejectionReason.QUESTION,
                            explanation="Mention occurs within an interrogative sentence/question.",
                        )
                    )
                    continue

                # 2. Check Hypothetical
                if any(re.search(pat, pre_context) for pat in HYPOTHETICAL_PATTERNS):
                    rejected_mentions.append(
                        RejectedMention(
                            mention_span=span,
                            candidate_category=canonical,
                            reason=MentionRejectionReason.HYPOTHETICAL,
                            explanation="Mention occurs within a hypothetical condition or clause.",
                        )
                    )
                    continue

                # 3. Check Uncertainty / Modal Hedges
                if any(re.search(pat, local_pre_context) for pat in UNCERTAINTY_PATTERNS):
                    rejected_mentions.append(
                        RejectedMention(
                            mention_span=span,
                            candidate_category=canonical,
                            reason=MentionRejectionReason.UNCERTAIN,
                            explanation="Mention occurs under uncertain or modal hedging scope.",
                        )
                    )
                    continue

                # 4. Check Negation & Coordinated Negation
                # Check preceding negation cues in the local clause window
                words_before = local_pre_context.split()[-10:]
                window_before = " ".join(words_before)
                has_negation = any(re.search(pat, window_before) for pat in NEGATION_PATTERNS)

                # Coordinated negation: if this mention is conjoined via 'or' / 'nor' to a preceding negated phrase in the same clause (without an intervening preposition)
                if not has_negation and not prep_match:
                    if re.search(r'\b(?:neither|nor|no|not)\s+[a-zA-Z\s,]+?\s+(?:or|nor)\s+(?:[a-zA-Z\s]+\s+)?$', pre_context):
                        has_negation = True

                # Check if immediately followed by 'not present' or 'nowhere'
                if re.search(r"^(?:\s+(?:is|are|was|were))?\s+(?:not\s+(?:visible|present|seen)|nowhere)", post_context):
                    has_negation = True

                if has_negation:
                    rejected_mentions.append(
                        RejectedMention(
                            mention_span=span,
                            candidate_category=canonical,
                            reason=MentionRejectionReason.NEGATED,
                            explanation="Mention occurs under direct or coordinated negation scope.",
                        )
                    )
                    continue

                # 5. Check Depictions
                if any(re.search(pat, pre_context) for pat in DEPICTION_PATTERNS):
                    rejected_mentions.append(
                        RejectedMention(
                            mention_span=span,
                            candidate_category=canonical,
                            reason=MentionRejectionReason.DEPICTION,
                            explanation="Mention is characterized as a depiction/artwork/drawing rather than real visual entity.",
                        )
                    )
                    continue

                # 6. Check Non-Visual references
                if any(re.search(pat, s_lower) for pat in NON_VISUAL_PATTERNS):
                    rejected_mentions.append(
                        RejectedMention(
                            mention_span=span,
                            candidate_category=canonical,
                            reason=MentionRejectionReason.NON_VISUAL,
                            explanation="Mention occurs in a non-visual or mental context.",
                        )
                    )
                    continue

                # 7. Ambiguous Word Senses & POS
                # Handle 'orange': if followed immediately by another noun (e.g. orange car, orange couch), it is a color modifier
                if canonical == "orange":
                    # Check next word
                    next_words = post_context.strip().split()
                    if next_words:
                        next_word = next_words[0].strip(",.!?")
                        # If next word matches another category
                        if self.registry.lookup_canonical(next_word):
                            rejected_mentions.append(
                                RejectedMention(
                                    mention_span=span,
                                    candidate_category=canonical,
                                    reason=MentionRejectionReason.AMBIGUOUS_SENSE,
                                    explanation=f"'orange' appears as a color modifier for '{next_word}'.",
                                )
                            )
                            continue

                # Handle 'train': if preceded by 'to', 'can', 'will', 'must' as a verb
                if canonical == "train":
                    if re.search(r"\b(?:to|can|will|must|should|going\s+to)\s+$", pre_context):
                        rejected_mentions.append(
                            RejectedMention(
                                mention_span=span,
                                candidate_category=canonical,
                                reason=MentionRejectionReason.AMBIGUOUS_SENSE,
                                explanation="'train' appears as an infinitive or modal verb.",
                            )
                        )
                        continue

                # If all filters pass: ACCEPT mention
                if canonical not in accepted_by_category:
                    accepted_by_category[canonical] = []
                    first_mention_text[canonical] = surface_text
                accepted_by_category[canonical].append(span)

        # Build ExtractedClaim instances
        accepted_claims: List[ExtractedClaim] = []
        for cat_name in sorted(accepted_by_category.keys()):
            spans = accepted_by_category[cat_name]
            claim_id = f"claim_{response.response_id}_{cat_name}"
            claim = ExtractedClaim(
                claim_id=claim_id,
                image_id=response.image_id,
                object_category=cat_name,
                response_id=response.response_id,
                spans=spans,
                raw_claim_text=first_mention_text[cat_name],
            )
            accepted_claims.append(claim)

        return ExtractionReport(
            response_id=response.response_id,
            image_id=response.image_id,
            num_accepted_claims=len(accepted_claims),
            num_rejected_mentions=len(rejected_mentions),
            accepted_claims=accepted_claims,
            rejected_mentions=rejected_mentions,
            raw_text=text,
            provenance={"rule_version": "1.0.0"},
        )

    def extract_claims_from_text(
        self,
        text: str,
        response_id: str = "resp_direct",
        image_id: str = "img_direct",
    ) -> ExtractionReport:
        """Convenience method to extract claims directly from a raw text string."""
        from src.data.schemas import GeneratedResponseRecord
        dummy_resp = GeneratedResponseRecord(
            response_id=response_id,
            image_id=image_id,
            model_name="direct_text",
            response_text=text,
        )
        return self.extract_from_response(dummy_resp)

