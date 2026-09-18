"""
Unit tests for ConservativeClaimExtractor.
"""

from pathlib import Path
import json
import pytest

from src.data.schemas import GeneratedResponseRecord
from src.claims.vocabulary import create_coco_category_registry
from src.claims.extraction import (
    ConservativeClaimExtractor,
    MentionRejectionReason,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "milestone3"
PARSER_TEST_CASES_PATH = FIXTURES_DIR / "parser_test_cases.json"


def test_claim_extraction_on_hand_authored_test_cases():
    """Verify claim extraction across all hand-authored edge case test sentences."""
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)

    with open(PARSER_TEST_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for case in cases:
        test_id = case["test_id"]
        text = case["text"]
        expected_accepted = set(case["expected_accepted"])
        expected_rejections = case["expected_rejected_reasons"]

        resp = GeneratedResponseRecord(
            response_id=f"resp_{test_id}",
            image_id="img_test",
            model_name="test_model",
            response_text=text,
        )

        report = extractor.extract_from_response(resp)
        actual_accepted = {c.object_category for c in report.accepted_claims}
        actual_rejections = [r.reason.value for r in report.rejected_mentions]

        assert actual_accepted == expected_accepted, (
            f"Case '{test_id}' failed accepted claims: expected {expected_accepted}, got {actual_accepted}"
        )

        for expected_reason in expected_rejections:
            assert expected_reason in actual_rejections, (
                f"Case '{test_id}' missing expected rejection reason '{expected_reason}': got {actual_rejections}"
            )


def test_character_offset_accuracy_and_unicode():
    """Verify precise Unicode-safe character offsets into original text."""
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)

    # Unicode text with emoji and accent
    text = "In the room, a cat 🐱 sits on the couch."
    resp = GeneratedResponseRecord(
        response_id="resp_unicode",
        image_id="img_unicode",
        model_name="test_model",
        response_text=text,
    )

    report = extractor.extract_from_response(resp)
    assert len(report.accepted_claims) == 2  # cat, couch

    cat_claim = next(c for c in report.accepted_claims if c.object_category == "cat")
    span = cat_claim.spans[0]
    extracted_slice = text[span.start_char : span.end_char]
    assert extracted_slice == "cat"

    couch_claim = next(c for c in report.accepted_claims if c.object_category == "couch")
    span_couch = couch_claim.spans[0]
    assert text[span_couch.start_char : span_couch.end_char] == "couch"


def test_repeated_mentions_deduplication():
    """Verify multiple mentions of the same category in one response are merged with spans preserved."""
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)

    text = "A dog barked at another dog while a third dog ran."
    resp = GeneratedResponseRecord(
        response_id="resp_rep",
        image_id="img_rep",
        model_name="test_model",
        response_text=text,
    )

    report = extractor.extract_from_response(resp)
    assert len(report.accepted_claims) == 1
    dog_claim = report.accepted_claims[0]
    assert dog_claim.object_category == "dog"
    assert len(dog_claim.spans) == 3

    for span in dog_claim.spans:
        assert text[span.start_char : span.end_char] == "dog"


def test_empty_and_ineligible_responses():
    """Verify clean handling of empty or no-object responses."""
    registry = create_coco_category_registry()
    extractor = ConservativeClaimExtractor(registry)

    resp_empty = GeneratedResponseRecord(
        response_id="resp_empty",
        image_id="img_empty",
        model_name="test_model",
        response_text="   ",
    )
    rep_empty = extractor.extract_from_response(resp_empty)
    assert rep_empty.num_accepted_claims == 0
    assert rep_empty.num_rejected_mentions == 0

    resp_no_objs = GeneratedResponseRecord(
        response_id="resp_no_objs",
        image_id="img_no_objs",
        model_name="test_model",
        response_text="A scenic sunset over misty mountains in the evening.",
    )
    rep_no_objs = extractor.extract_from_response(resp_no_objs)
    assert rep_no_objs.num_accepted_claims == 0
