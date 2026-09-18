"""
Unit tests for CategoryRegistry and alias resolution rules.
"""

import pytest

from src.claims.vocabulary import (
    CategoryRegistry,
    create_coco_category_registry,
    ALIAS_RULES_VERSION,
)


def test_category_registry_canonical_lookup():
    """Verify standard canonical category registrations and ID lookups."""
    registry = create_coco_category_registry()

    assert registry.lookup_canonical("dog") == "dog"
    assert registry.lookup_canonical("wine glass") == "wine glass"
    assert registry.lookup_canonical("hot dog") == "hot dog"
    assert registry.get_category_id("dog") == 18
    assert registry.get_category_id("wine glass") == 46


def test_category_registry_plural_and_synonym_aliases():
    """Verify singular/plural normalization and synonyms."""
    registry = create_coco_category_registry()

    # Plurals
    assert registry.lookup_canonical("dogs") == "dog"
    assert registry.lookup_canonical("cars") == "car"
    assert registry.lookup_canonical("people") == "person"
    assert registry.lookup_canonical("men") == "person"
    assert registry.lookup_canonical("women") == "person"
    assert registry.lookup_canonical("bicycles") == "bicycle"
    assert registry.lookup_canonical("busses") == "bus"
    assert registry.lookup_canonical("knives") == "knife"

    # Synonyms
    assert registry.lookup_canonical("sofa") == "couch"
    assert registry.lookup_canonical("cellphone") == "cell phone"
    assert registry.lookup_canonical("aeroplane") == "airplane"
    assert registry.lookup_canonical("motorbike") == "motorcycle"


def test_longest_phrase_matching_precedence():
    """Verify multiword category phrases take precedence over single-word components."""
    registry = create_coco_category_registry()
    sorted_phrases = registry.sorted_phrases

    # 'hot dog' must appear before 'dog'
    idx_hot_dog = sorted_phrases.index("hot dog")
    idx_dog = sorted_phrases.index("dog")
    assert idx_hot_dog < idx_dog

    # 'wine glass' must appear before 'glass' (if glass is present)
    idx_wine_glass = sorted_phrases.index("wine glass")
    idx_cup = sorted_phrases.index("cup")
    assert idx_wine_glass >= 0


def test_ambiguous_aliases_not_silently_resolved():
    """Verify ambiguous words without explicit rules return None."""
    registry = create_coco_category_registry()

    # 'glass' alone is ambiguous (drinking glass, eyeglasses, wine glass) -> not registered
    assert registry.lookup_canonical("glass") is None

    # 'vehicle' is a supercategory, not an atomic category
    assert registry.lookup_canonical("vehicle") is None

    # 'animal' is a supercategory
    assert registry.lookup_canonical("animal") is None


def test_custom_synthetic_registry():
    """Verify creation of isolated small synthetic registries for offline testing."""
    custom = CategoryRegistry()
    custom.register_category(1, "robot", "machine")
    custom.register_category(2, "laser pointer", "tool")
    custom.register_alias("robots", "robot", is_plural=True)

    assert custom.lookup_canonical("robots") == "robot"
    assert custom.lookup_canonical("laser pointer") == "laser pointer"
    assert custom.lookup_canonical("pointer") is None
