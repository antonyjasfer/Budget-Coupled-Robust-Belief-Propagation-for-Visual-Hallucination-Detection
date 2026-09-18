"""
Canonical category vocabulary management and alias registry.

Provides versioned category aliases, singular/plural canonicalization,
longest-matching phrase resolution, and explicit ambiguity guards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
import re

ALIAS_RULES_VERSION: str = "1.0.0"


@dataclass
class CategoryInfo:
    """
    Canonical category record.

    Attributes:
        category_id: Integer category ID.
        canonical_name: Normalized lowercase category name (e.g. 'wine glass').
        supercategory: High-level category grouping (e.g. 'kitchen').
    """
    category_id: int
    canonical_name: str
    supercategory: str = ""

    def __post_init__(self):
        self.canonical_name = self.canonical_name.strip().lower()


@dataclass
class AliasRule:
    """
    Explicit versioned alias rule mapping an surface alias phrase to a canonical category.

    Attributes:
        alias_phrase: Surface phrase (e.g. 'dogs', 'cellphone', 'sofa').
        canonical_name: Target canonical category name (e.g. 'dog', 'cell phone', 'couch').
        rule_version: Version string of the rule set.
        is_plural: Whether the alias is an inflected plural form.
        notes: Clarification or justification.
    """
    alias_phrase: str
    canonical_name: str
    rule_version: str = ALIAS_RULES_VERSION
    is_plural: bool = False
    notes: str = ""

    def __post_init__(self):
        self.alias_phrase = self.alias_phrase.strip().lower()
        self.canonical_name = self.canonical_name.strip().lower()


class CategoryRegistry:
    """
    Registry maintaining canonical object categories and explicit alias rules.

    Guarantees:
    - Longest-matching category phrases take precedence (e.g., 'hot dog' over 'dog').
    - Ambiguous aliases are never silently resolved without explicit rules.
    - Preserves bidirectional mappings between category IDs and canonical names.
    """

    def __init__(self, alias_rules_version: str = ALIAS_RULES_VERSION):
        self.alias_rules_version = alias_rules_version
        self.categories_by_id: Dict[int, CategoryInfo] = {}
        self.categories_by_name: Dict[str, CategoryInfo] = {}
        self.aliases: Dict[str, AliasRule] = {}
        self._sorted_phrases: List[str] = []

    def register_category(self, category_id: int, canonical_name: str, supercategory: str = "") -> None:
        """Register a canonical category."""
        cname = canonical_name.strip().lower()
        if not cname:
            raise ValueError("canonical_name cannot be empty")

        if category_id in self.categories_by_id and self.categories_by_id[category_id].canonical_name != cname:
            raise ValueError(
                f"Conflict: Category ID {category_id} already registered as '{self.categories_by_id[category_id].canonical_name}'"
            )

        info = CategoryInfo(category_id=category_id, canonical_name=cname, supercategory=supercategory)
        self.categories_by_id[category_id] = info
        self.categories_by_name[cname] = info
        self._update_sorted_phrases()

    def register_alias(
        self,
        alias_phrase: str,
        canonical_name: str,
        is_plural: bool = False,
        notes: str = "",
        rule_version: Optional[str] = None
    ) -> None:
        """Register an explicit alias mapping."""
        alias = alias_phrase.strip().lower()
        cname = canonical_name.strip().lower()

        if cname not in self.categories_by_name:
            raise ValueError(f"Cannot register alias '{alias}' for unregistered canonical category '{cname}'")

        if alias in self.categories_by_name and alias != cname:
            raise ValueError(f"Cannot register alias '{alias}' because it is already a distinct canonical category")

        rule = AliasRule(
            alias_phrase=alias,
            canonical_name=cname,
            rule_version=rule_version or self.alias_rules_version,
            is_plural=is_plural,
            notes=notes,
        )
        self.aliases[alias] = rule
        self._update_sorted_phrases()

    def _update_sorted_phrases(self) -> None:
        """Keep matchable phrases sorted in descending order of length/word count for longest match."""
        all_phrases = set(self.categories_by_name.keys()).union(self.aliases.keys())
        # Sort by length descending, then alphabetically for determinism
        self._sorted_phrases = sorted(list(all_phrases), key=lambda p: (-len(p.split()), -len(p), p))

    @property
    def sorted_phrases(self) -> List[str]:
        """List of all registered category phrases and aliases sorted longest-first."""
        return self._sorted_phrases

    def lookup_canonical(self, phrase: str) -> Optional[str]:
        """Look up canonical category name for a given phrase or alias."""
        norm = phrase.strip().lower()
        if norm in self.categories_by_name:
            return norm
        if norm in self.aliases:
            return self.aliases[norm].canonical_name
        return None

    def get_category_id(self, canonical_name: str) -> Optional[int]:
        """Get integer category ID for a canonical name."""
        info = self.categories_by_name.get(canonical_name.strip().lower())
        return info.category_id if info else None


def create_coco_category_registry() -> CategoryRegistry:
    """
    Factory creating a CategoryRegistry populated with the standard 80 MS COCO object categories
    and versioned, conservative singular/plural and synonym alias rules.
    """
    registry = CategoryRegistry(alias_rules_version=ALIAS_RULES_VERSION)

    # Standard 80 COCO categories (ID, canonical name, supercategory)
    coco_80 = [
        (1, "person", "person"),
        (2, "bicycle", "vehicle"),
        (3, "car", "vehicle"),
        (4, "motorcycle", "vehicle"),
        (5, "airplane", "vehicle"),
        (6, "bus", "vehicle"),
        (7, "train", "vehicle"),
        (8, "truck", "vehicle"),
        (9, "boat", "vehicle"),
        (10, "traffic light", "outdoor"),
        (11, "fire hydrant", "outdoor"),
        (13, "stop sign", "outdoor"),
        (14, "parking meter", "outdoor"),
        (15, "bench", "outdoor"),
        (16, "bird", "animal"),
        (17, "cat", "animal"),
        (18, "dog", "animal"),
        (19, "horse", "animal"),
        (20, "sheep", "animal"),
        (21, "cow", "animal"),
        (22, "elephant", "animal"),
        (23, "bear", "animal"),
        (24, "zebra", "animal"),
        (25, "giraffe", "animal"),
        (27, "backpack", "accessory"),
        (28, "umbrella", "accessory"),
        (31, "handbag", "accessory"),
        (32, "tie", "accessory"),
        (33, "suitcase", "accessory"),
        (34, "frisbee", "sports"),
        (35, "skis", "sports"),
        (36, "snowboard", "sports"),
        (37, "sports ball", "sports"),
        (38, "kite", "sports"),
        (39, "baseball bat", "sports"),
        (40, "baseball glove", "sports"),
        (41, "skateboard", "sports"),
        (42, "surfboard", "sports"),
        (43, "tennis racket", "sports"),
        (44, "bottle", "kitchen"),
        (46, "wine glass", "kitchen"),
        (47, "cup", "kitchen"),
        (48, "fork", "kitchen"),
        (49, "knife", "kitchen"),
        (50, "spoon", "kitchen"),
        (51, "bowl", "kitchen"),
        (52, "banana", "food"),
        (53, "apple", "food"),
        (54, "sandwich", "food"),
        (55, "orange", "food"),
        (56, "broccoli", "food"),
        (57, "carrot", "food"),
        (58, "hot dog", "food"),
        (59, "pizza", "food"),
        (60, "donut", "food"),
        (61, "cake", "food"),
        (62, "chair", "furniture"),
        (63, "couch", "furniture"),
        (64, "potted plant", "furniture"),
        (65, "bed", "furniture"),
        (67, "dining table", "furniture"),
        (70, "toilet", "furniture"),
        (72, "tv", "electronic"),
        (73, "laptop", "electronic"),
        (74, "mouse", "electronic"),
        (75, "remote", "electronic"),
        (76, "keyboard", "electronic"),
        (77, "cell phone", "electronic"),
        (78, "microwave", "appliance"),
        (79, "oven", "appliance"),
        (80, "toaster", "appliance"),
        (81, "sink", "appliance"),
        (82, "refrigerator", "appliance"),
        (84, "book", "indoor"),
        (85, "clock", "indoor"),
        (86, "vase", "indoor"),
        (87, "scissors", "indoor"),
        (88, "teddy bear", "indoor"),
        (89, "hair drier", "indoor"),
        (90, "toothbrush", "indoor"),
    ]

    for cid, cname, supercat in coco_80:
        registry.register_category(cid, cname, supercat)

    # Explicit alias rules (Plurals)
    plurals = [
        ("people", "person"),
        ("persons", "person"),
        ("men", "person"),
        ("man", "person"),
        ("women", "person"),
        ("woman", "person"),
        ("children", "person"),
        ("child", "person"),
        ("bicycles", "bicycle"),
        ("cars", "car"),
        ("motorcycles", "motorcycle"),
        ("airplanes", "airplane"),
        ("buses", "bus"),
        ("busses", "bus"),
        ("trains", "train"),
        ("trucks", "truck"),
        ("boats", "boat"),
        ("traffic lights", "traffic light"),
        ("fire hydrants", "fire hydrant"),
        ("stop signs", "stop sign"),
        ("parking meters", "parking meter"),
        ("benches", "bench"),
        ("birds", "bird"),
        ("cats", "cat"),
        ("kittens", "cat"),
        ("dogs", "dog"),
        ("puppies", "dog"),
        ("horses", "horse"),
        ("sheep", "sheep"),
        ("cows", "cow"),
        ("cattle", "cow"),
        ("elephants", "elephant"),
        ("bears", "bear"),
        ("zebras", "zebra"),
        ("giraffes", "giraffe"),
        ("backpacks", "backpack"),
        ("umbrellas", "umbrella"),
        ("handbags", "handbag"),
        ("purses", "handbag"),
        ("ties", "tie"),
        ("suitcases", "suitcase"),
        ("frisbees", "frisbee"),
        ("ski", "skis"),
        ("snowboards", "snowboard"),
        ("sports balls", "sports ball"),
        ("kites", "kite"),
        ("baseball bats", "baseball bat"),
        ("baseball gloves", "baseball glove"),
        ("skateboards", "skateboard"),
        ("surfboards", "surfboard"),
        ("tennis rackets", "tennis racket"),
        ("bottles", "bottle"),
        ("wine glasses", "wine glass"),
        ("cups", "cup"),
        ("forks", "fork"),
        ("knives", "knife"),
        ("spoons", "spoon"),
        ("bowls", "bowl"),
        ("bananas", "banana"),
        ("apples", "apple"),
        ("sandwiches", "sandwich"),
        ("oranges", "orange"),
        ("carrots", "carrot"),
        ("hot dogs", "hot dog"),
        ("pizzas", "pizza"),
        ("donuts", "donut"),
        ("doughnuts", "donut"),
        ("cakes", "cake"),
        ("chairs", "chair"),
        ("couches", "couch"),
        ("potted plants", "potted plant"),
        ("houseplants", "potted plant"),
        ("beds", "bed"),
        ("dining tables", "dining table"),
        ("toilets", "toilet"),
        ("tvs", "tv"),
        ("televisions", "tv"),
        ("television", "tv"),
        ("laptops", "laptop"),
        ("mice", "mouse"),
        ("remotes", "remote"),
        ("keyboards", "keyboard"),
        ("cell phones", "cell phone"),
        ("cellphones", "cell phone"),
        ("mobile phones", "cell phone"),
        ("smartphones", "cell phone"),
        ("microwaves", "microwave"),
        ("ovens", "oven"),
        ("toasters", "toaster"),
        ("sinks", "sink"),
        ("refrigerators", "refrigerator"),
        ("fridges", "refrigerator"),
        ("books", "book"),
        ("clocks", "clock"),
        ("vases", "vase"),
        ("teddy bears", "teddy bear"),
        ("hair driers", "hair drier"),
        ("hair dryers", "hair drier"),
        ("hair dryer", "hair drier"),
        ("toothbrushes", "toothbrush"),
    ]

    for alias, cname in plurals:
        registry.register_alias(alias, cname, is_plural=True, notes="Standard plural/inflection alias")

    # Explicit synonyms
    synonyms = [
        ("sofa", "couch"),
        ("sofas", "couch"),
        ("aeroplane", "airplane"),
        ("aeroplanes", "airplane"),
        ("plane", "airplane"),
        ("planes", "airplane"),
        ("motorbike", "motorcycle"),
        ("motorbikes", "motorcycle"),
        ("bike", "bicycle"),
        ("cellphone", "cell phone"),
        ("mobile phone", "cell phone"),
        ("smartphone", "cell phone"),
        ("fridge", "refrigerator"),
    ]

    for alias, cname in synonyms:
        if alias not in registry.aliases:
            registry.register_alias(alias, cname, is_plural=False, notes="Standard synonym alias")

    return registry
