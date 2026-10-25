"""CHAIR-faithful object-claim extraction and labelling.

Reimplements the rules of the official CHAIR code (Rohrbach et al., 2018,
https://github.com/LisaAnne/Hallucination, utils/chair.py):

* lower-case, tokenise, singularise every word;
* merge MSCOCO "double words" (e.g. "teddy bear", "hot dog");
* "baby/adult <animal>" -> animal, "passenger jet/train" -> jet/train,
  "bow tie" -> tie, "toilet seat" -> toilet;
* drop "seat" when "toilet" is also in the caption;
* map every remaining word that is a COCO object or synonym to its
  category ("node word") via data/synonyms.txt;
* ground truth objects of an image = categories in its instance annotations
  UNION node words found in its five human captions;
* a mentioned node word is HALLUCINATED iff it is not in the ground truth.

Documented deviations (record in docs/DECISIONS.md):
1. Tokeniser: a regex instead of nltk.word_tokenize (same words for normal prose).
2. Singulariser: `inflect` instead of the Python-2 `pattern.en.singularize`,
   with a small rule fallback when inflect is not installed.
The synonyms file is NOT vendored (no licence stated upstream); it is
downloaded at run time and its SHA-256 is checked (SYNONYMS_SHA256).

Unit of analysis: one CLAIM = one unique node word per caption, with all its
mention character spans. CHAIR_i (per mention) is also reported for
comparability with the literature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

SYNONYMS_URL = "https://raw.githubusercontent.com/LisaAnne/Hallucination/master/data/synonyms.txt"
SYNONYMS_SHA256 = "b685537b36fe21b791c87151b96a991ceb582edff19f3da204dcc7a7905502f2"

COCO_DOUBLE_WORDS = [
    "motor bike", "motor cycle", "air plane", "traffic light", "street light", "traffic signal",
    "stop light", "fire hydrant", "stop sign", "parking meter", "suit case", "sports ball",
    "baseball bat", "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie", "laptop computer",
    "stove top oven", "hot dog", "teddy bear", "home plate", "train track",
]
ANIMAL_WORDS = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                "giraffe", "animal", "cub"]
VEHICLE_WORDS = ["jet", "train"]

_TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[^\sa-z]")

_IRREGULAR = {
    "people": "person", "men": "man", "women": "woman", "children": "child", "mice": "mouse",
    "knives": "knife", "leaves": "leaf", "shelves": "shelf", "wolves": "wolf", "buses": "bus",
    "geese": "goose", "feet": "foot", "teeth": "tooth", "oxen": "ox", "skis": "ski",
    "ties": "tie", "glasses": "glass", "dishes": "dish", "benches": "bench", "boxes": "box",
    "sandwiches": "sandwich", "couches": "couch", "giraffes": "giraffe", "vases": "vase",
}
_UNCHANGED = {"bus", "glass", "grass", "dress", "class", "bicycles", "sheep", "skis",
              "this", "is", "was", "has", "his", "its", "us", "gas", "lens", "boss", "chess"}


def _rule_singularize(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _UNCHANGED or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suf in ("sses", "ches", "shes", "xes", "zes"):
        if word.endswith(suf):
            return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


try:  # pragma: no cover - depends on environment
    import inflect as _inflect

    _ENGINE = _inflect.engine()

    def singularize(word: str) -> str:
        if word in _IRREGULAR:
            return _IRREGULAR[word]
        if word in _UNCHANGED:
            return word
        out = _ENGINE.singular_noun(word)
        return out if out else word

except Exception:  # pragma: no cover
    singularize = _rule_singularize


@dataclass
class Token:
    text: str
    start: int
    end: int


@dataclass
class Claim:
    """One unique COCO object (node word) mentioned in one caption."""
    node_word: str
    surface_forms: List[str]
    char_spans: List[Tuple[int, int]]
    first_char: int
    hallucinated: Optional[bool] = None  # None until ground truth is attached


@dataclass
class CaptionClaims:
    claims: List[Claim] = field(default_factory=list)
    mention_count: int = 0           # CHAIR_i denominator
    hallucinated_mentions: int = 0   # CHAIR_i numerator


class ChairVocabulary:
    def __init__(self, synonym_lines: Iterable[str]):
        self.inverse: Dict[str, str] = {}
        self.objects: Set[str] = set()
        for line in synonym_lines:
            line = line.strip()
            if not line:
                continue
            syns = [s.strip() for s in line.split(",") if s.strip()]
            for s in syns:
                self.objects.add(s)
                self.inverse[s] = syns[0]
        self.double: Dict[str, str] = {w: w for w in COCO_DOUBLE_WORDS}
        for a in ANIMAL_WORDS:
            self.double[f"baby {a}"] = a
            self.double[f"adult {a}"] = a
        for v in VEHICLE_WORDS:
            self.double[f"passenger {v}"] = v
        self.double["bow tie"] = "tie"
        self.double["toilet seat"] = "toilet"
        self.double["wine glas"] = "wine glass"
        self.categories: Set[str] = set(self.inverse.values())

    @classmethod
    def from_file(cls, path) -> "ChairVocabulary":
        with open(path, encoding="utf-8") as f:
            return cls(f.readlines())

    # ------------------------------------------------------------------
    def tokens(self, text: str) -> List[Token]:
        low = text.lower()
        return [Token(singularize(m.group(0)), m.start(), m.end()) for m in _TOKEN_RE.finditer(low)]

    def mentions(self, text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
        """Return (surface word, node word, char span) for every COCO mention, in order."""
        toks = self.tokens(text)
        merged: List[Token] = []
        i = 0
        while i < len(toks):
            if i + 1 < len(toks):
                pair = f"{toks[i].text} {toks[i + 1].text}"
                if pair in self.double:
                    merged.append(Token(self.double[pair], toks[i].start, toks[i + 1].end))
                    i += 2
                    continue
            merged.append(toks[i])
            i += 1
        words = [t.text for t in merged]
        if "toilet" in words and "seat" in words:
            merged = [t for t in merged if t.text != "seat"]
        out = []
        for t in merged:
            if t.text in self.objects:
                out.append((t.text, self.inverse[t.text], (t.start, t.end)))
        return out

    def node_words(self, text: str) -> List[str]:
        return [n for _, n, _ in self.mentions(text)]

    def extract_claims(self, caption: str, gt_objects: Optional[Set[str]] = None) -> CaptionClaims:
        by_node: Dict[str, Claim] = {}
        ms = self.mentions(caption)
        for surface, node, span in ms:
            c = by_node.get(node)
            if c is None:
                c = Claim(node_word=node, surface_forms=[], char_spans=[], first_char=span[0])
                by_node[node] = c
            c.surface_forms.append(surface)
            c.char_spans.append(span)
        result = CaptionClaims(claims=sorted(by_node.values(), key=lambda c: c.first_char),
                               mention_count=len(ms))
        if gt_objects is not None:
            for c in result.claims:
                c.hallucinated = c.node_word not in gt_objects
            result.hallucinated_mentions = sum(1 for _, n, _ in ms if n not in gt_objects)
        return result


def ground_truth_objects(
    vocab: ChairVocabulary,
    image_ids: Sequence[int],
    instances: dict,
    captions: dict,
) -> Dict[int, Set[str]]:
    """CHAIR ground truth: instance-annotation categories UNION objects in human captions."""
    want = {int(i) for i in image_ids}
    gt: Dict[int, Set[str]] = {i: set() for i in want}
    id_to_name = {c["id"]: c["name"] for c in instances["categories"]}
    for ann in instances["annotations"]:
        iid = int(ann["image_id"])
        if iid in want:
            name = id_to_name[ann["category_id"]]
            gt[iid].add(vocab.inverse.get(name, name))
    for ann in captions["annotations"]:
        iid = int(ann["image_id"])
        if iid in want:
            gt[iid].update(vocab.node_words(ann["caption"]))
    return gt
