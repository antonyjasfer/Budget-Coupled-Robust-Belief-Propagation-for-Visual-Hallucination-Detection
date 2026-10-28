"""AMBER (Wang et al. 2023, arXiv:2311.07397) generative split as a domain shift.

AMBER images are not from COCO. Each generative image (id 1..1004) is annotated
with `truth` (objects present) and `hallu` (plausible objects verified ABSENT).
We map both lists to COCO node words with the same CHAIR vocabulary as Phase 1,
then label a caption claim as
    supported     if its node word is in the truth set,
    hallucinated  if it is in the hallu set and not in the truth set,
    unknown       otherwise (excluded from risk metrics, counted in the report).
Only COCO categories can be claims (CHAIR vocabulary), so objects such as "sky"
or "tree" are never claims.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.phase1.chair import ChairVocabulary

ANNOTATIONS_URL = "https://raw.githubusercontent.com/junyangwang0410/AMBER/534babf6bbfcce2e735c26289dedfb21cef3c939/data/annotations.json"
ANNOTATIONS_SHA256 = "db22a436c4836fbb295e60b4e80d519573077579fb7032f01b193c1bd022b797"
IMAGES_GDRIVE_ID = "1MaCHgtupcZUjf007anNl4_MV0o4DjXvl"  # from the AMBER README ("Image Download")
N_GENERATIVE = 1004


def image_filename(amber_id: int) -> str:
    return f"AMBER_{int(amber_id)}.jpg"


def generative_entries(annotations: Iterable[dict]) -> Dict[int, dict]:
    return {int(a["id"]): a for a in annotations if a.get("type") == "generative"}


def node_sets(vocab: ChairVocabulary, entry: dict) -> Tuple[Set[str], Set[str]]:
    truth = {n for w in entry.get("truth", []) for n in vocab.node_words(w)}
    hallu = {n for w in entry.get("hallu", []) for n in vocab.node_words(w)} - truth
    return truth, hallu


def label_claim(node: str, truth: Set[str], hallu: Set[str]) -> Optional[int]:
    if node in truth:
        return 0
    if node in hallu:
        return 1
    return None


def select_images(vocab: ChairVocabulary, entries: Dict[int, dict], n: int, seed: int = 42) -> List[int]:
    """Fixed random sample of `n` generative images with >= 1 COCO object present,
    so that captions have labelable claims. Deterministic given the annotations."""
    eligible = sorted(i for i, e in entries.items() if node_sets(vocab, e)[0])
    rng = random.Random(seed)
    return sorted(rng.sample(eligible, min(n, len(eligible))))


def build_claims(vocab: ChairVocabulary, entries: Dict[int, dict], captions: Sequence[dict],
                 evidence: Sequence[dict], source: str) -> Tuple[List[dict], Dict[str, int]]:
    """Claims table rows in the Phase 1 format (image_id, category, hallucinated,
    evidence[shift] = {...}); unknown claims are dropped and counted."""
    ev = {(int(r["image_id"]), r["shift"], r["category"]): r for r in evidence}
    rows, counts = [], {"captions": 0, "claims": 0, "supported": 0, "hallucinated": 0, "unknown": 0}
    for cap in captions:
        iid = int(cap["image_id"])
        if iid not in entries:
            continue
        counts["captions"] += 1
        truth, hallu = node_sets(vocab, entries[iid])
        for c in vocab.extract_claims(cap["caption"]).claims:
            counts["claims"] += 1
            lab = label_claim(c.node_word, truth, hallu)
            if lab is None:
                counts["unknown"] += 1
                continue
            counts["hallucinated" if lab else "supported"] += 1
            evd = {}
            for s in ("clean", "blur", "jpeg", "downsample"):
                r = ev.get((iid, s, c.node_word))
                if r is not None:
                    evd[s] = {k: float(r[k]) for k in ("owl_score", "clip_crop", "clip_global")}
            rows.append({"image_id": iid, "source": source, "category": c.node_word,
                         "hallucinated": lab, "evidence": evd})
    return rows, counts
