"""PREREG image splits over COCO val2014 and the Phase 1 pilot sample.

docs/PREREG.md section 3 fixes, before any model fitting:
    train 1,500 · validation 300 · Cal-A 300 · Cal-B 400 · test 500 (locked), seed 42.

The Phase 1 pilot draws its 200 images ONLY from the train split, so the
locked test split is never looked at. The pilot is divided into
pilot_cal (100) and pilot_eval (100).

Determinism: the candidate universe is the sorted list of val2014 image ids;
a numpy Generator(PCG64(seed)) permutation selects and orders images.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SPLIT_SIZES: Dict[str, int] = {
    "train": 1500,
    "validation": 300,
    "cal_a": 300,
    "cal_b": 400,
    "test": 500,
}
SPLIT_ORDER = ["train", "validation", "cal_a", "cal_b", "test"]
PREREG_SEED = 42
PILOT_SIZE = 200
PILOT_CAL_SIZE = 100


@dataclass
class SplitManifest:
    seed: int
    universe_size: int
    universe_sha256: str
    splits: Dict[str, List[int]]
    pilot_cal: List[int]
    pilot_eval: List[int]
    manifest_sha256: str = ""
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _sha256_ints(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


def build_prereg_splits(
    val2014_image_ids: Sequence[int],
    seed: int = PREREG_SEED,
    sizes: Dict[str, int] = SPLIT_SIZES,
    pilot_size: int = PILOT_SIZE,
    pilot_cal_size: int = PILOT_CAL_SIZE,
) -> SplitManifest:
    """Deterministically assign val2014 images to PREREG splits and draw the pilot."""
    universe = sorted({int(i) for i in val2014_image_ids})
    total = sum(sizes[s] for s in SPLIT_ORDER)
    if len(universe) < total:
        raise ValueError(f"Universe has {len(universe)} images, need {total}.")
    if pilot_size > sizes["train"] or pilot_cal_size >= pilot_size:
        raise ValueError("Pilot must fit inside the train split and leave pilot_eval non-empty.")

    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(universe))[:total]
    chosen = [universe[i] for i in order]

    splits: Dict[str, List[int]] = {}
    start = 0
    for name in SPLIT_ORDER:
        splits[name] = chosen[start:start + sizes[name]]
        start += sizes[name]

    pilot = splits["train"][:pilot_size]  # train is already in random order
    manifest = SplitManifest(
        seed=seed,
        universe_size=len(universe),
        universe_sha256=_sha256_ints(universe),
        splits=splits,
        pilot_cal=pilot[:pilot_cal_size],
        pilot_eval=pilot[pilot_cal_size:],
        notes=[
            "Universe = all COCO val2014 image ids (sorted).",
            "Pilot images are drawn only from the train split; the test split stays locked.",
        ],
    )
    body = json.dumps({"splits": splits, "pilot_cal": manifest.pilot_cal,
                       "pilot_eval": manifest.pilot_eval, "seed": seed}, sort_keys=True)
    manifest.manifest_sha256 = hashlib.sha256(body.encode()).hexdigest()
    return manifest


def assert_pilot_isolated(manifest: SplitManifest) -> None:
    """Raise if any pilot image lies outside the train split or inside the test split."""
    train = set(manifest.splits["train"])
    test = set(manifest.splits["test"])
    pilot = set(manifest.pilot_cal) | set(manifest.pilot_eval)
    if not pilot <= train:
        raise AssertionError("Pilot images must come from the train split only.")
    if pilot & test:
        raise AssertionError("Pilot touches the locked test split.")
    if set(manifest.pilot_cal) & set(manifest.pilot_eval):
        raise AssertionError("pilot_cal and pilot_eval overlap.")


def save_manifest(manifest: SplitManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.to_json(), encoding="utf-8")


def load_manifest(path: Path) -> SplitManifest:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d["splits"] = {k: [int(x) for x in v] for k, v in d["splits"].items()}
    return SplitManifest(**d)


def coco_val2014_filename(image_id: int) -> str:
    return f"COCO_val2014_{int(image_id):012d}.jpg"
