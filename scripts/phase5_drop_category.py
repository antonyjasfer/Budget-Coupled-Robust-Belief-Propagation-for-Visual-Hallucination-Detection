"""Phase 5 exploratory sensitivity (NOT pre-registered): re-run the frozen analysis with the
claims of one CHAIR category removed, e.g. "orange", which newer captioners mostly use as a
colour word ("orange traffic cones", "orange and white") but CHAIR labels as the fruit.

    python scripts/phase5_drop_category.py --src OUT --dst OUT_S2 --drop orange

DST must be a new, empty folder; the shards, prep5.json, synonyms.txt and amber/ are copied
from SRC so that SRC's results are never overwritten.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import phase4 as P4  # noqa: E402
import phase5 as P5  # noqa: E402

GENS = ["llava", "qwen25", "internvl3", "llava_t1", "qwen3_4b", "instructblip"]


class _Filtered:
    """The CHAIR vocabulary with some categories' claims removed."""

    def __init__(self, vocab, drop):
        self._v, self._drop = vocab, set(drop)

    def __getattr__(self, name):
        return getattr(self._v, name)

    def extract_claims(self, *args, **kw):
        cc = self._v.extract_claims(*args, **kw)
        cc.claims[:] = [c for c in cc.claims if c.node_word not in self._drop]
        return cc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--drop", nargs="+", required=True)
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    if a.dst.exists() and any(a.dst.iterdir()):
        sys.exit(f"{a.dst} is not empty")
    a.dst.mkdir(parents=True, exist_ok=True)
    for p in a.src.iterdir():
        if p.suffix == ".jsonl" or p.name in ("prep5.json", "synonyms.txt"):
            shutil.copy2(p, a.dst / p.name)
    shutil.copytree(a.src / "amber", a.dst / "amber")
    base = P4._vocab
    P4._vocab = lambda out: _Filtered(base(out), a.drop)
    P5.stage_analyze(a.dst, GENS, a.resamples, a.boot)


if __name__ == "__main__":
    main()
