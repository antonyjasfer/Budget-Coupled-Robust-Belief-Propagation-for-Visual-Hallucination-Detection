"""Run the Phase 5 analysis code on the Phase 4 raw files (phase4_results.zip).

Two purposes: (1) validate src/phase5/stats.py against the published Phase 4 numbers,
(2) compute the exact prevalence decomposition, label-free warning signal, grouped CRC
and the (small-sample) recalibration curve for the two Phase 4 generator swaps.
All of it is exploratory (post hoc relative to prereg-v2).

    python scripts/phase5_reanalyze_phase4.py --data DIR_WITH_UNZIPPED_PHASE4 [--out reports/phase5]
DIR must also hold synonyms.txt (scripts/phase1_pilot.py downloads and checks it).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase1.chair import ChairVocabulary  # noqa: E402
from src.phase3.shift_validity import logit  # noqa: E402
from src.phase5 import stats as S  # noqa: E402


def jsonl(d: Path, stem: str):
    for p in sorted(d.glob(f"{stem}.s*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def table(d: Path, vocab, prep, cap_stem: str, owl_set: str, ids: set) -> S.Table:
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    owl = {int(r["image_id"]): r["scores"] for r in jsonl(d, "owl") if r["set"] == owl_set and r["cond"] == "clean"}
    rows = []
    for r in jsonl(d, cap_stem):
        i = int(r["image_id"])
        if i not in ids:
            continue
        for c in vocab.extract_claims(r["caption"]).claims:
            s = owl.get(i, {}).get(c.node_word)
            if s is not None and s == s:
                rows.append((i, c.node_word not in gt.get(i, set()), logit(s)))
    return S.Table.of(*zip(*rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "phase5")
    a = ap.parse_args()
    prep = json.loads((a.data / "prep.json").read_text())
    vocab = ChairVocabulary.from_file(a.data / "synonyms.txt")
    cal_ids, test_ids = set(prep["cal_a"]) | set(prep["cal_b"]), set(prep["test"])
    cal = table(a.data, vocab, prep, "captions_llava_coco", "coco_llava", cal_ids)
    test = {"llava": table(a.data, vocab, prep, "captions_llava_coco", "coco_llava", test_ids),
            "qwen25": table(a.data, vocab, prep, "captions_qwen", "coco_qwen", test_ids),
            "internvl3": table(a.data, vocab, prep, "captions_internvl", "coco_internvl", test_ids)}
    res = {"note": "exploratory re-analysis of Phase 4 raw files with src/phase5/stats.py", "cells": {}, "crc": {},
           "recalibration": {}}
    lam = S.crc_threshold(cal, 0.10)
    for g, t in test.items():
        c = S.transfer_cell(cal, t, n_resamples=200, n_boot=2000, seed=0)
        c["hallucinated_share"] = float(t.hall.mean())
        res["cells"][g] = c
        res["crc"][g] = S.crc_risk(t, lam)
        print(f"{g:10s} claim {c['claim']['point']:.3f} [{c['claim']['lo']:.3f},{c['claim']['hi']:.3f}] {c['claim']['decision']:12s}"
              f" image {c['image']['point']:.3f} {c['image']['decision']:12s} pi {c['prevalence']:.3f} rho {c['rho']:.3f}"
              f" local {c['local_gap']:+.3f} warn {c['warning']:+.3f} crc {res['crc'][g]:.3f}"
              f" n_hall {c['n_test_hallucinated']}")
    for g in ("qwen25", "internvl3"):
        res["recalibration"][g] = S.recalibration_curve(test[g], [25, 50, 75], repeats=500)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "phase4_reanalysis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["recalibration"], indent=1))


if __name__ == "__main__":
    main()
