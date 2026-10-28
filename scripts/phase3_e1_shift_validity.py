"""Phase 3 / E1: do clean-calibrated conformal hallucination filters stay valid
under measured visual-evidence shifts? (CPU only; uses the Phase 1 pilot data.)

    python scripts/phase3_e1_shift_validity.py \
        --claims reports/phase1/claims_table.jsonl --out reports/phase3

Writes e1_results.json and e1_report.md. EXPLORATORY: the pilot images come from
the PREREG train split; the locked test split is not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase3.shift_validity import SHIFTS, load_claims, run_splits, score_shift_stats  # noqa: E402

LABEL = {
    "cebc_naive": "CEBC-style, clean-calibrated (target δ)",
    "conflvlm_naive": "ConfLVLM-style, clean-calibrated (target α)",
    "cebc_robust_directional": "CEBC + one-sided shift radius (target δ+β)",
    "cebc_robust_pool": "CEBC + two-sided image radius, all shifts (target δ+β)",
    "conflvlm_robust_pool": "ConfLVLM + two-sided image radius, all shifts (target α+β)",
    "cebc_robust_loo": "CEBC + image radius from the other two shifts",
    "conflvlm_robust_loo": "ConfLVLM + image radius from the other two shifts",
}


def fmt(d) -> str:
    return f"{d['mean']:.3f} [{d['q025']:.3f}, {d['q975']:.3f}]"


def write_report(res: dict, shift_stats: dict, path: Path) -> None:
    st = res["settings"]
    L = []
    L.append("# Phase 3 / E1 — Do clean-calibrated hallucination filters stay valid under evidence shift?\n")
    L.append("**Status: exploratory pilot** (PREREG train split, 200 images; test split untouched). "
             f"{st['n_claims']} claims, {st['n_hallucinated']} hallucinated; {st['n_splits']} random "
             f"image-level splits, {int(st['cal_frac']*100)}% calibration. Evidence score = logit(OWLv2). "
             f"Targets: δ = α = {st['delta']}, β = {st['beta']}. Calibration always on CLEAN images; "
             "brackets are the 2.5–97.5% range across splits.\n")
    L.append("## How each shift moves the evidence score\n")
    L.append("| Shift | Hallucinated: mean change | share that went up | Supported: mean change | share that went up |")
    L.append("|---|---|---|---|---|")
    for s in SHIFTS:
        h, p = shift_stats[f"{s}/hallucinated"], shift_stats[f"{s}/supported"]
        L.append(f"| {s} | {h['mean']:+.3f} | {h['share_up']:.2f} | {p['mean']:+.3f} | {p['share_up']:.2f} |")
    L.append("")
    L.append("## Realised risk and share of supported claims kept\n")
    L.append("Risk: CEBC rows = share of hallucinated claims that pass; ConfLVLM rows = share of images whose kept set contains a hallucinated claim.\n")
    L.append("| Method | Test condition | Risk (mean [2.5%, 97.5%]) | Supported claims kept |")
    L.append("|---|---|---|---|")
    for m, cd in res["results"].items():
        for c, d in cd.items():
            L.append(f"| {LABEL.get(m, m)} | {c} | {fmt(d['risk'])} | {d['retained_supported']['mean']:.3f} |")
    L.append("")
    L.append("## Calibrated radii (logit units)\n")
    for k, d in res["eps"].items():
        L.append(f"- {k}: {fmt(d)}")
    L.append("")
    L.append("## Reading guide (mechanical; conclusions are the author's)\n")
    L.append("- A clean-calibrated method is *broken* by a shift if its mean risk under that shift exceeds its target.")
    L.append("- A robust method is *useful* only if it restores the target where the naive one fails, at an acceptable loss in supported claims kept.")
    path.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--claims", type=Path, default=Path("reports/phase1/claims_table.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase3"))
    ap.add_argument("--splits", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    claims = load_claims(json.loads(l) for l in a.claims.read_text().splitlines() if l.strip())
    res = run_splits(claims, n_splits=a.splits, seed=a.seed)
    ss = score_shift_stats(claims)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e1_results.json").write_text(json.dumps({**res, "shift_stats": ss}, indent=2))
    write_report(res, ss, a.out / "e1_report.md")
    print(f"wrote {a.out/'e1_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
