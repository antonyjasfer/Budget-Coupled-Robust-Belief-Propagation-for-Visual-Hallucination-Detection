"""Phase 5 exploratory diagnostics (NOT pre-registered; prereg-v3 decisions are unchanged).

Run after `scripts/phase5.py analyze`, on the same folder:
    python scripts/phase5_exploratory.py --out OUT [--boot 2000]
Writes OUT/phase5_exploratory.json.

E2  Joint calibration + test bootstrap of every COCO claim-level cell (all three detectors):
    calibration images and test images are resampled together, so the interval covers the
    calibration draw as well as the test sample. Diagonal cells should then not be violated.
E3  Truncation and prevalence (H9): pi of truncated against non-truncated captions.
E4  Clustering of hallucinated claims per target (H11): mean hallucinated claims per
    hallucinating image, images needed for 170 claims, and P(realised risk > 0.13) at
    m = 170 re-estimated with 1,000 repeats (Monte Carlo error of the 200-repeat value).
E5  Detector dependence (H12): per detector, the median hallucinated and supported score of
    each generator's calibration split and the calibrated claim threshold.
E6  Operating points of the signed warning W (H8): alarms, caught violations and false
    alarms at W > 0.01, 0.02, 0.03 over the OWLv2 COCO off-diagonal cells.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import phase4 as P4  # noqa: E402
import phase5 as P5  # noqa: E402
from src.phase3.shift_validity import conformal_upper  # noqa: E402
from src.phase5 import stats as S  # noqa: E402

GENS = ["llava", "qwen25", "internvl3", "llava_t1", "qwen3_4b", "instructblip"]


def _hall(t: S.Table):
    """Hallucinated scores with the index of their image, and the number of images."""
    ids, inv = np.unique(t.image, return_inverse=True)
    return t.score[t.hall], inv[t.hall], len(ids)


def joint_cell(cal: S.Table, test: S.Table, n_boot: int, rng, delta: float = 0.10) -> dict:
    """Claim-level risk with calibration and test images resampled together.
    The weighted conformal quantile equals conformal_upper on the replicated scores."""
    cs, ci, nc = _hall(cal)
    ts, ti, nt = _hall(test)
    order = np.argsort(cs)
    cs_sorted, ci_sorted = cs[order], ci[order]
    point = float(np.mean(ts > conformal_upper(cs, 1 - delta)))
    risks = np.empty(n_boot)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, nc, nc), minlength=nc)[ci_sorted].astype(float)
        cum = np.cumsum(w)
        k = math.ceil((cum[-1] + 1) * (1 - delta) - 1e-9)
        tau = math.inf if k > cum[-1] else cs_sorted[np.searchsorted(cum, k)]
        wt = np.bincount(rng.integers(0, nt, nt), minlength=nt)[ti].astype(float)
        risks[b] = float(wt[ts > tau].sum() / max(wt.sum(), 1e-12))
    lo, hi = float(np.quantile(risks, 0.025)), float(np.quantile(risks, 0.975))
    return {"point": point, "lo": lo, "hi": hi, "decision": S.decide({"lo": lo, "hi": hi}, delta)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    out = a.out
    res = json.loads((out / "phase5_results.json").read_text())
    T = P5.build_tables(out, GENS)
    rng = np.random.default_rng(2028)
    ex: dict = {}

    # E2 joint bootstrap, every COCO cell of every detector.
    ex["E2_joint_bootstrap"] = {
        d: {s: {t: joint_cell(T[(s, "coco", "cal", d)], T[(t, "coco", "test", d)], a.boot, rng)
                for t in GENS} for s in GENS} for d in P5.DETECTORS}
    P4.log("E2 done")

    # E3 truncation and prevalence, COCO test split (images that produced claims).
    prep, vocab = P5._prep(out), P4._vocab(out)
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    test_ids = set(prep["test"])
    e3 = {}
    for g in GENS:
        n = {True: [0, 0], False: [0, 0]}          # truncated -> [images with claims, with hallucination]
        for r in P4.read_all(out, P5.cap_file(g, "coco")):
            iid = int(r["image_id"])
            if iid not in test_ids:
                continue
            cl = vocab.extract_claims(r["caption"]).claims
            if not cl:
                continue
            tr = bool(r.get("hit_max_tokens"))
            n[tr][0] += 1
            n[tr][1] += any(c.node_word not in gt.get(iid, set()) for c in cl)
        e3[g] = {k: {"images": v[0], "pi": (v[1] / v[0]) if v[0] else None}
                 for k, v in (("truncated", n[True]), ("not_truncated", n[False]))}
    ex["E3_truncation_prevalence"] = e3

    # E4 clustering per target of a violated OWLv2 cell.
    targets = sorted({k.split("->")[1] for k in res["recalibration"]})
    e4 = {}
    for t in targets:
        tab = T[(t, "coco", "test", "owlv2")]
        _, hi, n = _hall(tab)
        k = np.bincount(hi, minlength=n).astype(float)
        kk = k[k > 0]
        row = S.recalibration_curve(tab, [170], repeats=1000, seed=4)[0]
        e4[t] = {"mean_hall_per_hall_image": float(kk.mean()), "share_hall_images_ge3": float(np.mean(kk >= 3)),
                 "images_for_170": float(170 / kk.mean()),
                 "p_over_1000_repeats": row["p_over"], "theory": row["theory_p_over"]}
    ex["E4_clustering_H11"] = e4
    P4.log("E4 done")

    # E5 detector dependence: median scores and the calibrated claim threshold.
    ex["E5_detector_scores"] = {
        d: {g: {"median_hall_score": float(np.median(T[(g, "coco", "cal", d)].score[T[(g, "coco", "cal", d)].hall])),
                "median_supported_score": float(np.median(T[(g, "coco", "cal", d)].score[~T[(g, "coco", "cal", d)].hall])),
                "threshold": float(conformal_upper(T[(g, "coco", "cal", d)].score[T[(g, "coco", "cal", d)].hall], 0.90))}
            for g in GENS} for d in P5.DETECTORS}

    # E6 operating points of W over the OWLv2 COCO off-diagonal cells.
    O = res["matrices"]["owlv2"]
    cells = [(s, t) for s in GENS for t in GENS if s != t]
    viol = {c for c in cells if O[c[0]][c[1]]["claim"]["decision"] == "violated"}
    e6 = {}
    for w0 in (0.01, 0.02, 0.03):
        alarm = {c for c in cells if O[c[0]][c[1]]["warning"] > w0}
        e6[str(w0)] = {"alarms": len(alarm), "caught": len(alarm & viol), "violated": len(viol),
                       "missed": sorted(f"{s}->{t}" for s, t in viol - alarm),
                       "false_alarms": sorted(f"{s}->{t}" for s, t in alarm - viol),
                       "max_risk_unflagged": max((O[s][t]["claim"]["point"] for s, t in set(cells) - alarm),
                                                 default=None)}
    ex["E6_warning_operating_points"] = e6

    (out / "phase5_exploratory.json").write_text(json.dumps(ex, indent=1, default=float))
    P4.log("exploratory: wrote phase5_exploratory.json")


if __name__ == "__main__":
    main()
