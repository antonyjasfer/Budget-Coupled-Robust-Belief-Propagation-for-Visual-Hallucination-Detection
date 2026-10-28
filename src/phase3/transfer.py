"""Phase 3b analysis: do clean-calibrated hallucination filters stay valid under
shifts that can RAISE the evidence for absent objects?

Calibration source (always): Phase 1 COCO pilot, LLaVA-1.5 captions, OWLv2
score on clean images. Test conditions:

  ref             held-out COCO images, same pipeline (sanity check: risk ~ target)
  swap_owlvit_v1  held-out COCO images, OWLv2 replaced by OWL-ViT v1 (no recalibration)
  swap_gdino      held-out COCO images, OWLv2 replaced by Grounding DINO
  vlm_qwen        held-out COCO images, captions from Qwen2.5-VL-3B instead of LLaVA
  domain_amber    AMBER images (non-COCO), LLaVA captions, OWLv2

For the detector swaps the change can be MEASURED on calibration images (same
claims, two detectors), so we also report the one-sided shift-calibrated
("robust") filters with target + beta. For vlm_qwen and domain_amber the claims
differ, so only the naive filters and diagnostics are reported.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.phase3.shift_validity import Claim, by_image, conformal_upper, logit

SWAPS = ("owlvit_v1", "grounding_dino")


def claims_from_rows(rows: Sequence[dict], swap: Optional[Dict[tuple, dict]] = None) -> List[Claim]:
    """Phase-1-format rows -> Claims with score['clean'] = logit(OWLv2 clean) and,
    when `swap` is given, score[<detector>] for each swap detector."""
    out = []
    for r in rows:
        ev = r.get("evidence", {}).get("clean")
        if not ev:
            continue
        sc = {"clean": logit(ev["owl_score"])}
        if swap is not None:
            s = swap.get((int(r["image_id"]), r["category"]))
            if s is None:
                continue
            for d in SWAPS:
                v = s.get(d)
                if v is None or not np.isfinite(v):
                    break
                sc[d] = logit(v)
            else:
                out.append(Claim(int(r["image_id"]), int(r["hallucinated"]), sc))
            continue
        out.append(Claim(int(r["image_id"]), int(r["hallucinated"]), sc))
    return out


def swap_index(swap_rows: Sequence[dict]) -> Dict[tuple, dict]:
    idx = {}
    for r in swap_rows:
        for c in r["claims"]:
            idx[(int(r["image_id"]), c["category"])] = c
    return idx


# --------------------------------------------------------------------------- filters

def tau_claim(cal: Sequence[Claim], delta: float, key: str = "clean") -> float:
    return conformal_upper([c.score[key] for c in cal if c.hallucinated], 1.0 - delta)


def tau_image(cal_imgs: Dict[int, List[Claim]], alpha: float, key: str = "clean") -> float:
    r = []
    for cs in cal_imgs.values():
        h = [c.score[key] for c in cs if c.hallucinated]
        r.append(max(h) if h else -math.inf)
    return conformal_upper(r, 1.0 - alpha)


def eps_claim_up(cal: Sequence[Claim], key: str, beta: float) -> float:
    """One-sided radius: (1-beta) quantile of (key - clean) over hallucinated claims."""
    return max(conformal_upper([c.score[key] - c.score["clean"] for c in cal if c.hallucinated], 1.0 - beta), 0.0)


def eps_image_up(cal_imgs: Dict[int, List[Claim]], key: str, beta: float) -> float:
    """Image-level one-sided radius: per image, the largest upward change among its
    hallucinated claims (-inf if none); conformal (1-beta) quantile over images."""
    r = []
    for cs in cal_imgs.values():
        h = [c.score[key] - c.score["clean"] for c in cs if c.hallucinated]
        r.append(max(h) if h else -math.inf)
    return max(conformal_upper(r, 1.0 - beta), 0.0)


def risk_claim(test: Sequence[Claim], key: str, tau: float, eps: float = 0.0) -> Dict[str, float]:
    h = [c.score[key] - eps > tau for c in test if c.hallucinated]
    s = [c.score[key] - eps > tau for c in test if not c.hallucinated]
    return {"risk": float(np.mean(h)) if h else float("nan"),
            "retained_supported": float(np.mean(s)) if s else float("nan")}


def risk_image(test_imgs: Dict[int, List[Claim]], key: str, tau: float, eps: float = 0.0) -> Dict[str, float]:
    bad, ks, ns = [], 0, 0
    for cs in test_imgs.values():
        bad.append(any(c.hallucinated and c.score[key] - eps > tau for c in cs))
        for c in cs:
            if not c.hallucinated:
                ns += 1
                ks += c.score[key] - eps > tau
    return {"risk": float(np.mean(bad)) if bad else float("nan"),
            "retained_supported": ks / ns if ns else float("nan")}


# --------------------------------------------------------------------------- diagnostics

def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    pos, neg = s[y == 0], s[y == 1]  # higher evidence should mean "supported"
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def describe(claims: Sequence[Claim], key: str) -> dict:
    h = np.array([c.score[key] for c in claims if c.hallucinated])
    s = np.array([c.score[key] for c in claims if not c.hallucinated])
    return {"n_claims": len(claims), "n_images": len({c.image_id for c in claims}),
            "hallucination_rate": float(len(h) / max(1, len(claims))),
            "auroc": auroc([c.hallucinated for c in claims], [c.score[key] for c in claims]),
            "median_logit_hallucinated": float(np.median(h)) if h.size else float("nan"),
            "median_logit_supported": float(np.median(s)) if s.size else float("nan")}


# --------------------------------------------------------------------------- experiment

def run(coco: Sequence[Claim], qwen: Sequence[Claim], amber: Sequence[Claim],
        n_splits: int = 500, cal_frac: float = 0.5, delta: float = 0.10, alpha: float = 0.10,
        beta: float = 0.05, seed: int = 0) -> dict:
    """`coco` must carry swap scores (claims_from_rows with swap index)."""
    imgs = by_image(coco)
    q_imgs = by_image(qwen)
    a_imgs = by_image(amber)
    ids = np.array(sorted(imgs))
    rng = np.random.default_rng(seed)
    rec: Dict[str, Dict[str, List[float]]] = {}

    def put(name, d):
        m = rec.setdefault(name, {"risk": [], "retained_supported": []})
        for k in m:
            m[k].append(d[k])

    n_cal = int(round(cal_frac * len(ids)))
    for _ in range(n_splits):
        perm = rng.permutation(ids)
        cal_ids, test_ids = perm[:n_cal].tolist(), set(perm[n_cal:].tolist())
        cal_imgs = {i: imgs[i] for i in cal_ids}
        cal = [c for cs in cal_imgs.values() for c in cs]
        t_imgs = {i: imgs[i] for i in test_ids}
        test = [c for cs in t_imgs.values() for c in cs]
        tq_imgs = {i: q_imgs[i] for i in test_ids if i in q_imgs}
        tq = [c for cs in tq_imgs.values() for c in cs]
        tc, ti = tau_claim(cal, delta), tau_image(cal_imgs, alpha)

        put("claim/ref", risk_claim(test, "clean", tc))
        put("image/ref", risk_image(t_imgs, "clean", ti))
        for d in SWAPS:
            put(f"claim/swap_{d}", risk_claim(test, d, tc))
            put(f"image/swap_{d}", risk_image(t_imgs, d, ti))
            put(f"claim/swap_{d}+robust", risk_claim(test, d, tc, eps_claim_up(cal, d, beta)))
            put(f"image/swap_{d}+robust", risk_image(t_imgs, d, ti, eps_image_up(cal_imgs, d, beta)))
        if tq:
            put("claim/vlm_qwen", risk_claim(tq, "clean", tc))
            put("image/vlm_qwen", risk_image(tq_imgs, "clean", ti))
        if amber:
            put("claim/domain_amber", risk_claim(amber, "clean", tc))
            put("image/domain_amber", risk_image(a_imgs, "clean", ti))

    def summ(v):
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"mean": float("nan"), "q025": float("nan"), "q975": float("nan"), "n": 0}
        return {"mean": float(a.mean()), "q025": float(np.quantile(a, 0.025)),
                "q975": float(np.quantile(a, 0.975)), "n": int(a.size)}

    diag = {"coco_llava_owlv2": describe(coco, "clean")}
    for d in SWAPS:
        diag[f"coco_llava_{d}"] = describe(coco, d)
        up = np.array([c.score[d] - c.score["clean"] for c in coco if c.hallucinated])
        diag[f"coco_llava_{d}"]["hallucinated_share_up_vs_owlv2"] = float((up > 0).mean()) if up.size else float("nan")
    if qwen:
        diag["coco_qwen_owlv2"] = describe(qwen, "clean")
    if amber:
        diag["amber_llava_owlv2"] = describe(amber, "clean")
    return {"settings": {"n_splits": n_splits, "cal_frac": cal_frac, "delta": delta, "alpha": alpha,
                         "beta": beta, "seed": seed},
            "targets": {"claim": delta, "image": alpha, "claim+robust": delta + beta, "image+robust": alpha + beta},
            "diagnostics": diag,
            "results": {k: {m: summ(v) for m, v in d.items()} for k, d in rec.items()}}
