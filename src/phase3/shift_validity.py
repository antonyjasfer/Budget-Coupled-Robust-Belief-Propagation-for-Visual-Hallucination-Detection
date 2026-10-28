"""Phase 3, experiment E1: do clean-calibrated conformal hallucination filters stay
valid when the visual evidence shifts, and does a shift-calibrated evidence radius
restore validity?

Two published-style baselines, both calibrated on CLEAN images only:

* CEBC-style (Mishra et al. 2026, Eq. 4): a detector threshold tau set from the
  scores of hallucinated mentions so that a new hallucinated claim passes with
  probability <= delta. Keep a claim iff score > tau.
* ConfLVLM-style (Li et al. 2025, lambda = 0): an image-level threshold so that
  the kept claim set of a new image contains a hallucinated claim with
  probability <= alpha. Nonconformity of image i is the highest score among its
  hallucinated claims (-inf if none). Keep a claim iff score > tau.

Shift-robust versions (BCRBP with J = 0, the per-claim case of the root-dominance
result): calibrate an evidence radius eps from measured clean -> shifted score
changes on calibration images, then keep a claim iff (shifted score - eps) > tau,
i.e. iff the LOWER end of its evidence interval clears the clean threshold.
If |shifted - clean| <= eps (probability >= 1 - beta), a hallucinated claim that
passes under shift would also have passed on the clean image, so by a union bound
the risk under shift is at most target + beta.

Scores are logit(OWLv2 score). All guarantees are marginal over exchangeable
images. This module only computes; the script writes the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

SHIFTS = ("blur", "jpeg", "downsample")
CLIP = 1e-4


def logit(p: float) -> float:
    p = min(max(float(p), CLIP), 1.0 - CLIP)
    return math.log(p / (1.0 - p))


def conformal_upper(scores: Sequence[float], level: float) -> float:
    """The ceil((n+1)*level)-th smallest score; +inf if that index exceeds n.

    For a new score S exchangeable with `scores`, P(S <= q) >= level.
    """
    s = np.sort(np.asarray(scores, dtype=float))  # -inf entries are allowed
    n = len(s)
    k = math.ceil((n + 1) * level - 1e-9)  # guard: 20 * 0.9 == 18.000000000000004
    if k > n:
        return math.inf
    return float(s[k - 1])


@dataclass
class Claim:
    image_id: int
    hallucinated: int
    score: Dict[str, float]  # shift name -> logit evidence score ("clean" included)


def load_claims(rows: Iterable[dict], feature: str = "owl_score") -> List[Claim]:
    out = []
    for r in rows:
        out.append(Claim(int(r["image_id"]), int(r["hallucinated"]),
                         {s: logit(v[feature]) for s, v in r["evidence"].items()}))
    return out


def by_image(claims: Sequence[Claim]) -> Dict[int, List[Claim]]:
    d: Dict[int, List[Claim]] = {}
    for c in claims:
        d.setdefault(c.image_id, []).append(c)
    return d


# --------------------------------------------------------------------------- calibration

def calibrate_cebc(cal: Sequence[Claim], delta: float) -> float:
    """Threshold so that P(new hallucinated claim has clean score > tau) <= delta."""
    h = [c.score["clean"] for c in cal if c.hallucinated]
    return conformal_upper(h, 1.0 - delta)


def calibrate_conflvlm(cal_imgs: Dict[int, List[Claim]], alpha: float) -> float:
    """Threshold so that P(new image keeps >= 1 hallucinated claim) <= alpha."""
    r = []
    for cs in cal_imgs.values():
        hs = [c.score["clean"] for c in cs if c.hallucinated]
        r.append(max(hs) if hs else -math.inf)
    return conformal_upper(r, 1.0 - alpha)


def calibrate_eps(cal_imgs: Dict[int, List[Claim]], beta: float, shifts: Sequence[str]) -> float:
    """Evidence radius: per image, the largest |shifted - clean| over its claims and
    over `shifts`; eps is the conformal (1 - beta) quantile across images, so a new
    image has all its claims inside the radius under every listed shift with
    probability >= 1 - beta."""
    r = []
    for cs in cal_imgs.values():
        r.append(max(abs(c.score[s] - c.score["clean"]) for c in cs for s in shifts))
    return conformal_upper(r, 1.0 - beta)


def calibrate_eps_directional(cal: Sequence[Claim], beta: float, shifts: Sequence[str]) -> float:
    """One-sided radius for the claim-level (CEBC-style) guarantee.

    A hallucinated claim can only slip through under shift if its score went UP.
    So it suffices that (shifted - clean) <= eps for hallucinated claims; eps is
    the conformal (1 - beta) quantile of those signed changes (pooled over
    `shifts`), floored at 0. Same claim-level exchangeability as CEBC itself.
    """
    up = [c.score[s] - c.score["clean"] for c in cal if c.hallucinated for s in shifts]
    return max(conformal_upper(up, 1.0 - beta), 0.0)


# --------------------------------------------------------------------------- evaluation

def keep(score: float, tau: float, eps: float = 0.0) -> bool:
    return (score - eps) > tau


def eval_cebc(test: Sequence[Claim], shift: str, tau: float, eps: float = 0.0) -> Dict[str, float]:
    h = [c for c in test if c.hallucinated]
    s = [c for c in test if not c.hallucinated]
    fp = np.mean([keep(c.score[shift], tau, eps) for c in h]) if h else float("nan")
    ret = np.mean([keep(c.score[shift], tau, eps) for c in s]) if s else float("nan")
    return {"risk": float(fp), "retained_supported": float(ret)}


def eval_conflvlm(test_imgs: Dict[int, List[Claim]], shift: str, tau: float, eps: float = 0.0) -> Dict[str, float]:
    bad, kept_sup, n_sup = [], 0, 0
    for cs in test_imgs.values():
        bad.append(any(c.hallucinated and keep(c.score[shift], tau, eps) for c in cs))
        for c in cs:
            if not c.hallucinated:
                n_sup += 1
                kept_sup += keep(c.score[shift], tau, eps)
    return {"risk": float(np.mean(bad)), "retained_supported": kept_sup / n_sup if n_sup else float("nan")}


# --------------------------------------------------------------------------- experiment

def run_splits(claims: Sequence[Claim], n_splits: int = 500, cal_frac: float = 0.5,
               delta: float = 0.10, alpha: float = 0.10, beta: float = 0.05,
               seed: int = 0) -> dict:
    """Random image-level cal/test splits. For each split and test condition, records
    the realised risk and the share of supported claims kept, for:
      naive        clean-calibrated threshold applied to (possibly shifted) scores
      robust_pool  eps calibrated jointly over all three shifts
      robust_loo   eps calibrated on the two OTHER shifts (unseen-shift stress test)
      robust_directional  one-sided eps from hallucinated claims only (CEBC-style)
    """
    imgs = by_image(claims)
    ids = np.array(sorted(imgs))
    rng = np.random.default_rng(seed)
    conds = ("clean",) + SHIFTS
    rec: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    eps_log: Dict[str, List[float]] = {"pool": [], "directional": [], **{f"loo_{s}": [] for s in SHIFTS}}

    def put(method, cond, d):
        m = rec.setdefault(method, {}).setdefault(cond, {"risk": [], "retained_supported": []})
        for k in m:
            m[k].append(d[k])

    n_cal = int(round(cal_frac * len(ids)))
    for _ in range(n_splits):
        perm = rng.permutation(ids)
        cal_ids, test_ids = set(perm[:n_cal].tolist()), perm[n_cal:].tolist()
        cal_imgs = {i: imgs[i] for i in cal_ids}
        test_imgs = {i: imgs[i] for i in test_ids}
        cal = [c for cs in cal_imgs.values() for c in cs]
        test = [c for cs in test_imgs.values() for c in cs]

        t_cebc = calibrate_cebc(cal, delta)
        t_conf = calibrate_conflvlm(cal_imgs, alpha)
        eps_pool = calibrate_eps(cal_imgs, beta, SHIFTS)
        eps_dir = calibrate_eps_directional(cal, beta, SHIFTS)
        eps_log["pool"].append(eps_pool)
        eps_log["directional"].append(eps_dir)
        for cond in conds:
            put("cebc_naive", cond, eval_cebc(test, cond, t_cebc))
            put("conflvlm_naive", cond, eval_conflvlm(test_imgs, cond, t_conf))
            put("cebc_robust_pool", cond, eval_cebc(test, cond, t_cebc, eps_pool))
            put("conflvlm_robust_pool", cond, eval_conflvlm(test_imgs, cond, t_conf, eps_pool))
            put("cebc_robust_directional", cond, eval_cebc(test, cond, t_cebc, eps_dir))
        for s in SHIFTS:
            others = [o for o in SHIFTS if o != s]
            e = calibrate_eps(cal_imgs, beta, others)
            eps_log[f"loo_{s}"].append(e)
            put("cebc_robust_loo", s, eval_cebc(test, s, t_cebc, e))
            put("conflvlm_robust_loo", s, eval_conflvlm(test_imgs, s, t_conf, e))

    def summ(v):
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "q025": float(np.quantile(a, 0.025)), "q975": float(np.quantile(a, 0.975)), "n": int(a.size)}

    out = {"settings": {"n_splits": n_splits, "cal_frac": cal_frac, "delta": delta, "alpha": alpha,
                        "beta": beta, "seed": seed, "n_images": int(len(ids)), "n_claims": len(claims),
                        "n_hallucinated": int(sum(c.hallucinated for c in claims))},
           "targets": {"cebc": delta, "conflvlm": alpha, "cebc_robust": delta + beta, "conflvlm_robust": alpha + beta},
           "eps": {k: summ(v) for k, v in eps_log.items()},
           "results": {m: {c: {k: summ(v) for k, v in d.items()} for c, d in cd.items()} for m, cd in rec.items()}}
    return out


def score_shift_stats(claims: Sequence[Claim]) -> dict:
    """How each shift moves the evidence score, separately for hallucinated and
    supported claims (positive = shift makes the object look MORE present)."""
    out = {}
    for s in SHIFTS:
        for lab, name in ((1, "hallucinated"), (0, "supported")):
            d = np.array([c.score[s] - c.score["clean"] for c in claims if c.hallucinated == lab])
            out[f"{s}/{name}"] = {"mean": float(d.mean()), "median": float(np.median(d)),
                                  "share_up": float((d > 0).mean()), "n": int(d.size)}
    return out
