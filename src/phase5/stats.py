"""Phase 5 statistics (CPU only): transfer matrix, prevalence decomposition, local gap,
recalibration curve, grouped conformal risk control, TOST, Benjamini-Hochberg and a
label-free warning signal.

A "pipeline table" is three aligned arrays over claims, one row per claim:
    image  int    image id (the exchangeable unit)
    hall   bool   CHAIR (or human) label, True = hallucinated
    score  float  detector logit for the claim (larger = more evidence)
Images with no claim never enter a table.

Theorem numbers refer to paper/tai/tai.tex.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import beta as _beta
from scipy.stats import norm as _norm

from src.phase3.shift_validity import conformal_upper

MARGIN = 0.03


@dataclass(frozen=True)
class Table:
    image: np.ndarray
    hall: np.ndarray
    score: np.ndarray

    @staticmethod
    def of(image: Iterable[int], hall: Iterable[bool], score: Iterable[float]) -> "Table":
        t = Table(np.asarray(list(image), dtype=np.int64), np.asarray(list(hall), dtype=bool),
                  np.asarray(list(score), dtype=float))
        if not (len(t.image) == len(t.hall) == len(t.score)):
            raise ValueError("image, hall and score must have the same length")
        return t

    def images(self) -> np.ndarray:
        return np.unique(self.image)

    def subset(self, ids: Iterable[int]) -> "Table":
        m = np.isin(self.image, np.asarray(list(ids)))
        return Table(self.image[m], self.hall[m], self.score[m])


def image_max(t: Table) -> Tuple[np.ndarray, np.ndarray]:
    """Per image: id and the largest hallucinated score (-inf if none)."""
    ids, inv = np.unique(t.image, return_inverse=True)
    m = np.full(len(ids), -np.inf)
    np.maximum.at(m, inv[t.hall], t.score[t.hall])
    return ids, m


# ------------------------------------------------------------------ calibration

def thresholds(cal: Table, delta: float, alpha: float, n_resamples: int = 200, frac: float = 0.8,
               seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Claim-level and image-level thresholds on `n_resamples` random image subsets."""
    ids = cal.images()
    rng = np.random.default_rng(seed)
    m = int(round(frac * len(ids)))
    tc, ti = [], []
    for _ in range(n_resamples):
        sub = cal.subset(rng.choice(ids, m, replace=False))
        tc.append(conformal_upper(sub.score[sub.hall], 1 - delta))
        ti.append(conformal_upper(image_max(sub)[1], 1 - alpha))
    return np.array(tc), np.array(ti)


# ------------------------------------------------------------------ evaluation

def per_image(test: Table, tc: np.ndarray, ti: np.ndarray) -> Dict[str, np.ndarray]:
    """Per test image, averaged over the threshold resamples."""
    ids, inv = np.unique(test.image, return_inverse=True)
    n = len(ids)
    out = {k: np.zeros(n) for k in ("kh", "ks", "nh", "ns", "bad", "abst")}
    for j in range(n):
        s, h = test.score[inv == j], test.hall[inv == j]
        keep_c = s[None, :] > tc[:, None]
        keep_i = s[None, :] > ti[:, None]
        out["kh"][j] = keep_c[:, h].sum(1).mean()
        out["ks"][j] = keep_c[:, ~h].sum(1).mean()
        out["nh"][j], out["ns"][j] = h.sum(), (~h).sum()
        out["bad"][j] = keep_i[:, h].any(1).mean() if h.any() else 0.0
        out["abst"][j] = (~keep_i.any(1)).mean()
    out["ids"] = ids
    return out


def metrics(st: Dict[str, np.ndarray], w: np.ndarray) -> Dict[str, np.ndarray]:
    """Risks for image weights w, shape (N,) or (B, N)."""
    def ratio(a, b):
        num, den = (w * st[a]).sum(-1), (w * st[b]).sum(-1)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    has = (st["nh"] > 0).astype(float)
    n_has = (w * has).sum(-1)
    return {
        "claim_risk": ratio("kh", "nh"),
        "image_risk": (w * st["bad"]).sum(-1) / w.sum(-1),
        "prevalence": n_has / w.sum(-1),
        "rho": np.where(n_has > 0, (w * st["bad"]).sum(-1) / np.maximum(n_has, 1e-12), np.nan),
        "supported_kept": ratio("ks", "ns"),
        "abstention": (w * st["abst"]).sum(-1) / w.sum(-1),
    }


def bootstrap(st: Dict[str, np.ndarray], n_boot: int = 2000, seed: int = 1) -> Dict[str, dict]:
    """Image-level cluster bootstrap: point, 95% interval and the bootstrap draws."""
    n = len(st["ids"])
    point = metrics(st, np.ones(n))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    boot = metrics(st, W)
    out = {}
    for k, v in point.items():
        b = boot[k][np.isfinite(boot[k])]
        out[k] = {"point": float(v), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)), "boot": b}
    return out


def decide(m: dict, target: float, margin: float = MARGIN) -> str:
    """Pre-registered interval rule (prereg-v2)."""
    if m["lo"] > target:
        return "violated"
    return "preserved" if m["hi"] <= target + margin else "inconclusive"


# ------------------------------------------------------------------ Theorem 1 (local gap)

def local_gap(src_hall: np.ndarray, tgt_hall: np.ndarray, taus: np.ndarray) -> float:
    """Estimate of E Delta(tau_hat): change in survival of hallucinated scores at each
    calibrated threshold, averaged over the threshold resamples (Theorem 1)."""
    taus = np.asarray(taus)[:, None]
    return float(np.mean((tgt_hall[None, :] > taus).mean(1) - (src_hall[None, :] > taus).mean(1)))


def d_plus(src: np.ndarray, tgt: np.ndarray) -> float:
    """Global one-sided gap sup_t [P(tgt > t) - P(src > t)] (hypothesis H2)."""
    t = np.unique(np.concatenate([src, tgt]))
    return float(max(0.0, ((tgt[None, :] > t[:, None]).mean(1) - (src[None, :] > t[:, None]).mean(1)).max()))


def warning_signal(src_all: np.ndarray, tgt_all: np.ndarray, taus: np.ndarray) -> float:
    """Label-free version of the local gap: change in the share of ALL claims (supported
    and hallucinated, no labels needed) scoring above the calibrated threshold."""
    return local_gap(src_all, tgt_all, taus)


# ------------------------------------------------------------------ Theorem 3 (label budget)

def beta_tail(m: int, delta: float = 0.10, eps: float = MARGIN) -> float:
    """P(realised risk > delta + eps) after recalibrating on m i.i.d. hallucinated claims."""
    k = math.ceil((m + 1) * (1 - delta) - 1e-9)
    if k > m:
        return 0.0
    return float(_beta.sf(delta + eps, m + 1 - k, k))


def label_budget(prob: float = 0.10, delta: float = 0.10, eps: float = MARGIN, horizon: int = 5000) -> int:
    """Smallest m from which beta_tail stays at or below `prob` up to `horizon`."""
    ok = np.array([beta_tail(m, delta, eps) <= prob for m in range(1, horizon + 1)])
    bad = np.flatnonzero(~ok)
    return int(bad[-1] + 2) if bad.size else 1


def recalibration_curve(tgt: Table, ms: Sequence[int], repeats: int = 200, delta: float = 0.10,
                        eps: float = MARGIN, seed: int = 3) -> List[dict]:
    """Empirical check of Theorem 3. Split the target table by image into a labelling pool
    and an evaluation half; draw m hallucinated claims from the pool, recalibrate, and
    measure the realised risk on the evaluation half. Draws are by image (cluster-aware):
    images are added in random order until m hallucinated claims are collected."""
    rng = np.random.default_rng(seed)
    ids = tgt.images()
    rows = []
    for m in ms:
        over, risks, short = 0, [], False
        for _ in range(repeats):
            perm = rng.permutation(ids)
            pool, ev = perm[: len(perm) // 2], perm[len(perm) // 2:]
            P, E = tgt.subset(pool), tgt.subset(ev)
            got: List[float] = []
            for i in rng.permutation(P.images()):
                got.extend(P.score[(P.image == i) & P.hall].tolist())
                if len(got) >= m:
                    break
            if len(got) < m:           # too few hallucinated claims in half the images
                short = True
                break
            tau = conformal_upper(got[:m], 1 - delta)
            r = float(np.mean(E.score[E.hall] > tau))
            risks.append(r)
            over += r > delta + eps
        row = {"m": int(m), "theory_p_over": beta_tail(int(m), delta, eps)}
        if short:
            row.update(insufficient=True, p_over=None, mean_risk=None)
        else:
            row.update(insufficient=False, p_over=over / repeats, mean_risk=float(np.mean(risks)))
        rows.append(row)
    return rows


# ------------------------------------------------------------------ grouped conformal risk control

def _image_loss(t: Table, lam: np.ndarray) -> np.ndarray:
    """L_i(lam) = (# hallucinated claims kept at threshold lam) / (# claims in image i),
    shape (len(lam), n_images). Non-increasing in lam, bounded by 1."""
    ids, inv = np.unique(t.image, return_inverse=True)
    n_claims = np.bincount(inv, minlength=len(ids)).astype(float)
    kept_h = (t.score[None, :] > lam[:, None]) & t.hall[None, :]
    num = np.zeros((len(lam), len(ids)))
    for j in range(len(ids)):
        num[:, j] = kept_h[:, inv == j].sum(1)
    return num / n_claims[None, :]


def crc_threshold(cal: Table, alpha: float, grid: Optional[np.ndarray] = None) -> float:
    """Conformal risk control (Angelopoulos et al., ICLR 2024) with images as the
    exchangeable unit: the smallest lam with (n/(n+1)) mean L_i(lam) + 1/(n+1) <= alpha.
    Gives E[L_test(lam_hat)] <= alpha exactly for exchangeable images."""
    grid = np.unique(np.concatenate([[-np.inf], cal.score, [np.inf]])) if grid is None else np.sort(grid)
    L = _image_loss(cal, grid)
    n = L.shape[1]
    ok = (n / (n + 1)) * L.mean(1) + 1 / (n + 1) <= alpha
    return float(grid[np.argmax(ok)]) if ok.any() else math.inf


def crc_risk(test: Table, lam: float) -> float:
    return float(_image_loss(test, np.array([lam]))[0].mean())


# ------------------------------------------------------------------ equivalence and multiplicity

def tost_upper(boot: np.ndarray, target: float, eps: float) -> float:
    """One-sided bootstrap p-value for H0: risk >= target + eps (the upper arm of TOST;
    for a risk only exceedance matters)."""
    boot = np.asarray(boot)
    return float((np.sum(boot >= target + eps) + 1) / (len(boot) + 1))


def benjamini_hochberg(pvals: Dict[str, float], q: float = 0.05) -> Dict[str, dict]:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    adj, run = {}, 1.0
    for r in range(m - 1, -1, -1):
        run = min(run, pvals[keys[r]] * m / (r + 1))
        adj[keys[r]] = run
    return {k: {"p": pvals[k], "p_bh": adj[k], "reject": adj[k] <= q} for k in pvals}


def required_claims(p0: float = 0.10, p1: float = 0.13, alpha: float = 0.05, power: float = 0.80,
                    deff: float = 1.0) -> int:
    """One-sided binomial sample size to detect p1 > p0, times a design effect."""
    za, zb = _norm.ppf(1 - alpha), _norm.ppf(power)
    n = ((za * math.sqrt(p0 * (1 - p0)) + zb * math.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2
    return int(math.ceil(n * deff))


def design_effect(t: Table, icc: float) -> float:
    """1 + (mbar - 1) ICC with mbar the mean hallucinated claims per image that has any."""
    _, inv = np.unique(t.image[t.hall], return_inverse=True)
    mbar = float(np.bincount(inv).mean()) if inv.size else 1.0
    return 1.0 + (mbar - 1.0) * icc


# ------------------------------------------------------------------ transfer matrix

def transfer_cell(cal: Table, test: Table, delta: float = 0.10, alpha: float = 0.10,
                  n_resamples: int = 200, n_boot: int = 2000, seed: int = 0) -> dict:
    """Calibrate on `cal` (source generator), test on `test` (target generator)."""
    tc, ti = thresholds(cal, delta, alpha, n_resamples, seed=seed)
    st = per_image(test, tc, ti)
    b = bootstrap(st, n_boot, seed + 1)
    cb = b["claim_risk"]["boot"]
    return {
        "claim": {k: v for k, v in b["claim_risk"].items() if k != "boot"} | {"decision": decide(b["claim_risk"], delta)},
        "image": {k: v for k, v in b["image_risk"].items() if k != "boot"} | {"decision": decide(b["image_risk"], alpha)},
        "prevalence": b["prevalence"]["point"], "rho": b["rho"]["point"],
        "supported_kept": b["supported_kept"]["point"], "abstention": b["abstention"]["point"],
        "local_gap": local_gap(cal.score[cal.hall], test.score[test.hall], tc),
        "warning": warning_signal(cal.score, test.score, tc),
        "p_tost_claim": tost_upper(cb, delta, MARGIN),
        "p_violated_claim": float((np.sum(cb <= delta) + 1) / (len(cb) + 1)),
        "n_test_hallucinated": int(test.hall.sum()), "n_test_images": int(len(test.images())),
    }


def transfer_matrix(cal: Dict[str, Table], test: Dict[str, Table], **kw) -> Dict[str, Dict[str, dict]]:
    """cal[g] and test[g] are the calibration and test tables of generator g (one detector)."""
    return {s: {t: transfer_cell(cal[s], test[t], **kw) for t in test} for s in cal}
