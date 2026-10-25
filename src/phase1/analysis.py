"""Phase 1 Gate 1 analysis (CPU only).

Inputs (all produced by scripts/phase1_pilot.py):
    manifest.json        PREREG splits + pilot_cal / pilot_eval image ids
    gt_objects.json      CHAIR ground-truth objects per pilot image
    captions.jsonl       LLaVA captions with per-token log-probs
    evidence.jsonl       OWLv2 / CLIP evidence per (image, shift, category)

Outputs: claims table, gate1_results.json, gate1_report.md.

Every statistic has a 95% image-level cluster-bootstrap CI. The gate rules are
mechanical (docs/PREREG.md section 8); interpretation is left to the authors.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

from src.phase1.chair import ChairVocabulary
from src.phase1.gpu_pipeline import SHIFTS, claim_token_features, span_token_logprobs
from src.pgm.tree_model import TreeModel
from src.pgm.standard_bp import compute_rooted_total_field
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds
from src.experiments.parameterization import compute_unary_theta

TAU = 0.5
N_BOOT = 2000
M8_DEFAULT = 0.5493061443340549  # atanh(0.5): current B = eps = J
FEATURES = ["owl_score", "clip_crop", "clip_global", "tok_logprob_mean"]


# ----------------------------------------------------------------------------
# Claims table
# ----------------------------------------------------------------------------

@dataclass
class ClaimRow:
    image_id: int
    pilot_split: str
    category: str
    hallucinated: int
    n_mentions: int
    first_char: int
    tok_logprob_first: Optional[float]
    tok_logprob_mean: Optional[float]
    tok_prob_min: Optional[float]
    evidence: Dict[str, Dict[str, float]]  # shift -> {owl_score, clip_crop, clip_global}


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def build_claims_table(
    vocab: ChairVocabulary,
    manifest: dict,
    gt_objects: Dict[int, List[str]],
    captions: List[dict],
    evidence: List[dict],
) -> Tuple[List[ClaimRow], Dict[str, float]]:
    split_of = {i: "pilot_cal" for i in manifest["pilot_cal"]}
    split_of.update({i: "pilot_eval" for i in manifest["pilot_eval"]})
    ev: Dict[Tuple[int, str, str], dict] = {}
    for r in evidence:
        ev[(int(r["image_id"]), r["shift"], r["category"])] = r

    rows: List[ClaimRow] = []
    mentions = halluc_mentions = caps_with_h = n_caps = 0
    for cap in captions:
        iid = int(cap["image_id"])
        if iid not in split_of:
            continue
        gt = set(gt_objects.get(str(iid), gt_objects.get(iid, [])))
        cc = vocab.extract_claims(cap["caption"], gt_objects=gt)
        n_caps += 1
        mentions += cc.mention_count
        halluc_mentions += cc.hallucinated_mentions
        caps_with_h += int(cc.hallucinated_mentions > 0)
        for c in cc.claims:
            lps: List[float] = []
            if cap.get("token_spans"):
                lps = span_token_logprobs(cap["token_spans"], cap["token_logprobs"],
                                          c.char_spans[0], cap.get("text_offset", 0))
            tf = claim_token_features(lps)
            evd = {}
            for s in SHIFTS:
                e = ev.get((iid, s, c.node_word))
                if e is not None:
                    evd[s] = {k: float(e[k]) for k in ("owl_score", "clip_crop", "clip_global")}
            rows.append(ClaimRow(iid, split_of[iid], c.node_word, int(bool(c.hallucinated)),
                                 len(c.char_spans), c.first_char, tf["tok_logprob_first"],
                                 tf["tok_logprob_mean"], tf["tok_prob_min"], evd))
    chair = {
        "captions": n_caps,
        "chair_i": halluc_mentions / mentions if mentions else float("nan"),
        "chair_s": caps_with_h / n_caps if n_caps else float("nan"),
    }
    return rows, chair


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------

def auroc(y: Sequence[int], score: Sequence[float]) -> float:
    """P(score of a positive > score of a negative), ties counted 1/2."""
    y = np.asarray(y, dtype=int)
    s = np.asarray(score, dtype=float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ece(y: Sequence[int], p: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton/IRLS. Returns weights incl. bias (last)."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    reg = l2 * np.eye(len(w))
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + reg @ w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def cluster_bootstrap(
    groups: Sequence[int],
    stat: Callable[..., float],
    n_boot: int = N_BOOT,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Point estimate and 95% percentile CI, resampling whole images.

    `stat(idx)` receives an index array into the original rows. A statistic
    that needs image membership may take a second argument `boot_groups`:
    each resampled COPY of an image gets its own group id, so an image drawn
    twice is two images, never one merged image.
    """
    import inspect

    groups = np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    required = [q for q in inspect.signature(stat).parameters.values()
                if q.default is inspect.Parameter.empty and q.kind in (q.POSITIONAL_ONLY, q.POSITIONAL_OR_KEYWORD)]
    two_args = len(required) >= 2
    call = (lambda idx, bg: stat(idx, bg)) if two_args else (lambda idx, bg: stat(idx))
    point = call(np.arange(len(groups)), groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        bg = np.concatenate([np.full(len(members[g]), k) for k, g in enumerate(pick)])
        v = call(idx, bg)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def within_image_correlation(y: np.ndarray, groups: np.ndarray) -> float:
    """Pairwise intra-image correlation of labels (Pearson over all within-image ordered pairs)."""
    a, b = [], []
    for g in np.unique(groups):
        v = y[groups == g]
        if len(v) < 2:
            continue
        for i in range(len(v)):
            for j in range(len(v)):
                if i != j:
                    a.append(v[i])
                    b.append(v[j])
    if len(a) < 2:
        return float("nan")
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def permutation_pvalue(y: np.ndarray, groups: np.ndarray, n_perm: int = 2000, seed: int = 1) -> float:
    obs = within_image_correlation(y, groups)
    if math.isnan(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        v = within_image_correlation(rng.permutation(y), groups)
        hits += int(v >= obs)
    return (hits + 1) / (n_perm + 1)


def risk_coverage(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Keep the k most confident claims for k = 1..n. Returns coverage, risk, AURC."""
    order = np.argsort(-confidence, kind="mergesort")
    err = (y_true[order] != y_pred[order]).astype(float)
    k = np.arange(1, len(err) + 1)
    risk = np.cumsum(err) / k
    return k / len(err), risk, float(risk.mean())


# ----------------------------------------------------------------------------
# Graph models
# ----------------------------------------------------------------------------

def _chain_model(theta: np.ndarray, eps: float, j: float) -> TreeModel:
    n = len(theta)
    edges = [(i, i + 1) for i in range(n - 1)]
    return TreeModel(num_nodes=n, theta=np.asarray(theta, float), edges=edges,
                     coupling={e: j for e in edges}, epsilon=np.full(n, eps))


def _bisect(crosses: Callable[[float], bool], hi: float = 8.0, iters: int = 30) -> float:
    """Smallest s in [0, hi] with crosses(s) True (monotone); hi if never."""
    if not crosses(hi):
        return hi
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if crosses(mid):
            hi = mid
        else:
            lo = mid
    return hi


def critical_radius_robust(theta: np.ndarray, j: float, node: int, k: int = 40) -> float:
    """Smallest s such that the robust interval with eps_i = B = s contains tau (M8 default shape)."""
    def crosses(s: float) -> bool:
        m = _chain_model(theta, s, j)
        r = solve_robust_bp(m, node, s, num_grid_steps=k)
        return r.lower_grid <= TAU <= r.upper_grid
    return _bisect(crosses)


def critical_radius_box(theta: np.ndarray, j: float, node: int) -> float:
    """Smallest per-claim radius s such that the independent box interval contains tau."""
    def crosses(s: float) -> bool:
        m = _chain_model(theta, s, j)
        lo, up, _, _ = solve_independent_box_bounds(m, node)
        return lo <= TAU <= up
    return _bisect(crosses)


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def _feature_matrix(rows: Sequence[ClaimRow], shift: str, fill: Dict[str, float]) -> np.ndarray:
    X = []
    for r in rows:
        e = r.evidence.get(shift, {})
        tok = r.tok_logprob_mean if r.tok_logprob_mean is not None else fill["tok_logprob_mean"]
        X.append([e.get("owl_score", fill["owl_score"]), e.get("clip_crop", fill["clip_crop"]),
                  e.get("clip_global", fill["clip_global"]), tok])
    return np.asarray(X, float)


def analyse(rows: List[ClaimRow], chair: Dict[str, float], n_boot: int = N_BOOT) -> dict:
    res: dict = {"chair": chair, "n_boot": n_boot}
    rows = [r for r in rows if "clean" in r.evidence]
    y = np.array([r.hallucinated for r in rows])
    g = np.array([r.image_id for r in rows])

    # --- 1. Descriptives --------------------------------------------------------
    imgs, counts = np.unique(g, return_counts=True)
    res["descriptives"] = {
        "images_with_claims": int(len(imgs)),
        "claims": int(len(rows)),
        "claims_per_image_mean": float(counts.mean()) if len(counts) else float("nan"),
        "claims_per_image_median": float(np.median(counts)) if len(counts) else float("nan"),
        "share_images_with_ge3_claims": float((counts >= 3).mean()) if len(counts) else float("nan"),
        "hallucination_rate": cluster_bootstrap(g, lambda i: float(y[i].mean()), n_boot),
        "claims_missing_token_span": int(sum(r.tok_logprob_mean is None for r in rows)),
    }

    # --- 2. Within-image dependence ---------------------------------------------
    res["dependence"] = {
        "within_image_corr": cluster_bootstrap(g, lambda i, bg: within_image_correlation(y[i], bg), n_boot),
        "permutation_p": permutation_pvalue(y, g),
    }

    # --- 3. Signal of each raw feature (higher support => fewer hallucinations) --
    fill = {f: float(np.nanmean([_feat(r, f) for r in rows])) for f in FEATURES}
    sig = {}
    for f in FEATURES:
        vals = np.array([_feat(r, f) if _feat(r, f) is not None else fill[f] for r in rows])
        sig[f] = {"auroc": cluster_bootstrap(g, lambda i: auroc(y[i], -vals[i]), n_boot)}
    owl = np.array([r.evidence["clean"]["owl_score"] for r in rows])
    sig["owl_score"]["ece_as_probability"] = ece(y, 1 - owl)
    res["signal"] = sig

    # --- 4. Learned unary model (fit on pilot_cal, evaluate on pilot_eval) ------
    cal = [r for r in rows if r.pilot_split == "pilot_cal"]
    evl = [r for r in rows if r.pilot_split == "pilot_eval"]
    Xc, Xe = _feature_matrix(cal, "clean", fill), _feature_matrix(evl, "clean", fill)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    yc, ye = np.array([r.hallucinated for r in cal]), np.array([r.hallucinated for r in evl])
    ge = np.array([r.image_id for r in evl])
    w = fit_logistic((Xc - mu) / sd, yc)
    pe = predict_logistic(w, (Xe - mu) / sd)
    res["learned_unary"] = {
        "features": FEATURES, "weights_standardised": w.tolist(),
        "n_cal": int(len(cal)), "n_eval": int(len(evl)),
        "auroc": cluster_bootstrap(ge, lambda i: auroc(ye[i], pe[i]), n_boot),
        "ece": ece(ye, pe), "brier": float(np.mean((pe - ye) ** 2)),
    }

    # --- 5. Selective prediction on pilot_eval ----------------------------------
    theta_learn = 0.5 * np.log(np.clip(pe, 1e-6, 1 - 1e-6) / np.clip(1 - pe, 1e-6, 1))
    by_img: Dict[int, List[int]] = {}
    for k, r in enumerate(evl):
        by_img.setdefault(r.image_id, []).append(k)
    for ks in by_img.values():
        ks.sort(key=lambda k: evl[k].category)  # M8 chain: alphabetical by claim id
    theta_m8 = np.array([compute_unary_theta(r.evidence["clean"]["owl_score"], None) for r in evl])
    p_m8 = np.zeros(len(evl))
    rad_rob, rad_box = np.zeros(len(evl)), np.zeros(len(evl))
    for ks in by_img.values():
        th = theta_m8[ks]
        model = _chain_model(th, M8_DEFAULT, M8_DEFAULT)
        for node, k in enumerate(ks):
            _, p_m8[k], _ = compute_rooted_total_field(model, root=node, delta=None)
            rad_rob[k] = critical_radius_robust(th, M8_DEFAULT, node)
            rad_box[k] = critical_radius_box(th, M8_DEFAULT, node)
    pred_learn = (pe > TAU).astype(int)
    pred_m8 = (p_m8 > TAU).astype(int)
    methods = {
        "margin_learned_J0": (pred_learn, np.abs(pe - TAU)),
        "margin_m8_point": (pred_m8, np.abs(p_m8 - TAU)),
        "robust_m8_current": (pred_m8, rad_rob),
        "box_m8": (pred_m8, rad_box),
    }
    sel = {}
    for name, (pred, conf) in methods.items():
        cov, risk, aurc = risk_coverage(ye, pred, conf)
        at = {}
        for c in (0.8, 0.9):
            kk = max(1, int(round(c * len(ye))))
            at[f"risk_at_{int(c * 100)}"] = float(risk[kk - 1])
        sel[name] = {"aurc": cluster_bootstrap(ge, lambda i, p=pred, q=conf: risk_coverage(ye[i], p[i], q[i])[2], n_boot), **at}
    # Ranking agreement between current robust BP and margin on the same posterior
    rr, rm = rankdata(rad_rob), rankdata(np.abs(np.arctanh(np.clip(2 * p_m8 - 1, -1 + 1e-12, 1 - 1e-12))))
    sel["spearman_robust_vs_margin_m8"] = float(np.corrcoef(rr, rm)[0, 1]) if len(rr) > 1 else float("nan")
    res["selective"] = sel

    # --- 6. Shift structure (H2 early read) -------------------------------------
    shift_res = {}
    theta_clean = 0.5 * np.log(np.clip(pe, 1e-6, 1 - 1e-6) / np.clip(1 - pe, 1e-6, 1))
    pooled_box, pooled_bud, pooled_n = [], [], []
    for s in SHIFTS[1:]:
        keep = [k for k, r in enumerate(evl) if s in r.evidence]
        if not keep:
            continue
        Xs = _feature_matrix([evl[k] for k in keep], s, fill)
        ps = predict_logistic(w, (Xs - mu) / sd)
        th_s = 0.5 * np.log(np.clip(ps, 1e-6, 1 - 1e-6) / np.clip(1 - ps, 1e-6, 1))
        delta = th_s - theta_clean[keep]
        gi = ge[keep]
        per_img_max, per_img_mean, per_img_n = [], [], []
        for img in np.unique(gi):
            d = np.abs(delta[gi == img])
            per_img_max.append(d.max())
            per_img_mean.append(d.mean())
            per_img_n.append(len(d))
        per_img_max, per_img_mean = np.array(per_img_max), np.array(per_img_mean)
        pooled_box += per_img_max.tolist()
        pooled_bud += per_img_mean.tolist()
        pooled_n += per_img_n
        shift_res[s] = {
            "abs_delta_q50": float(np.quantile(np.abs(delta), 0.5)),
            "abs_delta_q90": float(np.quantile(np.abs(delta), 0.9)),
            "budget_to_box_ratio_q90": _ratio_ci(per_img_mean, per_img_max, n_boot),
        }
    if pooled_box:
        pb, pm, pn = np.array(pooled_box), np.array(pooled_bud), np.array(pooled_n)
        ratio = _ratio_ci(pm, pb, n_boot)
        t_box, t_bud = float(np.quantile(pb, 0.9)), float(np.quantile(pm, 0.9))
        shift_res["pooled"] = {
            "budget_to_box_ratio_q90": ratio,
            "t_box_q90": t_box, "t_budget_per_claim_q90": t_bud,
            # Regime check for T3: calibrated per-claim radius eps = t_box vs total budget B = n * t_bud
            "share_images_B_gt_eps": float(np.mean(pn * t_bud > t_box)),
            "width_combined_vs_box": _width_ratio(evl, theta_clean, by_img, t_box, t_bud, ge, n_boot),
        }
    res["shifts"] = shift_res
    res["gate1"] = gate1(res)
    return res


def _width_ratio(evl, theta, by_img, t_box, t_bud, ge, n_boot) -> dict:
    """Mean interval width, calibrated combined set (eps = t_box, B = n * t_bud) / box set.

    Uses the learned unary fields on an M8-style chain with J = atanh(0.5).
    With J = 0 the ratio is exactly 1 whenever B >= eps (root dominance), so any
    gain here comes only through claim coupling.
    """
    if t_box <= 0:
        return {"ratio": [float("nan")] * 3, "note": "no measurable shift"}
    w_box, w_grid, w_cert = np.zeros(len(evl)), np.zeros(len(evl)), np.zeros(len(evl))
    for ks in by_img.values():
        th = theta[ks]
        model = _chain_model(th, t_box, M8_DEFAULT)
        budget = len(ks) * t_bud
        for node, k in enumerate(ks):
            lo, up, _, _ = solve_independent_box_bounds(model, node)
            w_box[k] = up - lo
            if budget > 0:
                b_eff = min(budget, len(ks) * t_box)
                k_grid = int(min(2000, max(200, math.ceil(200 * b_eff / t_box))))  # step <= eps/200 (capped)
                r = solve_robust_bp(model, node, b_eff, num_grid_steps=k_grid)
                w_grid[k] = r.upper_grid - r.lower_grid          # slightly optimistic
                w_cert[k] = r.upper_certified - r.lower_certified  # conservative
    ratio = lambda w: (lambda i: float(w[i].mean() / w_box[i].mean()) if w_box[i].mean() > 0 else float("nan"))
    return {"ratio": cluster_bootstrap(ge, ratio(w_grid), n_boot),
            "ratio_certified": cluster_bootstrap(ge, ratio(w_cert), n_boot),
            "mean_width_box": float(w_box.mean()), "mean_width_combined": float(w_grid.mean()),
            "mean_width_combined_certified": float(w_cert.mean())}


def _feat(r: ClaimRow, f: str) -> Optional[float]:
    if f == "tok_logprob_mean":
        return r.tok_logprob_mean
    return r.evidence.get("clean", {}).get(f)


def _ratio_ci(per_img_mean: np.ndarray, per_img_max: np.ndarray, n_boot: int) -> Tuple[float, float, float]:
    """q90(mean |delta|) / q90(max |delta|): <1 means a budget set is smaller than a box at equal coverage."""
    idx_groups = np.arange(len(per_img_max))

    def stat(i):
        b = np.quantile(per_img_max[i], 0.9)
        return float(np.quantile(per_img_mean[i], 0.9) / b) if b > 0 else float("nan")
    return cluster_bootstrap(idx_groups, stat, n_boot)


# ----------------------------------------------------------------------------
# Gate 1
# ----------------------------------------------------------------------------

def gate1(res: dict) -> List[dict]:
    d, dep, sel = res["descriptives"], res["dependence"], res["learned_unary"]
    corr = dep["within_image_corr"]
    checks = [
        {
            "check": "Graph exists",
            "rule": "mean claims per image >= 4",
            "value": f"{d['claims_per_image_mean']:.2f}",
            "pass": bool(d["claims_per_image_mean"] >= 4),
            "if_fail": "Change prompt/dataset; drop H4 if still < 3",
        },
        {
            "check": "Coupling is real",
            "rule": "within-image label correlation: 95% CI lower bound > 0",
            "value": f"{corr[0]:.3f} [{corr[1]:.3f}, {corr[2]:.3f}], perm p={dep['permutation_p']:.4f}",
            "pass": bool(not math.isnan(corr[1]) and corr[1] > 0),
            "if_fail": "Keep J = 0 as the main model; graph becomes an ablation",
        },
    ]
    pooled = res.get("shifts", {}).get("pooled")
    if pooled:
        r = pooled["budget_to_box_ratio_q90"]
        checks.append({
            "check": "Shifts not all simultaneously extreme",
            "rule": "q90 budget/box ratio: 95% CI upper bound < 0.9",
            "value": f"{r[0]:.3f} [{r[1]:.3f}, {r[2]:.3f}]",
            "pass": bool(not math.isnan(r[2]) and r[2] < 0.9),
            "if_fail": "H2 likely false; reposition around H1 + H3",
        })
    a = sel["auroc"]
    checks.append({
        "check": "Signal exists",
        "rule": "learned unary AUROC on pilot_eval: 95% CI lower bound > 0.6",
        "value": f"{a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]",
        "pass": bool(not math.isnan(a[1]) and a[1] > 0.6),
        "if_fail": "Fix evidence features before anything else",
    })
    return checks


def _ci(t) -> str:
    if isinstance(t, (list, tuple)) and len(t) == 3:
        return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    return f"{t:.3f}" if isinstance(t, float) else str(t)


def write_report(res: dict, path: Path, meta: dict) -> None:
    d, dep = res["descriptives"], res["dependence"]
    L = [
        "# Phase 1 — Gate 1 Report (falsification pilot)",
        "",
        f"Generated from result files. Images: pilot_cal {meta.get('n_pilot_cal')} / pilot_eval {meta.get('n_pilot_eval')} "
        f"(train split only; test split untouched). Bootstrap: {res['n_boot']} image-level resamples, 95% CI.",
        "",
        "## Gate 1 checks",
        "",
        "| Check | Rule | Value | Result | If it fails |",
        "|---|---|---|---|---|",
    ]
    for c in res["gate1"]:
        L.append(f"| {c['check']} | {c['rule']} | {c['value']} | {'PASS' if c['pass'] else 'FAIL'} | {c['if_fail']} |")
    L += [
        "",
        "## Descriptives",
        "",
        f"- Captions: {res['chair']['captions']}; CHAIR_i = {res['chair']['chair_i']:.3f}; CHAIR_s = {res['chair']['chair_s']:.3f}",
        f"- Claims: {d['claims']} on {d['images_with_claims']} images; per image mean {d['claims_per_image_mean']:.2f}, "
        f"median {d['claims_per_image_median']:.1f}; images with >= 3 claims {d['share_images_with_ge3_claims']:.2f}",
        f"- Claim hallucination rate: {_ci(d['hallucination_rate'])}",
        f"- Claims without a matched token span: {d['claims_missing_token_span']}",
        "",
        "## Within-image dependence",
        "",
        f"- Intra-image label correlation: {_ci(dep['within_image_corr'])}; permutation p = {dep['permutation_p']:.4f}",
        "",
        "## Evidence signal (AUROC for predicting hallucination; higher support = lower risk)",
        "",
        "| Feature | AUROC [95% CI] |",
        "|---|---|",
    ]
    for f, v in res["signal"].items():
        L.append(f"| {f} | {_ci(v['auroc'])} |")
    L += [
        "",
        f"- Raw OWLv2 score used as P(supported) (current M8 θ mapping): ECE = {res['signal']['owl_score']['ece_as_probability']:.3f}",
        "",
        "## Learned unary model (fit on pilot_cal, evaluated on pilot_eval)",
        "",
        f"- AUROC {_ci(res['learned_unary']['auroc'])}; ECE {res['learned_unary']['ece']:.3f}; "
        f"Brier {res['learned_unary']['brier']:.3f} (n_cal={res['learned_unary']['n_cal']}, n_eval={res['learned_unary']['n_eval']})",
        "",
        "## Selective prediction on pilot_eval (lower AURC is better)",
        "",
        "| Method | AURC [95% CI] | Risk @ 80% coverage | Risk @ 90% coverage |",
        "|---|---|---|---|",
    ]
    for name, v in res["selective"].items():
        if isinstance(v, dict):
            L.append(f"| {name} | {_ci(v['aurc'])} | {v['risk_at_80']:.3f} | {v['risk_at_90']:.3f} |")
    L += [
        "",
        f"- Spearman rank agreement, current robust BP vs margin on the same posterior: "
        f"{res['selective']['spearman_robust_vs_margin_m8']:.4f} (root dominance predicts 1.0).",
        "",
        "## Evidence shifts (H2 early read; δ in θ of the learned unary model)",
        "",
        "| Shift | median abs(δ) | q90 abs(δ) | q90 budget/box ratio [95% CI] |",
        "|---|---|---|---|",
    ]
    for s, v in res["shifts"].items():
        if s == "pooled":
            continue
        L.append(f"| {s} | {v['abs_delta_q50']:.3f} | {v['abs_delta_q90']:.3f} | {_ci(v['budget_to_box_ratio_q90'])} |")
    if "pooled" in res["shifts"]:
        p = res["shifts"]["pooled"]
        L += ["", f"- Pooled ratio {_ci(p['budget_to_box_ratio_q90'])}; share of images where calibrated B > ε "
                  f"(graph can matter, T3): {p['share_images_B_gt_eps']:.2f}"]
        wr = p.get("width_combined_vs_box", {})
        if "mean_width_box" in wr:
            L += [f"- **Direct H2 read:** mean interval width, calibrated combined set / box set "
                  f"(learned θ, chain J = 0.549): grid {_ci(wr['ratio'])}, certified (conservative) {_ci(wr['ratio_certified'])} "
                  f"(box {wr['mean_width_box']:.3f} vs combined {wr['mean_width_combined']:.3f}). "
                  "With J = 0 this ratio is exactly 1 (root dominance), so any gain comes from claim coupling."]
    L += ["", "Interpretation and the Gate 1 decision are recorded by the authors in docs/DECISIONS.md."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def rows_to_jsonl(rows: Sequence[ClaimRow], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")
