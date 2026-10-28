"""Phase 4 confirmatory analysis (PREREG v2, Sections 2, 5, 6, 7).

Filters are calibrated on clean COCO Cal-A + Cal-B (LLaVA claims, OWLv2) with
`n_resamples` random 80% subsamples. Each test condition is evaluated once on
the locked test split (or the 804 AMBER images). Point estimates average the
resamples; 95% CIs come from an image-level cluster bootstrap over test images.

Decision rule (PREREG v2 Section 5): violated if CI lower > target; preserved if
CI upper <= target + MARGIN; otherwise inconclusive. Bootstrap one-sided
p-values feed a Holm correction across H1, H3, H4, H5, H6 within each filter.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.phase3.shift_validity import Claim, by_image, conformal_upper, logit
from src.phase4.shifts import S1_CONDITIONS

TARGETS = (0.10, 0.05, 0.20)   # first is primary
MARGIN = 0.03
SWAPS = ("owlvit_v1", "grounding_dino")


# ----------------------------------------------------------------------------- calibration

def tau_claim(cal: Sequence[Claim], level: float, key: str = "clean") -> float:
    return conformal_upper([c.score[key] for c in cal if c.hallucinated and key in c.score], 1.0 - level)


def tau_image(cal_imgs: Dict[int, List[Claim]], level: float, key: str = "clean") -> float:
    r = []
    for cs in cal_imgs.values():
        h = [c.score[key] for c in cs if c.hallucinated and key in c.score]
        r.append(max(h) if h else -math.inf)
    return conformal_upper(r, 1.0 - level)


def resample_taus(cal: Sequence[Claim], level: float, n_resamples: int, frac: float = 0.8,
                  key: str = "clean", seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    imgs = by_image([c for c in cal if key in c.score])
    ids = np.array(sorted(imgs))
    rng = np.random.default_rng(seed)
    m = int(round(frac * len(ids)))
    tc, ti = [], []
    for _ in range(n_resamples):
        sub = rng.choice(ids, m, replace=False)
        ci = {int(i): imgs[int(i)] for i in sub}
        tc.append(tau_claim([c for cs in ci.values() for c in cs], level, key))
        ti.append(tau_image(ci, level, key))
    return np.array(tc), np.array(ti)


# ----------------------------------------------------------------------------- evaluation

def per_image_stats(test: Sequence[Claim], key: str, tc: np.ndarray, ti: np.ndarray) -> dict:
    """Per test image, averaged over calibration resamples: kept hallucinated
    claims, kept supported claims, counts, image-level failure and abstention."""
    imgs = by_image([c for c in test if key in c.score])
    ids = sorted(imgs)
    kh, ks, nh, ns, bad, abst = [], [], [], [], [], []
    for i in ids:
        cs = imgs[i]
        s = np.array([c.score[key] for c in cs])
        h = np.array([c.hallucinated for c in cs], dtype=bool)
        keep_c = s[None, :] > tc[:, None]            # (R, claims) claim-level filter
        keep_i = s[None, :] > ti[:, None]            # image-level filter
        kh.append(keep_c[:, h].sum(1).mean())
        ks.append(keep_c[:, ~h].sum(1).mean())
        nh.append(h.sum())
        ns.append((~h).sum())
        bad.append((keep_i[:, h].any(1)).mean() if h.any() else 0.0)
        abst.append((~keep_i.any(1)).mean())
    return {k: np.asarray(v, dtype=float) for k, v in
            dict(kh=kh, ks=ks, nh=nh, ns=ns, bad=bad, abst=abst).items()} | {"n_images": len(ids)}


def _metrics(st: dict, w: np.ndarray) -> dict:
    """Metrics for image weights w (shape (B, N) or (N,))."""
    def ratio(a, b):
        num, den = (w * st[a]).sum(-1), (w * st[b]).sum(-1)
        return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    kept = (w * (st["kh"] + st["ks"])).sum(-1)
    return {
        "claim_risk": ratio("kh", "nh"),
        "image_risk": (w * st["bad"]).sum(-1) / w.sum(-1),
        "supported_kept": ratio("ks", "ns"),
        "abstention": (w * st["abst"]).sum(-1) / w.sum(-1),
        "kept_hallucinated_share": np.where(kept > 0, (w * st["kh"]).sum(-1) / np.maximum(kept, 1e-12), np.nan),
    }


def summarise(st: dict, n_boot: int, seed: int = 1) -> dict:
    n = st["n_images"]
    if n == 0:
        return {}
    point = _metrics(st, np.ones(n))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)
    boot = _metrics(st, W)
    out = {"n_images": n, "n_hallucinated": int(st["nh"].sum()), "n_supported": int(st["ns"].sum())}
    for k, v in point.items():
        b = boot[k][np.isfinite(boot[k])]
        out[k] = {"point": float(v), "lo": float(np.quantile(b, 0.025)) if b.size else float("nan"),
                  "hi": float(np.quantile(b, 0.975)) if b.size else float("nan"), "boot": b}
    return out


def decide(m: dict, target: float) -> dict:
    b = m["boot"]
    return {"target": target, "point": m["point"], "lo": m["lo"], "hi": m["hi"],
            "decision": "violated" if m["lo"] > target else ("preserved" if m["hi"] <= target + MARGIN else "inconclusive"),
            "p_preserved": float(np.mean(b > target + MARGIN)) if b.size else float("nan"),
            "p_violated": float(np.mean(b <= target)) if b.size else float("nan")}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted((p, k) for k, p in pvals.items() if np.isfinite(p))
    m, adj, run = len(items), {}, 0.0
    for r, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - r) * p))
        adj[k] = run
    return adj


# ----------------------------------------------------------------------------- H2 (dominance)

def d_plus(clean: np.ndarray, shifted: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """sup_t [P(shifted > t) - P(clean > t)] with optional claim weights."""
    w = np.ones(len(clean)) if w is None else w
    t = np.unique(np.concatenate([clean, shifted]))
    tw = w.sum()
    ps = ((shifted[None, :] > t[:, None]) * w).sum(1) / tw
    pc = ((clean[None, :] > t[:, None]) * w).sum(1) / tw
    return float(max(0.0, (ps - pc).max()))


def dominance_test(test: Sequence[Claim], cond: str, tau: float, n_boot: int, seed: int = 2) -> dict:
    hs = [c for c in test if c.hallucinated and cond in c.score]
    if not hs:
        return {}
    img = np.array([c.image_id for c in hs])
    a = np.array([c.score["clean"] for c in hs])
    b = np.array([c.score[cond] for c in hs])
    ids, inv = np.unique(img, return_inverse=True)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        cnt = rng.multinomial(len(ids), np.full(len(ids), 1.0 / len(ids)))
        w = cnt[inv].astype(float)
        if w.sum() > 0:
            boots.append(d_plus(a, b, w))
    boots = np.array(boots)
    return {"d_plus": d_plus(a, b), "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "supported": bool(np.quantile(boots, 0.975) <= MARGIN),
            "diff_at_tau": float(np.mean(b > tau) - np.mean(a > tau)),
            "share_up": float(np.mean(b > a)), "n_hallucinated": len(hs)}


# ----------------------------------------------------------------------------- loading

def load(out: Path) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import phase4 as P
    from src.phase3 import amber as am
    vocab = P._vocab(out)
    prep = P._prep(out)
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    owl: Dict[Tuple[str, int, str], dict] = {}
    for r in P.read_all(out, "owl"):
        owl[(r["set"], int(r["image_id"]), r["cond"])] = r["scores"]
    swap: Dict[Tuple[int, str], dict] = {}
    for r in P.read_all(out, "swap"):
        for c in r["claims"]:
            swap[(int(r["image_id"]), c["category"])] = c

    def coco_claims(stem: str, set_name: str, ids: set, with_swap: bool) -> List[Claim]:
        cl = []
        for r in P.read_all(out, stem):
            iid = int(r["image_id"])
            if iid not in ids:
                continue
            for c in vocab.extract_claims(r["caption"]).claims:
                sc = {}
                for cond in ["clean"] + S1_CONDITIONS:
                    s = owl.get((set_name, iid, cond), {}).get(c.node_word)
                    if s is not None:
                        sc[cond] = logit(s)
                if "clean" not in sc:
                    continue
                if with_swap:
                    sw = swap.get((iid, c.node_word))
                    if sw:
                        for d in SWAPS:
                            v = sw.get(d)
                            if v is not None and np.isfinite(v):
                                sc[d] = logit(v)
                cl.append(Claim(iid, int(c.node_word not in gt.get(iid, set())), sc))
        return cl

    cal_ids = set(prep["cal_a"]) | set(prep["cal_b"])
    test_ids = set(prep["test"])
    data = {
        "cal": coco_claims("captions_llava_coco", "coco_llava", cal_ids, True),
        "cal_a_ids": set(prep["cal_a"]),
        "test": coco_claims("captions_llava_coco", "coco_llava", test_ids, True),
        "qwen": coco_claims("captions_qwen", "coco_qwen", test_ids, False),
        "internvl": coco_claims("captions_internvl", "coco_internvl", test_ids, False),
    }
    # S5: the removed object's claim, hallucinated by construction.
    s5 = []
    for (sname, iid, cond), scores in owl.items():
        if sname == "coco_llava" and cond == "s5":
            for cat, s in scores.items():
                clean = owl.get(("coco_llava", iid, "clean"), {}).get(cat)
                if clean is not None:
                    s5.append(Claim(iid, 1, {"clean": logit(clean), "s5": logit(s)}))
    data["s5"] = s5
    # AMBER
    entries = am.generative_entries(json.loads((out / "amber" / "annotations.json").read_text()))
    amb, counts = [], {"claims": 0, "unknown": 0}
    keep = set(prep["amber"])
    for r in P.read_all(out, "captions_llava_amber"):
        iid = int(r["image_id"])
        if iid not in keep:
            continue
        truth, hallu = am.node_sets(vocab, entries[iid])
        for c in vocab.extract_claims(r["caption"]).claims:
            counts["claims"] += 1
            lab = am.label_claim(c.node_word, truth, hallu)
            s = owl.get(("amber_llava", iid, "clean"), {}).get(c.node_word)
            if lab is None:
                counts["unknown"] += 1
            elif s is not None:
                amb.append(Claim(iid, lab, {"clean": logit(s)}))
    data["amber"], data["amber_counts"] = amb, counts
    data["chair"] = {name: _chair(vocab, P.read_all(out, stem), gt, test_ids)
                     for name, stem in (("llava", "captions_llava_coco"), ("qwen", "captions_qwen"),
                                        ("internvl", "captions_internvl"))}
    data["truncated"] = {name: int(sum(bool(r.get("hit_max_tokens")) for r in P.read_all(out, stem)
                                       if int(r["image_id"]) in test_ids))
                         for name, stem in (("llava", "captions_llava_coco"), ("qwen", "captions_qwen"),
                                            ("internvl", "captions_internvl"))}
    return data


def _chair(vocab, caps, gt, ids) -> dict:
    men = hm = capsh = n = 0
    for r in caps:
        iid = int(r["image_id"])
        if iid not in ids:
            continue
        cc = vocab.extract_claims(r["caption"], gt_objects=gt.get(iid, set()))
        n += 1
        men += cc.mention_count
        hm += cc.hallucinated_mentions
        capsh += int(cc.hallucinated_mentions > 0)
    return {"captions": n, "chair_i": hm / men if men else float("nan"), "chair_s": capsh / n if n else float("nan")}


# ----------------------------------------------------------------------------- descriptive PGM

def _auroc(y, p) -> float:
    from src.phase3.transfer import auroc
    # transfer.auroc treats higher score as label 0 (supported); p is P(hallucinated).
    return auroc(np.asarray(y), -np.asarray(p, dtype=float))


def class_metrics(y: Sequence[int], p: Sequence[float], thr: float = 0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)
    tp, fp = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 1) & (y == 0)).sum())
    tn, fn = int(((yhat == 0) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"accuracy": (tp + tn) / len(y) if len(y) else float("nan"), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec and np.isfinite(prec + rec) else float("nan"),
            "roc_auc": _auroc(y, p), "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}}


def _chain_loglik(theta: np.ndarray, h: np.ndarray, J: float) -> float:
    """Exact log P(h) for a chain Ising model with fields theta and coupling J."""
    n = len(theta)
    ll = float((theta * h).sum() + J * (h[:-1] * h[1:]).sum())
    s = np.array([-1.0, 1.0])
    alpha = np.exp(theta[0] * s)
    logz = 0.0
    for i in range(1, n):
        alpha = (alpha[:, None] * np.exp(J * np.outer(s, s))).sum(0) * np.exp(theta[i] * s)
        c = alpha.sum()
        logz += math.log(c)
        alpha = alpha / c
    logz += math.log(alpha.sum()) if n == 1 else 0.0
    return ll - logz


def pgm_descriptive(data: dict) -> dict:
    from src.phase1.analysis import fit_logistic, predict_logistic, within_image_correlation
    from src.pgm.standard_bp import compute_rooted_total_field
    from src.pgm.tree_model import TreeModel

    cal_a = [c for c in data["cal"] if c.image_id in data["cal_a_ids"]]
    test = data["test"]
    X = lambda cs: np.array([[c.score["clean"]] for c in cs])
    y = lambda cs: np.array([c.hallucinated for c in cs])
    w = fit_logistic(X(cal_a), y(cal_a))
    p_unary = lambda cs: predict_logistic(w, X(cs))

    def fields(cs):
        p = np.clip(p_unary(cs), 1e-6, 1 - 1e-6)
        return 0.5 * np.log(p / (1 - p))  # P(h = +1 = hallucinated) = sigmoid(2 theta)

    def groups(cs):
        g = {}
        for k, c in enumerate(cs):
            g.setdefault(c.image_id, []).append(k)
        return g

    th_a, ga = fields(cal_a), groups(cal_a)
    ha = 2 * y(cal_a) - 1
    grid = np.round(np.arange(0.0, 2.0001, 0.01), 2)
    ll = [sum(_chain_loglik(th_a[ks], ha[ks], J) for ks in ga.values()) for J in grid]
    J = float(grid[int(np.argmax(ll))])

    def posteriors(cs, Jv):
        th, out = fields(cs), np.zeros(len(cs))
        for ks in groups(cs).values():
            n = len(ks)
            edges = [(i, i + 1) for i in range(n - 1)]
            m = TreeModel(num_nodes=n, theta=th[ks], edges=edges, coupling={e: Jv for e in edges})
            for node, k in enumerate(ks):
                out[k] = compute_rooted_total_field(m, node)[1]
        return out

    res = {"J_learned": J, "loglik_gain_vs_J0": float(max(ll) - ll[0]),
           "within_image_corr": {"cal": within_image_correlation(y(data["cal"]), np.array([c.image_id for c in data["cal"]])),
                                 "test": within_image_correlation(y(test), np.array([c.image_id for c in test]))}}
    res["robust_intervals"] = robust_widths(cal_a, test, w, J, groups, fields)
    for name, cs in (("train_cal_a", cal_a), ("test", test)):
        p0, pJ = posteriors(cs, 0.0), posteriors(cs, J)
        res[name] = {"unary_J0": class_metrics(y(cs), p0), "ising_tree_J": class_metrics(y(cs), pJ),
                     "max_abs_posterior_change": float(np.abs(pJ - p0).max()) if len(cs) else float("nan")}
    return res


def robust_widths(cal_a, test, w, J, groups, fields, max_images: int = 150) -> dict:
    """(iii) Robust BP intervals with (eps, B) calibrated on Cal-A S1 shifts.

    theta = 0.5 * (w0 * s + b), so a score change ds moves theta by 0.5 * w0 * ds.
    eps = 0.9 quantile of |d theta| over Cal-A claims and S1 conditions; B = 0.9
    quantile of the per-(image, condition) sum of |d theta|. Widths are computed on
    the first `max_images` test images (sorted ids) for the learned J and for J = 0.
    """
    from src.pgm.tree_model import TreeModel
    from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds
    k = 0.5 * abs(float(w[0]))
    per_claim, per_img = [], []
    for ks in groups(cal_a).values():
        for cond in S1_CONDITIONS:
            d = [k * abs(cal_a[i].score[cond] - cal_a[i].score["clean"]) for i in ks if cond in cal_a[i].score]
            if d:
                per_claim += d
                per_img.append(sum(d))
    if not per_claim:
        return {"note": "no Cal-A shift scores"}
    eps, B = float(np.quantile(per_claim, 0.9)), float(np.quantile(per_img, 0.9))
    g = groups(test)
    sel = sorted(g)[:max_images]
    th = fields(test)
    out = {"eps": eps, "B": B, "images": len(sel), "share_B_gt_eps": float(B > eps)}
    for name, Jv in (("J_learned", J), ("J0", 0.0)):
        wr, wb = [], []
        for iid in sel:
            ks = g[iid]
            n = len(ks)
            edges = [(i, i + 1) for i in range(n - 1)]
            m = TreeModel(num_nodes=n, theta=th[ks], edges=edges, coupling={e: Jv for e in edges},
                          epsilon=np.full(n, eps))
            for node in range(n):
                lo, up, _, _ = solve_independent_box_bounds(m, node)
                wb.append(up - lo)
                r = solve_robust_bp(m, node, min(B, n * eps), num_grid_steps=200)
                wr.append(r.upper_certified - r.lower_certified)
        out[name] = {"mean_width_robust": float(np.mean(wr)), "mean_width_box": float(np.mean(wb))}
    return out


# ----------------------------------------------------------------------------- main

def run_all(out: Path, n_resamples: int = 200, n_boot: int = 2000) -> dict:
    data = load(out)
    cal, test = data["cal"], data["test"]
    res = {"counts": {k: len(data[k]) for k in ("cal", "test", "qwen", "internvl", "amber", "s5")},
           "amber_counts": data["amber_counts"], "chair": data["chair"], "truncated_captions": data["truncated"],
           "levels": {}}
    for level in TARGETS:
        tc, ti = resample_taus(cal, level, n_resamples)
        conds = {"clean": (test, "clean", tc, ti)}
        for s in S1_CONDITIONS:
            conds[s] = (test, s, tc, ti)
        for d in SWAPS:
            conds[f"swap_{d}"] = (test, d, tc, ti)
            rc, ri = resample_taus(cal, level, n_resamples, key=d)
            conds[f"swap_{d}+recalibrated"] = (test, d, rc, ri)
        conds["vlm_qwen"] = (data["qwen"], "clean", tc, ti)
        conds["vlm_internvl"] = (data["internvl"], "clean", tc, ti)
        conds["domain_amber"] = (data["amber"], "clean", tc, ti)
        conds["s5_object_removed"] = (data["s5"], "s5", tc, ti)
        lv = {}
        for name, (cs, key, a, b) in conds.items():
            sm = summarise(per_image_stats(cs, key, a, b), n_boot)
            if not sm:
                continue
            lv[name] = {"n_images": sm["n_images"], "n_hallucinated": sm["n_hallucinated"], "n_supported": sm["n_supported"]}
            for f in ("claim_risk", "image_risk"):
                lv[name][f] = decide(sm[f], level)
            for f in ("supported_kept", "abstention", "kept_hallucinated_share"):
                lv[name][f] = {k: sm[f][k] for k in ("point", "lo", "hi")}
        res["levels"][str(level)] = lv

    lv = res["levels"][str(TARGETS[0])]
    hyp = {}
    for f in ("claim_risk", "image_risk"):
        s1 = [lv[s][f] for s in S1_CONDITIONS if s in lv]
        h = {
            "H1": {"supported": all(x["decision"] == "preserved" for x in s1),
                   "p": max(x["p_preserved"] for x in s1) if s1 else float("nan")},
            "H3": {"supported": any(lv.get(f"swap_{d}", {}).get(f, {}).get("decision") == "violated" for d in SWAPS),
                   "p": min(lv[f"swap_{d}"][f]["p_violated"] for d in SWAPS if f"swap_{d}" in lv)},
            "H4": {"supported": all(lv.get(f"swap_{d}+recalibrated", {}).get(f, {}).get("decision") == "preserved" for d in SWAPS),
                   "p": max(lv[f"swap_{d}+recalibrated"][f]["p_preserved"] for d in SWAPS if f"swap_{d}+recalibrated" in lv)},
            "H5": {"supported": all(lv.get(v, {}).get(f, {}).get("decision") == "preserved" for v in ("vlm_qwen", "vlm_internvl")),
                   "p": max(lv[v][f]["p_preserved"] for v in ("vlm_qwen", "vlm_internvl") if v in lv)},
            "H6": {"supported": lv.get("domain_amber", {}).get(f, {}).get("decision") == "preserved",
                   "p": lv["domain_amber"][f]["p_preserved"] if "domain_amber" in lv else float("nan")},
        }
        adj = holm({k: v["p"] for k, v in h.items()})
        for k in h:
            h[k]["p_holm"] = adj.get(k, float("nan"))
        hyp[f] = h
    tau_full = tau_claim(cal, TARGETS[0])
    h2 = {s: dominance_test(test, s, tau_full, n_boot) for s in S1_CONDITIONS}
    hyp["H2"] = {"supported": all(v.get("supported", False) for v in h2.values()), "per_condition": h2}
    res["hypotheses"] = hyp
    res["pgm"] = pgm_descriptive(data)
    return res


# ----------------------------------------------------------------------------- report

def write_report(res: dict, path: Path) -> None:
    f3 = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and np.isfinite(x) else "–"
    L = ["# Phase 4 — When do conformal hallucination guarantees survive distribution shift?\n",
         "Confirmatory analysis under PREREG v2. Calibration: clean COCO Cal-A + Cal-B (LLaVA-1.5-7B, OWLv2), "
         "200 resamples of 80%. Test: locked PREREG test split (500 images) and 804 AMBER images. "
         f"95% CIs: image-level cluster bootstrap. Decision: violated if CI lower > target; preserved if CI upper ≤ target + {MARGIN}.\n",
         "## Hypotheses (target 0.10)\n", "| Hypothesis | Claim-level | Image-level |", "|---|---|---|"]
    for h in ("H1", "H3", "H4", "H5", "H6"):
        c, i = res["hypotheses"]["claim_risk"][h], res["hypotheses"]["image_risk"][h]
        L.append(f"| {h} | {'supported' if c['supported'] else 'not supported'} (Holm p {f3(c['p_holm'])}) | "
                 f"{'supported' if i['supported'] else 'not supported'} (Holm p {f3(i['p_holm'])}) |")
    L.append(f"| H2 (dominance, all S1 conditions) | {'supported' if res['hypotheses']['H2']['supported'] else 'not supported'} | |")
    L += ["\n## Results by condition (target 0.10)\n",
          "| Condition | Claim risk [95% CI] | Decision | Image risk [95% CI] | Decision | Supported kept | Abstention | Hallucinated share of kept |",
          "|---|---|---|---|---|---|---|---|"]
    for name, d in res["levels"]["0.1"].items():
        c, i = d["claim_risk"], d["image_risk"]
        L.append(f"| {name} | {f3(c['point'])} [{f3(c['lo'])}, {f3(c['hi'])}] | {c['decision']} | "
                 f"{f3(i['point'])} [{f3(i['lo'])}, {f3(i['hi'])}] | {i['decision']} | {f3(d['supported_kept']['point'])} | "
                 f"{f3(d['abstention']['point'])} | {f3(d['kept_hallucinated_share']['point'])} |")
    L += ["\n## H2: stochastic dominance (P1's condition) per S1 condition\n",
          "| Condition | D⁺ [95% CI] | Supported | Δ at τ | Share of hallucinated scores that went up |", "|---|---|---|---|---|"]
    for s, v in res["hypotheses"]["H2"]["per_condition"].items():
        if v:
            L.append(f"| {s} | {f3(v['d_plus'])} [{f3(v['lo'])}, {f3(v['hi'])}] | {v['supported']} | {f3(v['diff_at_tau'])} | {f3(v['share_up'])} |")
    L += ["\n## Secondary targets (claim-level risk, point estimate)\n", "| Condition | δ = 0.05 | δ = 0.20 |", "|---|---|---|"]
    for name in res["levels"]["0.1"]:
        a = res["levels"]["0.05"].get(name, {}).get("claim_risk", {}).get("point", float("nan"))
        b = res["levels"]["0.2"].get(name, {}).get("claim_risk", {}).get("point", float("nan"))
        L.append(f"| {name} | {f3(a)} | {f3(b)} |")
    p = res["pgm"]
    L += ["\n## Descriptive: graphical model (Proposition P2) and classification metrics\n",
          f"- Learned chain coupling J (Cal-A, exact likelihood): {p['J_learned']:.2f}; log-likelihood gain over J = 0: {p['loglik_gain_vs_J0']:.2f}",
          f"- Within-image label correlation: cal {f3(p['within_image_corr']['cal'])}, test {f3(p['within_image_corr']['test'])}",
          "", "| Data | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP | FP | TN | FN |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for dname in ("train_cal_a", "test"):
        for mname in ("unary_J0", "ising_tree_J"):
            m = p[dname][mname]
            cm = m["confusion"]
            L.append(f"| {dname} | {mname} | {f3(m['accuracy'])} | {f3(m['precision'])} | {f3(m['recall'])} | {f3(m['f1'])} | "
                     f"{f3(m['roc_auc'])} | {cm['tp']} | {cm['fp']} | {cm['tn']} | {cm['fn']} |")
        L.append(f"| {dname} | max \\|posterior change\\| | {f3(p[dname]['max_abs_posterior_change'])} | | | | | | | | |")
    ri = p.get("robust_intervals", {})
    if "J0" in ri:
        L.append(f"\n- Robust BP intervals (ε = {ri['eps']:.3f}, B = {ri['B']:.3f} from Cal-A S1 shifts; {ri['images']} test images): "
                 f"mean certified width with learned J {ri['J_learned']['mean_width_robust']:.4f} vs J = 0 {ri['J0']['mean_width_robust']:.4f}; "
                 f"independent box: {ri['J_learned']['mean_width_box']:.4f} vs {ri['J0']['mean_width_box']:.4f}")
    L += ["\n## Data\n", f"```\n{json.dumps({k: res[k] for k in ('counts', 'amber_counts', 'chair', 'truncated_captions')}, indent=1)}\n```"]
    path.write_text("\n".join(L) + "\n")
