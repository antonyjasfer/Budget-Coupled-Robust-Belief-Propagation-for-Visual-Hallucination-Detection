"""Human label audit (Q1 plan, protocol E.3).

Sample, with a fixed seed and before anyone sees labels, 200 claims per generator:
100 CHAIR-hallucinated claims above the calibrated threshold, 50 CHAIR-hallucinated
below it and 50 CHAIR-supported. Two annotators label every claim blind to the model,
the score and the CHAIR label: present / absent / ambiguous / synonym_mismatch.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

LABELS = ("present", "absent", "ambiguous", "synonym_mismatch")
STRATA = (("hall_above", 100), ("hall_below", 50), ("supported", 50))


def sample(claims: Sequence[dict], tau: Dict[str, float], per_stratum=STRATA, seed: int = 2026) -> List[dict]:
    """claims: dicts with generator, image_id, category, hall (bool), score, caption.
    Returns the audit rows in a seeded random order with an opaque id; labels, scores and
    generator names stay in the key file, not in the annotator file."""
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for g in sorted({c["generator"] for c in claims}):
        cs = [c for c in claims if c["generator"] == g]
        pools = {"hall_above": [c for c in cs if c["hall"] and c["score"] > tau[g]],
                 "hall_below": [c for c in cs if c["hall"] and c["score"] <= tau[g]],
                 "supported": [c for c in cs if not c["hall"]]}
        for stratum, n in per_stratum:
            pool = pools[stratum]
            if len(pool) < n:
                raise ValueError(f"{g}/{stratum}: need {n} claims, have {len(pool)}")
            for i in rng.choice(len(pool), n, replace=False):
                rows.append({**pool[int(i)], "stratum": stratum})
    order = rng.permutation(len(rows))
    out = []
    for k, i in enumerate(order):
        out.append({**rows[int(i)], "audit_id": f"A{k:04d}"})
    return out


def write_files(rows: Sequence[dict], out_dir: Path, image_url) -> None:
    """annotator_tasks.csv (blind) and audit_key.csv (labels, scores, generator)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "annotator_tasks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "image_url", "object", "caption", "label (present/absent/ambiguous/synonym_mismatch)"])
        for r in rows:
            w.writerow([r["audit_id"], image_url(r["image_id"]), r["category"], r["caption"], ""])
    with open(out_dir / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "generator", "image_id", "category", "chair_hallucinated", "score", "stratum"])
        for r in rows:
            w.writerow([r["audit_id"], r["generator"], r["image_id"], r["category"], int(r["hall"]), r["score"], r["stratum"]])


def cohen_kappa(a: Sequence[str], b: Sequence[str], labels: Iterable[str] = LABELS) -> float:
    labels = list(labels)
    if len(a) != len(b) or not a:
        raise ValueError("need two equal, non-empty label lists")
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / n ** 2
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def noise_rates(key: Sequence[dict], final: Dict[str, str]) -> Dict[str, dict]:
    """Per stratum: share of claims whose adjudicated human label disagrees with CHAIR
    (ambiguous and synonym_mismatch counted separately, not as errors)."""
    out: Dict[str, dict] = {}
    for s, _ in STRATA:
        ks = [k for k in key if k["stratum"] == s and k["audit_id"] in final]
        if not ks:
            continue
        lab = [final[k["audit_id"]] for k in ks]
        chair_absent = s != "supported"
        wrong = sum(1 for l in lab if l == ("present" if chair_absent else "absent"))
        out[s] = {"n": len(ks), "chair_wrong": wrong / len(ks),
                  "ambiguous": lab.count("ambiguous") / len(ks), "synonym_mismatch": lab.count("synonym_mismatch") / len(ks)}
    return out
