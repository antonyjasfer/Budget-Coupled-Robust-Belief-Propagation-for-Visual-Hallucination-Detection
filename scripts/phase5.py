"""Phase 5 (Q1 extension, to be pre-registered as prereg-v3 before `analyze` is run on
the test split): generator x generator transfer matrix for conformal hallucination
filters, with prevalence decomposition, local gap, recalibration curve, grouped CRC,
BH-controlled decisions and the human label audit sample.

Stages:
    setup     COCO annotations + CHAIR synonyms; NEW seeded COCO val2014 split
              (1,000 calibration + 3,000 test images, disjoint from every Phase 1-4
              image), ground truth; AMBER annotations and its 804 confirmatory ids  [CPU]
    images    download the Phase 5 COCO images (AMBER images as in Phase 4)        [CPU]
    import    copy *.jsonl from a previous Kaggle session                         [CPU]
    captions  captions for --gens on COCO cal + test and AMBER (one shard)       [GPU]
    detect    OWLv2, OWL-ViT v1 and Grounding DINO scores for every generator's
              claims (one shard)                                                   [GPU]
    pilot     CHAIR and truncation per generator (week-1 Gate 1)                  [CPU]
    analyze   transfer matrices, prevalence, recalibration curves, CRC, BH        [CPU]
    audit     the 600-claim blind audit sample and key                            [CPU]

    python scripts/phase5.py setup  --out OUT
    python scripts/phase5.py images --out OUT
    CUDA_VISIBLE_DEVICES=0 python scripts/phase5.py captions --out OUT --gens llava,qwen25 --shard 0 --nshards 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/phase5.py captions --out OUT --gens llava,qwen25 --shard 1 --nshards 2
    (same for detect)
    python scripts/phase5.py analyze --out OUT
    python scripts/phase5.py audit   --out OUT --gens llava,qwen25,internvl3

Week-1 pilot: `captions --limit 100 --out PILOT` with every generator, then `pilot`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import phase4 as P4  # noqa: E402  (read_all, log, coco/amber paths, import, claims_by_image)

N_CAL, N_TEST, SEED = 1000, 3000, 2027
GENS_DEFAULT = "llava,llava_t1,qwen25,qwen3_4b,internvl3,instructblip"
DETECTORS = ("owlv2", "owlvit_v1", "grounding_dino")
log = P4.log


def _prep(out: Path) -> dict:
    return json.loads((out / "prep5.json").read_text())


def cap_file(gen: str, dataset: str) -> str:
    return f"captions5_{gen}_{dataset}"


# ----------------------------------------------------------------------------- CPU stages

def _used_ids() -> set:
    m = json.loads((ROOT / "reports" / "phase1" / "manifest.json").read_text())
    ids = set(m["pilot_cal"]) | set(m["pilot_eval"])
    for v in m["splits"].values():
        ids |= set(v)
    return {int(i) for i in ids}


def stage_setup(out: Path) -> None:
    import numpy as np
    from src.phase1.chair import ground_truth_objects
    P4.stage_setup(out)                       # annotations, synonyms, AMBER ids (Phase 4 prep.json)
    old = _used_ids()
    ann = out / "annotations"
    inst = json.loads((ann / "instances_val2014.json").read_text())
    caps = json.loads((ann / "captions_val2014.json").read_text())
    pool = sorted({int(im["id"]) for im in inst["images"]} - old)
    rng = np.random.default_rng(SEED)
    pick = [int(i) for i in rng.choice(pool, N_CAL + N_TEST, replace=False)]
    cal, test = sorted(pick[:N_CAL]), sorted(pick[N_CAL:])
    gt = ground_truth_objects(P4._vocab(out), cal + test, inst, caps)
    prep = {"seed": SEED, "cal": cal, "test": test, "amber": P4._prep(out)["amber"],
            "gt": {str(k): sorted(v) for k, v in gt.items()}, "excluded_previous_ids": len(old)}
    (out / "prep5.json").write_text(json.dumps(prep))
    log(f"setup: Phase 5 COCO cal {len(cal)}, test {len(test)} (disjoint from {len(old)} earlier ids); "
        f"AMBER {len(prep['amber'])}")


def stage_images(out: Path) -> None:
    import phase1_pilot as p1
    prep = _prep(out)
    paths = P4.coco_paths(out, prep["cal"] + prep["test"])
    for n, (_, p) in enumerate(paths.items(), 1):
        if not p.exists():
            p1._download(p1.IMG_URL.format(p.name), p)
        if n % 250 == 0:
            log(f"  coco images {n}/{len(paths)}")
    P4.stage_images(out)                      # AMBER images (Phase 4 COCO images are reused if cached)
    log(f"images: {len(paths)} Phase 5 COCO images ready")


def _datasets(out: Path, prep: dict) -> dict:
    return {"coco": P4.coco_paths(out, prep["cal"] + prep["test"]), "amber": P4.amber_paths(out, prep["amber"])}


# ----------------------------------------------------------------------------- GPU stages

def stage_captions(out: Path, gens: list, k: int, n: int, limit=None, four_bit=None, only_ds=None) -> None:
    """Each (generator, dataset) runs in a fresh Python process: a model deleted inside a
    process does not reliably give its GPU memory back, and on a 15 GB T4 the next model
    then fails with CUDA out of memory (seen in the first Kaggle pilot)."""
    import subprocess
    prep = _prep(out)
    pairs = [(g, ds) for g in gens for ds in _datasets(out, prep) if only_ds in (None, ds)]
    if len(pairs) > 1:
        for g, ds in pairs:
            cmd = [sys.executable, str(Path(__file__).resolve()), "captions", "--out", str(out), "--gens", g,
                   "--only-dataset", ds, "--shard", str(k), "--nshards", str(n)]
            if limit:
                cmd += ["--limit", str(limit)]
            if four_bit:
                cmd += ["--four-bit"]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                log(f"captions: {g}/{ds} failed with exit code {r.returncode}; continuing with the next generator")
        return
    from src.phase4.gpu import shard
    from src.phase5.gpu import run_generator
    for g, ds in pairs:
        paths = _datasets(out, prep)[ds]
        ids = list(paths)[:limit] if limit else list(paths)
        mine = {i: paths[i] for i in shard(ids, k, n)}
        run_generator(g, mine, out / f"{cap_file(g, ds)}.s{k}.jsonl", four_bit=four_bit, log=log)


def stage_detect(out: Path, gens: list, k: int, n: int, only_ds=None) -> None:
    """One fresh process per (generator, dataset), for the same reason as stage_captions."""
    import subprocess
    prep = _prep(out)
    pairs = [(g, ds) for g in gens for ds in _datasets(out, prep) if only_ds in (None, ds)]
    if len(pairs) > 1:
        for g, ds in pairs:
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "detect", "--out", str(out), "--gens", g,
                                "--only-dataset", ds, "--shard", str(k), "--nshards", str(n)])
            if r.returncode != 0:
                log(f"detect: {g}/{ds} failed with exit code {r.returncode}; continuing")
        return
    from src.phase3.gpu_extra import run_swap_detectors
    from src.phase4.gpu import run_owl_tasks, shard
    vocab = P4._vocab(out)
    for g, ds in pairs:
        paths = _datasets(out, prep)[ds]
        claims = P4.claims_by_image(vocab, P4.read_all(out, cap_file(g, ds)))
        ids = [i for i in shard(list(paths), k, n) if i in claims]
        tasks = [{"set": f"{ds}_{g}", "image_id": i, "cond": "clean", "path": str(paths[i]), "cats": claims[i]}
                 for i in ids]
        run_owl_tasks(tasks, out / f"owl5_{g}_{ds}.s{k}.jsonl", log=log)
        run_swap_detectors({i: paths[i] for i in ids}, claims, out / f"swap5_{g}_{ds}.s{k}.jsonl", log=log)


# ----------------------------------------------------------------------------- tables

def build_tables(out: Path, gens: list):
    """tables[(gen, dataset, split, detector)] -> stats.Table of (image, hallucinated, logit score)."""
    from src.phase3 import amber as am
    from src.phase3.shift_validity import logit
    from src.phase5.stats import Table
    prep, vocab = _prep(out), P4._vocab(out)
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    entries = am.generative_entries(json.loads((out / "amber" / "annotations.json").read_text()))
    split_of = {**{i: "cal" for i in prep["cal"]}, **{i: "test" for i in prep["test"]}}
    rows: dict = {}
    for g in gens:
        for ds in ("coco", "amber"):
            owl = {int(r["image_id"]): r["scores"] for r in P4.read_all(out, f"owl5_{g}_{ds}")}
            swap = {int(r["image_id"]): {c["category"]: c for c in r["claims"]}
                    for r in P4.read_all(out, f"swap5_{g}_{ds}")}
            for r in P4.read_all(out, cap_file(g, ds)):
                iid = int(r["image_id"])
                truth = hallu = None
                if ds == "amber":
                    truth, hallu = am.node_sets(vocab, entries[iid])
                for c in vocab.extract_claims(r["caption"]).claims:
                    w = c.node_word
                    lab = am.label_claim(w, truth, hallu) if ds == "amber" else int(w not in gt.get(iid, set()))
                    if lab is None:
                        continue
                    sc = {"owlv2": owl.get(iid, {}).get(w),
                          "owlvit_v1": swap.get(iid, {}).get(w, {}).get("owlvit_v1"),
                          "grounding_dino": swap.get(iid, {}).get(w, {}).get("grounding_dino")}
                    split = split_of.get(iid, "test") if ds == "coco" else "test"
                    for d, s in sc.items():
                        if s is not None and s == s:      # skip missing and NaN
                            rows.setdefault((g, ds, split, d), []).append((iid, bool(lab), logit(s)))
    return {k: Table.of(*zip(*v)) for k, v in rows.items() if v}


def stage_analyze(out: Path, gens: list, n_resamples: int, n_boot: int) -> None:
    from src.phase5 import stats as S
    T = build_tables(out, gens)
    res: dict = {"gens": gens, "matrices": {}, "recalibration": {}, "crc": {}, "budget": S.label_budget()}
    for d in DETECTORS:
        cal = {g: T[(g, "coco", "cal", d)] for g in gens if (g, "coco", "cal", d) in T}
        test = {g: T[(g, "coco", "test", d)] for g in gens if (g, "coco", "test", d) in T}
        for g in gens:
            if (g, "amber", "test", d) in T:
                test[f"{g}@amber"] = T[(g, "amber", "test", d)]
        res["matrices"][d] = S.transfer_matrix(cal, test, n_resamples=n_resamples, n_boot=n_boot)
        res["crc"][d] = {s: {t: S.crc_risk(test[t], S.crc_threshold(cal[s], 0.10)) for t in test} for s in cal}
    import numpy as np
    # Benjamini-Hochberg over the COCO off-diagonal claim-level tests of all three matrices
    # (prereg-v3 H7: 30 cells x 3 detectors for six generators; AMBER cells are secondary).
    pv = {f"{d}|{s}|{t}": c["p_violated_claim"] for d, M in res["matrices"].items()
          for s, row in M.items() for t, c in row.items() if s != t and "@" not in t}
    res["bh"] = S.benjamini_hochberg(pv, q=0.05)
    # Theorem 3 check in every OWLv2 COCO cell that broke at claim level (m = 170 is H11).
    for s, row in res["matrices"].get("owlv2", {}).items():
        for t, c in row.items():
            if c["claim"]["decision"] == "violated" and "@" not in t:
                res["recalibration"][f"{s}->{t}"] = S.recalibration_curve(
                    T[(t, "coco", "test", "owlv2")], [25, 50, 100, 150, 170, 200, 300], repeats=200)
    # H10 part 2 (Theorem 2 prediction) and exploratory E1, greedy LLaVA -> sampled LLaVA, OWLv2.
    k = lambda g, sp: (g, "coco", sp, "owlv2")
    if all(x in T for x in (k("llava", "cal"), k("llava_t1", "cal"), k("llava", "test"), k("llava_t1", "test"))):
        h10 = {"part2": S.theorem2_check(T[k("llava", "cal")], T[k("llava_t1", "cal")],
                                         T[k("llava_t1", "test")], n_resamples=n_resamples, n_boot=n_boot)}
        tc, ti = S.thresholds(T[k("llava", "cal")], 0.10, 0.10, n_resamples)
        bg = S.bootstrap(S.per_image(T[k("llava", "test")], tc, ti), n_boot, 11)["image_risk"]
        bt = S.bootstrap(S.per_image(T[k("llava_t1", "test")], tc, ti), n_boot, 12)["image_risk"]
        n = min(len(bg["boot"]), len(bt["boot"]))
        h10["E1"] = {"greedy": bg["point"], "sampled": bt["point"],
                     "p_one_sided": float((np.sum(bt["boot"][:n] - bg["boot"][:n] <= 0) + 1) / (n + 1))}
        res["h10"] = h10
    res["hypotheses"] = S.hypotheses(res)
    (out / "phase5_results.json").write_text(json.dumps(res, indent=1, default=float))
    log("analyze: wrote phase5_results.json")


def stage_pilot(out: Path, gens: list) -> None:
    vocab, prep = P4._vocab(out), _prep(out)
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    for g in gens:
        caps = P4.read_all(out, cap_file(g, "coco"))
        men = hal = capsh = 0
        for r in caps:
            g_obj = gt.get(int(r["image_id"]), set())
            cc = vocab.extract_claims(r["caption"], gt_objects=g_obj)
            h = sum(1 for c in cc.claims if c.node_word not in g_obj)
            men, hal, capsh = men + cc.mention_count, hal + h, capsh + (h > 0)
        log(f"pilot {g}: {len(caps)} captions, CHAIR_i {hal / max(men, 1):.3f}, "
            f"CHAIR_s {capsh / max(len(caps), 1):.3f}, truncated {sum(bool(r.get('hit_max_tokens')) for r in caps)}")


def stage_audit(out: Path, gens: list) -> None:
    from src.phase1 import splits as sp
    from src.phase3.shift_validity import conformal_upper
    from src.phase5 import audit as A
    from src.phase3.shift_validity import logit
    T = build_tables(out, ["llava"])
    src = T[("llava", "coco", "cal", "owlv2")]
    t_src = conformal_upper(src.score[src.hall], 0.90)   # the clean LLaVA/OWLv2 threshold
    prep, vocab = _prep(out), P4._vocab(out)
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    test = set(prep["test"])
    full = []
    for g in gens:
        owl = {int(r["image_id"]): r["scores"] for r in P4.read_all(out, f"owl5_{g}_coco")}
        for r in P4.read_all(out, cap_file(g, "coco")):
            iid = int(r["image_id"])
            if iid not in test:
                continue
            for c in vocab.extract_claims(r["caption"]).claims:
                s = owl.get(iid, {}).get(c.node_word)
                if s is not None and s == s:
                    full.append({"generator": g, "image_id": iid, "category": c.node_word, "caption": r["caption"],
                                 "hall": c.node_word not in gt.get(iid, set()), "score": logit(s)})
    rows = A.sample(full, {g: t_src for g in gens})
    A.write_files(rows, out / "audit",
                  lambda i: f"http://images.cocodataset.org/val2014/{sp.coco_val2014_filename(int(i))}")
    log(f"audit: {len(rows)} claims written to {out / 'audit'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["setup", "images", "import", "captions", "detect", "pilot", "analyze", "audit"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gens", default=GENS_DEFAULT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="pilot/smoke test: first N images per dataset")
    ap.add_argument("--four-bit", dest="four_bit", action="store_true", default=None)
    ap.add_argument("--only-dataset", dest="only_ds", choices=["coco", "amber"], default=None,
                    help="internal: one dataset per process")
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    gens = a.gens.split(",")
    {"setup": lambda: stage_setup(a.out), "images": lambda: stage_images(a.out),
     "import": lambda: P4.stage_import(a.out),
     "captions": lambda: stage_captions(a.out, gens, a.shard, a.nshards, a.limit, a.four_bit, a.only_ds),
     "detect": lambda: stage_detect(a.out, gens, a.shard, a.nshards, a.only_ds),
     "pilot": lambda: stage_pilot(a.out, gens),
     "analyze": lambda: stage_analyze(a.out, gens, a.resamples, a.boot),
     "audit": lambda: stage_audit(a.out, gens)}[a.stage]()


if __name__ == "__main__":
    main()
