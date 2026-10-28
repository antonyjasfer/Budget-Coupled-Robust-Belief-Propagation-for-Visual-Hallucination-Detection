"""Phase 4 (PREREG v2, confirmatory): when do conformal hallucination guarantees
survive distribution shift?

Stages:
    setup     COCO annotations + synonyms, PREREG manifest, ground truth, COCO
              instance boxes, AMBER annotations and the 804 confirmatory ids   [CPU]
    images    COCO Cal-A + Cal-B + test (1,200) and AMBER (804) images          [CPU]
    import    copy *.jsonl from a previous Kaggle session (/kaggle/input/...)  [CPU]
    gpu       everything below for one shard (run one process per GPU):        [GPU]
                llava   LLaVA-1.5-7B captions: COCO cal + test, AMBER
                vlms    Qwen2.5-VL-3B and InternVL3-2B captions: COCO test
                owl     OWLv2 scores: clean (all sets), S1 x 25 (test + Cal-A),
                        S5 object removal (test)
                swap    OWL-ViT v1 + Grounding DINO: COCO cal + test, LLaVA claims
    analyze   H1-H6 and descriptive PGM metrics                                 [CPU]

    python scripts/phase4.py setup  --out OUT
    python scripts/phase4.py images --out OUT
    CUDA_VISIBLE_DEVICES=0 python scripts/phase4.py gpu --out OUT --shard 0 --nshards 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/phase4.py gpu --out OUT --shard 1 --nshards 2
    python scripts/phase4.py analyze --out OUT

The PREREG test split and the 804 AMBER images are only analysed by `analyze`,
which must run from a commit at or after the `prereg-v2` tag.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.phase1.chair import ChairVocabulary  # noqa: E402
from src.phase3 import amber as am  # noqa: E402
from src.phase4.shifts import S1_CONDITIONS, pick_removal  # noqa: E402

CAPTION_FILES = {"llava_coco": "captions_llava_coco", "llava_amber": "captions_llava_amber",
                 "qwen": "captions_qwen", "internvl": "captions_internvl"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def read_all(out: Path, stem: str) -> list:
    """Rows from every shard file OUT/<stem>.s*.jsonl (half-written lines skipped).
    Duplicates (same set, image and condition; e.g. after a resumed or repeated
    run) are dropped, keeping the first occurrence."""
    rows, seen = [], set()
    for p in sorted(glob.glob(str(out / f"{stem}.s*.jsonl"))):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (r.get("set"), int(r["image_id"]), r.get("cond"))
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def _vocab(out: Path) -> ChairVocabulary:
    return ChairVocabulary.from_file(out / "synonyms.txt")


def _prep(out: Path) -> dict:
    return json.loads((out / "prep.json").read_text())


# ----------------------------------------------------------------------------- CPU stages

def stage_setup(out: Path) -> None:
    import phase1_pilot as p1
    from src.phase1.chair import ground_truth_objects
    p1.stage_fetch(out)
    manifest = json.loads((ROOT / "reports" / "phase1" / "manifest.json").read_text())
    (out / "manifest.json").write_text(json.dumps(manifest))
    vocab = _vocab(out)
    sp = manifest["splits"]
    coco_ids = sp["cal_a"] + sp["cal_b"] + sp["test"]
    pilot = set(manifest["pilot_cal"] + manifest["pilot_eval"])
    assert not pilot & set(coco_ids), "pilot images must not be in Cal-A/Cal-B/test"
    ann = out / "annotations"
    caps = json.loads((ann / "captions_val2014.json").read_text())
    inst = json.loads((ann / "instances_val2014.json").read_text())
    gt = ground_truth_objects(vocab, coco_ids, inst, caps)
    names = {c["id"]: c["name"] for c in inst["categories"]}
    test = set(sp["test"])
    boxes: dict = {}
    for a in inst["annotations"]:
        iid = int(a["image_id"])
        if iid in test:
            node = vocab.inverse.get(names[a["category_id"]], names[a["category_id"]])
            boxes.setdefault(str(iid), {}).setdefault(node, []).append(a["bbox"])
    d = out / "amber"
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "annotations.json").exists():
        urllib.request.urlretrieve(am.ANNOTATIONS_URL, d / "annotations.json")
    import hashlib
    if hashlib.sha256((d / "annotations.json").read_bytes()).hexdigest() != am.ANNOTATIONS_SHA256:
        raise RuntimeError("AMBER annotations checksum mismatch")
    pilot_amber = set(json.loads((ROOT / "reports" / "phase3b" / "amber" / "selected_ids.json").read_text()))
    amber_ids = sorted(i for i in range(1, am.N_GENERATIVE + 1) if i not in pilot_amber)
    assert len(amber_ids) == 804
    prep = {"cal_a": sp["cal_a"], "cal_b": sp["cal_b"], "test": sp["test"], "amber": amber_ids,
            "gt": {str(k): sorted(v) for k, v in gt.items()}, "boxes": boxes}
    (out / "prep.json").write_text(json.dumps(prep))
    log(f"setup: COCO cal {len(sp['cal_a']) + len(sp['cal_b'])}, test {len(sp['test'])}, AMBER {len(amber_ids)}")


def coco_paths(out: Path, ids) -> dict:
    from src.phase1 import splits as sp
    return {int(i): out / "images" / sp.coco_val2014_filename(int(i)) for i in ids}


def amber_paths(out: Path, ids) -> dict:
    return {int(i): out / "amber" / "images" / am.image_filename(int(i)) for i in ids}


def stage_images(out: Path) -> None:
    import zipfile
    import phase1_pilot as p1
    import phase3b as p3
    prep = _prep(out)
    coco = coco_paths(out, prep["cal_a"] + prep["cal_b"] + prep["test"])
    for n, (iid, p) in enumerate(coco.items(), 1):
        if not p.exists():
            p1._download(p1.IMG_URL.format(p.name), p)
        if n % 200 == 0:
            log(f"  coco images {n}/{len(coco)}")
    ap = amber_paths(out, prep["amber"])
    missing = [i for i, p in ap.items() if not p.exists()]
    if missing:
        z = p3._find_amber_zip(out / "amber")
        (out / "amber" / "images").mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            members = {Path(m).name: m for m in zf.namelist()}
            for i in missing:
                with zf.open(members[am.image_filename(i)]) as s, open(ap[i], "wb") as d:
                    shutil.copyfileobj(s, d)
    log(f"images: {len(coco)} COCO, {len(ap)} AMBER ready")


def stage_import(out: Path) -> None:
    """Resume across Kaggle sessions: attach the previous session's output as an
    input; its shard files are copied here if not already present."""
    root = Path("/kaggle/input")
    n = 0
    if root.exists():
        for p in root.rglob("*.s*.jsonl"):
            if p.parent.name != out.name:   # never import smoke-test files
                continue
            dst = out / p.name
            if not dst.exists() or dst.stat().st_size < p.stat().st_size:
                shutil.copy(p, dst)
                n += 1
    log(f"import: {n} shard files copied from previous sessions")


# ----------------------------------------------------------------------------- claims

def claims_by_image(vocab: ChairVocabulary, captions: list, keep=None) -> dict:
    out = {}
    for r in captions:
        iid = int(r["image_id"])
        if keep is None or iid in keep:
            cats = [c.node_word for c in vocab.extract_claims(r["caption"]).claims]
            if cats:
                out[iid] = cats
    return out


def owl_tasks(out: Path) -> list:
    vocab, prep = _vocab(out), _prep(out)
    tasks = []
    cal = prep["cal_a"] + prep["cal_b"]
    test = prep["test"]
    cp = coco_paths(out, cal + test)
    llava = claims_by_image(vocab, read_all(out, CAPTION_FILES["llava_coco"]))
    for iid in cal + test:
        cats = llava.get(iid)
        if not cats:
            continue
        conds = ["clean"]
        if iid in set(test) or iid in set(prep["cal_a"]):
            conds += S1_CONDITIONS
        for c in conds:
            tasks.append({"set": "coco_llava", "image_id": iid, "cond": c, "path": str(cp[iid]), "cats": cats})
        if iid in set(test):
            gt = set(prep["gt"].get(str(iid), []))
            bx = prep["boxes"].get(str(iid), {})
            cand = [c for c in cats if c in gt and bx.get(c)]
            if cand:
                rm = pick_removal(iid, cand)
                tasks.append({"set": "coco_llava", "image_id": iid, "cond": "s5", "path": str(cp[iid]),
                              "cats": [rm], "boxes": bx[rm]})
    for name in ("qwen", "internvl"):
        cl = claims_by_image(vocab, read_all(out, CAPTION_FILES[name]))
        for iid in test:
            if cl.get(iid):
                tasks.append({"set": f"coco_{name}", "image_id": iid, "cond": "clean", "path": str(cp[iid]), "cats": cl[iid]})
    ap = amber_paths(out, prep["amber"])
    cl = claims_by_image(vocab, read_all(out, CAPTION_FILES["llava_amber"]))
    for iid in prep["amber"]:
        if cl.get(iid):
            tasks.append({"set": "amber_llava", "image_id": iid, "cond": "clean", "path": str(ap[iid]), "cats": cl[iid]})
    return tasks


# ----------------------------------------------------------------------------- GPU stage

def stage_gpu(out: Path, k: int, n: int, parts: list, limit=None) -> None:
    from src.phase1.gpu_pipeline import run_captions
    from src.phase3.gpu_extra import run_swap_detectors
    from src.phase4.gpu import run_owl_tasks, run_vlm_captions, shard
    prep = _prep(out)
    if limit:  # smoke test: a few images of each set (use a separate --out folder)
        prep = {**prep, "cal_a": prep["cal_a"][:limit], "cal_b": prep["cal_b"][:limit],
                "test": prep["test"][:limit], "amber": prep["amber"][:limit]}
        (out / "prep.json").write_text(json.dumps(prep))
    cal, test = prep["cal_a"] + prep["cal_b"], prep["test"]
    sfx = f".s{k}.jsonl"
    if "llava" in parts:
        run_captions(coco_paths(out, shard(cal + test, k, n)), out / (CAPTION_FILES["llava_coco"] + sfx), log=log)
        run_captions(amber_paths(out, shard(prep["amber"], k, n)), out / (CAPTION_FILES["llava_amber"] + sfx), log=log)
    if "vlms" in parts:
        for name in ("qwen", "internvl"):
            run_vlm_captions(name, coco_paths(out, shard(test, k, n)), out / (CAPTION_FILES[name] + sfx), log=log)
    if "owl" in parts:
        # Score only the images this shard captioned itself, so the two GPU
        # processes never wait on each other's captions.
        own = {"coco_llava": set(shard(cal + test, k, n)), "coco_qwen": set(shard(test, k, n)),
               "coco_internvl": set(shard(test, k, n)), "amber_llava": set(shard(prep["amber"], k, n))}
        mine = [t for t in owl_tasks(out) if t["image_id"] in own[t["set"]]]
        run_owl_tasks(mine, out / ("owl" + sfx), log=log)
    if "swap" in parts:
        claims = claims_by_image(_vocab(out), read_all(out, CAPTION_FILES["llava_coco"]))
        ids = [i for i in shard(cal + test, k, n) if i in claims]
        run_swap_detectors(coco_paths(out, ids), claims, out / ("swap" + sfx), log=log)


def stage_analyze(out: Path, n_resamples: int, n_boot: int) -> None:
    from src.phase4 import analysis as an
    res = an.run_all(out, n_resamples=n_resamples, n_boot=n_boot)
    (out / "phase4_results.json").write_text(json.dumps(res, indent=2, default=float))
    an.write_report(res, out / "phase4_report.md")
    log("analyze: wrote phase4_results.json and phase4_report.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["setup", "images", "import", "gpu", "analyze"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--parts", default="llava,vlms,owl,swap")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N images per set (separate --out!)")
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.stage == "setup":
        stage_setup(a.out)
    elif a.stage == "images":
        stage_images(a.out)
    elif a.stage == "import":
        stage_import(a.out)
    elif a.stage == "gpu":
        stage_gpu(a.out, a.shard, a.nshards, a.parts.split(","), a.limit)
    elif a.stage == "analyze":
        stage_analyze(a.out, a.resamples, a.boot)
    log(f"[{a.stage}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
