"""Phase 1 falsification pilot: one resumable command per stage.

Stages (run in order; each skips work already done):
    fetch      download COCO 2014 annotations + CHAIR synonyms (checksum verified)
    prepare    PREREG splits (seed 42) and CHAIR ground truth for the 200 pilot images
    images     download the pilot images (val2014)
    captions   LLaVA-1.5-7B detailed captions with token log-probs          [GPU]
    evidence   OWLv2 + CLIP evidence on clean / blur / jpeg / downsample    [GPU]
    analyze    claims table, Gate 1 statistics and report                   [CPU]
    all        everything above

Example (Colab / Kaggle):
    python scripts/phase1_pilot.py all --out /content/drive/MyDrive/BCRBP_phase1
    python scripts/phase1_pilot.py all --out OUT --limit 3     # smoke test first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase1 import splits as sp  # noqa: E402
from src.phase1.chair import SYNONYMS_SHA256, SYNONYMS_URL, ChairVocabulary, ground_truth_objects  # noqa: E402

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
IMG_URL = "http://images.cocodataset.org/val2014/{}"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _download(url: str, dest: Path, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(retries):
        for u in (url, url.replace("http://", "https://", 1)):
            try:
                with urllib.request.urlopen(u, timeout=120) as r, open(tmp, "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                tmp.rename(dest)
                return
            except Exception as e:  # network hiccups: retry
                last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------

def stage_fetch(out: Path) -> None:
    ann_dir = out / "annotations"
    need = [ann_dir / "captions_val2014.json", ann_dir / "instances_val2014.json"]
    if not all(p.exists() for p in need):
        z = out / "annotations_trainval2014.zip"
        if not z.exists():
            log("downloading COCO 2014 annotations (~240 MB)")
            _download(ANN_URL, z)
        with zipfile.ZipFile(z) as zf:
            for name in ("annotations/captions_val2014.json", "annotations/instances_val2014.json"):
                zf.extract(name, out)
        z.unlink()
    syn = out / "synonyms.txt"
    if not syn.exists():
        _download(SYNONYMS_URL, syn)
    if _sha256(syn) != SYNONYMS_SHA256:
        raise RuntimeError("CHAIR synonyms.txt checksum mismatch; upstream file changed.")
    log("fetch: annotations and synonyms ready")


def stage_prepare(out: Path, limit: int | None) -> None:
    ann = out / "annotations"
    caps = json.loads((ann / "captions_val2014.json").read_text())
    inst = json.loads((ann / "instances_val2014.json").read_text())
    ids = [im["id"] for im in caps["images"]]
    m = sp.build_prereg_splits(ids)
    sp.assert_pilot_isolated(m)
    sp.save_manifest(m, out / "manifest.json")
    vocab = ChairVocabulary.from_file(out / "synonyms.txt")
    pilot = m.pilot_cal + m.pilot_eval
    gt = ground_truth_objects(vocab, pilot, inst, caps)
    (out / "gt_objects.json").write_text(json.dumps({str(k): sorted(v) for k, v in gt.items()}, indent=1))
    log(f"prepare: universe {m.universe_size} images, manifest sha256 {m.manifest_sha256[:12]}…, "
        f"pilot {len(pilot)} images (train split only)")


def _pilot_ids(out: Path, limit: int | None) -> list[int]:
    m = json.loads((out / "manifest.json").read_text())
    if limit:  # smoke test: a few images from each half
        half = max(1, limit // 2)
        return m["pilot_cal"][:half] + m["pilot_eval"][: max(1, limit - half)]
    return m["pilot_cal"] + m["pilot_eval"]


def stage_images(out: Path, limit: int | None) -> dict[int, Path]:
    paths = {}
    for iid in _pilot_ids(out, limit):
        p = out / "images" / sp.coco_val2014_filename(iid)
        if not p.exists():
            _download(IMG_URL.format(p.name), p)
        paths[iid] = p
    log(f"images: {len(paths)} ready")
    return paths


def stage_captions(out: Path, limit: int | None) -> None:
    from src.phase1.gpu_pipeline import run_captions
    run_captions(stage_images(out, limit), out / "captions.jsonl", log=log)


def stage_evidence(out: Path, limit: int | None) -> None:
    from src.phase1.gpu_pipeline import read_done, run_evidence
    vocab = ChairVocabulary.from_file(out / "synonyms.txt")
    paths = stage_images(out, limit)
    claims = {}
    for line in (out / "captions.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if int(r["image_id"]) in paths:
                claims[int(r["image_id"])] = [c.node_word for c in vocab.extract_claims(r["caption"]).claims]
    run_evidence(paths, claims, out / "evidence.jsonl", log=log)


def stage_analyze(out: Path, n_boot: int) -> None:
    from src.phase1 import analysis as an
    vocab = ChairVocabulary.from_file(out / "synonyms.txt")
    manifest = json.loads((out / "manifest.json").read_text())
    gt = json.loads((out / "gt_objects.json").read_text())
    captions = an._read_jsonl(out / "captions.jsonl")
    evidence = an._read_jsonl(out / "evidence.jsonl")
    rows, chair = an.build_claims_table(vocab, manifest, gt, captions, evidence)
    an.rows_to_jsonl(rows, out / "claims_table.jsonl")
    res = an.analyse(rows, chair, n_boot=n_boot)
    (out / "gate1_results.json").write_text(json.dumps(res, indent=2, default=float))
    an.write_report(res, out / "gate1_report.md",
                    {"n_pilot_cal": len(manifest["pilot_cal"]), "n_pilot_eval": len(manifest["pilot_eval"])})
    log("analyze: wrote claims_table.jsonl, gate1_results.json, gate1_report.md")
    for c in res["gate1"]:
        log(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}: {c['value']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fetch", "prepare", "images", "captions", "evidence", "analyze", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N images")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    order = ["fetch", "prepare", "images", "captions", "evidence", "analyze"]
    for st in (order if a.stage == "all" else [a.stage]):
        t0 = time.time()
        if st == "fetch":
            stage_fetch(a.out)
        elif st == "prepare":
            stage_prepare(a.out, a.limit)
        elif st == "images":
            stage_images(a.out, a.limit)
        elif st == "captions":
            stage_captions(a.out, a.limit)
        elif st == "evidence":
            stage_evidence(a.out, a.limit)
        elif st == "analyze":
            stage_analyze(a.out, 200 if a.limit else a.n_boot)
        log(f"[{st}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
