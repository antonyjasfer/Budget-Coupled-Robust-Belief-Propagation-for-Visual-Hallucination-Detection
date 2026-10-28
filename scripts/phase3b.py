"""Phase 3b: shifts that can raise the evidence for absent objects.

Stages (run in order; each resumes and skips finished work):
    setup          COCO annotations + CHAIR synonyms; copy the committed Phase 1
                   manifest / ground truth / captions / claims into OUT/phase1
    coco_images    the 200 Phase 1 pilot images (COCO val2014)
    swap           OWL-ViT v1 + Grounding DINO scores for the Phase 1 claims  [GPU]
    qwen           Qwen2.5-VL-3B captions + OWLv2 evidence on the pilot images [GPU]
    amber_fetch    AMBER annotations (checksum) + images (Google Drive via gdown)
    amber          LLaVA-1.5 captions + OWLv2 evidence on AMBER images         [GPU]
    analyze        calibrate on Phase 1, test every condition, write report    [CPU]
    all            everything above

    python scripts/phase3b.py all --out OUT [--limit 4] [--amber-n 200] [--qwen-4bit]

EXPLORATORY: COCO images are the Phase 1 pilot (PREREG train split); AMBER is an
external benchmark. The PREREG test split is never touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.phase1.chair import ChairVocabulary  # noqa: E402
from src.phase3 import amber as am  # noqa: E402

P1_FILES = ("manifest.json", "gt_objects.json", "captions.jsonl", "claims_table.jsonl")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _jsonl(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _vocab(out: Path) -> ChairVocabulary:
    return ChairVocabulary.from_file(out / "synonyms.txt")


def _pilot_ids(out: Path, limit):
    m = json.loads((out / "phase1" / "manifest.json").read_text())
    ids = m["pilot_cal"] + m["pilot_eval"]
    if limit:
        half = max(1, limit // 2)
        ids = m["pilot_cal"][:half] + m["pilot_eval"][: max(1, limit - half)]
    return ids


# ----------------------------------------------------------------------------

def stage_setup(out: Path) -> None:
    import phase1_pilot as p1
    p1.stage_fetch(out)
    dst = out / "phase1"
    dst.mkdir(parents=True, exist_ok=True)
    for f in P1_FILES:
        src = ROOT / "reports" / "phase1" / f
        if not (dst / f).exists():
            shutil.copy(src, dst / f)
    log("setup: COCO annotations, synonyms and Phase 1 results ready")


def stage_coco_images(out: Path, limit) -> dict:
    import phase1_pilot as p1
    from src.phase1 import splits as sp
    paths = {}
    for iid in _pilot_ids(out, limit):
        p = out / "images" / sp.coco_val2014_filename(iid)
        if not p.exists():
            p1._download(p1.IMG_URL.format(p.name), p)
        paths[iid] = p
    log(f"coco_images: {len(paths)} ready")
    return paths


def _claims_by_image(vocab, captions, keep) -> dict:
    out = {}
    for r in captions:
        iid = int(r["image_id"])
        if iid in keep:
            cats = [c.node_word for c in vocab.extract_claims(r["caption"]).claims]
            if cats:
                out[iid] = cats
    return out


def stage_swap(out: Path, limit) -> None:
    from src.phase3.gpu_extra import run_swap_detectors
    paths = stage_coco_images(out, limit)
    claims = _claims_by_image(_vocab(out), _jsonl(out / "phase1" / "captions.jsonl"), set(paths))
    run_swap_detectors(paths, claims, out / "swap_evidence.jsonl", log=log)


def stage_qwen(out: Path, limit, four_bit: bool) -> None:
    from src.phase1.gpu_pipeline import run_evidence
    from src.phase3.gpu_extra import run_qwen_captions
    paths = stage_coco_images(out, limit)
    run_qwen_captions(paths, out / "qwen_captions.jsonl", four_bit=four_bit, log=log)
    claims = _claims_by_image(_vocab(out), _jsonl(out / "qwen_captions.jsonl"), set(paths))
    run_evidence(paths, claims, out / "qwen_evidence.jsonl", shifts=["clean"], log=log)


def stage_amber_fetch(out: Path, n: int, limit) -> dict:
    d = out / "amber"
    d.mkdir(parents=True, exist_ok=True)
    ann = d / "annotations.json"
    if not ann.exists():
        urllib.request.urlretrieve(am.ANNOTATIONS_URL, ann)
    if _sha256(ann) != am.ANNOTATIONS_SHA256:
        raise RuntimeError("AMBER annotations.json checksum mismatch")
    entries = am.generative_entries(json.loads(ann.read_text()))
    ids = am.select_images(_vocab(out), entries, n)
    if limit:
        ids = ids[:limit]
    (d / "selected_ids.json").write_text(json.dumps(ids))
    img_dir = d / "images"
    img_dir.mkdir(exist_ok=True)
    missing = [i for i in ids if not (img_dir / am.image_filename(i)).exists()]
    if missing:
        z = _find_amber_zip(d)
        with zipfile.ZipFile(z) as zf:
            members = {Path(m).name: m for m in zf.namelist()}
            for i in missing:
                name = am.image_filename(i)
                if name not in members:
                    raise RuntimeError(f"{name} not in {z}")
                with zf.open(members[name]) as src, open(img_dir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    log(f"amber_fetch: {len(ids)} AMBER images ready")
    return {i: img_dir / am.image_filename(i) for i in ids}


def _find_amber_zip(d: Path) -> Path:
    # 1) a zip the user added as a Kaggle dataset / uploaded by hand
    for root in (Path("/kaggle/input"), d):
        if root.exists():
            for z in root.rglob("*.zip"):
                try:
                    with zipfile.ZipFile(z) as zf:
                        if any(Path(m).name == "AMBER_1.jpg" for m in zf.namelist()):
                            return z
                except zipfile.BadZipFile:
                    continue
    # 2) download from the AMBER README's Google Drive link
    z = d / "amber_images.zip"
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "gdown"], check=True)
    import gdown
    log("amber_fetch: downloading AMBER images from Google Drive (gdown)")
    gdown.download(id=am.IMAGES_GDRIVE_ID, output=str(z), quiet=False)
    if not z.exists() or not zipfile.is_zipfile(z):
        raise RuntimeError(
            "Could not download the AMBER images. Download the zip by hand from "
            f"https://drive.google.com/file/d/{am.IMAGES_GDRIVE_ID}/view, add it to this "
            "notebook as a Kaggle dataset (Add Input), and run this cell again.")
    return z


def stage_amber(out: Path, n: int, limit) -> None:
    from src.phase1.gpu_pipeline import run_captions, run_evidence
    paths = stage_amber_fetch(out, n, limit)
    run_captions(paths, out / "amber_captions.jsonl", log=log)
    claims = _claims_by_image(_vocab(out), _jsonl(out / "amber_captions.jsonl"), set(paths))
    run_evidence(paths, claims, out / "amber_evidence.jsonl", shifts=["clean"], log=log)


def stage_analyze(out: Path, splits: int) -> None:
    from src.phase1.analysis import build_claims_table
    from src.phase3 import transfer as tr
    vocab = _vocab(out)
    p1 = out / "phase1"
    manifest = json.loads((p1 / "manifest.json").read_text())
    gt = json.loads((p1 / "gt_objects.json").read_text())
    coco_rows = _jsonl(p1 / "claims_table.jsonl")
    coco = tr.claims_from_rows(coco_rows, tr.swap_index(_jsonl(out / "swap_evidence.jsonl")))

    qrows, qchair = build_claims_table(vocab, manifest, gt, _jsonl(out / "qwen_captions.jsonl"),
                                       _jsonl(out / "qwen_evidence.jsonl"))
    qwen = tr.claims_from_rows([{"image_id": r.image_id, "category": r.category,
                                 "hallucinated": r.hallucinated, "evidence": r.evidence} for r in qrows])

    entries = am.generative_entries(json.loads((out / "amber" / "annotations.json").read_text())) \
        if (out / "amber" / "annotations.json").exists() else {}
    arows, acounts = am.build_claims(vocab, entries, _jsonl(out / "amber_captions.jsonl"),
                                     _jsonl(out / "amber_evidence.jsonl"), "amber_llava")
    amber = tr.claims_from_rows(arows)

    res = tr.run(coco, qwen, amber, n_splits=splits)
    res["counts"] = {"coco_claims_with_swap": len(coco), "qwen_claims": len(qwen),
                     "qwen_chair": qchair, "amber": acounts}
    (out / "phase3b_results.json").write_text(json.dumps(res, indent=2, default=float))
    with open(out / "amber_claims.jsonl", "w") as f:
        for r in arows:
            f.write(json.dumps(r) + "\n")
    write_report(res, out / "phase3b_report.md")
    log("analyze: wrote phase3b_results.json and phase3b_report.md")


NAMES = {
    "ref": "COCO, same pipeline (reference)",
    "swap_owlvit_v1": "Detector swap → OWL-ViT v1",
    "swap_grounding_dino": "Detector swap → Grounding DINO",
    "swap_owlvit_v1+robust": "  + one-sided measured radius",
    "swap_grounding_dino+robust": "  + one-sided measured radius",
    "vlm_qwen": "VLM swap → Qwen2.5-VL-3B",
    "domain_amber": "Domain shift → AMBER images",
}


def write_report(res: dict, path: Path) -> None:
    st, tg = res["settings"], res["targets"]
    L = ["# Phase 3b — Shifts that can raise the evidence for absent objects\n",
         "**Status: exploratory.** Calibration always on the Phase 1 COCO pilot (LLaVA-1.5 captions, OWLv2, clean "
         f"images); {st['n_splits']} random image splits ({int(st['cal_frac']*100)}% calibration). "
         f"Targets: claim-level δ = {st['delta']}, image-level α = {st['alpha']}; robust rows target +β = {st['beta']}. "
         "Brackets: 2.5–97.5% range across splits.\n",
         "## Diagnostics\n",
         "| Condition | Claims | Images | Hallucination rate | AUROC (evidence) | Median logit, hallucinated | Median logit, supported |",
         "|---|---|---|---|---|---|---|"]
    for k, d in res["diagnostics"].items():
        L.append(f"| {k} | {d['n_claims']} | {d['n_images']} | {d['hallucination_rate']:.3f} | {d['auroc']:.3f} | "
                 f"{d['median_logit_hallucinated']:.2f} | {d['median_logit_supported']:.2f} |")
    for k, d in res["diagnostics"].items():
        if "hallucinated_share_up_vs_owlv2" in d:
            L.append(f"\n- {k}: share of hallucinated claims scored HIGHER than by OWLv2: {d['hallucinated_share_up_vs_owlv2']:.2f}")
    L += ["\n## Realised risk\n",
          "Claim-level (CEBC-style): share of hallucinated claims that pass. Image-level (ConfLVLM-style, λ = 0): share of images whose kept claims include a hallucination.\n",
          "| Level | Condition | Target | Risk (mean [2.5%, 97.5%]) | Supported claims kept | Exceeds target? |",
          "|---|---|---|---|---|---|"]
    for key, d in res["results"].items():
        level, cond = key.split("/", 1)
        t = tg[level + ("+robust" if cond.endswith("+robust") else "")]
        r = d["risk"]
        L.append(f"| {level} | {NAMES.get(cond, cond)} | {t:.2f} | {r['mean']:.3f} [{r['q025']:.3f}, {r['q975']:.3f}] | "
                 f"{d['retained_supported']['mean']:.3f} | {'YES' if r['mean'] > t else 'no'} |")
    c = res.get("counts", {})
    L += ["\n## Counts\n", f"```\n{json.dumps(c, indent=1)}\n```",
          "\n## Reading guide (mechanical; conclusions are the author's)\n",
          "- A shift *breaks* a clean-calibrated filter if its mean risk exceeds the target.",
          "- Detector swaps: the robust rows show whether a radius measured on calibration images restores the target, and at what cost in supported claims kept.",
          "- AMBER labels: claims whose object is in neither AMBER's present list nor its verified-absent list are excluded (see Counts)."]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["setup", "coco_images", "swap", "qwen", "amber_fetch", "amber", "analyze", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test on N images per arm")
    ap.add_argument("--amber-n", type=int, default=200)
    ap.add_argument("--qwen-4bit", action="store_true", help="load Qwen in 4-bit (use if fp16 runs out of memory)")
    ap.add_argument("--splits", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    order = ["setup", "coco_images", "swap", "qwen", "amber", "analyze"]
    for st in (order if a.stage == "all" else [a.stage]):
        t0 = time.time()
        if st == "setup":
            stage_setup(a.out)
        elif st == "coco_images":
            stage_coco_images(a.out, a.limit)
        elif st == "swap":
            stage_swap(a.out, a.limit)
        elif st == "qwen":
            stage_qwen(a.out, a.limit, a.qwen_4bit)
        elif st == "amber_fetch":
            stage_amber_fetch(a.out, a.amber_n, a.limit)
        elif st == "amber":
            stage_amber(a.out, a.amber_n, a.limit)
        elif st == "analyze":
            stage_analyze(a.out, 100 if a.limit else a.splits)
        log(f"[{st}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
