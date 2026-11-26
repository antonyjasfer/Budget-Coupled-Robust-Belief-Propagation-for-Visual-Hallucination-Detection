"""End-to-end CPU check of scripts/phase5.py on a synthetic output folder written in the
formats the GPU stages produce (caption, OWLv2 and swap-detector JSONL shards)."""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase5 as P5  # noqa: E402

OBJECTS = ["dog", "cat", "cup", "table", "chair", "car", "bus", "bird"]


def _sig(x):
    return 1 / (1 + math.exp(-x))


def make_out(tmp: Path, n_cal=120, n_test=200, gens=("llava", "qwen25")):
    rng = np.random.default_rng(0)
    (tmp / "synonyms.txt").write_text("\n".join(OBJECTS) + "\n")
    cal, test = list(range(1, n_cal + 1)), list(range(1000, 1000 + n_test))
    amber = list(range(1, 41))
    gt = {}
    for i in cal + test:
        gt[str(i)] = sorted(rng.choice(OBJECTS, 3, replace=False).tolist())
    (tmp / "prep5.json").write_text(json.dumps({"seed": 0, "cal": cal, "test": test, "amber": amber, "gt": gt}))
    (tmp / "amber").mkdir()
    ann = [{"id": i, "type": "generative", "truth": OBJECTS[:3], "hallu": OBJECTS[3:6]} for i in amber]
    (tmp / "amber" / "annotations.json").write_text(json.dumps(ann))
    for gi, g in enumerate(gens):
        shift = 0.0 if g == "llava" else 1.2       # the second generator's hallucinations score higher
        for ds, ids in (("coco", cal + test), ("amber", amber)):
            truth = (lambda i: set(OBJECTS[:3])) if ds == "amber" else (lambda i: set(gt[str(i)]))
            caps, owl, swap = [], [], []
            for i in ids:
                words = rng.choice(OBJECTS, 4, replace=False).tolist()
                caps.append({"image_id": i, "caption": "there is a " + ", a ".join(words) + "."})
                sc = {}
                for w in words:
                    x = rng.normal(1.0, 1.0) if w in truth(i) else rng.normal(-2.0 + shift, 1.0)
                    sc[w] = _sig(x)
                owl.append({"set": f"{ds}_{g}", "image_id": i, "cond": "clean", "scores": sc})
                swap.append({"image_id": i, "claims": [{"category": w, "owlvit_v1": _sig(math.log(p / (1 - p)) - 1),
                                                        "grounding_dino": _sig(math.log(p / (1 - p)) + 3)}
                                                       for w, p in sc.items()]})
            for stem, rows in ((P5.cap_file(g, ds), caps), (f"owl5_{g}_{ds}", owl), (f"swap5_{g}_{ds}", swap)):
                (tmp / f"{stem}.s0.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp


def test_build_tables_and_analyze(tmp_path):
    out = make_out(tmp_path)
    T = P5.build_tables(out, ["llava", "qwen25"])
    assert ("llava", "coco", "cal", "owlv2") in T and ("qwen25", "amber", "test", "grounding_dino") in T
    t = T[("llava", "coco", "test", "owlv2")]
    assert 0.2 < t.hall.mean() < 0.8 and len(t.images()) == 200
    P5.stage_analyze(out, ["llava", "qwen25"], n_resamples=20, n_boot=100)
    res = json.loads((out / "phase5_results.json").read_text())
    m = res["matrices"]["owlv2"]
    assert m["llava"]["qwen25"]["claim"]["decision"] == "violated"
    assert m["llava"]["qwen25"]["local_gap"] > 0
    assert set(m["llava"]) == {"llava", "qwen25", "llava@amber", "qwen25@amber"}
    assert "llava->qwen25" in res["recalibration"]
    assert res["budget"] == 170


def test_pilot_and_audit(tmp_path):
    out = make_out(tmp_path, n_cal=150, n_test=500)
    P5.stage_pilot(out, ["llava", "qwen25"])
    P5.stage_audit(out, ["llava", "qwen25"])
    rows = (out / "audit" / "annotator_tasks.csv").read_text().splitlines()
    assert len(rows) == 1 + 400
