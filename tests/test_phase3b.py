"""Phase 3b: AMBER labelling, Grounding DINO token mask, transfer analysis, report wiring."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.phase1.chair import ChairVocabulary
from src.phase3 import amber as am
from src.phase3 import transfer as tr
from src.phase3.gpu_extra import gdino_phrase_positions
from src.phase3.shift_validity import Claim

SYN = ["person, man, woman, people, child", "dog, puppy", "cat, kitten", "boat, ship",
       "dining table, table", "car, automobile", "bird", "chair"]


@pytest.fixture
def vocab():
    return ChairVocabulary(SYN)


def test_amber_node_sets_and_labels(vocab):
    e = {"id": 1, "type": "generative", "truth": ["sky", "man", "ship", "ship"], "hallu": ["dog", "bird", "cloud", "man"]}
    truth, hallu = am.node_sets(vocab, e)
    assert truth == {"person", "boat"} and hallu == {"dog", "bird"}  # 'man' in both -> present wins
    assert am.label_claim("person", truth, hallu) == 0
    assert am.label_claim("dog", truth, hallu) == 1
    assert am.label_claim("cat", truth, hallu) is None


def test_amber_select_deterministic_and_eligible(vocab):
    entries = {i: {"id": i, "type": "generative", "truth": ["dog"] if i % 2 else ["sky"], "hallu": []} for i in range(1, 51)}
    a, b = am.select_images(vocab, entries, 10), am.select_images(vocab, entries, 10)
    assert a == b and len(a) == 10 and all(i % 2 for i in a)


def test_amber_build_claims_drops_unknown(vocab):
    entries = {7: {"id": 7, "type": "generative", "truth": ["dog"], "hallu": ["cat"]}}
    caps = [{"image_id": 7, "caption": "A dog chases a cat past a car."}]
    ev = [{"image_id": 7, "shift": "clean", "category": c, "owl_score": s, "clip_crop": 0.2, "clip_global": 0.2}
          for c, s in (("dog", 0.8), ("cat", 0.1), ("car", 0.5))]
    rows, counts = am.build_claims(vocab, entries, caps, ev, "amber")
    assert {(r["category"], r["hallucinated"]) for r in rows} == {("dog", 0), ("cat", 1)}
    assert counts["unknown"] == 1 and counts["claims"] == 3


def test_gdino_phrase_positions():
    # [CLS]=101 dog=3899 .=1012 [SEP]=102
    assert gdino_phrase_positions([101, 3899, 1012, 102], [101, 102, 0, 1012]) == [1]
    assert gdino_phrase_positions([101, 2829, 3899, 1012, 102], [101, 102, 0, 1012]) == [1, 2]


def _coco(rng, n_img=200, inflate=1.5):
    out = []
    for i in range(n_img):
        for _ in range(3):
            h = int(rng.random() < 0.3)
            clean = rng.normal(-1.5 if h else 1.0, 1.0)
            out.append(Claim(i, h, {"clean": clean,
                                    "owlvit_v1": clean + rng.normal(inflate if h else 0.0, 0.3),
                                    "grounding_dino": clean + rng.normal(-1.0 if h else 0.0, 0.3)}))
    return out


def test_transfer_detects_inflating_swap_and_robust_restores():
    rng = np.random.default_rng(0)
    coco = _coco(rng)
    qwen = [Claim(c.image_id, c.hallucinated, {"clean": c.score["clean"] + (1.0 if c.hallucinated else 0)}) for c in coco]
    amber = [Claim(1000 + c.image_id, c.hallucinated, {"clean": c.score["clean"]}) for c in coco[:150]]
    res = tr.run(coco, qwen, amber, n_splits=60, seed=1)
    r = res["results"]
    assert r["claim/ref"]["risk"]["mean"] <= 0.12
    assert r["claim/swap_owlvit_v1"]["risk"]["mean"] > 0.2               # inflating swap breaks it
    assert r["claim/swap_owlvit_v1+robust"]["risk"]["mean"] <= 0.15 + 0.03  # measured radius restores
    assert r["claim/swap_grounding_dino"]["risk"]["mean"] < 0.1            # deflating swap is benign
    assert r["claim/vlm_qwen"]["risk"]["mean"] > r["claim/ref"]["risk"]["mean"]
    assert "domain_amber" in "".join(r)
    assert 0.8 < res["diagnostics"]["coco_llava_owlv2"]["auroc"] < 1.0
    assert res["diagnostics"]["coco_llava_owlvit_v1"]["hallucinated_share_up_vs_owlv2"] > 0.9


def test_auroc_matches_simple_cases():
    assert tr.auroc([0, 0, 1, 1], [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert tr.auroc([0, 1], [0.5, 0.5]) == 0.5


def _load_script():
    p = Path(__file__).resolve().parents[1] / "scripts" / "phase3b.py"
    spec = importlib.util.spec_from_file_location("phase3b", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_analyze_stage_end_to_end(tmp_path):
    """Synthetic OUT folder in the exact file formats the GPU stages write."""
    s = _load_script()
    out = tmp_path
    (out / "synonyms.txt").write_text("\n".join(SYN))
    p1 = out / "phase1"
    p1.mkdir()
    rng = np.random.default_rng(3)
    ids = list(range(1, 41))
    (p1 / "manifest.json").write_text(json.dumps({"pilot_cal": ids[:20], "pilot_eval": ids[20:]}))
    cats = ["person", "dog", "cat", "car", "bird", "chair"]
    gt, rows, swap, qcaps, qev = {}, [], [], [], []
    for i in ids:
        present = set(rng.choice(cats, 2, replace=False))
        gt[str(i)] = sorted(present)
        mentioned = list(rng.choice(cats, 3, replace=False))
        sw = []
        for c in mentioned:
            h = int(c not in present)
            o = float(1 / (1 + math.exp(-rng.normal(-1.5 if h else 1.0, 1.0))))
            rows.append({"image_id": i, "category": c, "hallucinated": h,
                         "evidence": {"clean": {"owl_score": o, "clip_crop": 0.2, "clip_global": 0.2}}})
            sw.append({"category": c, "owlvit_v1": min(0.99, o * 1.2), "grounding_dino": o * 0.8})
        swap.append({"image_id": i, "shift": "clean", "claims": sw})
        qcaps.append({"image_id": i, "caption": "There is a " + " and a ".join(mentioned) + "."})
        for c in mentioned:
            qev.append({"image_id": i, "shift": "clean", "category": c,
                        "owl_score": 0.9 if c in present else 0.2, "clip_crop": 0.2, "clip_global": 0.2})
    (p1 / "gt_objects.json").write_text(json.dumps(gt))
    (p1 / "claims_table.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (out / "swap_evidence.jsonl").write_text("\n".join(json.dumps(r) for r in swap))
    (out / "qwen_captions.jsonl").write_text("\n".join(json.dumps(r) for r in qcaps))
    (out / "qwen_evidence.jsonl").write_text("\n".join(json.dumps(r) for r in qev))
    (out / "amber").mkdir()
    ann = [{"id": k, "type": "generative", "truth": ["dog", "man"], "hallu": ["cat", "car"]} for k in range(1, 11)]
    (out / "amber" / "annotations.json").write_text(json.dumps(ann))
    (out / "amber_captions.jsonl").write_text("\n".join(json.dumps({"image_id": k, "caption": "A dog, a man and a cat."}) for k in range(1, 11)))
    (out / "amber_evidence.jsonl").write_text("\n".join(json.dumps({"image_id": k, "shift": "clean", "category": c,
        "owl_score": {"dog": 0.8, "person": 0.7, "cat": 0.3}[c], "clip_crop": 0.2, "clip_global": 0.2})
        for k in range(1, 11) for c in ("dog", "person", "cat")))
    s.stage_analyze(out, splits=20)
    res = json.loads((out / "phase3b_results.json").read_text())
    assert res["counts"]["amber"]["hallucinated"] == 10 and res["counts"]["amber"]["supported"] == 20
    assert res["counts"]["qwen_claims"] == 120 and res["counts"]["coco_claims_with_swap"] == 120
    text = (out / "phase3b_report.md").read_text()
    assert "Detector swap → Grounding DINO" in text and "AMBER" in text


def test_cap_pixels_keeps_aspect_and_limit():
    from PIL import Image
    from src.phase3.gpu_extra import cap_pixels
    small = Image.new("RGB", (100, 50))
    assert cap_pixels(small, 10_000) is small
    big = cap_pixels(Image.new("RGB", (2000, 1000)), 600_000)
    assert big.size[0] * big.size[1] <= 600_000 and abs(big.size[0] / big.size[1] - 2) < 0.02
