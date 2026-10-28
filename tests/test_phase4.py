"""Phase 4: corruptions, analysis primitives, and the full analysis on synthetic files."""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.phase3.shift_validity import Claim
from src.phase4 import analysis as an
from src.phase4 import shifts as sh
from src.phase4.gpu import shard, task_key


# ----------------------------------------------------------------------------- shifts

def test_s1_conditions_and_determinism():
    assert len(sh.S1_CONDITIONS) == 25 and sh.parse_condition("jpeg_3") == ("jpeg", 3)
    img = Image.fromarray((np.random.default_rng(0).random((48, 64, 3)) * 255).astype("uint8"))
    for c in sh.S1_CONDITIONS:
        out = sh.corrupt(img, c, image_id=7)
        assert out.size == img.size
    a = np.asarray(sh.corrupt(img, "gaussian_noise_5", 7), float)
    b = np.asarray(sh.corrupt(img, "gaussian_noise_5", 7), float)
    assert np.array_equal(a, b)
    # severity increases the damage
    d1 = np.abs(np.asarray(sh.corrupt(img, "contrast_1"), float) - np.asarray(img, float)).mean()
    d5 = np.abs(np.asarray(sh.corrupt(img, "contrast_5"), float) - np.asarray(img, float)).mean()
    assert d5 > d1 > 0
    with pytest.raises(ValueError):
        sh.corrupt(img, "rain_2")


def test_mask_boxes_and_pick_removal():
    img = Image.new("RGB", (100, 100), (200, 10, 10))
    img.paste((0, 0, 255), (40, 40, 60, 60))
    out = np.asarray(sh.mask_boxes(img, [[40, 40, 20, 20]]))
    assert (out[50, 50] != [0, 0, 255]).any() and (out[5, 5] == [200, 10, 10]).all()
    assert sh.pick_removal(5, ["dog", "cat"]) == sh.pick_removal(5, ["cat", "dog"])


def test_shard_partitions():
    items = list(range(11))
    parts = [shard(items, k, 3) for k in range(3)]
    assert sorted(sum(parts, [])) == items
    assert task_key({"set": "a", "image_id": "3", "cond": "clean"}) == ("a", 3, "clean")


# ----------------------------------------------------------------------------- primitives

def test_d_plus_and_dominance():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    assert an.d_plus(a, a - 1.0) == 0.0
    assert an.d_plus(a, a + 10.0) == pytest.approx(1.0)


def test_holm():
    adj = an.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)


def test_chain_loglik_matches_bruteforce():
    th, J = np.array([0.3, -0.7, 1.1]), 0.6
    def score(h):
        return (th * h).sum() + J * (h[:-1] * h[1:]).sum()
    states = [np.array(s) for s in itertools.product([-1.0, 1.0], repeat=3)]
    logz = math.log(sum(math.exp(score(s)) for s in states))
    h = np.array([1.0, -1.0, 1.0])
    assert an._chain_loglik(th, h, J) == pytest.approx(score(h) - logz)
    assert an._chain_loglik(np.array([0.4]), np.array([1.0]), J) == pytest.approx(0.4 - math.log(2 * math.cosh(0.4)))


def test_per_image_stats_and_decide():
    cs = [Claim(1, 1, {"clean": 2.0}), Claim(1, 0, {"clean": 3.0}), Claim(2, 1, {"clean": -1.0}), Claim(2, 0, {"clean": 0.5})]
    st = an.per_image_stats(cs, "clean", np.array([0.0, 2.5]), np.array([0.0, 2.5]))
    assert st["kh"].tolist() == [0.5, 0.0] and st["ks"].tolist() == [1.0, 0.5]
    sm = an.summarise(st, 50)
    assert sm["claim_risk"]["point"] == pytest.approx(0.25)
    d = an.decide(sm["claim_risk"], 0.10)
    assert d["decision"] in ("violated", "inconclusive")


def test_class_metrics():
    m = an.class_metrics([1, 0, 1, 0], [0.9, 0.2, 0.4, 0.6])
    assert m["confusion"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1} and m["accuracy"] == 0.5
    assert an.class_metrics([1, 0], [0.9, 0.1])["roc_auc"] == 1.0


# ----------------------------------------------------------------------------- end to end

SYN = ["person, man, woman, people", "dog, puppy", "cat, kitten", "car, automobile", "bird", "chair", "boat, ship"]
CATS = ["person", "dog", "cat", "car", "bird", "chair"]


def _write(out: Path, stem: str, rows, k=0):
    (out / f"{stem}.s{k}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def _sig(x):
    return float(1 / (1 + math.exp(-x)))


def test_analyze_end_to_end(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("phase4", Path(__file__).resolve().parents[1] / "scripts" / "phase4.py")
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)

    out = tmp_path
    rng = np.random.default_rng(0)
    (out / "synonyms.txt").write_text("\n".join(SYN))
    cal_a, cal_b, test = list(range(1, 31)), list(range(31, 61)), list(range(101, 151))
    amber = list(range(1, 21))
    gt, boxes = {}, {}
    caps, owl, swap, q, iv = [], [], [], [], []
    for iid in cal_a + cal_b + test:
        present = list(rng.choice(CATS, 3, replace=False))
        gt[str(iid)] = sorted(present)
        mentioned = list(rng.choice(CATS, 3, replace=False))
        caps.append({"image_id": iid, "caption": "There is a " + " and a ".join(mentioned) + ".", "hit_max_tokens": False})
        clean = {c: _sig(rng.normal(-1.5 if c not in present else 1.0, 1.0)) for c in mentioned}
        owl.append({"set": "coco_llava", "image_id": iid, "cond": "clean", "scores": clean})
        swap.append({"image_id": iid, "shift": "clean", "claims": [
            {"category": c, "owlvit_v1": _sig(math.log(s / (1 - s)) - 0.5), "grounding_dino": _sig(math.log(s / (1 - s)) + 2.0)}
            for c, s in clean.items()]})
        if iid in test or iid in cal_a:
            for cond in sh.S1_CONDITIONS:
                owl.append({"set": "coco_llava", "image_id": iid, "cond": cond,
                            "scores": {c: _sig(math.log(s / (1 - s)) - 0.3) for c, s in clean.items()}})
        if iid in test:
            sup = [c for c in mentioned if c in present]
            if sup:
                boxes[str(iid)] = {sup[0]: [[1, 1, 5, 5]]}
                owl.append({"set": "coco_llava", "image_id": iid, "cond": "s5", "scores": {sup[0]: 0.05}})
            for lst, name in ((q, "coco_qwen"), (iv, "coco_internvl")):
                lst.append({"image_id": iid, "caption": "A " + " and a ".join(mentioned[:2]) + ".", "hit_max_tokens": False})
                owl.append({"set": name, "image_id": iid, "cond": "clean", "scores": {c: clean[c] for c in mentioned[:2]}})
    amb_caps, ann = [], []
    for k in amber:
        ann.append({"id": k, "type": "generative", "truth": ["dog", "man"], "hallu": ["cat"]})
        amb_caps.append({"image_id": k, "caption": "A dog, a man and a cat.", "hit_max_tokens": False})
        owl.append({"set": "amber_llava", "image_id": k, "cond": "clean", "scores": {"dog": 0.8, "person": 0.7, "cat": 0.05}})
    (out / "amber").mkdir()
    (out / "amber" / "annotations.json").write_text(json.dumps(ann))
    (out / "prep.json").write_text(json.dumps({"cal_a": cal_a, "cal_b": cal_b, "test": test, "amber": amber,
                                              "gt": gt, "boxes": boxes}))
    _write(out, "captions_llava_coco", caps[:70], 0)
    _write(out, "captions_llava_coco", caps[70:], 1)
    _write(out, "captions_llava_amber", amb_caps)
    _write(out, "captions_qwen", q)
    _write(out, "captions_internvl", iv)
    _write(out, "owl", owl[: len(owl) // 2], 0)
    _write(out, "owl", owl[len(owl) // 2:], 1)
    _write(out, "swap", swap)

    P.stage_analyze(out, n_resamples=20, n_boot=100)
    res = json.loads((out / "phase4_results.json").read_text())
    lv = res["levels"]["0.1"]
    assert res["counts"]["test"] == 150 and res["counts"]["amber"] == 60 and res["counts"]["s5"] > 0
    assert lv["swap_grounding_dino"]["claim_risk"]["point"] > 0.3          # +2 logits breaks it
    assert lv["swap_grounding_dino+recalibrated"]["claim_risk"]["point"] < 0.25
    assert lv["jpeg_3"]["claim_risk"]["point"] <= lv["clean"]["claim_risk"]["point"] + 1e-9  # scores lowered
    assert res["hypotheses"]["H2"]["per_condition"]["jpeg_3"]["d_plus"] == 0.0
    assert lv["s5_object_removed"]["claim_risk"]["point"] < 0.2
    assert set(res["hypotheses"]["claim_risk"]) == {"H1", "H3", "H4", "H5", "H6"}
    assert "J_learned" in res["pgm"] and "confusion" in res["pgm"]["test"]["ising_tree_J"]
    assert res["pgm"]["robust_intervals"]["images"] > 0
    assert res["pgm"]["test"]["unary_J0"]["roc_auc"] > 0.7
    text = (out / "phase4_report.md").read_text()
    assert "H1" in text and "ising_tree_J" in text and "Robust BP intervals" in text
