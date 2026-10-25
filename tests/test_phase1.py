"""Phase 1 pilot: splits, CHAIR labelling, GPU-stage helpers and Gate 1 analysis."""
import json
import math
import random

import numpy as np
import pytest
from PIL import Image

from src.phase1 import splits as sp
from src.phase1.chair import ChairVocabulary, ground_truth_objects
from src.phase1.gpu_pipeline import (apply_shift, claim_token_features, owl_box_to_pixels,
                                     read_done, append_jsonl, span_token_logprobs, token_char_offsets)
from src.phase1 import analysis as an

# Small synonym table in the CHAIR file format (first entry = category).
SYN = [
    "person, man, woman, people, child, boy, girl",
    "dog, puppy",
    "cat, kitten",
    "car, automobile, taxi",
    "dining table, table",
    "teddy bear",
    "toilet",
    "chair, seat",
    "elephant",
    "train",
    "wine glass",
    "cup, mug",
]
CATS = [line.split(",")[0].strip() for line in SYN]


@pytest.fixture
def vocab():
    return ChairVocabulary(SYN)


# --------------------------------------------------------------------------- splits

def test_prereg_splits_deterministic_and_isolated():
    ids = list(range(100000, 140504))
    m1, m2 = sp.build_prereg_splits(ids), sp.build_prereg_splits(list(reversed(ids)))
    assert m1.manifest_sha256 == m2.manifest_sha256
    assert {k: len(v) for k, v in m1.splits.items()} == sp.SPLIT_SIZES
    all_ids = [i for v in m1.splits.values() for i in v]
    assert len(all_ids) == len(set(all_ids))
    assert len(m1.pilot_cal) == 100 and len(m1.pilot_eval) == 100
    sp.assert_pilot_isolated(m1)


def test_pilot_isolation_detects_test_leak():
    m = sp.build_prereg_splits(list(range(5000)))
    m.pilot_eval[0] = m.splits["test"][0]
    with pytest.raises(AssertionError):
        sp.assert_pilot_isolated(m)


def test_manifest_roundtrip(tmp_path):
    m = sp.build_prereg_splits(list(range(5000)))
    sp.save_manifest(m, tmp_path / "m.json")
    assert sp.load_manifest(tmp_path / "m.json").manifest_sha256 == m.manifest_sha256


# --------------------------------------------------------------------------- CHAIR

def test_chair_rules(vocab):
    s = "Two women sit at a table with wine glasses, a teddy bear on the toilet seat and a passenger train."
    nodes = vocab.node_words(s)
    assert nodes == ["person", "dining table", "wine glass", "teddy bear", "toilet", "train"]
    assert "chair" not in nodes  # toilet-seat rule


def test_chair_baby_animal_and_spans(vocab):
    s = "A baby elephant and a man. The man pets a puppy."
    cc = vocab.extract_claims(s, gt_objects={"elephant", "person"})
    by = {c.node_word: c for c in cc.claims}
    assert by["elephant"].hallucinated is False
    assert s[slice(*by["elephant"].char_spans[0])].lower() == "baby elephant"
    assert by["person"].hallucinated is False and len(by["person"].char_spans) == 2
    assert by["dog"].hallucinated is True
    assert cc.mention_count == 4 and cc.hallucinated_mentions == 1


def test_ground_truth_union_of_instances_and_captions(vocab):
    inst = {"categories": [{"id": 1, "name": "person"}, {"id": 18, "name": "dog"}],
            "annotations": [{"image_id": 7, "category_id": 18}]}
    caps = {"annotations": [{"image_id": 7, "caption": "A man walking his puppy."},
                            {"image_id": 8, "caption": "A cat."}]}
    gt = ground_truth_objects(vocab, [7], inst, caps)
    assert gt == {7: {"dog", "person"}}


# --------------------------------------------------------------------------- GPU helpers

def test_token_offsets_and_span_logprobs():
    pieces = [" A", " dog", " and", " a", " cat", "."]
    decode = lambda ids: "".join(pieces[i] for i in ids)
    spans = token_char_offsets(decode, list(range(len(pieces))))
    assert spans[1] == (2, 6)
    raw = decode(list(range(len(pieces))))
    offset = len(raw) - len(raw.lstrip())
    caption = raw.strip()
    a = caption.index("cat")
    lps = span_token_logprobs(spans, [-0.1, -0.2, -0.3, -0.4, -2.0, -0.5], (a, a + 3), offset)
    assert lps == [-2.0]
    f = claim_token_features(lps)
    assert f["tok_prob_min"] == pytest.approx(math.exp(-2.0))
    assert claim_token_features([])["tok_logprob_mean"] is None


def test_owl_box_uses_padded_square():
    # 400x200 image is padded to 400x400; a box at the image centre in pixels:
    assert owl_box_to_pixels([0.5, 0.25, 0.5, 0.25], 400, 200) == (100, 50, 300, 150)


def test_shifts_keep_size_and_change_pixels():
    img = Image.fromarray((np.random.default_rng(0).random((64, 80, 3)) * 255).astype("uint8"))
    for s in ("clean", "blur", "jpeg", "downsample"):
        out = apply_shift(img, s)
        assert out.size == img.size
        if s != "clean":
            assert np.abs(np.asarray(out, float) - np.asarray(img, float)).mean() > 1.0
    with pytest.raises(ValueError):
        apply_shift(img, "rain")


def test_jsonl_resume_ignores_partial_line(tmp_path):
    p = tmp_path / "x.jsonl"
    append_jsonl(p, {"image_id": 1})
    with open(p, "a") as f:
        f.write('{"image_id": 2')  # interrupted write
    assert read_done(p, lambda r: r["image_id"]) == {1}


# --------------------------------------------------------------------------- analysis

def test_auroc_ece_logistic():
    assert an.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert an.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert an.ece([0, 1], [0.0, 1.0]) == 0.0
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(int)
    w = an.fit_logistic(X, y)
    assert w[0] > 2 and abs(w[1]) < 1


def test_within_image_correlation_detects_clustering():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(60), 5)
    clustered = np.repeat(rng.integers(0, 2, 60), 5)
    indep = rng.integers(0, 2, 300)
    assert an.within_image_correlation(clustered, groups) == pytest.approx(1.0)
    assert abs(an.within_image_correlation(indep, groups)) < 0.15


def _synthetic_run(seed=0, n_images=40, claims_per_image=5):
    """Fake pilot: planted signal (owl high = supported) and clustered hallucinations."""
    rng = random.Random(seed)
    ids = list(range(1, 6001))
    m = sp.build_prereg_splits(ids)
    pilot_cal, pilot_eval = m.pilot_cal[: n_images // 2], m.pilot_eval[: n_images // 2]
    manifest = {"pilot_cal": pilot_cal, "pilot_eval": pilot_eval}
    vocab = ChairVocabulary(SYN)
    gt, captions, evidence = {}, [], []
    for iid in pilot_cal + pilot_eval:
        cats = rng.sample(CATS, claims_per_image)
        bad_image = rng.random() < 0.4
        halluc = {c for c in cats if rng.random() < (0.6 if bad_image else 0.1)}
        gt[str(iid)] = sorted(set(cats) - halluc)
        cap = " ".join(f"There is a {c}." for c in cats)
        words = cap.split(" ")
        pieces = [w if k == 0 else " " + w for k, w in enumerate(words)]
        decode = lambda ids_, p=pieces: "".join(p[i] for i in ids_)
        tok_ids = list(range(len(pieces)))
        lps = [-0.05] * len(pieces)
        captions.append({"image_id": iid, "caption": cap, "text_offset": 0, "token_ids": tok_ids,
                         "token_logprobs": lps, "token_spans": token_char_offsets(decode, tok_ids)})
        for c in cats:
            base = rng.uniform(0.0, 0.3) if c in halluc else rng.uniform(0.3, 0.9)
            for s, drop in (("clean", 0), ("blur", 0.1), ("jpeg", 0.05), ("downsample", 0.08)):
                noise = rng.uniform(0, drop) if rng.random() < 0.3 else 0.0
                evidence.append({"image_id": iid, "shift": s, "category": c,
                                 "owl_score": max(0.0, base - noise), "clip_crop": 0.2 + 0.1 * base,
                                 "clip_global": 0.2 + 0.05 * base})
    return vocab, manifest, gt, captions, evidence


def test_end_to_end_analysis_finds_planted_effects(tmp_path):
    vocab, manifest, gt, captions, evidence = _synthetic_run()
    rows, chair = an.build_claims_table(vocab, manifest, gt, captions, evidence)
    assert len(rows) == 200 and all(r.tok_logprob_mean is not None for r in rows)
    res = an.analyse(rows, chair, n_boot=100)
    assert res["descriptives"]["claims_per_image_mean"] == 5
    assert res["signal"]["owl_score"]["auroc"][0] > 0.9
    assert res["learned_unary"]["auroc"][0] > 0.85
    assert res["dependence"]["within_image_corr"][0] > 0.05
    # Root dominance: current robust BP ranks claims exactly like margin on the same posterior.
    assert res["selective"]["spearman_robust_vs_margin_m8"] == pytest.approx(1.0, abs=1e-6)
    assert res["selective"]["robust_m8_current"]["aurc"][0] == pytest.approx(
        res["selective"]["margin_m8_point"]["aurc"][0], abs=1e-9)
    names = [c["check"] for c in res["gate1"]]
    assert names == ["Graph exists", "Coupling is real", "Shifts not all simultaneously extreme", "Signal exists"]
    an.write_report(res, tmp_path / "r.md", {"n_pilot_cal": 20, "n_pilot_eval": 20})
    text = (tmp_path / "r.md").read_text()
    assert "Gate 1 checks" in text and "PASS" in text
    json.dumps(res, default=float)  # serialisable


def test_width_ratio_is_one_without_coupling():
    """Root dominance: with J = 0 and B >= eps, combined-set widths equal box widths."""
    from types import SimpleNamespace
    evl = [SimpleNamespace(category=c) for c in "abcd"]
    theta = np.array([0.3, -0.5, 1.0, 0.1])
    by_img = {1: [0, 1], 2: [2, 3]}
    ge = np.array([1, 1, 2, 2])
    old = an.M8_DEFAULT
    try:
        an.M8_DEFAULT = 0.0
        out = an._width_ratio(evl, theta, by_img, t_box=0.4, t_bud=0.3, ge=ge, n_boot=20)
    finally:
        an.M8_DEFAULT = old
    assert out["ratio"][0] == pytest.approx(1.0, abs=5e-3)            # grid resolution eps/200
    assert out["ratio_certified"][0] >= out["ratio"][0]                 # certified is conservative


def test_cluster_bootstrap_treats_resampled_copies_as_distinct_images():
    """Regression: duplicates must not merge into one image (that inflated the correlation).

    Under independent labels the 95% CI must cover 0 in most datasets and contain its own point.
    """
    covered = 0
    for seed in range(30):
        rng = np.random.default_rng(seed)
        groups = np.repeat(np.arange(60), 4)
        y = rng.integers(0, 2, len(groups))
        point, lo, hi = an.cluster_bootstrap(groups, lambda i, bg: an.within_image_correlation(y[i], bg), 200, seed=seed)
        assert lo - 1e-9 <= point <= hi + 1e-9
        covered += int(lo <= 0 <= hi)
    assert covered >= 25
