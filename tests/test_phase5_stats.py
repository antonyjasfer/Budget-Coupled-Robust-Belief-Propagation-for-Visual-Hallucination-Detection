"""Tests for src/phase5/stats.py on synthetic data with known answers."""
import math

import numpy as np
import pytest

from src.phase5 import stats as S


def synth(n_images, p_hall=0.3, shift=0.0, seed=0, max_claims=4):
    """Each image has 1..max_claims claims; hallucinated scores ~ N(-2 + shift, 1),
    supported scores ~ N(1, 1)."""
    rng = np.random.default_rng(seed)
    img, hall, score = [], [], []
    for i in range(n_images):
        for _ in range(rng.integers(1, max_claims + 1)):
            h = rng.random() < p_hall
            img.append(i)
            hall.append(h)
            score.append(rng.normal(-2 + shift, 1) if h else rng.normal(1, 1))
    return S.Table.of(img, hall, score)


def test_table_validates_lengths():
    with pytest.raises(ValueError):
        S.Table.of([1, 2], [True], [0.1, 0.2])


def test_image_max_minus_inf_without_hallucination():
    t = S.Table.of([1, 1, 2], [True, False, False], [0.5, 3.0, 9.0])
    ids, m = S.image_max(t)
    assert ids.tolist() == [1, 2] and m[0] == 0.5 and m[1] == -np.inf


def test_exchangeable_image_risk_is_valid():
    cal, test = synth(700, seed=1), synth(3000, seed=2)
    tc, ti = S.thresholds(cal, 0.10, 0.10, n_resamples=50)
    b = S.bootstrap(S.per_image(test, tc, ti), n_boot=200)
    assert b["image_risk"]["point"] <= 0.10 + 0.02
    assert b["claim_risk"]["point"] <= 0.10 + 0.02


def test_prevalence_identity():
    cal, test = synth(500, seed=3), synth(1000, seed=4)
    tc, ti = S.thresholds(cal, 0.10, 0.10, n_resamples=20)
    b = S.bootstrap(S.per_image(test, tc, ti), n_boot=50)
    assert b["image_risk"]["point"] == pytest.approx(b["prevalence"]["point"] * b["rho"]["point"], rel=1e-12)


def test_local_gap_identity_matches_realised_excess():
    cal, src_test, tgt = synth(600, seed=5), synth(800, seed=6), synth(800, shift=0.6, seed=7)
    tc, ti = S.thresholds(cal, 0.10, 0.10, n_resamples=30)
    r_src = S.bootstrap(S.per_image(src_test, tc, ti), n_boot=20)["claim_risk"]["point"]
    r_tgt = S.bootstrap(S.per_image(tgt, tc, ti), n_boot=20)["claim_risk"]["point"]
    gap = S.local_gap(src_test.score[src_test.hall], tgt.score[tgt.hall], tc)
    assert r_tgt - r_src == pytest.approx(gap, abs=1e-12)
    assert gap > 0


def test_beta_tail_matches_simulation():
    rng = np.random.default_rng(8)
    m, d, e = 120, 0.10, 0.03
    k = math.ceil((m + 1) * (1 - d))
    sims = [1 - np.sort(rng.random(m))[k - 1] for _ in range(20000)]
    assert np.mean(np.array(sims) > d + e) == pytest.approx(S.beta_tail(m, d, e), abs=0.01)


def test_label_budget_is_170():
    assert S.label_budget(0.10, 0.10, 0.03) == 170


def test_recalibration_curve_tracks_theory_for_iid_claims():
    tgt = synth(6000, p_hall=0.3, max_claims=1, seed=9)  # one claim per image: i.i.d.
    rows = S.recalibration_curve(tgt, [50, 200], repeats=150)
    for r in rows:
        assert r["p_over"] == pytest.approx(r["theory_p_over"], abs=0.08)
        assert r["mean_risk"] <= 0.10 + 0.01


def test_crc_controls_expected_image_loss():
    risks = []
    for s in range(30):
        cal, test = synth(300, seed=100 + s), synth(300, seed=200 + s)
        risks.append(S.crc_risk(test, S.crc_threshold(cal, 0.10)))
    assert np.mean(risks) <= 0.10 + 0.01


def test_benjamini_hochberg_known_example():
    p = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
    r = S.benjamini_hochberg(p, q=0.05)
    assert r["a"]["p_bh"] == pytest.approx(0.04)
    assert r["c"]["p_bh"] == pytest.approx(0.0533333, rel=1e-4)
    assert [k for k in p if r[k]["reject"]] == ["a"]


def test_required_claims_matches_power_calculation():
    assert S.required_claims(0.10, 0.13) == 670


def test_transfer_matrix_flags_shifted_generator():
    cal = {"src": synth(700, seed=11), "tgt": synth(700, shift=1.0, seed=12)}
    test = {"src": synth(1500, seed=13), "tgt": synth(1500, shift=1.0, seed=14)}
    m = S.transfer_matrix(cal, test, n_resamples=30, n_boot=300)
    assert m["src"]["tgt"]["claim"]["decision"] == "violated"
    assert m["src"]["tgt"]["local_gap"] > 0.05
    # A single calibration draw can be unlucky (the realised risk has the Beta spread of
    # Theorem 3), so check the same-generator cells on average over calibration draws.
    diag = [S.transfer_cell(synth(700, shift=1.0, seed=50 + s), synth(1500, shift=1.0, seed=80 + s),
                            n_resamples=20, n_boot=50)["claim"]["point"] for s in range(12)]
    assert np.mean(diag) <= 0.10 + 0.01


# ------------------------------------------------------------------ prereg-v3 checks

def test_auroc_known_values():
    assert S.auroc([3, 4], [1, 2]) == 1.0
    assert S.auroc([1, 2], [3, 4]) == 0.0
    assert S.auroc([1], [1]) == 0.5
    assert S.auroc([], [1]) is None


def test_theorem2_check_holds_when_scores_iid_within_image():
    src = synth(800, seed=1)
    tgt_cal = synth(800, shift=0.5, seed=2)
    tgt_test = synth(1500, shift=0.5, seed=3)
    r = S.theorem2_check(src, tgt_cal, tgt_test, n_resamples=20, n_boot=300)
    assert r["holds"]
    assert abs(r["r_pred"] - r["r_test"]) < 0.03
    assert r["lo"] <= r["diff"] <= r["hi"]


def test_theorem2_check_fails_when_target_test_differs():
    src = synth(800, seed=1)
    tgt_cal = synth(800, shift=0.0, seed=2)
    tgt_test = synth(1500, shift=1.5, seed=3)       # test split much worse than calibration split
    r = S.theorem2_check(src, tgt_cal, tgt_test, n_resamples=20, n_boot=300)
    assert not r["holds"] and r["diff"] > 0


def _cell(claim_dec, image_dec="violated", point=0.2, prev=0.5, warn=0.0):
    return {"claim": {"decision": claim_dec, "point": point}, "image": {"decision": image_dec},
            "prevalence": prev, "warning": warn}


def test_hypotheses_decisions():
    g = ["a", "b", "c"]
    owl = {s: {t: _cell("preserved", "preserved", 0.08, prev={"a": 0.5, "b": 0.3, "c": 0.6}[t], warn=-0.01)
               for t in g} for s in g}
    owl["a"]["b"] = _cell("violated", "preserved", 0.18, prev=0.3, warn=0.05)   # violated, image preserved
    other = {s: {t: {"claim": {"point": 0.15}} for t in g} for s in g}
    res = {"gens": g, "matrices": {"owlv2": owl, "owlvit_v1": other, "grounding_dino": other},
           "bh": {"owlv2|a|b": {"reject": True}},
           "recalibration": {"a->b": [{"m": 170, "insufficient": False, "p_over": 0.09}]}}
    h = S.hypotheses(res)
    assert h["H7"]["holds"] and h["H7"]["cells"] == ["a->b"]
    assert h["H8"]["auroc"] == 1.0 and h["H8"]["holds"]
    assert h["H9"]["holds"] and h["H9"]["cells"]["a->b"]["pi_target"] == 0.3
    assert h["H11"]["holds"]
    assert h["H12"]["holds"]


def test_hypotheses_no_violation_gives_undecided_followups():
    g = ["a", "b"]
    owl = {s: {t: _cell("preserved", "preserved", 0.08) for t in g} for s in g}
    h = S.hypotheses({"gens": g, "matrices": {"owlv2": owl}, "bh": {}, "recalibration": {}})
    assert not h["H7"]["holds"]
    assert h["H8"]["holds"] is None and h["H9"]["holds"] is None and h["H11"]["holds"] is None
