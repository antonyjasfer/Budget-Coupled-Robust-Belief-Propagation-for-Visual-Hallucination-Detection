"""Phase 3 / E1: conformal quantiles, the filters, and the union-bound guarantee."""
import math

import numpy as np
import pytest

from src.phase3 import shift_validity as sv


def test_conformal_upper_index_and_inf():
    s = list(range(1, 20))  # n = 19
    assert sv.conformal_upper(s, 0.9) == 18  # ceil(20*0.9) = 18th smallest
    assert sv.conformal_upper([1.0, 2.0], 0.9) == math.inf
    assert sv.conformal_upper([-math.inf, 0.0, 1.0, 2.0], 0.5) == 1.0  # ceil(5*0.5)=3rd


def _claims(rng, n_img, shift_up_halluc):
    out = []
    for i in range(n_img):
        for _ in range(3):
            h = int(rng.random() < 0.3)
            clean = rng.normal(-1.5 if h else 1.0, 1.0)
            d = rng.normal(shift_up_halluc if h else 0.0, 0.3)
            out.append(sv.Claim(i, h, {"clean": clean, "blur": clean + d, "jpeg": clean + d, "downsample": clean + d}))
    return out


def test_naive_valid_on_clean_and_broken_by_upward_shift_robust_restores():
    rng = np.random.default_rng(0)
    claims = _claims(rng, 400, shift_up_halluc=1.5)
    r = sv.run_splits(claims, n_splits=60, delta=0.1, beta=0.05, seed=1)["results"]
    assert r["cebc_naive"]["clean"]["risk"]["mean"] <= 0.1 + 0.02
    assert r["cebc_naive"]["blur"]["risk"]["mean"] > 0.2          # shift inflates hallucinated scores
    assert r["cebc_robust_directional"]["blur"]["risk"]["mean"] <= 0.15 + 0.02  # delta + beta
    assert r["conflvlm_naive"]["clean"]["risk"]["mean"] <= 0.1 + 0.03
    assert r["conflvlm_robust_pool"]["blur"]["risk"]["mean"] <= 0.15 + 0.03


def test_downward_shift_is_benign():
    rng = np.random.default_rng(2)
    claims = _claims(rng, 400, shift_up_halluc=-1.0)
    r = sv.run_splits(claims, n_splits=40, seed=3)["results"]
    assert r["cebc_naive"]["blur"]["risk"]["mean"] < r["cebc_naive"]["clean"]["risk"]["mean"]


def test_directional_eps_floor_and_keep_rule():
    c = [sv.Claim(0, 1, {"clean": 0.0, "blur": -1.0, "jpeg": -1.0, "downsample": -1.0})] * 30
    assert sv.calibrate_eps_directional(c, 0.05, sv.SHIFTS) == 0.0
    assert sv.keep(1.0, 0.5) and not sv.keep(1.0, 0.5, eps=0.6)


def test_logit_clipped():
    assert math.isfinite(sv.logit(0.0)) and math.isfinite(sv.logit(1.0))
