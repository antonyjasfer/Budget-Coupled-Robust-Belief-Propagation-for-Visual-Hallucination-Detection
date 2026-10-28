# Phase 3 / E1 — Do clean-calibrated hallucination filters stay valid under evidence shift?

**Status: exploratory pilot** (PREREG train split, 200 images; test split untouched). 660 claims, 179 hallucinated; 500 random image-level splits, 50% calibration. Evidence score = logit(OWLv2). Targets: δ = α = 0.1, β = 0.05. Calibration always on CLEAN images; brackets are the 2.5–97.5% range across splits.

## How each shift moves the evidence score

| Shift | Hallucinated: mean change | share that went up | Supported: mean change | share that went up |
|---|---|---|---|---|
| blur | -0.486 | 0.29 | -0.109 | 0.56 |
| jpeg | -0.072 | 0.48 | -0.197 | 0.39 |
| downsample | -0.371 | 0.28 | -0.097 | 0.48 |

## Realised risk and share of supported claims kept

Risk: CEBC rows = share of hallucinated claims that pass; ConfLVLM rows = share of images whose kept set contains a hallucinated claim.

| Method | Test condition | Risk (mean [2.5%, 97.5%]) | Supported claims kept |
|---|---|---|---|
| CEBC-style, clean-calibrated (target δ) | clean | 0.096 [0.020, 0.207] | 0.613 |
| CEBC-style, clean-calibrated (target δ) | blur | 0.070 [0.022, 0.152] | 0.572 |
| CEBC-style, clean-calibrated (target δ) | jpeg | 0.074 [0.032, 0.145] | 0.547 |
| CEBC-style, clean-calibrated (target δ) | downsample | 0.066 [0.022, 0.167] | 0.588 |
| ConfLVLM-style, clean-calibrated (target α) | clean | 0.099 [0.030, 0.180] | 0.672 |
| ConfLVLM-style, clean-calibrated (target α) | blur | 0.074 [0.020, 0.170] | 0.627 |
| ConfLVLM-style, clean-calibrated (target α) | jpeg | 0.087 [0.030, 0.170] | 0.612 |
| ConfLVLM-style, clean-calibrated (target α) | downsample | 0.079 [0.020, 0.160] | 0.636 |
| CEBC + two-sided image radius, all shifts (target δ+β) | clean | 0.000 [0.000, 0.000] | 0.008 |
| CEBC + two-sided image radius, all shifts (target δ+β) | blur | 0.000 [0.000, 0.000] | 0.024 |
| CEBC + two-sided image radius, all shifts (target δ+β) | jpeg | 0.000 [0.000, 0.000] | 0.009 |
| CEBC + two-sided image radius, all shifts (target δ+β) | downsample | 0.000 [0.000, 0.000] | 0.015 |
| ConfLVLM + two-sided image radius, all shifts (target α+β) | clean | 0.000 [0.000, 0.000] | 0.015 |
| ConfLVLM + two-sided image radius, all shifts (target α+β) | blur | 0.000 [0.000, 0.000] | 0.036 |
| ConfLVLM + two-sided image radius, all shifts (target α+β) | jpeg | 0.000 [0.000, 0.000] | 0.012 |
| ConfLVLM + two-sided image radius, all shifts (target α+β) | downsample | 0.000 [0.000, 0.000] | 0.023 |
| CEBC + one-sided shift radius (target δ+β) | clean | 0.001 [0.000, 0.011] | 0.244 |
| CEBC + one-sided shift radius (target δ+β) | blur | 0.004 [0.000, 0.013] | 0.262 |
| CEBC + one-sided shift radius (target δ+β) | jpeg | 0.001 [0.000, 0.011] | 0.202 |
| CEBC + one-sided shift radius (target δ+β) | downsample | 0.002 [0.000, 0.013] | 0.261 |
| CEBC + image radius from the other two shifts | blur | 0.000 [0.000, 0.000] | 0.045 |
| CEBC + image radius from the other two shifts | jpeg | 0.000 [0.000, 0.000] | 0.009 |
| CEBC + image radius from the other two shifts | downsample | 0.000 [0.000, 0.000] | 0.015 |
| ConfLVLM + image radius from the other two shifts | blur | 0.000 [0.000, 0.000] | 0.068 |
| ConfLVLM + image radius from the other two shifts | jpeg | 0.000 [0.000, 0.000] | 0.013 |
| ConfLVLM + image radius from the other two shifts | downsample | 0.000 [0.000, 0.000] | 0.024 |

## Calibrated radii (logit units)

- pool: 2.670 [2.309, 3.033]
- directional: 1.173 [0.949, 1.395]
- loo_blur: 2.326 [2.131, 2.527]
- loo_jpeg: 2.655 [2.298, 2.878]
- loo_downsample: 2.646 [2.309, 2.984]

## Reading guide (mechanical; conclusions are the author's)

- A clean-calibrated method is *broken* by a shift if its mean risk under that shift exceeds its target.
- A robust method is *useful* only if it restores the target where the naive one fails, at an acceptable loss in supported claims kept.
