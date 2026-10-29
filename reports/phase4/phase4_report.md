# Phase 4 — When do conformal hallucination guarantees survive distribution shift?

Confirmatory analysis under PREREG v2. Calibration: clean COCO Cal-A + Cal-B (LLaVA-1.5-7B, OWLv2), 200 resamples of 80%. Test: locked PREREG test split (500 images) and 804 AMBER images. 95% CIs: image-level cluster bootstrap. Decision: violated if CI lower > target; preserved if CI upper ≤ target + 0.03.

## Hypotheses (target 0.10)

| Hypothesis | Claim-level | Image-level |
|---|---|---|
| H1 | supported (Holm p 0.038) | not supported (Holm p 0.111) |
| H3 | supported (Holm p 0.000) | supported (Holm p 0.000) |
| H4 | supported (Holm p 0.011) | supported (Holm p 0.002) |
| H5 | not supported (Holm p 0.895) | supported (Holm p 0.007) |
| H6 | supported (Holm p 0.000) | supported (Holm p 0.000) |
| H2 (dominance, all S1 conditions) | not supported | |

## Results by condition (target 0.10)

| Condition | Claim risk [95% CI] | Decision | Image risk [95% CI] | Decision | Supported kept | Abstention | Hallucinated share of kept |
|---|---|---|---|---|---|---|---|
| clean | 0.080 [0.057, 0.106] | preserved | 0.100 [0.075, 0.126] | preserved | 0.587 | 0.030 | 0.044 |
| gaussian_blur_1 | 0.095 [0.069, 0.125] | preserved | 0.100 [0.076, 0.125] | preserved | 0.616 | 0.025 | 0.049 |
| gaussian_blur_2 | 0.063 [0.042, 0.087] | preserved | 0.080 [0.057, 0.104] | preserved | 0.594 | 0.030 | 0.035 |
| gaussian_blur_3 | 0.049 [0.030, 0.072] | preserved | 0.056 [0.037, 0.076] | preserved | 0.549 | 0.045 | 0.029 |
| gaussian_blur_4 | 0.055 [0.033, 0.077] | preserved | 0.053 [0.034, 0.073] | preserved | 0.504 | 0.065 | 0.035 |
| gaussian_blur_5 | 0.036 [0.018, 0.054] | preserved | 0.041 [0.024, 0.058] | preserved | 0.408 | 0.155 | 0.029 |
| jpeg_1 | 0.096 [0.066, 0.127] | preserved | 0.096 [0.070, 0.120] | preserved | 0.591 | 0.029 | 0.052 |
| jpeg_2 | 0.067 [0.046, 0.090] | preserved | 0.086 [0.062, 0.108] | preserved | 0.572 | 0.040 | 0.038 |
| jpeg_3 | 0.063 [0.039, 0.087] | preserved | 0.085 [0.062, 0.108] | preserved | 0.563 | 0.043 | 0.036 |
| jpeg_4 | 0.044 [0.026, 0.066] | preserved | 0.063 [0.043, 0.084] | preserved | 0.526 | 0.061 | 0.028 |
| jpeg_5 | 0.039 [0.022, 0.058] | preserved | 0.059 [0.040, 0.080] | preserved | 0.460 | 0.110 | 0.028 |
| pixelate_1 | 0.042 [0.025, 0.062] | preserved | 0.066 [0.045, 0.088] | preserved | 0.540 | 0.048 | 0.026 |
| pixelate_2 | 0.047 [0.028, 0.068] | preserved | 0.051 [0.033, 0.070] | preserved | 0.542 | 0.050 | 0.029 |
| pixelate_3 | 0.032 [0.016, 0.049] | preserved | 0.039 [0.023, 0.055] | preserved | 0.512 | 0.072 | 0.021 |
| pixelate_4 | 0.022 [0.009, 0.037] | preserved | 0.039 [0.023, 0.057] | preserved | 0.413 | 0.153 | 0.018 |
| pixelate_5 | 0.019 [0.007, 0.033] | preserved | 0.026 [0.013, 0.039] | preserved | 0.340 | 0.258 | 0.018 |
| gaussian_noise_1 | 0.092 [0.067, 0.119] | preserved | 0.107 [0.082, 0.133] | inconclusive | 0.615 | 0.035 | 0.048 |
| gaussian_noise_2 | 0.084 [0.058, 0.113] | preserved | 0.089 [0.066, 0.113] | preserved | 0.587 | 0.044 | 0.046 |
| gaussian_noise_3 | 0.067 [0.045, 0.093] | preserved | 0.080 [0.058, 0.105] | preserved | 0.530 | 0.079 | 0.041 |
| gaussian_noise_4 | 0.056 [0.035, 0.079] | preserved | 0.062 [0.043, 0.083] | preserved | 0.402 | 0.181 | 0.045 |
| gaussian_noise_5 | 0.027 [0.012, 0.045] | preserved | 0.027 [0.015, 0.042] | preserved | 0.211 | 0.480 | 0.041 |
| contrast_1 | 0.089 [0.063, 0.118] | preserved | 0.114 [0.088, 0.140] | inconclusive | 0.603 | 0.024 | 0.048 |
| contrast_2 | 0.096 [0.069, 0.128] | preserved | 0.106 [0.082, 0.132] | inconclusive | 0.612 | 0.023 | 0.050 |
| contrast_3 | 0.077 [0.053, 0.103] | preserved | 0.097 [0.073, 0.122] | preserved | 0.599 | 0.026 | 0.042 |
| contrast_4 | 0.061 [0.041, 0.084] | preserved | 0.075 [0.053, 0.098] | preserved | 0.553 | 0.060 | 0.036 |
| contrast_5 | 0.043 [0.024, 0.066] | preserved | 0.048 [0.030, 0.066] | preserved | 0.357 | 0.262 | 0.039 |
| swap_owlvit_v1 | 0.012 [0.003, 0.023] | preserved | 0.019 [0.008, 0.031] | preserved | 0.524 | 0.103 | 0.008 |
| swap_owlvit_v1+recalibrated | 0.030 [0.015, 0.046] | preserved | 0.041 [0.026, 0.059] | preserved | 0.606 | 0.069 | 0.016 |
| swap_grounding_dino | 0.609 [0.558, 0.657] | violated | 0.409 [0.365, 0.450] | violated | 0.970 | 0.000 | 0.175 |
| swap_grounding_dino+recalibrated | 0.087 [0.060, 0.117] | preserved | 0.083 [0.060, 0.107] | preserved | 0.599 | 0.036 | 0.047 |
| vlm_qwen | 0.129 [0.094, 0.165] | inconclusive | 0.093 [0.070, 0.119] | preserved | 0.604 | 0.060 | 0.054 |
| vlm_internvl | 0.160 [0.112, 0.208] | violated | 0.074 [0.053, 0.098] | preserved | 0.602 | 0.048 | 0.042 |
| domain_amber | 0.036 [0.008, 0.074] | preserved | 0.007 [0.002, 0.014] | preserved | 0.683 | 0.152 | 0.005 |
| s5_object_removed | 0.105 [0.079, 0.132] | inconclusive | 0.142 [0.112, 0.174] | violated | – | 0.858 | 1.000 |

## H2: stochastic dominance (P1's condition) per S1 condition

| Condition | D⁺ [95% CI] | Supported | Δ at τ | Share of hallucinated scores that went up |
|---|---|---|---|---|
| gaussian_blur_1 | 0.038 [0.026, 0.081] | False | 0.018 | 0.504 |
| gaussian_blur_2 | 0.003 [0.000, 0.031] | False | -0.018 | 0.377 |
| gaussian_blur_3 | 0.003 [0.000, 0.020] | True | -0.033 | 0.316 |
| gaussian_blur_4 | 0.005 [0.000, 0.019] | True | -0.023 | 0.276 |
| gaussian_blur_5 | 0.000 [0.000, 0.013] | True | -0.043 | 0.238 |
| jpeg_1 | 0.033 [0.028, 0.079] | False | 0.015 | 0.499 |
| jpeg_2 | 0.043 [0.028, 0.083] | False | -0.015 | 0.511 |
| jpeg_3 | 0.041 [0.025, 0.085] | False | -0.018 | 0.471 |
| jpeg_4 | 0.035 [0.017, 0.077] | False | -0.035 | 0.499 |
| jpeg_5 | 0.030 [0.012, 0.084] | False | -0.043 | 0.441 |
| pixelate_1 | 0.003 [0.000, 0.015] | True | -0.041 | 0.253 |
| pixelate_2 | 0.008 [0.003, 0.030] | False | -0.030 | 0.349 |
| pixelate_3 | 0.008 [0.005, 0.033] | False | -0.048 | 0.362 |
| pixelate_4 | 0.008 [0.003, 0.038] | False | -0.061 | 0.367 |
| pixelate_5 | 0.013 [0.002, 0.041] | False | -0.061 | 0.362 |
| gaussian_noise_1 | 0.089 [0.070, 0.137] | False | 0.008 | 0.605 |
| gaussian_noise_2 | 0.078 [0.055, 0.130] | False | 0.003 | 0.597 |
| gaussian_noise_3 | 0.063 [0.039, 0.121] | False | -0.015 | 0.562 |
| gaussian_noise_4 | 0.023 [0.007, 0.070] | False | -0.025 | 0.430 |
| gaussian_noise_5 | 0.000 [0.000, 0.005] | True | -0.051 | 0.291 |
| contrast_1 | 0.041 [0.037, 0.077] | False | 0.008 | 0.552 |
| contrast_2 | 0.035 [0.031, 0.077] | False | 0.015 | 0.537 |
| contrast_3 | 0.023 [0.018, 0.069] | False | -0.005 | 0.481 |
| contrast_4 | 0.000 [0.000, 0.038] | False | -0.018 | 0.430 |
| contrast_5 | 0.003 [0.000, 0.018] | True | -0.035 | 0.306 |

## Secondary targets (claim-level risk, point estimate)

| Condition | δ = 0.05 | δ = 0.20 |
|---|---|---|
| clean | 0.027 | 0.197 |
| gaussian_blur_1 | 0.026 | 0.218 |
| gaussian_blur_2 | 0.015 | 0.151 |
| gaussian_blur_3 | 0.016 | 0.103 |
| gaussian_blur_4 | 0.013 | 0.074 |
| gaussian_blur_5 | 0.011 | 0.063 |
| jpeg_1 | 0.024 | 0.185 |
| jpeg_2 | 0.013 | 0.178 |
| jpeg_3 | 0.010 | 0.176 |
| jpeg_4 | 0.011 | 0.145 |
| jpeg_5 | 0.010 | 0.117 |
| pixelate_1 | 0.011 | 0.115 |
| pixelate_2 | 0.016 | 0.104 |
| pixelate_3 | 0.007 | 0.090 |
| pixelate_4 | 0.006 | 0.084 |
| pixelate_5 | 0.008 | 0.063 |
| gaussian_noise_1 | 0.020 | 0.213 |
| gaussian_noise_2 | 0.025 | 0.180 |
| gaussian_noise_3 | 0.023 | 0.159 |
| gaussian_noise_4 | 0.019 | 0.123 |
| gaussian_noise_5 | 0.006 | 0.050 |
| contrast_1 | 0.020 | 0.232 |
| contrast_2 | 0.018 | 0.214 |
| contrast_3 | 0.021 | 0.189 |
| contrast_4 | 0.016 | 0.140 |
| contrast_5 | 0.016 | 0.094 |
| swap_owlvit_v1 | 0.003 | 0.045 |
| swap_owlvit_v1+recalibrated | 0.005 | 0.086 |
| swap_grounding_dino | 0.240 | 0.874 |
| swap_grounding_dino+recalibrated | 0.048 | 0.169 |
| vlm_qwen | 0.041 | 0.205 |
| vlm_internvl | 0.093 | 0.262 |
| domain_amber | 0.025 | 0.076 |
| s5_object_removed | 0.042 | 0.175 |

## Descriptive: graphical model (Proposition P2) and classification metrics

- Learned chain coupling J (Cal-A, exact likelihood): 0.14; log-likelihood gain over J = 0: 5.45
- Within-image label correlation: cal 0.017, test 0.015

| Data | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| train_cal_a | unary_J0 | 0.820 | 0.753 | 0.502 | 0.602 | 0.859 | 128 | 42 | 641 | 127 |
| train_cal_a | ising_tree_J | 0.823 | 0.795 | 0.471 | 0.591 | 0.861 | 120 | 31 | 652 | 135 |
| train_cal_a | max \|posterior change\| | 0.127 | | | | | | | | |
| test | unary_J0 | 0.842 | 0.784 | 0.514 | 0.621 | 0.870 | 203 | 56 | 1115 | 192 |
| test | ising_tree_J | 0.836 | 0.797 | 0.468 | 0.590 | 0.871 | 185 | 47 | 1124 | 210 |
| test | max \|posterior change\| | 0.123 | | | | | | | | |

- Robust BP intervals (ε = 0.910, B = 2.711 from Cal-A S1 shifts; 150 test images): mean certified width with learned J 0.5063 vs J = 0 0.4774; independent box: 0.5022 vs 0.4724

## Data

```
{
 "counts": {
  "cal": 2179,
  "test": 1566,
  "qwen": 1424,
  "internvl": 1333,
  "amber": 1257,
  "s5": 496
 },
 "amber_counts": {
  "claims": 1568,
  "unknown": 311
 },
 "chair": {
  "llava": {
   "captions": 500,
   "chair_i": 0.14458177637252084,
   "chair_s": 0.466
  },
  "qwen": {
   "captions": 500,
   "chair_i": 0.09079878665318504,
   "chair_s": 0.424
  },
  "internvl": {
   "captions": 500,
   "chair_i": 0.07569444444444444,
   "chair_s": 0.284
  }
 },
 "truncated_captions": {
  "llava": 0,
  "qwen": 47,
  "internvl": 0
 }
}
```
