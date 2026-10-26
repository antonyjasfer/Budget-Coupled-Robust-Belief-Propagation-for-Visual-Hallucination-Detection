# Phase 1 — Gate 1 Report (falsification pilot)

Generated from result files. Images: pilot_cal 100 / pilot_eval 100 (train split only; test split untouched). Bootstrap: 2000 image-level resamples, 95% CI.

## Gate 1 checks

| Check | Rule | Value | Result | If it fails |
|---|---|---|---|---|
| Graph exists | mean claims per image >= 4 | 3.30 | FAIL | Change prompt/dataset; drop H4 if still < 3 |
| Coupling is real | within-image label correlation: 95% CI lower bound > 0 | 0.031 [-0.031, 0.091], perm p=0.1314 | FAIL | Keep J = 0 as the main model; graph becomes an ablation |
| Shifts not all simultaneously extreme | q90 budget/box ratio: 95% CI upper bound < 0.9 | 0.518 [0.482, 0.601] | PASS | H2 likely false; reposition around H1 + H3 |
| Signal exists | learned unary AUROC on pilot_eval: 95% CI lower bound > 0.6 | 0.887 [0.842, 0.924] | PASS | Fix evidence features before anything else |

## Descriptives

- Captions: 200; CHAIR_i = 0.164; CHAIR_s = 0.500
- Claims: 660 on 200 images; per image mean 3.30, median 3.0; images with >= 3 claims 0.60
- Claim hallucination rate: 0.271 [0.234, 0.307]
- Claims without a matched token span: 0

## Within-image dependence

- Intra-image label correlation: 0.031 [-0.031, 0.091]; permutation p = 0.1314

## Evidence signal (AUROC for predicting hallucination; higher support = lower risk)

| Feature | AUROC [95% CI] |
|---|---|
| owl_score | 0.884 [0.856, 0.911] |
| clip_crop | 0.744 [0.701, 0.785] |
| clip_global | 0.747 [0.704, 0.788] |
| tok_logprob_mean | 0.631 [0.581, 0.678] |

- Raw OWLv2 score used as P(supported) (current M8 θ mapping): ECE = 0.388

## Learned unary model (fit on pilot_cal, evaluated on pilot_eval)

- AUROC 0.887 [0.842, 0.924]; ECE 0.071; Brier 0.118 (n_cal=332, n_eval=328)

## Selective prediction on pilot_eval (lower AURC is better)

| Method | AURC [95% CI] | Risk @ 80% coverage | Risk @ 90% coverage |
|---|---|---|---|
| margin_learned_J0 | 0.053 [0.035, 0.074] | 0.111 | 0.139 |
| margin_m8_point | 0.490 [0.406, 0.582] | 0.615 | 0.620 |
| robust_m8_current | 0.490 [0.406, 0.582] | 0.615 | 0.620 |
| box_m8 | 0.465 [0.387, 0.548] | 0.607 | 0.624 |

- Spearman rank agreement, current robust BP vs margin on the same posterior: 1.0000 (root dominance predicts 1.0).

## Evidence shifts (H2 early read; δ in θ of the learned unary model)

| Shift | median abs(δ) | q90 abs(δ) | q90 budget/box ratio [95% CI] |
|---|---|---|---|
| blur | 0.344 | 1.057 | 0.584 [0.458, 0.655] |
| jpeg | 0.277 | 0.961 | 0.539 [0.443, 0.708] |
| downsample | 0.219 | 0.803 | 0.538 [0.446, 0.627] |

- Pooled ratio 0.518 [0.482, 0.601]; share of images where calibrated B > ε (graph can matter, T3): 0.87
- **Direct H2 read:** mean interval width, calibrated combined set / box set (learned θ, chain J = 0.549): grid 0.920 [0.899, 0.938], certified (conservative) 0.925 [0.905, 0.943] (box 0.653 vs combined 0.601). With J = 0 this ratio is exactly 1 (root dominance), so any gain comes from claim coupling.

Interpretation and the Gate 1 decision are recorded by the authors in docs/DECISIONS.md.
