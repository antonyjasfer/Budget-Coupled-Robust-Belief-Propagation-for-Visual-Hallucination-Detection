# Pre-registration: Calibrated Robust Claim Inference for VLM Hallucination Detection

**Version:** v1 · **Drafted:** 2026-10-25 · **Freeze:** tag this file's commit as `prereg-v1` after all four authors sign below.
Any later change goes in a new version (`prereg-v2`, …) with a dated entry in `docs/DECISIONS.md` explaining why. Changes made after looking at test-split results must be reported in the paper as deviations.

## 1. Research question

When the visual evidence behind a VLM's object-existence claims shifts under realistic conditions (blur, compression, low resolution, a different detector, a different crop or prompt), can we compute per-claim posterior intervals that (a) contain the shifted posterior with probability at least 1 − α, and (b) are narrower than simpler alternatives at the same coverage, and do those intervals support better abstention decisions?

## 2. Hypotheses (each may fail; all are reported either way)

| ID | Hypothesis | Primary test | Supported if |
|---|---|---|---|
| H1 Validity | With ε and B set by split-conformal calibration on measured evidence shifts, empirical coverage of shifted posteriors on held-out images is ≥ 1 − α | Coverage at α ∈ {0.05, 0.10, 0.20}, per in-family shift type, 5 random cal/test re-splits | Mean coverage ≥ 1 − α − 0.02 for every α and in-family shift type |
| H2 Efficiency | At equal coverage, the box + global ℓ1-budget set gives narrower intervals than the independent box set | Paired difference in mean width, image-level cluster bootstrap | 95% CI of the relative width reduction excludes 0 and its point estimate is ≥ 10% |
| H3 Decisions | Abstaining when L < τ < U yields lower risk at fixed coverage than (a) margin abstention on the point posterior and (b) a conformal claim filter | AURC and hallucination rate among kept claims at 80% and 90% coverage | Paired bootstrap CI of the AURC difference excludes 0 vs both baselines |
| H4 Structure | Learned claim coupling (J ≠ 0) improves detection on images with ≥ 3 claims vs J = 0 | AUROC and ECE, paired by image | CI of the AUROC difference excludes 0 |

Exploratory (not confirmatory): which graph topology works best; regime analysis (fraction of claims with B > ε_r; see `docs/theory/root_dominance.md`); subgroup results by object size and claims per image.

## 3. Data and splits

- **Main:** COCO val2014 images (CHAIR protocol). Claims: object-existence mentions over the 80 COCO categories plus CHAIR synonyms. Labels: COCO instance + caption annotations (CHAIR rule).
- **Splits (by image, seed 42, fixed before any model fitting):** train 1,500 · validation 300 · Cal-A 300 (shape of the uncertainty set) · Cal-B 400 (conformal scale) · **test 500 (locked)**.
- **Cross-dataset:** AMBER generative split; POPE (random / popular / adversarial) as secondary.
- **Human audit:** 400–600 claims from the test split, two independent annotators blind to model outputs, adjudication by a third; Cohen's κ reported. UNKNOWN is excluded from binary metrics and counted.
- **VLMs:** LLaVA-1.5-7B (primary), plus Qwen2.5-VL-7B and one more model. Greedy decoding; prompt "Please describe this image in detail."
- COCO annotations are never used as model features.

**Test-split lock:** no metric is computed on the test split until the model, calibration procedure, thresholds and all hyperparameters are frozen and hashed (Gate 3, planned 2027-01-10). The test split is evaluated once.

## 4. Methods compared

Proposed: learned tree CRF + calibrated box + ℓ1-budget uncertainty set + robust BP abstention.
Baselines: detector threshold; learned unary model with J = 0; standard tree BP; margin abstention on the point posterior; independent box intervals at equal coverage; conformal claim filter (ConfLVLM-style); next-token-probability feature classifier; detector-ensemble disagreement.

## 5. Shift families (for calibration and coverage)

Image corruptions (Gaussian blur, JPEG, Gaussian noise, downsample, occlusion; severities 1–5), evidence-source shift (OWLv2 ↔ Grounding DINO; CLIP ↔ SigLIP), view shift (crops / rescales keeping the object), prompt shift (3–5 templates). Leave-one-family-out coverage is reported as a stress test, not as part of H1.

## 6. Metrics

AUROC, AUPRC, F1, accuracy, ECE, Brier; empirical coverage; mean / median interval width; risk–coverage curves, AURC, hallucination rate at 80% / 90% coverage; runtime; certificate gap G per interval.

## 7. Statistical analysis

- 95% CIs by image-level cluster bootstrap (2,000 resamples).
- Paired comparisons on identical images; Holm correction across the confirmatory comparisons in Section 2.
- Conformal results averaged over 5 random Cal-B / test re-splits (mean ± s.d.).
- Effect sizes are always reported alongside significance.

## 8. Decision gates

| Gate | Date (planned) | Decides |
|---|---|---|
| Gate 1 | 2026-11-08 | Claims per image (≥ 4), within-image label dependence, shift structure, signal exists → scope of H2/H4 |
| Gate 2 | 2026-12-13 | H1 holds on validation-side data; H2 plausible; share of claims with B > ε_r |
| Gate 3 | 2027-01-10 | Freeze everything, evaluate test split once, choose paper framing |

## 9. Integrity rules

Every reported number is produced by a committed script from a committed results file. AI tools may write code and suggest text, never results, conclusions or citations that the authors have not verified. Negative results for H1–H4 are reported.

## 10. Sign-off

| Author | Date | Signature (initials) |
|---|---|---|
| A. Antony Selva Jasfer | | |
| Nishanth | | |
| Priyan U | | |
| Akash Narayanan C | | |
