# Pre-registration v3 (DRAFT, not yet frozen)

Status: draft for review by all authors. Freeze by renaming to `prereg-v3.md`, committing,
and tagging the commit `prereg-v3` **before** `scripts/phase5.py analyze` is run on any
Phase 5 test data. Captioning and detection may start before the tag; looking at test
results may not. Pilot runs (`--limit 100`, separate folder) may be inspected.

## Question

When a conformal hallucination filter calibrated on one caption generator is used behind
another, when is its guarantee preserved, and does the threshold-local and prevalence
account (Theorems 1 and 2 of the Phase 4 journal paper) predict which cells break?

## Design

- Images: COCO val2014, 1,000 calibration + 3,000 test, drawn with seed 2027 from images not
  used in Phases 1 to 4 (`scripts/phase5.py setup`); AMBER generative set, 804 images, as a
  secondary domain.
- Generators (greedy unless noted, prompt "Please describe this image in detail.", at most 512
  new tokens): LLaVA-1.5-7B (4-bit, pinned revision, as in Phase 4), LLaVA-1.5-7B with nucleus
  sampling (T = 1.0, top-p 0.9, seed = image id), Qwen2.5-VL-3B, Qwen3-VL-4B, InternVL3-2B,
  InstructBLIP-Vicuna-7B (4-bit). Wave 1 = LLaVA, Qwen2.5-VL-3B, InternVL3-2B; wave 2 = the rest.
- Detectors: OWLv2 (primary), OWL-ViT v1, Grounding DINO; each matrix uses one detector for
  both calibration and test.
- Labels: CHAIR with the official synonym list; AMBER labels as in Phase 4.
- Filters, targets and estimation as in prereg-v2: claim-level and image-level split conformal,
  delta = alpha = 0.10; 200 calibration resamples (80%); 2,000-sample image-level cluster
  bootstrap; interval decision rule with margin 0.03.

## Hypotheses (OWLv2 matrix unless stated)

- **H7 (generator shift).** Among off-diagonal claim-level cells, at least one is violated
  after Benjamini-Hochberg control at q = 0.05 over all off-diagonal cells of the three matrices.
- **H8 (warning signal).** The label-free warning (change in the share of all claims above
  the calibrated threshold) predicts claim-level violation across off-diagonal cells with
  AUROC >= 0.80.
- **H9 (prevalence shield).** In every off-diagonal cell where the claim-level filter is
  violated but the image-level filter is preserved, the target's prevalence pi is lower than
  the source's.
- **H10 (controlled prevalence).** LLaVA with sampling has higher prevalence than greedy LLaVA,
  and the image-level risk of the greedy-LLaVA threshold applied to sampled LLaVA is higher
  than on greedy LLaVA (one-sided bootstrap, alpha = 0.05).
- **H11 (label budget).** In each violated cell, recalibrating on 170 labelled hallucinated
  claims of the target generator gives P(realised risk > 0.13) <= 0.15 (Theorem 3 predicts 0.10).
- **H12 (detector robustness).** The direction of every H7 violation (claim-level risk above
  0.10) is the same in the OWL-ViT v1 and Grounding DINO matrices.

## Sample size

670 hallucinated test claims per generator give 80% power to detect a claim-level risk of
0.13 against 0.10 (one-sided, alpha = 0.05), before the design effect (about 1.2). 3,000 test
images give about 1,100 (InternVL3-2B) to 2,400 (LLaVA-1.5-7B) hallucinated claims per generator at Phase 4 rates.

## Human label audit

600 claims (200 per wave-1 generator: 100 hallucinated above the clean threshold, 50 below,
50 supported), drawn by `scripts/phase5.py audit` with seed 2026, labelled independently by
two authors blind to model, score and CHAIR label; Cohen's kappa reported; disagreements
adjudicated by a third author. The main analysis is repeated with audited labels where they
exist; any change of decision is reported.

## Deviations

Any deviation is listed in the paper's deviations table with its reason.
