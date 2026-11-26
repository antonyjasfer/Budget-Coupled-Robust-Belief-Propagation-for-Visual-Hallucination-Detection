# Pre-registration v3 (frozen)

Status: frozen. This file replaces `prereg-v3-DRAFT.md`. It is committed and the commit is
tagged `prereg-v3` **before** `scripts/phase5.py analyze` is run on any Phase 5 test data.
Captioning and detection were completed before the tag (allowed by the draft). No conformal
risk, threshold or coverage result on Phase 5 test data has been computed before the tag.
What was seen before the tag is listed under "Information seen before freezing".

Freeze date: 2026-11-26   Git tag: `prereg-v3` (the commit it points to is recorded in the
GitHub release note and the README, not here, because a file cannot contain its own commit).

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
  InstructBLIP-Vicuna-7B (4-bit NF4, float16 compute, loaded with its own model and processor
  classes on a single GPU; see Deviation D1). Wave 1 = LLaVA, Qwen2.5-VL-3B, InternVL3-2B;
  wave 2 = the rest.
- Detectors: OWLv2 (primary), OWL-ViT v1, Grounding DINO; each matrix uses one detector for
  both calibration and test. With six generators each matrix has 6 x 6 cells, 30 of them
  off-diagonal.
- Labels: CHAIR with the official synonym list; AMBER labels as in Phase 4.
- Filters, targets and estimation as in prereg-v2: claim-level and image-level split conformal,
  delta = alpha = 0.10; 200 calibration resamples (80%); 2,000-sample image-level cluster
  bootstrap; interval decision rule with margin 0.03 (a cell is **violated** when the lower
  end of the 95% bootstrap interval exceeds the target, **preserved** when the upper end is at
  most target + 0.03, otherwise **inconclusive**).
- Software: torch 2.11.0+cu130, transformers 5.20.2, Tesla T4, identical in every session.
  Analysis code: `scripts/phase5.py` (`stage_analyze`) and `src/phase5/stats.py`
  (`transfer_cell`, `theorem2_check`, `hypotheses`) at the tagged commit.

### Definitions used below

All quantities are over the images of a split that produced at least one claim, as in the
Phase 4 paper. For an image, K is its number of hallucinated claims and M its largest
hallucinated score (M = -inf when K = 0).

- Claim-level risk R_c(t): share of hallucinated claims with score above t.
- Image-level risk R_I(t) = P(M > t): share of images that keep at least one hallucinated claim.
- **Prevalence pi = P(K >= 1)**: share of images with at least one hallucinated claim. This is
  image-level prevalence (CHAIR_s-type), **not** CHAIR_i. It differs from the monitor's CHAIR_s
  only in the denominator (images that produced a claim, not all captions).
- rho(t) = P(M > t | K >= 1), so that R_I(t) = pi rho(t) (Theorem 2).

### Completed data (verified before freezing)

All six generators captioned the same 4,000 COCO and 804 AMBER images (4,804 captions each,
28,824 in total), and all three detectors scored them. Captions that mention no COCO category
produce no claims: 24 to 38 COCO images and 44 to 61 AMBER images per generator. As in Phase 4,
such images enter no table and are outside the denominators defined above; their number is
reported per generator.

### Truncated captions (rule fixed now)

Captions that reached the 512-token limit (`hit_max_tokens = true`), COCO + AMBER:
Qwen2.5-VL-3B 352 + 14, Qwen3-VL-4B 380 + 22, InternVL3-2B 1 + 0, InstructBLIP 1 + 0,
LLaVA and sampled LLaVA 0. Rule: truncated captions are **kept** in the main analysis (the
cut is part of the generator's deployed behaviour). Every hypothesis is repeated with
truncated captions removed (sensitivity analysis S1); any change of decision is reported.

## Hypotheses (OWLv2 matrix, COCO test, unless stated)

- **H7 (generator shift).** For every COCO off-diagonal cell of the three matrices (30 x 3 =
  90 tests) the one-sided bootstrap p-value for H0: R_c <= 0.10 is
  p = (#{b : R_c*(b) <= 0.10} + 1) / (B + 1), B = 2,000. Benjamini-Hochberg is applied at
  q = 0.05 to these 90 p-values together. H7 holds if at least one **OWLv2** off-diagonal cell
  is both BH-rejected and violated by the interval rule. Rejections in the other two matrices
  are reported and feed H12.
- **H8 (warning signal).** The warning of a cell is the **signed** change
  W = mean over the 200 calibrated claim thresholds t of [P_target(score > t) - P_source(score > t)],
  over all claims (no labels). Positive W means more claims pass on the target. Across OWLv2
  off-diagonal cells, AUROC of W for violated (positive class) against preserved (negative
  class) cells; inconclusive cells are left out. H8 holds if AUROC >= 0.80. If either class is
  empty, H8 is reported as not evaluable.
- **H9 (prevalence shield).** In every OWLv2 off-diagonal cell where the claim-level filter is
  violated but the image-level filter is preserved, the target's prevalence pi (on its own
  test split) is lower than the source's. If no such cell exists, H9 is not evaluable.
- **H10 (controlled prevalence), source = greedy LLaVA, target = sampled LLaVA.**
  - *Part 1 (reported as observed, not confirmatory; see "Information seen before freezing").*
    Sampled LLaVA has a higher prevalence pi than greedy LLaVA.
  - *Part 2 (confirmatory, Theorem 2).* With t_I the greedy-LLaVA image-level thresholds (200
    resamples) and, on the **calibration split of sampled LLaVA**, pi, Sbar(t) the share of
    hallucinated claims scoring above t and P(K = j | K >= 1) the distribution of hallucinated
    claims per image, Theorem 2(b) predicts
    R_pred = mean over t in t_I of  pi (1 - sum_{j>=1} P(K = j | K >= 1) (1 - Sbar(t))^j).
    R_test is the observed image-level risk of the same thresholds on the **test split of
    sampled LLaVA**. Calibration and test images are bootstrapped independently (2,000 draws).
    Part 2 holds if 0 lies in the 95% bootstrap interval of R_test - R_pred; |R_test - R_pred|
    is reported with it.
  - *Exploratory E1 (reported, not a hypothesis).* Whether the image-level risk of the
    greedy-LLaVA thresholds is higher on sampled LLaVA than on greedy LLaVA (one-sided
    bootstrap p-value, alpha = 0.05), i.e. the draft's original wording.
- **H11 (label budget).** For the target generator of each violated OWLv2 cell: 200 repeats;
  in each, the target's test images are split at random into two halves, images from the
  first half are taken in random order until 170 hallucinated claims are collected, the
  claim threshold is recalibrated on those claims, and the realised claim-level risk is
  measured on the second half. P(realised risk > 0.13) is estimated as the share of the 200
  repeats above 0.13. H11 holds if this share is <= 0.15 in every such cell (Theorem 3
  predicts 0.10). A cell whose half holds fewer than 170 hallucinated claims is not evaluable.
- **H12 (detector robustness).** For every cell counted by H7, the claim-level point risk is
  above 0.10 in both the OWL-ViT v1 and the Grounding DINO matrices.

AMBER results are reported as secondary, with the same rules and no separate hypotheses, and
are not part of the BH family.

## Sample size

670 hallucinated test claims per generator give 80% power to detect a claim-level risk of
0.13 against 0.10 (one-sided, alpha = 0.05), before the design effect (about 1.2). From the
pilot (100 images per generator), 3,000 test images are expected to give about 1,350 to 3,100
hallucinated claims per generator.

## Human label audit

600 claims (200 per wave-1 generator: 100 hallucinated above the clean threshold, 50 below,
50 supported), drawn by `scripts/phase5.py audit` with seed 2026, labelled independently by
two authors blind to model, score and CHAIR label; Cohen's kappa reported; disagreements
adjudicated by a third author. The main analysis is repeated with audited labels where they
exist; any change of decision is reported.

## Information seen before freezing

1. Pilot runs (`--limit 100`, separate folder), as allowed by the draft.
2. The run monitor printed, for each generator, the CHAIR_i and CHAIR_s rates and the number
   of truncated captions over all 4,000 COCO captions (calibration and test together). Values
   seen: LLaVA CHAIR_i 0.110, CHAIR_s 0.456; sampled LLaVA 0.136, 0.516; Qwen2.5-VL-3B 0.061,
   0.423; Qwen3-VL-4B 0.075, 0.523; InternVL3-2B 0.071, 0.309. InstructBLIP's monitor line
   was not recorded; its rates were computed on 2026-11-26, before the tag, with the monitor's own
   function (`stage_pilot`) in the analysis environment: CHAIR_i 0.098, CHAIR_s 0.412. The same
   computation reproduces LLaVA's CHAIR_i (0.110) and gives CHAIR_s 0.454 against the
   monitor's 0.456 (8 of 4,000 captions; text-processing library version).
   These are hallucination rates only. No threshold, filter, risk, coverage or warning
   statistic was computed. Because these rates reveal H10 part 1, that part is reported as
   observed and is not counted as a confirmed prediction.
3. Caption text and counts were checked for completeness and for broken output (see D1).

## Deviations

| # | Deviation | Reason | Effect |
|---|---|---|---|
| D1 | InstructBLIP is loaded with `InstructBlipForConditionalGeneration` and `InstructBlipProcessor` on one GPU per shard, instead of the generic `AutoModelForImageTextToText` loader split across two GPUs. | With transformers 5.20 the generic route first crashed on a 5-D image tensor and, once patched, produced the same non-text caption ("6666...") for every image. A one-image test showed the dedicated classes on one GPU give correct captions. | All 1,658 broken captions were discarded and InstructBLIP was re-run from zero. Model, weights, quantisation, prompt, decoding and token limit are unchanged. |
| D2 | Per-generator CHAIR rates were seen before freezing. | Printed automatically by the run monitor. | H10 part 1 reported as observed, not confirmatory. |
| D3 | Truncation rule and sensitivity analysis S1 added. | About 9% of Qwen captions reach 512 tokens. | Fixed before any analysis; main analysis unchanged. |
| D4 | H10 part 2 restated as a quantitative test of Theorem 2; the draft wording kept as exploratory E1. | By Theorem 2, R_I = pi rho. A higher prevalence pi raises image-level risk only if the conditional survival rho does not fall enough to offset it, so the draft's monotone statement is a special case that the theory does not imply. The restated part 2 tests the theorem's own prediction of the combined effect. The pilot (inspection allowed by the draft) also suggested that the special case may not hold with OWLv2. | Decided before the tag and before any Phase 5 risk statistic was computed; E1 is still reported in full. |
| D5 | Analysis code completed before the tag: the BH family is limited to the 90 COCO off-diagonal cells (AMBER cells had been included), m = 170 added to the recalibration grid, and H10 part 2, E1 and the H7 to H12 decisions computed by `hypotheses()`. | Make the code compute exactly the tests stated here. | Code fixed at the tagged commit; no result had been computed. |

Any further deviation is added to this table in the paper with its reason.
