# Pre-registration v3 (frozen)

Status: frozen. This file replaces `prereg-v3-DRAFT.md`. It is committed and the commit is
tagged `prereg-v3` **before** `scripts/phase5.py analyze` is run on any Phase 5 test data.
Captioning and detection were completed before the tag (allowed by the draft). No conformal
risk, threshold or coverage result on Phase 5 test data has been computed before the tag.
What was seen before the tag is listed under "Information seen before freezing".

Freeze date: ____ (fill in)   Commit: ____ (fill in after tagging)

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
  both calibration and test.
- Labels: CHAIR with the official synonym list; AMBER labels as in Phase 4.
- Filters, targets and estimation as in prereg-v2: claim-level and image-level split conformal,
  delta = alpha = 0.10; 200 calibration resamples (80%); 2,000-sample image-level cluster
  bootstrap; interval decision rule with margin 0.03.
- Software: torch 2.11.0+cu130, transformers 5.20.2, Tesla T4, identical in every session.

### Completed data (verified before freezing)

All six generators captioned the same 4,000 COCO and 804 AMBER images (4,804 captions each,
28,824 in total), and all three detectors scored them. Captions that mention no COCO category
have no claims and therefore no detector rows: 24 to 38 COCO images and 44 to 61 AMBER images
per generator. These images enter the image-level analysis as images with zero claims.

### Truncated captions (rule fixed now)

Captions that reached the 512-token limit (`hit_max_tokens = true`), COCO + AMBER:
Qwen2.5-VL-3B 352 + 14, Qwen3-VL-4B 380 + 22, InternVL3-2B 1 + 0, InstructBLIP 1 + 0,
LLaVA and sampled LLaVA 0. Rule: truncated captions are **kept** in the main analysis (the
cut is part of the generator's deployed behaviour). Every hypothesis is repeated with
truncated captions removed (sensitivity analysis S1); any change of decision is reported.

## Hypotheses (OWLv2 matrix unless stated)

- **H7 (generator shift).** Among off-diagonal claim-level cells, at least one is violated
  after Benjamini-Hochberg control at q = 0.05 over all off-diagonal cells of the three matrices.
- **H8 (warning signal).** The label-free warning (change in the share of all claims above
  the calibrated threshold) predicts claim-level violation across off-diagonal cells with
  AUROC >= 0.80.
- **H9 (prevalence shield).** In every off-diagonal cell where the claim-level filter is
  violated but the image-level filter is preserved, the target's prevalence pi is lower than
  the source's.
- **H10 (controlled prevalence).**
  - *Part 1 (reported as observed, not confirmatory; see "Information seen before freezing").*
    LLaVA with sampling has higher prevalence than greedy LLaVA.
  - *Part 2 (confirmatory).* The image-level risk of the greedy-LLaVA threshold applied to
    sampled LLaVA lies inside the 95% bootstrap interval of the value predicted by Theorem 2
    from the target's prevalence and the source threshold's conditional miss rate, both as
    defined in Theorem 2 and estimated on the target's calibration split.
  - *Exploratory E1 (reported, not a hypothesis).* Whether that image-level risk is higher than
    on greedy LLaVA (one-sided bootstrap, alpha = 0.05), i.e. the draft's original wording.
- **H11 (label budget).** In each violated cell, recalibrating on 170 labelled hallucinated
  claims of the target generator gives P(realised risk > 0.13) <= 0.15 (Theorem 3 predicts 0.10).
- **H12 (detector robustness).** The direction of every H7 violation (claim-level risk above
  0.10) is the same in the OWL-ViT v1 and Grounding DINO matrices.

All hypotheses are evaluated on COCO test. AMBER results are reported as secondary, with the
same rules and no separate hypotheses.

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
   0.423; Qwen3-VL-4B 0.075, 0.523; InternVL3-2B 0.071, 0.309; InstructBLIP ____, ____ (copy
   from the final InstructBLIP run log, or write "not seen"). These are hallucination rates
   only. No threshold, filter, risk, coverage or warning statistic was computed. Because these
   rates reveal H10 part 1, that part is reported as observed and is not counted as a
   confirmed prediction.
3. Caption text and counts were checked for completeness and for broken output (see D1).

## Deviations

| # | Deviation | Reason | Effect |
|---|---|---|---|
| D1 | InstructBLIP is loaded with `InstructBlipForConditionalGeneration` and `InstructBlipProcessor` on one GPU per shard, instead of the generic `AutoModelForImageTextToText` loader split across two GPUs. | With transformers 5.20 the generic route first crashed on a 5-D image tensor and, once patched, produced the same non-text caption ("6666...") for every image. A one-image test showed the dedicated classes on one GPU give correct captions. | All 1,658 broken captions were discarded and InstructBLIP was re-run from zero. Model, weights, quantisation, prompt, decoding and token limit are unchanged. |
| D2 | Per-generator CHAIR rates were seen before freezing. | Printed automatically by the run monitor. | H10 part 1 reported as observed, not confirmatory. |
| D3 | Truncation rule and sensitivity analysis S1 added. | About 9% of Qwen captions reach 512 tokens. | Fixed before any analysis; main analysis unchanged. |
| D4 | H10 part 2 restated as a quantitative test of Theorem 2; the draft wording kept as exploratory E1. | The pilot (inspection allowed by the draft) suggested the draft wording may not hold with OWLv2, and the draft wording does not test the theory the paper proposes. | Decided before the tag and before any Phase 5 risk statistic was computed; E1 is still reported in full. |

Any further deviation is added to this table in the paper with its reason.
