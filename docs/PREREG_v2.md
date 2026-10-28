# Pre-registration v2: When do conformal hallucination guarantees survive distribution shift?

**Version:** v2 (DRAFT) · **Drafted:** 2026-10-28 · **Supersedes:** `docs/PREREG.md` (v1), whose H1–H4 were tested on the pilot and are retired (see DECISIONS 2026-10-25 to 2026-10-28).
**Freeze:** the author signs Section 10; the commit is tagged `prereg-v2`. Nothing in Section 3's locked test data is computed before the tag.

## 1. Research question

Conformal filters give finite-sample guarantees on the hallucination rate of kept claims under exchangeable data (CEBC, Mishra et al. 2026; ConfLVLM, Li et al. 2025). Deployed systems face shifts: image degradation, a different VLM, a different detector, a different image domain. **Which realistic shifts preserve these guarantees without recalibration, which break them, why, and what is the cheapest valid repair?**

Pilot evidence (exploratory, PREREG train split, `reports/phase3`, `reports/phase3b`) motivated the hypotheses below; the pilot images are excluded from all confirmatory analyses.

## 2. Filters under study (both calibrated on CLEAN source data)

- **Claim-level (CEBC-style):** threshold τ = conformal (1 − δ) quantile of the source evidence score of hallucinated claims; keep a claim iff score > τ. Risk = share of hallucinated claims kept.
- **Image-level (ConfLVLM-style, λ = 0):** τ = conformal (1 − α) quantile over images of the highest score among each image's hallucinated claims; risk = share of images whose kept claims include a hallucination.
- Source pipeline: LLaVA-1.5-7B captions (greedy, prompt "Please describe this image in detail."), CHAIR labels (Phase 1 code), OWLv2 score, logit scale. δ = α = 0.10 (primary); 0.05 and 0.20 secondary.

## 3. Data and splits

- **Calibration:** COCO val2014 PREREG **Cal-A ∪ Cal-B** (700 images), clean.
- **Development:** PREREG train/validation (pilot images included); used freely for debugging only.
- **Confirmatory test:** PREREG **test split (500 images, locked)**, evaluated once after the tag; and **AMBER generative, all 1,004 images** (labels: AMBER `truth` = present, `hallu` = verified absent; other claims excluded and counted).
- Seeds: PREREG splits seed 42; calibration resamples seed 0.

## 4. Shifts (applied to the test data only)

| Family | Conditions |
|---|---|
| S1 Degradation | Gaussian blur, JPEG, downsample, Gaussian noise, contrast; severities 1–5 (ImageNet-C style parameters, fixed in code before the tag) |
| S2 Detector swap | OWLv2 → OWL-ViT v1; OWLv2 → Grounding DINO (base) |
| S3 VLM swap | LLaVA-1.5-7B → Qwen2.5-VL-7B (4-bit) and → InternVL2.5-4B (or the closest model that runs on a T4; recorded before the tag) |
| S4 Domain | COCO → AMBER |
| S5 Semantic stress (exploratory) | inpainting-removed objects; added distractor context (defined in code before the tag; no hypothesis) |

## 5. Hypotheses (each may fail; all are reported)

Decision rule for "violated": the image-level cluster-bootstrap 95% CI (2,000 resamples) of the realised risk lies entirely above the target. "Preserved": the CI upper bound is ≤ target + 0.03.

| ID | Hypothesis | Test |
|---|---|---|
| H1 | **Degradation is safe.** For every S1 condition, both filters are preserved. | Risk per condition, both filters, LLaVA source |
| H2 | **Mechanism.** Under S1, hallucinated claims' scores are stochastically smaller than on clean images (paired), which by Proposition P1 implies validity. | Share of paired score increases < 0.5 and one-sided sign test per condition; P1 stated and proved in `docs/theory/` |
| H3 | **Evidence-source change breaks guarantees.** At least one S2 swap violates the claim-level target without recalibration. | Risk per swap |
| H4 | **Recalibration repairs.** Recomputing τ on the new detector's scores for the same calibration claims preserves the target under S2. | Risk after recalibration |
| H5 | **Cross-VLM transfer.** Thresholds calibrated on LLaVA claims are preserved on each S3 VLM. | Risk per VLM (reported either way) |
| H6 | **Cross-domain transfer.** Preserved on AMBER. | Risk on AMBER |

**Proposition P1 (to be proved before the tag).** If, for hallucinated claims, the shifted score is stochastically no larger than the clean score, then a threshold that is valid on clean data remains valid under the shift. **Proposition P2 (root dominance, credited to Chan & Darwiche 2002, Thm 3.3, for gain ≤ 1).** In the claim tree, coupling cannot change robust bounds when B ≤ ε_root; with the measured coupling (pilot: within-image correlation 0.031), graph-based propagation gives no measurable benefit. Reported descriptively on the test split.

## 6. Metrics

Realised risk (both filters), share of supported claims kept, AUROC of the evidence score, CHAIR_i / CHAIR_s per VLM, paired score-shift statistics. Averages over 200 calibration resamples (80% of Cal-A ∪ Cal-B each), with image-level cluster-bootstrap CIs on the test data.

## 7. Analysis

All confirmatory tests use the locked test split and AMBER once. Holm correction across H1, H3, H4, H5, H6 within each filter. Effect sizes are always reported. Exploratory analyses (S5, severity trends, other δ) are labelled as such.

## 8. Gates

| Gate | Planned date | Decides |
|---|---|---|
| Gate 2 | after the tag | Pipeline frozen and hashed; test evaluation runs once |
| Gate 3 | after the test run | Paper framing given the outcomes of H1–H6 |

## 9. Integrity rules

As in v1: every number comes from a committed script and a committed results file. AI tools may write code and suggest text, never results, conclusions or citations the author has not verified. Negative results are reported.

## 10. Sign-off

| Role | Name | Date | Initials |
|---|---|---|---|
| Author | A. Antony Selva Jasfer | | |
| Supervisor (optional) | | | |
