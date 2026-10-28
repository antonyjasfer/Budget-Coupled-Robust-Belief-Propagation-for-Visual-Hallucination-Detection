# Pre-registration v2: When do conformal hallucination guarantees survive distribution shift?

**Version:** v2 (frozen at tag `prereg-v2`) · **Drafted:** 2026-10-28 · **Supersedes:** `docs/PREREG.md` (v1), whose H1–H4 were tested on the pilot and are retired (see DECISIONS 2026-10-25 to 2026-10-28).
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
- **Confirmatory test:** PREREG **test split (500 images, locked)**, evaluated once after the tag; and **AMBER generative, 804 images** = all 1,004 minus the 200 images already used in the Phase 3b pilot (`reports/phase3b/amber/selected_ids.json`, excluded by code). Labels: AMBER `truth` = present, `hallu` = verified absent; other claims excluded and counted.
- **Pilot exclusion:** the 200 COCO pilot images are in the PREREG train split and cannot enter the test split (`assert_pilot_isolated`); the 200 AMBER pilot images are excluded as above.
- Seeds: PREREG splits seed 42; calibration resamples seed 0.

## 4. Shifts (applied to the test data only)

| Family | Conditions |
|---|---|
| S1 Degradation | Gaussian blur, JPEG, downsample, Gaussian noise, contrast; severities 1–5 (ImageNet-C style parameters, fixed in code before the tag) |
| S2 Detector swap | OWLv2 → OWL-ViT v1; OWLv2 → Grounding DINO (base) |
| S3 VLM swap | LLaVA-1.5-7B → Qwen2.5-VL-3B-Instruct (fp16) and → InternVL3-2B-hf (fp16); greedy, same prompt, ≤ 512 new tokens (see Section 9a) |
| S4 Domain | COCO → AMBER |
| S5 Semantic stress (exploratory) | inpainting-removed objects; added distractor context (defined in code before the tag; no hypothesis) |

## 5. Hypotheses (each may fail; all are reported)

Decision rule for "violated": the image-level cluster-bootstrap 95% CI (2,000 resamples) of the realised risk lies entirely above the target. "Preserved": the CI upper bound is ≤ target + 0.03.

| ID | Hypothesis | Test |
|---|---|---|
| H1 | **Degradation is safe.** For every S1 condition, both filters are preserved. Supported by the data only if the share of supported claims kept and the abstention rate are reported for the same conditions (a filter that keeps nothing is trivially "safe"). | Risk, share of supported claims kept and abstention rate per condition, both filters, LLaVA source |
| H2 | **Mechanism.** Under S1, hallucinated claims' scores are stochastically no larger than on clean images (P1's condition), which by Proposition P1 implies validity. | Per condition: D⁺ = max over t of [P(S̃ > t) − P(S > t)] on hallucinated test claims (one-sided KS statistic, paired cluster bootstrap); H2 supported if the 95% CI upper bound of D⁺ is ≤ 0.03. Also reported: the same difference at the calibrated τ, and the share of paired increases. P1 proved in `docs/theory/` |
| H3 | **Evidence-source change breaks guarantees.** At least one S2 swap violates the claim-level target without recalibration. | Risk per swap |
| H4 | **Recalibration repairs.** Recomputing τ on the new detector's scores for the same calibration claims preserves the target under S2. | Risk after recalibration |
| H5 | **Cross-VLM transfer.** Thresholds calibrated on LLaVA claims are preserved on each S3 VLM. | Risk per VLM (reported either way) |
| H6 | **Cross-domain transfer.** Preserved on AMBER. | Risk on AMBER |

**Proposition P1 (to be proved before the tag).** Both guaranteed quantities are *unconditional* false-acceptance probabilities, not precision among kept claims.
(a) Claim level: if the score S̃ of a hallucinated claim under the shift is stochastically no larger than its clean score S (P(S̃ > t) ≤ P(S > t) for all t), then P(S̃ > τ) ≤ P(S > τ) ≤ δ, so the clean guarantee is preserved.
(b) Image level: the same holds with S replaced by M = the largest score among an image's hallucinated claims (M̃ stochastically ≤ M).
No condition on supported claims is needed. The precision among kept claims (share of kept claims that are hallucinated) is **not** covered by P1: it can rise if supported claims lose more score than hallucinated ones. It is reported as a secondary metric, not a guarantee. **Proposition P2 (root dominance, credited to Chan & Darwiche 2002, Thm 3.3, for gain ≤ 1).** In the claim tree, coupling cannot change robust bounds when B ≤ ε_root; with the measured coupling (pilot: within-image correlation 0.031), graph-based propagation gives no measurable benefit. Reported descriptively on the test split.

## 6. Metrics

Realised risk (both filters), share of supported claims kept, abstention rate (images with no kept claim), precision among kept claims (not guaranteed; see P1), AURC of the evidence score, AUROC, CHAIR_i / CHAIR_s per VLM, paired score-shift statistics (for P1's condition).

**Graphical model (descriptive, for P2 and the PGM course deliverables).** On calibration and test data: (i) unary model with J = 0; (ii) learned attractive Ising tree over each image's claims (Chow–Liu tree, J fitted on Cal-A) with exact BP; (iii) robust BP intervals with the calibrated (ε, B). Reported: within-image label correlation, learned J, difference in AUROC and interval width between (ii)/(iii) and (i), and for each model accuracy, precision, recall, F1, ROC-AUC and confusion matrices (calibration and test, decision threshold 0.5). These are descriptive and are not hypotheses. Averages over 200 calibration resamples (80% of Cal-A ∪ Cal-B each), with image-level cluster-bootstrap CIs on the test data.

## 7. Analysis

All confirmatory tests use the locked test split and AMBER once. Holm correction across H1, H3, H4, H5, H6 within each filter. Effect sizes are always reported. Exploratory analyses (S5, severity trends, other δ) are labelled as such.

## 7a. Implementation (frozen at the tag)

- Code: `scripts/phase4.py`, `src/phase4/{shifts,gpu,analysis}.py`, notebook `notebooks/phase4_confirmatory.ipynb`; tests `tests/test_phase4.py`.
- S1 parameters: `src/phase4/shifts.py` (ImageNet-C severities; noise seeded by image id and severity). S1 is also computed on Cal-A, only to calibrate (ε, B) for the descriptive robust-BP intervals.
- S5: one supported LLaVA claim per test image whose category has COCO instance boxes; all its boxes (+10%) are filled with the image's mean colour; the claim is labelled hallucinated. Risk = share of removed objects still kept by the clean-calibrated filter.
- Calibration resamples: 200 × 80% of Cal-A ∪ Cal-B images, seed 0; bootstrap: 2,000 image-level resamples, seed 1 (H2: seed 2).
- Descriptive PGM: logistic unary model on the OWLv2 logit fitted on Cal-A; one shared chain coupling J ∈ [0, 2] (step 0.01) chosen by exact likelihood on Cal-A (claims in caption order); exact BP. Robust intervals on the first 150 test images (sorted ids), ε and B = 0.9 quantiles of Cal-A S1 changes in θ.

## 8. Gates

| Gate | Planned date | Decides |
|---|---|---|
| Gate 2 | after the tag | Pipeline frozen and hashed; test evaluation runs once |
| Gate 3 | after the test run | Paper framing given the outcomes of H1–H6 |

## 9. Integrity rules

As in v1: every number comes from a committed script and a committed results file. AI tools may write code and suggest text, never results, conclusions or citations the author has not verified. Negative results are reported.

## 9a. Changes made before the tag (no test data had been generated)

| Change | Reason |
|---|---|
| Qwen2.5-VL-7B (4-bit) → Qwen2.5-VL-3B (fp16) | 7B in 4-bit on a T4 takes about 3× longer per caption; 3B already ran in the pilot. |
| InternVL2.5-4B → InternVL3-2B-hf | HF-native implementation (no remote code) that runs on transformers 5.x; fits a T4 in fp16. |
| VLM caption limit 384 → 512 tokens | 44 / 200 pilot Qwen captions hit 384 tokens. |

## 10. Sign-off

| Role | Name | Date | Initials |
|---|---|---|---|
| Author | A. Antony Selva Jasfer | 2026-10-28 | approved in writing ("PREREG v2 approved"); tag `prereg-v2` applied once P1 proof and Phase 4 code (shift parameters, models, S5) are committed |
| Supervisor (optional) | | | |
