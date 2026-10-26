# Phase 1 results — falsification pilot (Gate 1)

Run on Kaggle (T4), 2026-10-26, code at commit `2935925` (`environment.txt`): torch 2.11.0, transformers 5.16.1, LLaVA-1.5-7B revision `b234b80`.
200 COCO val2014 images from the PREREG train split (100 pilot_cal / 100 pilot_eval); the locked test split was not touched.
Re-running `src/phase1/analysis.py` on these files reproduces `gate1_report.md` exactly.

| File | Content |
|---|---|
| `gate1_report.md` | Gate 1 checks and statistics (95% image-level bootstrap CIs) |
| `gate1_results.json` | All numbers behind the report |
| `captions.jsonl` | 200 LLaVA captions with per-token log-probs |
| `evidence.jsonl` | OWLv2 / CLIP evidence for 660 claims × 4 image conditions |
| `claims_table.jsonl` | Claim-level table (CHAIR label, token features, evidence) |
| `gt_objects.json` | CHAIR ground-truth objects per image |
| `manifest.json` | PREREG splits and pilot image ids (seed 42) |

## Gate 1 at a glance

| Check | Result | Value |
|---|---|---|
| Graph exists (mean claims/image ≥ 4) | FAIL | 3.30 |
| Coupling is real (within-image correlation CI > 0) | FAIL | 0.031 [−0.031, 0.091] |
| Shifts not all simultaneously extreme | PASS | 0.518 [0.482, 0.601] |
| Signal exists (learned unary AUROC CI > 0.6) | PASS | 0.887 [0.842, 0.924] |

Known label caveat: CHAIR does not handle negation (e.g. "not carrying any passengers" counts as a person mention). The Phase 6 human audit measures this label noise.
