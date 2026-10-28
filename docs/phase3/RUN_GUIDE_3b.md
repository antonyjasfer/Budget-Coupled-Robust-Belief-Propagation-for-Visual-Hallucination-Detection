# Phase 3b — How to run (Kaggle)

**Goal:** `phase3b_report.md`, which tests whether hallucination filters calibrated on clean COCO + LLaVA + OWLv2 (Phase 1 data) stay valid under three shifts that can *raise* the evidence for absent objects:

| Arm | Shift | Data | Approx. GPU time (T4) |
|---|---|---|---|
| A | Detector swap: OWLv2 → OWL-ViT v1, OWLv2 → Grounding DINO | 200 Phase 1 COCO pilot images, Phase 1 LLaVA captions | 15–20 min |
| B | VLM swap: LLaVA-1.5-7B → Qwen2.5-VL-3B | same 200 COCO images | ~1–1.5 h |
| C | Domain shift: COCO → AMBER (non-COCO photos) | 200 AMBER generative images (fixed seed 42) | ~1–1.5 h |

The PREREG test split is never used. The Phase 1 manifest, ground truth and captions are read from the repository (`reports/phase1/`), so Phase 3b uses exactly the Phase 1 claims.

## Steps

1. Kaggle → **New Notebook** → File → **Import Notebook** → upload `notebooks/phase3b_shifts.ipynb` (from the repo, branch `phase3b-shifts`).
2. Right panel → Settings: **Accelerator: GPU T4 x2**, **Internet: ON**.
3. Run cells **1 to 4** one by one (smoke test, about 10–15 minutes). Cell 4 should print one swap-detector row, one Qwen caption and one AMBER caption.
4. If cell 4 worked: top right **Save Version** → **Save & Run All (Commit)** → Save. The full run continues in the background; you can close the browser.
5. When the version finishes (Kaggle emails you, or check *Your Work*), open it → **Output** → download `BCRBP_phase3b/phase3b_results.zip`.
6. Put the zip in Drive → `BCRBP_phase2` (or attach it in the chat).

## If something goes wrong

| Problem | Fix |
|---|---|
| Cell 4: `Could not download the AMBER images` | Open https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view, download `AMBER.zip`. In Kaggle: right panel → **Add Input** → **Upload** → New Dataset → select the zip → Create. Run cell 4 again; the script finds the zip under `/kaggle/input`. |
| `CUDA out of memory` in the Qwen step | In cell 5 change the qwen line to `!python scripts/phase3b.py qwen --out "$OUT" --qwen-4bit`. |
| Any other error in cell 4 | Copy the last 30 lines of the error into the chat. Nothing is lost; finished work is skipped on re-run. |
| Session stopped in the middle | Run all again. Each stage resumes. |

## Outputs

| File | Content |
|---|---|
| `swap_evidence.jsonl` | OWL-ViT v1 and Grounding DINO scores for every Phase 1 claim (clean images) |
| `qwen_captions.jsonl`, `qwen_evidence.jsonl` | Qwen2.5-VL-3B captions and OWLv2 evidence |
| `amber/selected_ids.json`, `amber_captions.jsonl`, `amber_evidence.jsonl`, `amber_claims.jsonl` | AMBER images used, LLaVA captions, OWLv2 evidence, labelled claims |
| `phase3b_results.json`, `phase3b_report.md` | Risk of each clean-calibrated filter under each shift (500 random splits) |
