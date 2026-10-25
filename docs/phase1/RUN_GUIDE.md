# Phase 1 — How to run the falsification pilot

**Goal:** the Gate 1 report (`gate1_report.md`) on 200 COCO val2014 images from the PREREG *train* split. The locked test split is never touched (`assert_pilot_isolated`).

## Option A — Google Colab (recommended)

1. Open `notebooks/phase1_pilot.ipynb` in Colab (File → Open notebook → GitHub tab → paste the repo URL).
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Runtime → **Run all**. Allow Google Drive access when asked.
4. Check cell 4 (smoke test on 4 images, ~5 min). If it prints a caption, the full run will work.
5. Cells 5–7 run the full pilot (~1–1.5 h), the analysis, and save `MyDrive/BCRBP_phase1/phase1_results.zip`.
6. If Colab disconnects: reconnect and **Run all** again. Finished work is skipped.

## Option B — Kaggle

1. New Notebook → File → Import Notebook → upload `notebooks/phase1_pilot.ipynb`.
2. Settings: Accelerator **GPU T4 x2**, **Internet ON** (requires a phone-verified account).
3. Run all. Download `BCRBP_phase1/phase1_results.zip` from the Output panel.

## Option C — command line (any GPU machine)

```bash
pip install -e . inflect bitsandbytes accelerate torch transformers pillow
python scripts/phase1_pilot.py all --out OUT_DIR
```

## What each stage produces

| Stage | Output | Notes |
|---|---|---|
| fetch | `annotations/`, `synonyms.txt` | COCO 2014 annotations; CHAIR synonyms checked by SHA-256 |
| prepare | `manifest.json`, `gt_objects.json` | PREREG splits (seed 42); CHAIR ground truth for pilot images |
| images | `images/` | 200 val2014 JPEGs |
| captions | `captions.jsonl` | LLaVA-1.5-7B (4-bit, greedy, ≤512 tokens) + per-token log-probs |
| evidence | `evidence.jsonl` | OWLv2 score/box, CLIP crop + global similarity; clean, blur (r=3), JPEG (q=10), 4× downsample |
| analyze | `claims_table.jsonl`, `gate1_results.json`, `gate1_report.md` | Gate 1 statistics with image-level bootstrap CIs |

## After the run

Send `phase1_results.zip` (Drive or chat). The Gate 1 decision is then recorded in `docs/DECISIONS.md` and the plan doc.

## Troubleshooting

- *"NO GPU FOUND"*: the runtime is CPU-only; change it to T4 and restart.
- *CUDA out of memory in captions*: restart the runtime (frees GPU memory) and run again; it resumes.
- *Download errors for COCO*: re-run; downloads retry over http and https.
