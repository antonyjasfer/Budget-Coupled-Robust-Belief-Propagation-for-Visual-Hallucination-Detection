# Phase 4 — How to run the confirmatory experiment (Kaggle)

The notebook clones the code at the git tag **`prereg-v2`**, so what runs is exactly what was pre-registered.

## Steps (about 3 sessions of ~4 hours)

**Session 1**
1. Kaggle → **New Notebook** → File → **Import Notebook** → upload `phase4_confirmatory.ipynb` (Drive → `BCRBP_phase2`, or `notebooks/` in the repo).
2. Settings: **Accelerator: GPU T4 x2**, **Internet: ON**.
3. Run **cells 1–5** one by one (setup and smoke test, ~25 min). Cell 5 must show a caption for each of `captions_llava_coco`, `captions_llava_amber`, `captions_qwen`, `captions_internvl`, and non-zero `owl rows` and `swap rows`.
4. **Save Version → Save & Run All (Commit)**. The GPU work stops by itself after `SESSION_HOURS` (cell 6, default 3.5 h) and everything finished is saved.

**Session 2, 3, … until cell 7 prints COMPLETE**
1. Open the notebook → right panel **Add Input** → **Your Work** → this notebook → choose the **latest version**.
2. **Save Version → Save & Run All (Commit)**. Cell 4 imports the finished work; the smoke test is skipped.
3. Open the finished version's log: cell 7 prints a progress table and **COMPLETE** or **NOT COMPLETE**.

**When COMPLETE:** the same session runs the confirmatory analysis. Download `BCRBP_phase4/phase4_results.zip` from that version's Output tab and put it in Drive → `BCRBP_phase2`.

To use fewer, longer sessions, change `SESSION_HOURS` in cell 6 (maximum about 11).

## If something fails

| Problem | Fix |
|---|---|
| Cell 4: `Could not download the AMBER images` | Download `AMBER.zip` from https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view, add it as a Kaggle dataset (**Add Input → Upload**), re-run cell 4. |
| Cell 5: an error in `captions_qwen` or `captions_internvl` | Copy the last 30 lines into the chat. Do not start the full run. |
| Cell 6: an exit code other than 0 or -15 (-15 = stopped at the session limit, which is normal) | The cell prints the log tail; copy it into the chat. Finished work is kept. |
| Out of GPU memory | Re-run; if it repeats, send the log. |

## Time and size

About 8–11 GPU hours per T4 (two in parallel), ~25,000 detector passes, 3,500 captions. The results zip contains no images (~10–20 MB).
