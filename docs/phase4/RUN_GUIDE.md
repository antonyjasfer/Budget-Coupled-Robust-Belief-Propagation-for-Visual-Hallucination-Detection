# Phase 4 — How to run the confirmatory experiment (Kaggle)

The notebook clones the code at the git tag **`prereg-v2`**, so what runs is exactly what was pre-registered.

## Steps

1. Kaggle → **New Notebook** → File → **Import Notebook** → upload `phase4_confirmatory.ipynb` (Drive → `BCRBP_phase2`, or `notebooks/` in the repo).
2. Settings: **Accelerator: GPU T4 x2**, **Internet: ON**.
3. Run **cells 1–5** one by one (setup and smoke test, ~25 min). Cell 5 must show a caption for each of `captions_llava_coco`, `captions_llava_amber`, `captions_qwen`, `captions_internvl`, and non-zero `owl rows` and `swap rows`.
4. **Save Version → Save & Run All (Commit)**. Both GPUs run in parallel for ~8–11 h; you can close the browser.
5. When it finishes: open the version → **Output** → download `BCRBP_phase4/phase4_results.zip` → put it in Drive → `BCRBP_phase2`.

## If the run stops at Kaggle's 12-hour limit

Edit the notebook → right panel **Add Input** → **Your Work** → choose this notebook's last version → **Save & Run All** again. Cell 4 copies the finished shard files and only the remaining work runs.

## If something fails

| Problem | Fix |
|---|---|
| Cell 4: `Could not download the AMBER images` | Download `AMBER.zip` from https://drive.google.com/file/d/1MaCHgtupcZUjf007anNl4_MV0o4DjXvl/view, add it as a Kaggle dataset (**Add Input → Upload**), re-run cell 4. |
| Cell 5: an error in `captions_qwen` or `captions_internvl` | Copy the last 30 lines into the chat. Do not start the full run. |
| Cell 6: `exit codes` not all 0 | The cell prints the log tail; copy it into the chat. Finished work is kept. |
| Out of GPU memory | Re-run; if it repeats, send the log. |

## Time and size

About 8–11 GPU hours per T4 (two in parallel), ~25,000 detector passes, 3,500 captions. The results zip contains no images (~10–20 MB).
