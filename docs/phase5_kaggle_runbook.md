# Phase 5 runbook (Q1 extension)

What this phase adds: a generator x generator transfer matrix (six caption generators x three
detectors), the prevalence decomposition, the local gap, the recalibration curve, grouped
conformal risk control, Benjamini-Hochberg decisions and a 600-claim human label audit.
Theorem numbers refer to `paper/tai/tai.tex`.

## Before any test data is analysed

1. Write `prereg/prereg-v3.md`: hypotheses, the six generators and three detectors, the
   1,000 + 3,000 COCO split (seed 2027), the decision rule (interval rule plus BH at q = 0.05
   over off-diagonal cells), and the audit protocol. Commit it and tag `prereg-v3`.
   `analyze` must run from a commit at or after that tag.

## Kaggle sessions (GPU T4 x2, Internet on)

```bash
git clone <repo> && cd <repo> && pip install -q -e ".[phase1,vlm]" bitsandbytes matplotlib
OUT=/kaggle/working/phase5
python scripts/phase5.py setup  --out $OUT
python scripts/phase5.py images --out $OUT
python scripts/phase5.py import --out $OUT        # from the second session on

# Week 1 pilot (Gate 1): 100 images per generator, then CHAIR per generator
P=/kaggle/working/pilot
python scripts/phase5.py setup --out $P && python scripts/phase5.py images --out $P
CUDA_VISIBLE_DEVICES=0 python scripts/phase5.py captions --out $P --limit 100 --gens llava,llava_t1,qwen25 &
CUDA_VISIBLE_DEVICES=1 python scripts/phase5.py captions --out $P --limit 100 --gens qwen3_4b,internvl3,instructblip
python scripts/phase5.py pilot --out $P

# Full run: one process per GPU, same generator list, different shard
G=llava,llava_t1,qwen25,qwen3_4b,internvl3,instructblip
CUDA_VISIBLE_DEVICES=0 python scripts/phase5.py captions --out $OUT --gens $G --shard 0 --nshards 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/phase5.py captions --out $OUT --gens $G --shard 1 --nshards 2
CUDA_VISIBLE_DEVICES=0 python scripts/phase5.py detect   --out $OUT --gens $G --shard 0 --nshards 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/phase5.py detect   --out $OUT --gens $G --shard 1 --nshards 2
```

Every stage appends to `*.s0.jsonl` / `*.s1.jsonl` and resumes, so a session can stop at any
time. Use "Save & Run All" so jobs run with the tab closed, and save the output folder as a
Kaggle Dataset version every few hours; attach it to the next session and run `import`.
LLaVA-OneVision needs `--four-bit` on one T4 (`llava_ov` in `src/phase5/gpu.py`).
The GPU code has not been run yet: start with the pilot and check its output by hand.

## CPU stages (GPU off)

```bash
python scripts/phase5.py analyze --out $OUT --gens $G      # writes phase5_results.json
python scripts/phase5.py audit   --out $OUT --gens llava,qwen25,internvl3
```

`audit` writes `audit/annotator_tasks.csv` (blind: image link, object, caption) and
`audit/audit_key.csv` (generator, CHAIR label, score, stratum). Two annotators label every
row independently; compute agreement with `src.phase5.audit.cohen_kappa` and noise per
stratum with `src.phase5.audit.noise_rates`.

## Sample size

`src.phase5.stats.required_claims(0.10, 0.13)` = 670 hallucinated test claims for 80% power
(one-sided, alpha = 0.05); multiply by the design effect (`design_effect`, about 1.2) and divide
by each generator's hallucinated claims per image to get test images per generator. The
3,000-image test set is sized for this.

## Not yet included

Open Images V7 (verified negatives), weighted-CP and CoFact baselines, and learn-then-test
control of the kept hallucinated share are in the Q1 plan but not implemented here.
