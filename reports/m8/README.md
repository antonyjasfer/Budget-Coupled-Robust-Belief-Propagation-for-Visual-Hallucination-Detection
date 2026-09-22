# Milestone 8: Experimental Evaluation Framework

## Overview

Milestone 8 (M8) implements the reproducible experimental evaluation pipeline comparing **Standard Belief Propagation (point posteriors)** against **Budget-Coupled Robust Belief Propagation (interval posteriors $[L_i, U_i]$)** for visual hallucination detection.

### Central Research Question

> **Does globally budget-coupled robust inference identify evidence-sensitive hallucination decisions that ordinary point posteriors do not expose?**

---

## Architecture & Modules

```
src/experiments/
├── configs.py           # Experiment configuration schemas & dataclasses
├── loaders.py           # Dataset loading, split validation & leakage prevention
├── parameterization.py  # Evidence-to-PGM mapping & tree construction
├── baselines.py         # Point baseline, Standard BP, Robust BP, Ablations
├── inference_runner.py  # Inference execution, atomic cache & result persistence
├── robust_runner.py     # Profile generation & witness extraction
├── evaluation.py        # Core classification metrics (Acc, Prec, Rec, F1, ROC-AUC)
├── interval_metrics.py  # Interval width statistics & correlation analysis
├── corruption.py        # Deterministic visual corruption suite
├── ablations.py         # 8-dimension ablation matrix runner
├── aggregation.py       # Image cluster bootstrap & case studies selection
└── provenance.py        # Hardware/software/git manifest generation

scripts/
├── run_m8_experiments.py    # Master experiment CLI orchestrator
├── run_m8_single.py         # Single image/claim diagnostic inspection
├── evaluate_m8_results.py   # Offline re-evaluation of saved results
├── generate_m8_plots.py     # Generation of Figures 1–12 (PNG & PDF)
├── generate_m8_tables.py    # Generation of Tables 1–7 (JSON, CSV, Markdown)
└── validate_m8_artifacts.py # Strict artifact verification CLI
```

---

## Execution Modes & Real Data Status

M8 enforces three mutually exclusive execution states:

1. **`SMOKE`**:
   - Uses a tiny deterministic synthetic fixture (5 images, 15 claims).
   - Validates all math, pipelines, table generation, plot generation, and schema compliance end-to-end in < 2 seconds.
   - All outputs are watermarked as `[SMOKE / SYNTHETIC]`.

2. **`DEVELOPMENT`**:
   - Uses currently available real artifacts from M6/M7 (10 genuine COCO train2017 images, 15 claim records).
   - Used for engineering validation, integration testing, and local sanity checking.
   - Reports clearly note partial dataset size.

3. **`FINAL`**:
   - Requires the complete, verified, and locked 600-image M7 dataset with full annotations.
   - Strict checksum and split isolation verification enforced.

---

## Quickstart & CLI Commands

### 1. Run Fast Synthetic Smoke Test
```bash
python scripts/run_m8_experiments.py --smoke --output-dir reports/m8_smoke
```

### 2. Run Local Development Experiments (Available Real Subset)
```bash
python scripts/run_m8_experiments.py --config configs/m8_default.yaml --output-dir reports/m8
```

### 3. Inspect a Single Claim
```bash
python scripts/run_m8_single.py --image-id coco_000000000064 --claim-id coco_000000000064_claim_0 --detailed-profile
```

### 4. Re-generate Tables & Plots Offline
```bash
python scripts/generate_m8_tables.py --results-dir reports/m8
python scripts/generate_m8_plots.py --results-dir reports/m8
```

### 5. Validate Output Artifacts
```bash
python scripts/validate_m8_artifacts.py --results-dir reports/m8
```

---

## Generated Tables & Figures

### Tables (`reports/m8/tables/`)
- **Table 1**: Dataset Summary (Images, Claims, Split distributions, Ground Truth counts).
- **Table 2**: Method Comparison (Point baseline, Standard BP, Robust BP, $J=0$, Independent Box).
- **Table 3**: Interval Statistics (Mean, Median, Std, Class Breakdown, Sensitive fraction).
- **Table 4**: Full Ablation Matrix (8 dimensions).
- **Table 5**: Budget Sensitivity Sweep ($B \in [0.0, 4.0]$).
- **Table 6**: Visual Corruption Robustness (Clean vs Blur, JPEG, Noise, Occlusion).
- **Table 7**: Computational Cost & Memory Profile.

### Figures (`reports/m8/figures/`)
- **Fig 1 & 2**: Confusion Matrices (Standard BP vs Robust BP).
- **Fig 3**: ROC Curves Comparison.
- **Fig 4**: Robust Interval Width Distribution.
- **Fig 5**: Standard BP Posterior vs Robust Midpoint.
- **Fig 6**: Representative Claim Intervals $[L_i, U_i]$ with Threshold $\tau$.
- **Fig 7**: Interval Width vs Corruption Severity.
- **Fig 8 & 9**: F1 Score & Interval Width vs Global Budget $B$.
- **Fig 10**: Performance (Accuracy/F1) vs Visual Degradation.
- **Fig 11**: Standard BP Error Rate vs Robust Interval Width Bins.
- **Fig 12**: Evidence-Sensitive Fraction vs Visual Degradation.
