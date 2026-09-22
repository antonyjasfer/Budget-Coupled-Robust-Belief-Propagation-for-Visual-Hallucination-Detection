# Milestone 10A — Engineering Pilot Acquisition Report

**Execution Timestamp:** `2026-09-22T18:30:50.908326+00:00`  
**Pilot Status:** `PASSED`  
**Runtime:** `241.66 seconds`  

## 1. Pilot Quality Checks

| Check | Expected | Observed | Status |
| :--- | :--- | :--- | :--- |
| **Pilot Images Tested** | ~15 | 15 | PASS |
| **Claims Extracted** | > 0 | 16 | PASS |
| **Detector Evidence Obtained** | > 0 | 16 | PASS |
| **CLIP Evidence Obtained** | > 0 | 16 | PASS |
| **NaN / Inf Anomalies** | 0 | 0 | PASS |
| **Duplicate Claim IDs** | 0 | 0 | PASS |
| **Memory Fragmentation** | Stable | Stable Singleton Models | PASS |

## 2. Sample Pilot Claim Evidence

| Image ID | Split | Claim Category | OWL-ViT (d_i) | CLIP (g_i) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `coco_000000004968` | `train` | `car` | `0.2252` | `0.1999` | AVAILABLE |
| `coco_000000010407` | `train` | `broccoli` | `0.7029` | `0.2711` | AVAILABLE |
| `coco_000000010495` | `train` | `elephant` | `0.8099` | `0.3103` | AVAILABLE |
| `coco_000000012284` | `train` | `umbrella` | `0.3181` | `0.2946` | AVAILABLE |
| `coco_000000028885` | `validation` | `giraffe` | `0.7562` | `0.3108` | AVAILABLE |
| `coco_000000028885` | `validation` | `person` | `0.1050` | `0.2407` | AVAILABLE |
| `coco_000000044464` | `validation` | `frisbee` | `0.3677` | `0.2523` | AVAILABLE |
| `coco_000000044464` | `validation` | `person` | `0.2970` | `0.1980` | AVAILABLE |
| `coco_000000010920` | `calibration` | `car` | `0.5047` | `0.2091` | AVAILABLE |
| `coco_000000010920` | `calibration` | `traffic light` | `0.4910` | `0.2335` | AVAILABLE |
| `coco_000000027235` | `calibration` | `bed` | `0.3070` | `0.2292` | AVAILABLE |
| `coco_000000027235` | `calibration` | `teddy bear` | `0.7713` | `0.2547` | AVAILABLE |
| `coco_000000093201` | `calibration` | `dog` | `0.8243` | `0.2755` | AVAILABLE |
| `coco_000000011205` | `test` | `cat` | `0.7797` | `0.2592` | AVAILABLE |
| `coco_000000011205` | `test` | `motorcycle` | `0.2644` | `0.2737` | AVAILABLE |
| `coco_000000016531` | `test` | `truck` | `0.5090` | `0.2381` | AVAILABLE |
