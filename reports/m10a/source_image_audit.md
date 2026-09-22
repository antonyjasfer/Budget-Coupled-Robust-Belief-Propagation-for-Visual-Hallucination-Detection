# Milestone 10A — Source Image Forensic Audit Report

**Audit Execution Timestamp:** `2026-09-22T18:06:38.134596+00:00`  
**Sampling Manifest:** `final_sampling_manifest.json`  
**Sampling Manifest Hash:** `869bf58a15e22df4abebdd5c7b67e9780e8063a43b14ebf9087c46cf0ad4b808`  

## 1. Primary Cohort Image Audit Summary

| Metric | Count | Rate (%) | Status |
| :--- | :--- | :--- | :--- |
| **Requested Images** | 600 | 100.0% | Frozen 600 Manifest |
| **Verified & Decodable Images** | 93 | 15.5% | PASS |
| **Missing Images** | 507 | 84.5% | Documented (No Silent Substitution) |
| **Corrupt Images** | 0 | 0.0% | Zero Corruption Detected |
| **Duplicate Hashes** | 0 | 0.0% | Zero Duplicate Content |

## 2. Partition Breakdown

| Split | Manifest Target | Found & Verified | Missing | Verification Rate |
| :--- | :--- | :--- | :--- | :--- |
| `train` | 300 | 39 | 261 | 13.0% |
| `validation` | 90 | 20 | 70 | 22.2% |
| `calibration` | 90 | 9 | 81 | 10.0% |
| `test` | 120 | 25 | 95 | 20.8% |

## 3. Storage & Source Breakdown

| Source Location | Verified Images |
| :--- | :--- |
| `remote:test2017` | 18 |
| `remote:train2017` | 74 |
| `remote:val2017` | 1 |

## 4. Methodological Compliance Guarantees

1. **No Silent Substitution:** Missing candidate images are explicitly flagged as `MISSING` in the forensic audit. Under no circumstances were alternative images substituted.
2. **Real COCO Provenance:** Every verified image was retrieved directly from official MS COCO 2017 distribution servers or authenticated local genuine COCO caches.
3. **Zero Label Inference:** COCO object detection annotations were **strictly excluded** from being used as ground-truth hallucination labels.
4. **Full SHA-256 Tracking:** Every verified image file has an individual SHA-256 digest recorded in `source_image_audit.json`.
