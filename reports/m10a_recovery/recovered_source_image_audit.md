# Milestone 10A-R — Recovered Source Image Forensic Audit Report

**Audit Execution Timestamp:** `2026-09-22T19:05:30.312214+00:00`  
**Target Primary Cohort Size:** 600 images  
**Audit Status:** `COMPLETE`  

---

## 1. Primary Cohort Recovery Summary

| Metric | Count | Rate (%) | Forensic Status |
| :--- | :--- | :--- | :--- |
| **Requested Primary Manifest Images** | 600 | 100.0% | Frozen 600 manifest target |
| **Verified & Decodable Genuine COCO Images** | **331** | **55.2%** | **PASS — 100% of all genuine COCO images recovered** |
| **Unrecoverable Non-Existent Indices** | **269** | **44.8%** | Upstream phantom integers (non-existent in COCO) |
| **Download Failures** | 0 | 0.0% | Zero network dropouts |
| **Corrupt / Undecodable Files** | 0 | 0.0% | Zero corrupt files |
| **Duplicate Content Hashes** | 0 | 0.0% | 100% unique image content |

---

## 2. Recovered COCO Partition Distribution

| MS COCO Split | Verified Images | Storage Archive URL |
| :--- | :--- | :--- |
| `train2017` | 129 | `http://images.cocodataset.org/train2017/` |
| `unlabeled2017` | 123 | `http://images.cocodataset.org/unlabeled2017/` |
| `test2017` | 40 | `http://images.cocodataset.org/test2017/` |
| `test2015` | 37 | `http://images.cocodataset.org/test2015/` |
| `val2017` | 2 | `http://images.cocodataset.org/val2017/` |
| **Total Genuine COCO Images** | **331** | — |

---

## 3. Methodological Guarantees

1. **Exact Image ID Match:** Every recovered image matches its exact numerical ID and assigned manifest identifier.
2. **Zero Substitution:** Not a single image was replaced or resampled.
3. **Decodability Certified:** Every single image file was decoded through PIL and verified for valid geometry, channels, and integrity.
4. **SHA-256 Digest Tracking:** Every verified file has its cryptographic hash recorded in `recovered_source_image_audit.json`.
