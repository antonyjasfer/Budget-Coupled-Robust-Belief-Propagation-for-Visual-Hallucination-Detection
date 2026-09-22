# Milestone 10A-R — Primary Cohort Source Recovery, Claim-Source Audit, and Pre-Annotation Refreeze Report

**Execution Timestamp:** `2026-09-22T19:25:00Z`  
**Repository:** `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  
**Head Commit SHA:** `6510402d106eb1b8e0716d7d648a3a6fa1b03896`  
**Phase Status:** `COMPLETE`  
**Dataset Lock:** `NOT LOCKED`  
**Final Experiment Readiness:** `NOT READY`  

---

## 1. Executive Summary

Phase 10A-R was executed to forensically audit, diagnose, and recover the 600-image frozen primary COCO cohort, resolve the root causes of the 507 missing images from Phase 10A, enforce the scientific prohibition against substituting COCO human ground truth captions for VLM generation, invalidate superseded partial annotation tasks, and seal a revised pre-annotation freeze.

---

## 2. Forensic Audit of the 600-ID Sampling Manifest (10A-R-1)

Cross-referencing the frozen sampling manifest ([`data/manifests/final_sampling_manifest.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/final_sampling_manifest.json)) against official MS COCO dataset catalogs revealed that the 600 IDs are distributed across multiple partitions:

| MS COCO Dataset Partition | Image Count | Official Storage URL | Verification Status |
| :--- | :--- | :--- | :--- |
| `train2017` | 129 | `http://images.cocodataset.org/train2017/` | Genuine MS COCO images |
| `unlabeled2017` | 123 | `http://images.cocodataset.org/unlabeled2017/` | Genuine MS COCO images |
| `test2017` | 40 | `http://images.cocodataset.org/test2017/` | Genuine MS COCO images |
| `test2015` | 37 | `http://images.cocodataset.org/test2015/` | Genuine MS COCO images (prefixed `COCO_test2015_`) |
| `val2017` | 2 | `http://images.cocodataset.org/val2017/` | Genuine MS COCO images |
| `NON_EXISTENT_COCO_INDEX` | 269 | N/A (HTTP 404 on official S3 buckets) | Upstream sampling artifact (Phase 9E uniform random range) |
| **Total Target Cohort** | **600** | — | — |

---

## 3. Root Cause Diagnosis of the 507 Missing Images (10A-R-2)

Every single missing image from Phase 10A was classified with an exact deterministic root cause:
- **Detailed JSONL:** [`reports/m10a_recovery/missing_source_root_causes.jsonl`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a_recovery/missing_source_root_causes.jsonl)
- **Summary Report:** [`reports/m10a_recovery/missing_source_root_causes.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a_recovery/missing_source_root_causes.md)

1. **`NON_EXISTENT_COCO_INDEX` (269 images):** In Phase 9E, `generate_candidate_coco_universe` sampled integer IDs uniformly from `np.arange(1000, 580000)`. Because COCO IDs are sparse (~50% density across Flickr imports), ~45% of generated numbers never existed in any release of MS COCO.
2. **`WRONG_COCO_SPLIT_DIRECTORY_UNLABELED2017` (123 images):** Genuine COCO images located in `unlabeled2017` that were missed because Phase 10A only inspected 2017 train/val/test directories.
3. **`FILENAME_RESOLUTION_ERROR_TEST2015` (37 images):** Genuine COCO test images located in `test2015` that use the legacy filename prefix `COCO_test2015_000000xxxxxx.jpg`.
4. **`DOWNLOAD_FAILED_OR_TIMEOUT` (78 images):** Genuine images (`train2017`: 55, `test2017`: 22, `val2017`: 1) that timed out during Phase 10A due to a short 5-second socket timeout and multi-threaded connection throttling.

---

## 4. Source Image Recovery Execution (10A-R-3 & 10A-R-4)

All 238 recoverable images were retrieved, verified through PIL decoding, and stored with SHA-256 digests in [`data/coco/images/`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/coco/images/):
- **Recovered Audit:** [`reports/m10a_recovery/recovered_source_image_audit.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a_recovery/recovered_source_image_audit.json)
- **Summary:** [`reports/m10a_recovery/recovered_source_image_audit.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a_recovery/recovered_source_image_audit.md)
- **Genuine Recovered Images:** **331 / 331 (100.0% of all real COCO images in the 600 cohort)**.
- **Corrupt / Undecodable Files:** 0.
- **Duplicate Content Hashes:** 0.

---

## 5. Caption Source & "BLIND_SPLIT_NO_CAPTION_AVAILABLE" Audit (10A-R-5)

- **Audit Finding:** In Phase 10A, human reference captions from `captions_train2017.json` and `captions_val2017.json` were mistakenly used as surrogates for LLaVA. Because test split images do not have public captions, 18 test images failed with `BLIND_SPLIT_NO_CAPTION_AVAILABLE`.
- **Methodological Correction:** In accordance with the scientific specification, substituting COCO human captions for VLM generation is strictly prohibited. The claim extraction pipeline must operate on VLM descriptions.
- **Regression Verification:** Added [`tests/test_vlm_claim_generation_no_coco_caption.py`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/tests/test_vlm_claim_generation_no_coco_caption.py), which verified that an image passes claim generation cleanly without any COCO reference caption.

---

## 6. Real LLaVA Execution Audit & Failure Policy (10A-R-6 & 10A-R-7)

- `llava-hf/llava-1.5-7b-hf` requires 14 GB of memory in FP16 or 4-bit CUDA execution with `bitsandbytes`.
- The local Windows workstation possesses 7.7 GB total physical RAM and no CUDA GPU (`torch.cuda.is_available() == False`).
- Real LLaVA cache hits: 0.
- In accordance with the M9E Predeclared Failure Policy, images requiring GPU VLM execution are cataloged as `VLM_GPU_INFERENCE_REQUIRED_COLAB_RUNBOOK` (`UNAVAILABLE`), preserving absolute scientific honesty without fabricating evidence or using human captions as pseudo-VLM outputs.
- Detailed failure log: [`reports/m10a_recovery/evidence_failures.jsonl`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a_recovery/evidence_failures.jsonl).

---

## 7. Zero-Claim Audit (10A-R-9)

The distinction between genuine zero-claim images and pipeline failure zero-claim images was enforced:
- **`GENUINE_ZERO_CLAIM`:** 0 (requires real LLaVA caption that produces no object mentions under conservative extractor).
- **`PIPELINE_FAILURE_SOURCE_UNAVAILABLE`:** 269 images (upstream non-existent COCO indices).
- **`PIPELINE_FAILURE_VLM_LOCAL_UNAVAILABLE`:** 331 images (genuine COCO images requiring GPU execution per Colab runbook).

---

## 8. Corruption Protocol Chronological Audit (10A-R-13)

Audited in [`reports/m10a_recovery/corruption_protocol_audit.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a_recovery/corruption_protocol_audit.md):
- 4 families (`gaussian_noise`, `gaussian_blur`, `jpeg_compression`, `contrast_reduction`) × 5 severities = 600 variants.
- Frozen in Phase 9E prior to any outcome evaluation.
- `occlusion` removed to prevent bounding-box obscuration confounding human annotation.
- `downsampling` replaced by `contrast_reduction` for orthogonal photometric stress testing.
- Chronology certified pre-experimental; no outcome-informed changes detected.

---

## 9. Historical Invalidation & Refreeze (10A-R-14, 10A-R-16, 10A-R-17)

- Old partial annotation task packages archived to [`data/annotations/archived_partial_10a/`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/annotations/archived_partial_10a/) and stamped `SUPERSEDED_PARTIAL_ACQUISITION_DO_NOT_ANNOTATE`.
- Updated evidence manifests created: [`data/manifests/final_evidence_manifest.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/final_evidence_manifest.json) and [`data/manifests/evidence_manifest.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/evidence_manifest.json).
- New pre-annotation freeze sealed in [`data/manifests/pre_annotation_freeze.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/pre_annotation_freeze.json) (version `1.1.0-recovery`).
- Masked annotation packages generated with 0 fabricated claims.

---

## 10. Verification Gates (10A-R-18)

- **Development Validator:** `PASSED` (`python scripts/validate_final_dataset.py --mode DEVELOPMENT`).
- **Final Validator:** `FAILED (EXPECTED)` (`python scripts/validate_final_dataset.py --mode FINAL`).
- **Regression Suite:** `4 passed, 0 failed` (`pytest tests/test_vlm_claim_generation_no_coco_caption.py tests/test_m10a_cache_resume.py`).
- **Dataset Lock:** `NOT LOCKED`.
- **Final Experiment:** `NOT READY`.
