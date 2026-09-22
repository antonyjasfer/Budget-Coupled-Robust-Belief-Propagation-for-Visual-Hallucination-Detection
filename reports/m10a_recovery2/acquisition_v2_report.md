# Milestone 10A-R2: Valid COCO Universe Reconstruction, Primary Cohort Resampling, and Pre-Annotation Freeze V2

**Document Version:** 2.0.0  
**Phase:** 10A-R2  
**Date:** 2026-09-22T20:01:03.202608+00:00  
**Repository:** antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection  
**Base SHA:** `6510402d106eb1b8e0716d7d648a3a6fa1b03896`  
**Status:** COMPLETE (Acquisition Software & Sampling Foundations) / GPU INFERENCE PENDING (Colab Runbook)  

---

## 1. Original Sampling Defect Summary

- **Mechanism**: In Phase 9E `scripts/generate_m9e_manifests.py`, candidate COCO IDs were drawn uniformly from `np.arange(1000, 580000)` instead of parsing actual MS COCO metadata.
- **Consequences**: Out of 600 requested IDs, only 331 were genuine COCO images, while 269 were phantom IDs (`NON_EXISTENT_COCO_INDEX`).
- **Archival**: The v1 manifest was marked `INVALID_SAMPLING_UNIVERSE` and archived under `data/manifests/archived_v1/final_sampling_manifest_v1_superseded.json`.
- **Integrity**: Resampling was conducted prior to any human annotation or outcome inspection.

---

## 2. Valid COCO Universe Reconstruction

- **Source Definition**: Parsed strictly from the `images` arrays of official MS COCO 2017 `captions_train2017.json` and `captions_val2017.json`.
- **Zero Caption Exposure**: `annotations` arrays were strictly ignored and never loaded into memory.
- **Parsed Image Records**:
  - `train2017`: 118,287 images
  - `val2017`: 5,000 images
  - `n_union`: **123,287 unique authoritative COCO images**
  - Duplicate IDs: 0
  - Overlapping Train/Val IDs: 0
- **Candidate Universe Manifest**: `data/manifests/coco_candidate_universe_v2.json`
- **Universe Hash**: `5f8700d1f59378abb95390917fd1bb5856210554b0f462d70d42ed2abdd3093f`

---

## 3. Deterministic 600-Image Resampling & Split Reconstruction

- **Sampling Rule**: 600 images sampled uniformly without replacement from the union of official MS COCO 2017 train and validation records (`seed=42`).
- **Realized Source-Split Composition**:
  - `sample_train2017`: 577 (96.17%)
  - `sample_val2017`: 23 (3.83%)
- **Separated Split Architecture**:
  - `coco_source_split`: `train2017` / `val2017`
  - `research_split`: `TRAIN` (300), `VALIDATION` (90), `CALIBRATION` (90), `TEST` (120)
- **Primary Sampling Manifest**: `data/manifests/final_sampling_manifest_v2.json`
- **Sampling Manifest V2 Hash**: `2a1fd7f1f0510f4cd6e04722b262a1e4be9d9a204a2a90457ca72cb0af563d3a`

---

## 4. Source Image Acquisition Gate Verification

All 600 images downloaded from official MS COCO endpoints (`images.cocodataset.org`) and verified on disk:
- **Requested Images**: 600
- **Valid Decoded Images (PIL)**: **600 (100.0%)**
- **Dimension Matches**: 600 / 600
- **Missing Images**: **0**
- **Corrupt Images**: **0**
- **Audit Report**: `reports/m10a_recovery2/source_image_audit_v2.json`
- **Audit Hash**: `1618a6a1ccf8c25a67b878aed0cff924705e575cc8877c62036b39432a773be0`

---

## 5. Frozen Model Revisions & Generation Hashes

| Model | Checkpoint Name | Frozen Revision |
| :--- | :--- | :--- |
| **VLM** | `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| **Detector** | `google/owlvit-base-patch32` | `cbc355fb364588351c5d51c7f74465e8e7ec6f72` |
| **CLIP** | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |

- **LLaVA Generation Configuration Hash**: `daf24c0c70b8f636d291d52218cc05d329d1732986592fd7f1f7016e24580d5c`
- **Claim Extractor Configuration Hash**: `8e69e79feb9d49c55066197068873b525ef4afc79deda03a0eb1493326b3b166`

---

## 6. GPU Execution Gate & Predeclared Failure Policy

- **Execution Host**: Windows 11 AMD64
- **Hardware Device**: `None (CPU Execution Host)`
- **CUDA Runtime**: `None`
- **PyTorch**: `2.14.0+cpu`
- **Execution Audit**:
  - Attempted Images: 600
  - Completed on Local Host: 0 (Requires CUDA bitsandbytes 4-bit VRAM)
  - Controlled Predeclared Failures: 600 (`VLM_GPU_INFERENCE_REQUIRED_COLAB_RUNBOOK`)
- **Zero-Claim Classification**:
  - `GENUINE_ZERO_CLAIM`: 0
  - `PIPELINE_FAILURE_ZERO_CLAIM`: 600
- **Evidence Manifest V2**: `data/manifests/final_evidence_manifest_v2.json`
- **Evidence Manifest Hash**: `3506bf4509c16a88ed9d0630e093b6bb5bfbf67d398be70e65174b94151e5fe1`

---

## 7. Graph Testability Analysis

- **Primary Cohort Preservation**: Primary 600-image cohort remains strictly unaltered.
- **Overall Statistics**:
  - Total Images: 600
  - Genuine Zero-Claim Images: 0
  - Pipeline Failure Images: 600
- **Test Partition (120 Images)**:
  - Pipeline Failure Images: 120
- **Rating**: `INSUFFICIENT` (locally pending GPU inference).
- **Rationale**: Local host logged controlled typed failures per the Predeclared Failure Policy; no caption fallback or synthetic substitution was permitted.

---

## 8. Corruption Protocol Verification

- **Families**: `gaussian_noise`, `gaussian_blur`, `jpeg_compression`, `contrast_reduction` (4 families × 5 severities = 20 variants per image).
- **Total Variants**: 12,000 (across 600 primary images).
- **Corruption Manifest V2**: `data/manifests/final_corruption_manifest_v2.json`
- **Corruption Manifest V2 Hash**: `3feb647b19e3db4db77d77eb14b2447954846f694c533616a21f099d4f2db035`

---

## 9. Pre-Annotation Freeze V2

Sealed in `data/manifests/pre_annotation_freeze_v2.json`:
- `candidate_universe_hash`: `5f8700d1f59378abb95390917fd1bb5856210554b0f462d70d42ed2abdd3093f`
- `sampling_manifest_hash`: `2a1fd7f1f0510f4cd6e04722b262a1e4be9d9a204a2a90457ca72cb0af563d3a`
- `source_image_audit_hash`: `1618a6a1ccf8c25a67b878aed0cff924705e575cc8877c62036b39432a773be0`
- `corruption_manifest_hash`: `3feb647b19e3db4db77d77eb14b2447954846f694c533616a21f099d4f2db035`
- `evidence_manifest_hash`: `3506bf4509c16a88ed9d0630e093b6bb5bfbf67d398be70e65174b94151e5fe1`
- `llava_generation_config_hash`: `daf24c0c70b8f636d291d52218cc05d329d1732986592fd7f1f7016e24580d5c`
- `claim_extraction_config_hash`: `8e69e79feb9d49c55066197068873b525ef4afc79deda03a0eb1493326b3b166`
- `task_claims_set_hash`: `8ec93db080a177dec2151fc40b9e6ea8c1e63cb80f9ae252d7155b5491b7e826`
- **Freeze Hash**: `df4caf28a3075b500a8dbba88105260f0d41071738c58e9ba79f541cd57cdea6`

---

## 10. Human Task Packages V2 Status

- Exported: `data/annotations/annotator_A_tasks_v2.json`, `annotator_B_tasks_v2.json`
- Masking: Model scores, splits, posteriors, and parameters are 100% masked.
- Claim Sets: A and B are identical.
- Readiness Definition: Per Approval Correction 17, human annotation begins only after GPU acquisition completes in Google Colab.
- Human Tasks Readiness: **PENDING GPU INFERENCE** (0 claims locally; task packages ready for population upon Colab run).

---

## 11. Scientific Lock & Dataset Gate Status

- **DEVELOPMENT Validator**: **PASSED** (`scripts/validate_final_dataset.py --mode DEVELOPMENT`)
- **FINAL Validator**: **CORRECTLY REJECTED** (`scripts/validate_final_dataset.py --mode FINAL`)
  - Missing human labels (by design before annotation)
  - Missing final dataset lock (by design before annotation completion)
- **Dataset Lock**: **NOT LOCKED**
- **Final Experiment**: **NOT READY**
