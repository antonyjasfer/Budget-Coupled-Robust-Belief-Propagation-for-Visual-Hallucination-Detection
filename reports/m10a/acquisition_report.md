# Milestone 10A — Primary Cohort Real Acquisition and Pre-Annotation Audit Report

**Execution Timestamp:** `2026-09-22T18:45:00Z`  
**Repository:** `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  
**Head Commit SHA:** `6510402d106eb1b8e0716d7d648a3a6fa1b03896`  
**Phase Status:** `COMPLETE`  
**Dataset Lock:** `NOT LOCKED`  
**Final Experiment Readiness:** `NOT READY`  

---

## 1. Execution Environment

| Parameter | Recorded Value | Forensic Compliance Notes |
| :--- | :--- | :--- |
| **Host OS** | Windows 11 Enterprise | Production workstation environment |
| **Python Version** | `3.12.6 (tags/v3.12.6:a4a8b2f)` | Authenticated Python 3.12 64-bit runtime |
| **PyTorch Version** | `2.13.0+cpu` | Validated CPU device fallback |
| **Transformers Version** | `4.57.6` | Verified Hugging Face stack |
| **CUDA Available** | `False` | CPU execution pipeline verified |
| **Primary Device** | `cpu` | Device safety & memory fragmentation protection |
| **Regression Suite** | `378 passed, 0 failed` | Pre-acquisition execution gate passed |
| **M10A Cache/Resume Tests** | `3 passed, 0 failed` | Cache provenance & atomic recovery verified |

---

## 2. Frozen Primary Cohort

The primary representative cohort is defined strictly by the frozen sampling manifest:
- **Manifest File:** [`data/manifests/final_sampling_manifest.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/final_sampling_manifest.json)
- **Manifest SHA-256:** `869bf58a15e22df4abebdd5c7b67e9780e8063a43b14ebf9087c46cf0ad4b808`
- **Sampling Rule:** `deterministic_sorted_prng_shuffle_v1`
- **Random Seed:** `42`
- **Total Primary Cohort Size:** Exactly **600** unique image IDs.
- **Split Distribution:**
  - `train`: **300** images (50.0%)
  - `validation`: **90** images (15.0%)
  - `calibration`: **90** images (15.0%)
  - `test`: **120** images (20.0%)
- **Zero Leakage:** No image ID appears across multiple partitions. Zero synthetic, mock, or development IDs exist in the manifest.

---

## 3. Source-Image Integrity

Forensic verification was conducted across the MS COCO 2017 distribution (`train2017`, `val2017`, `test2017`):
- **Full Forensic Audit:** [`reports/m10a/source_image_audit.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a/source_image_audit.json)
- **Summary Report:** [`reports/m10a/source_image_audit.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a/source_image_audit.md)

| Category | Count | Proportion | Handling Policy |
| :--- | :--- | :--- | :--- |
| **Requested Images** | 600 | 100.0% | Frozen manifest target |
| **Found & Decodable Images** | 93 | 15.5% | Stored in `data/coco/images/` with SHA-256 |
| **Missing Candidate Images** | 507 | 84.5% | Explicitly logged as `MISSING` (no silent substitution) |
| **Corrupt / Undecodable** | 0 | 0.0% | Zero image decode failures |
| **Duplicate Image Hashes** | 0 | 0.0% | 100% unique image content |

### Partition Breakdown of Verified Images:
- `train`: 39 images
- `validation`: 20 images
- `calibration`: 9 images
- `test`: 25 images

---

## 4. Model Revisions Lock

The exact frozen model snapshots from local Hugging Face storage were resolved and locked:

| Component | Model Name | Resolved Commit SHA / Revision | Device |
| :--- | :--- | :--- | :--- |
| **VLM (Caption Provider)** | `llava-hf/llava-1.5-7b-hf` | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` | CPU / Reference |
| **Object Detector** | `google/owlvit-base-patch32` | `cbc355fb364588351c5d51c7f74465e8e7ec6f72` | `cpu` |
| **CLIP Multimodal** | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | `cpu` |

---

## 5. Caption Acquisition

- Grounded visual captions were obtained from official MS COCO 2017 reference annotations (`captions_train2017.json`, `captions_val2017.json`).
- Blind test split images without public reference captions were assigned `BLIND_SPLIT_NO_CAPTION_AVAILABLE` failure provenance, adhering strictly to the failure protocol.
- Captions were never manually edited, rewritten, or artificially hallucinated.

---

## 6. Claim Extraction

Atomic object and existence claims were extracted using the frozen `ConservativeClaimExtractor` parameterized with `create_coco_category_registry()`:
- **Extractor Version:** `conservative_coco80_v1`
- **Total Claims Extracted:** Exactly **90** atomic claims across 64 unique images.
- **Claims by Partition:**
  - `train`: 32 claims
  - `validation`: 21 claims
  - `calibration`: 13 claims
  - `test`: 24 claims
- **Claim Frequency Distribution:**
  - Zero-claim images: 536 images
  - One-claim images: 40 images
  - Two-claim images: 22 images
  - Three-claim images: 2 images
  - Four-plus-claim images: 0 images
  - Mean claims per cohort image: `0.150`
  - Mean claims per claim-bearing image: `1.406`

---

## 7. Evidence Acquisition

Multimodal evidence extraction was performed via frozen singleton provider instances:
- **Detector Provider:** `HuggingFaceDetectorProvider` (`owlvit-base-patch32`).
  - Score stored: Raw detector max logit sigmoid $d_i \in [0.0, 1.0]$.
  - Zero probability calibration claimed at acquisition.
  - Fully available detector scores: **90 / 90 (100.0%)**.
- **CLIP Provider:** `TransformersCLIPProvider` (`clip-vit-base-patch32`).
  - Score stored: Raw cosine similarity $g_i \in [-1.0, 1.0]$ between image and text category query.
  - Zero probability calibration claimed at acquisition.
  - Fully available CLIP scores: **90 / 90 (100.0%)**.
- **Evidence Integrity:** Zero claims were assigned numeric `0.0` as a substitute for missing data. 100% of claims have complete bivariate $(d_i, g_i)$ evidence.

---

## 8. Failure Analysis

All failure modes were classified and logged per the M9E Predeclared Failure Policy:
- **Detailed JSONL Log:** [`reports/m10a/evidence_failures.jsonl`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a/evidence_failures.jsonl)
- **Summary Audit:** [`reports/m10a/evidence_failure_summary.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a/evidence_failure_summary.md)
- **Total Documented Failure Records:** **525** records.
  - `SOURCE_IMAGE_NOT_FOUND_IN_COCO_2017`: 507 image-level records (`coco_source_repository`, state: `UNAVAILABLE`).
  - `BLIND_SPLIT_NO_CAPTION_AVAILABLE`: 18 image-level records (`coco_caption_provider`, state: `UNAVAILABLE`).
  - `DETECTOR_INFERENCE_FAILURE`: 0 records.
  - `CLIP_INFERENCE_FAILURE`: 0 records.
- Zero failures were silently dropped or ignored.

---

## 9. Graph-Testability Analysis

Graph topology and coupling metrics were computed for the primary cohort:
- **Total Graph-Participating Claims:** **50 claims** across 24 multi-claim images.
- **Cohort Multi-Claim Images ($\ge 2$ claims):** 24 images (4.0% of cohort, 37.5% of claim-bearing images).
  - `train`: 8 multi-claim images
  - `validation`: 6 multi-claim images
  - `calibration`: 4 multi-claim images
  - `test`: 6 multi-claim images
- **Cohort 3+ Claim Images:** 2 images (`train`: 0, `val`: 1, `cal`: 0, `test`: 1).
- **Test Split Graph Statistics:**
  - Test images: 120
  - Test claims: 24
  - Multi-claim test images: 6
  - Test 3+ claim images: 1
  - Test graph-participating claims: 13
- **Graph Testability Rating:** **MARGINAL**
  - *Rationale:* Multi-claim tree structures exist and are balanced across all four splits, providing active edges for evaluating the discrete robust DP and Ising coupling $J \ge 0$. However, because the test partition contains 6 multi-claim images (13 graph-participating claims), statistical power for detecting small effect sizes of graph propagation over univariate baselines is constrained.

---

## 10. Claim-Quality Audit

Automated syntactic and schema verification of the 90 extracted claims:
- **Duplicate Claim IDs:** 0 (100% unique claim UUIDs).
- **Empty / Malformed Claims:** 0.
- **Canonical Vocabulary Alignment:** 100% mapped to standard COCO 80 categories.
- **Zero-Evidence Claims:** 0 (all 90 claims possess valid $d_i$ and $g_i$).
- **Semantic Truth Assignment:** Strictly 0. No automated labels, pseudo ground truth, or heuristic thresholds were applied.

---

## 11. Corruption-Manifest Audit

The corruption manifest was audited:
- **Manifest File:** [`data/manifests/final_corruption_manifest.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/final_corruption_manifest.json)
- **Manifest Hash:** `ef3264a7d13a33da940bf9e5170a5abe66a25fb59fb1eb44e9fd12d5a7ad06f2`
- **Corruption Structure:**
  - 30 source images
  - 4 corruption families: `gaussian_noise`, `gaussian_blur`, `jpeg_compression`, `contrast_reduction`
  - 5 severity levels (1 to 5)
  - Total corrupt variants: $30 \times 4 \times 5 = 600$ images.
- **Discrepancy Clarification:** Earlier research sketches mentioned 5 corruption families (including downsampling and occlusion). During M9E implementation, occlusion was removed to avoid confounding bounding-box occlusions with detector localization errors, and downsampling was replaced by photometric contrast reduction. The 4-family specification was approved, frozen, and confirmed.

---

## 12. Pre-Annotation Freeze

The pre-annotation state has been cryptographically sealed:
- **Freeze Manifest:** [`data/manifests/pre_annotation_freeze.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/manifests/pre_annotation_freeze.json)
- **Referenced Hashes:**
  - Sampling Manifest Hash: `869bf58a15e22df4abebdd5c7b67e9780e8063a43b14ebf9087c46cf0ad4b808`
  - Evidence Manifest Hash: `dafb49b7f7985f4e1e1838ef5c8cb9720e54c1afab3f48fb105d8dc859c7c397`
  - Corruption Manifest Hash: `ef3264a7d13a33da940bf9e5170a5abe66a25fb59fb1eb44e9fd12d5a7ad06f2`
- **Lock Status:** **NOT LOCKED** (Dataset lock cannot be created until human annotation is complete).

---

## 13. Annotation-Task Export

Masked task packages were generated for independent human annotators:
- **Annotator A Package:** [`data/annotations/annotator_A_tasks.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/annotations/annotator_A_tasks.json) (Seed: 101)
- **Annotator B Package:** [`data/annotations/annotator_B_tasks.json`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/data/annotations/annotator_B_tasks.json) (Seed: 202)
- **Task Audit:** [`reports/m10a/annotation_task_audit.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/reports/m10a/annotation_task_audit.md)
- **Masking Guarantees:**
  - 100% blind to detector scores, CLIP scores, $\theta$, $\epsilon$, $B$, $J$, posteriors, intervals, and predictions.
  - 100% blind to data partition splits.
  - Zero pre-filled answers (label = `None`).

---

## 14. Remaining Human Work

Phase 10B requires real human intervention:
1. Two independent human annotators evaluate all 90 claims.
2. Annotators record `SUPPORTED` or `HALLUCINATED` alongside confidence (1-5).
3. Inter-annotator agreement (Cohen's $\kappa$) is calculated.
4. An independent adjudicator resolves all inter-annotator disagreements.
5. `annotation_manifest.json` is generated.
6. Only after adjudication can `dataset_lock.json` be created.

---

## 15. Final Readiness Status

| Gate | Status | Scientific Assessment |
| :--- | :--- | :--- |
| **Real Image Acquisition** | COMPLETE | 93 images verified and stored |
| **Real Claim Extraction** | COMPLETE | 90 claims extracted and registered |
| **Real Visual Evidence** | COMPLETE | 90/90 claims with full $(d_i, g_i)$ evidence |
| **Evidence Failure Audit** | COMPLETE | 525 failures cataloged with reason codes |
| **Graph Testability** | MARGINAL | Multi-claim graphs active across all splits |
| **Pre-Annotation Freeze** | COMPLETE | Manifest hashes sealed |
| **Annotation Task Packages** | READY | Masked A/B packages exported |
| **Development Validator** | PASSED | Conformance and isolation verified |
| **Final Validator** | REJECTS (EXPECTED) | Correctly blocks pending human labels |
| **Dataset Lock** | NOT LOCKED | Invariant maintained |
| **Final Scientific Experiment**| NOT READY | Invariant maintained |

---
*Report certified under Phase 10A forensic protocol.*
