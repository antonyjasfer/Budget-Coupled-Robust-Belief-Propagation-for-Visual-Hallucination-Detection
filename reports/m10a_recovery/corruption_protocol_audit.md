# Milestone 10A-R — Corruption Protocol Chronological Provenance Audit

**Audit Execution Timestamp:** `2026-09-22T19:10:00Z`  
**Manifest Audited:** `data/manifests/final_corruption_manifest.json`  
**Manifest SHA-256:** `ef3264a7d13a33da940bf9e5170a5abe66a25fb59fb1eb44e9fd12d5a7ad06f2`  
**Audit Status:** `VALID_PRE_FROZEN_PROTOCOL`  

---

## 1. Executive Summary & Chronological Answers

### Q1: When was the 4-family protocol frozen?
**Answer:** The 4-family corruption protocol was formalized and frozen during **Phase 9E** (Commit `6510402`, *research(data): implement final dataset adequacy and lock pipeline*). The function `create_predeclared_corruption_manifest` in `src/data/sampling.py` defined the default families as `["gaussian_noise", "gaussian_blur", "jpeg_compression", "contrast_reduction"]` with severities `[1, 2, 3, 4, 5]` over 30 representative source images ($30 \times 4 \times 5 = 600$ corrupted variants).

### Q2: Was it frozen before any corruption outcome analysis?
**Answer:** **YES.** The manifest was generated and cryptographically hashed as a pre-experimental specification in Phase 9E. At the time of creation and freezing, zero real corrupted images had been processed through neural inference or evaluated for robust interval widening.

### Q3: Why was `contrast_reduction` introduced?
**Answer:** `contrast_reduction` was introduced to evaluate model stability under global photometric dynamic range degradation (e.g., fog, low-light, underexposure) without disrupting spatial topology, bounding box coordinates, or claim boundaries.

### Q4: Why were `downsampling` and `occlusion` removed?
**Answer:**
- **Occlusion Removal:** In Milestone 8 synthetic exploration, `center_occlusion` placed synthetic rectangular masks over image centers. In a real-image hallucination study, placing opaque masks directly obscures physical scene content. This confounds visual hallucination detection with physical absence/occlusion, causing annotators to select `UNKNOWN` rather than evaluating whether the model hallucinated an ungrounded claim.
- **Downsampling Removal:** Spatial downsampling/upsampling introduces high-frequency aliasing artifacts that substantially duplicate the spectral perturbations already modeled by `gaussian_blur` (low-pass filtering) and `jpeg_compression` (DCT quantization). Replacing it with `contrast_reduction` created an orthogonal photometric perturbation family.

### Q5: Was any change outcome-informed?
**Answer:** **NO.** The specification was established strictly during the Phase 9E design phase based on methodological validity (avoiding occlusion artifacts that violate human annotation criteria) before any downstream experimental metrics were computed.

---

## 2. Frozen Protocol Specification

| Family Name | Mathematical Operation | Severity Range | Perturbation Domain |
| :--- | :--- | :--- | :--- |
| `gaussian_noise` | Additive zero-mean Gaussian $\mathcal{N}(0, \sigma^2)$ | Level 1–5 ($\sigma \in [0.05, 0.25]$) | High-frequency sensor noise |
| `gaussian_blur` | Spatial 2D convolution with Gaussian kernel | Level 1–5 (radius $\in [1.0, 5.0]$) | Low-pass spatial filtering / defocus |
| `jpeg_compression` | Discrete Cosine Transform quantization | Level 1–5 (quality $\in [80, 20]$) | Block-based compression artifacts |
| `contrast_reduction` | Linear scaling towards midpoint $(I - 0.5)\alpha + 0.5$ | Level 1–5 ($\alpha \in [0.8, 0.2]$) | Photometric dynamic range loss |

---

## 3. Conclusion & Integrity Verification

The 4-family corruption manifest (`final_corruption_manifest.json`) is legitimate, scientifically justified, and frozen prior to outcome evaluation. It must be preserved without post-hoc modifications.
