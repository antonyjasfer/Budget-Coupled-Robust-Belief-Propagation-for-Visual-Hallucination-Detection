# Milestone 9B: Parameterization and Evidence Calibration Audit Report

**Repository**: `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  
**Project**: Budget-Coupled Robust Belief Propagation for Visual Hallucination Detection  
**Phase**: Phase 9B — Principled Evidence Calibration and Robust Parameterization  
**Date**: September 2026  

---

## 1. Executive Summary

This audit evaluates the current implementation of model parameters ($\theta_i$, $\epsilon_i$, $J_{ij}$, $B$) in [`src/experiments/parameterization.py`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/src/experiments/parameterization.py) and [`src/experiments/configs.py`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/src/experiments/configs.py).

The purpose of Phase 9B is to transition from heuristic, hand-crafted parameter settings to a **principled, split-safe, auditable parameterization pipeline** grounded in:
1. Supervised evidence-fusion models fit strictly on `TRAIN`.
2. Calibration curves and empirical perturbation distributions evaluated on `CALIBRATION`.
3. Hyperparameters and coupling strengths tuned on `VALIDATION`.
4. Zero influence from `TEST` data.

---

## 2. Parameter-by-Parameter Audit

### 2.1. Unary Potential Field $\theta_i$

| Attribute | Current Implementation Status |
| :--- | :--- |
| **Code Location** | `src/experiments/parameterization.py:compute_unary_theta()` |
| **Current Formula** | $s_{\text{det}} = \operatorname{clip}(d_i, 10^{-4}, 1 - 10^{-4})$<br>$s_{\text{clip}} = \operatorname{clip}((g_i + 1)/2, 10^{-4}, 1 - 10^{-4})$<br>$v_{\text{support}} = \frac{w_{\text{det}} s_{\text{det}} + w_{\text{clip}} s_{\text{clip}}}{w_{\text{det}} + w_{\text{clip}}}$<br>$\theta_i = \frac{1}{2} \ln \left( \frac{1 - v_{\text{support}}}{v_{\text{support}}} \right)$, clipped to $[-\theta_{\text{max}}, \theta_{\text{max}}]$ |
| **Data Source** | Raw OWL-ViT detector score $d_i \in [0, 1]$, raw CLIP cosine similarity $g_i \in [-1, 1]$ |
| **Heuristic vs Learned** | **Heuristic**: $w_{\text{det}} = 1.0, w_{\text{clip}} = 0.0$ by default; affine mapping $(g_i + 1)/2$ is an uncalibrated heuristic. |
| **Calibrated?** | **No**. Raw scores are treated directly as probabilities of visual presence. |
| **Split Used** | Evaluated per claim at inference time. |
| **Test Leakage Risk** | **None currently** (since parameters are hard-coded heuristics), but vulnerable if weights are tuned on test. |
| **Units & Range** | Unary energy / log-odds scale, range $[-\theta_{\text{max}}, \theta_{\text{max}}] = [-5.0, 5.0]$. |
| **Assumptions** | Assumes $d_i$ linearly measures probability of object support; assumes CLIP cosine similarity is centered at 0 and maps linearly to support probability. |
| **Scientific Weakness** | OWL-ViT detection logits/sigmoid scores are well-known to be uncalibrated across categories. CLIP similarity distributions vary dramatically across open-vocabulary object categories. Treating raw scores as probabilities distorts the true belief scale. |

---

### 2.2. Local Perturbation Bound $\epsilon_i$

| Attribute | Current Implementation Status |
| :--- | :--- |
| **Code Location** | `src/experiments/parameterization.py:compute_uncertainty_epsilon()` |
| **Current Formula** | $\epsilon_i = \max\left(\epsilon_{\text{min}}, \, \epsilon_{\text{base}} \cdot s_\epsilon + w_{\text{disc}} \cdot |s_{\text{det}} - s_{\text{clip}}|\right)$<br>Default: $\epsilon_{\text{base}} = \frac{1}{2} \ln 3 \approx 0.549306$, $s_\epsilon = 1.0$, $w_{\text{disc}} = 0.0$ |
| **Data Source** | Hard-coded theoretical constant $\frac{1}{2} \ln 3$ (corresponding to an odds-ratio factor of 3). |
| **Heuristic vs Learned** | **Heuristic**: Chosen theoretically as a "standard unit" of logit perturbation. |
| **Calibrated?** | **No**. It has no measured relationship to actual sensor noise, image resolution, JPEG artifacts, or bounding box jitter. |
| **Split Used** | Fixed constant across all splits. |
| **Test Leakage Risk** | **None** (constant). |
| **Units & Range** | Field perturbation units, $\epsilon_i \in [0.01, \infty)$. |
| **Assumptions** | Assumes all claims have identical evidence uncertainty regardless of object size, image contrast, or feature detector reliability. |
| **Scientific Weakness** | Unary uncertainty cannot be defended as "robust to visual evidence noise" if $\epsilon_i$ has never been compared to empirical variations under image corruptions. |

---

### 2.3. Pairwise Coupling Strength $J_{ij}$

| Attribute | Current Implementation Status |
| :--- | :--- |
| **Code Location** | `src/experiments/parameterization.py:build_image_pgm_context()` |
| **Current Formula** | $J_{ij} = J_{\text{base}} = \frac{1}{2} \ln 3 \approx 0.549306$ for all edges $(i, j) \in \mathcal{E}$. |
| **Data Source** | Fixed configuration constant. |
| **Heuristic vs Learned** | **Heuristic**: Uniform constant across all edges. |
| **Calibrated?** | **No**. Not fitted or tuned against validation likelihood or co-occurrence statistics. |
| **Split Used** | Fixed constant across all splits. |
| **Test Leakage Risk** | **None** (constant). |
| **Units & Range** | Interaction energy $J \ge 0$, dimensionally matching $\theta$. |
| **Assumptions** | Assumes every pair of connected claims has identical attractive tendency to co-hallucinate or co-support. |
| **Scientific Weakness** | Claims of totally unrelated objects (e.g., "giraffe" and "microwave") should not have the same positive coupling as semantically linked objects (e.g., "fork" and "bowl"). Furthermore, edge structure is currently constructed on arbitrary alphabetical claim order! |

---

### 2.4. Global Perturbation Budget $B$

| Attribute | Current Implementation Status |
| :--- | :--- |
| **Code Location** | `src/experiments/parameterization.py:build_image_pgm_context()` |
| **Current Formula** | $B = \rho \cdot \epsilon_{\text{base}}$ where default $\rho = 1.0 \implies B = 0.549306$. |
| **Data Source** | Configuration scalar. |
| **Heuristic vs Learned** | **Heuristic**. |
| **Calibrated?** | **No**. Not derived from empirical aggregate perturbation bounds. |
| **Split Used** | Fixed constant across all splits. |
| **Test Leakage Risk** | **None** (constant). |
| **Units & Range** | Total $L_1$ field budget, $B \ge 0$. |
| **Assumptions** | Assumes total budget across an image is independent of tree size $n$, or equal to 1 single node's perturbation limit. |
| **Scientific Weakness** | If an image has 1 claim, $B = 0.55$ allows full local perturbation. If an image has 10 claims, $B = 0.55$ forces average perturbation to be $0.055$, heavily over-constraining larger images without empirical justification. |

---

## 3. Action Plan for Phase 9B

1. **Split-Safe Calibration Framework**: Establish explicit dataset roles:
   - `TRAIN`: Fit logistic evidence-fusion models ($d_i, g_i \mapsto p_i$).
   - `CALIBRATION`: Calibrate probabilities (Platt / Isotonic) and estimate perturbation distributions ($r_{ic} = |\theta_{ic} - \theta_{i,\text{clean}}|$).
   - `VALIDATION`: Tune coupling strength $\lambda$ and budget quantile $q$.
   - `TEST`: Evaluation only.
2. **Probability-to-$\theta$ Transformation**: Mathematically grounded mapping $\theta_i = \frac{1}{2} \ln(p_i / (1 - p_i))$ with verified inverse $\sigma(2\theta_i) = p_i$.
3. **Empirical $\epsilon_i$ Calibration**: Compute quantiles of residual field shifts under standardized image corruptions (blur, noise, compression).
4. **Empirical Budget $B$ Calibration**: Compute tree-level aggregate residual sums $S = \sum_j r_j$ under corruptions.
5. **Relationship-Weighted Couplings $J_{ij} = \lambda s_{ij}$**: Ground $s_{ij} \ge 0$ in semantic co-occurrence and visual overlap, guaranteeing $J_{ij} \ge 0$.
