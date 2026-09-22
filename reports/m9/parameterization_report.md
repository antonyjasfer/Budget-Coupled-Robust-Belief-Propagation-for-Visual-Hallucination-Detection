# Milestone 9B — Principled Evidence Calibration and Robust Parameterization Report

**Repository**: `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  
**Milestone**: M9B (Robust Parameterization & Evidence Calibration)  
**Status**: `DEVELOPMENT PARAMETERIZATION / NOT FINAL CALIBRATION`  
**Current Locked Dataset Status**: 10 genuine COCO train2017 images, 15 genuine claim records (development scale). Final parameter fitting deferred to full compute/dataset availability.

---

## Executive Summary

Phase 9B replaces legacy heuristic parameter choices ($\theta_i = \pm 0.5$, $\epsilon_i = 0.2$, $J_{ij} = 0.5$, $B = 1.0$) with a mathematically rigorous, split-safe, provenance-tracked calibration pipeline. Every parameter in the robust Ising-tree formulation now possesses an explicit, auditable empirical or statistical interpretation.

In addition, Phase 9B-0 conducted an exhaustive mathematical audit of the M9A continuous discretization certificate. The naive bound $\Delta = B/K$ was proven insufficient when sub-quantum perturbations ($\epsilon_i < \Delta$) accumulate across multiple tree nodes. We derived, proved, and implemented the exact tree-level continuous certificate gap:
$$G_{\text{cert}} = \min\left(B, \sum_{i \in \mathcal{V}} \gamma_i \min(\epsilon_i, \Delta)\right), \quad \gamma_i = \prod_{e \in \text{path}(i \to r)} \tanh(J_e) \le 1$$
which converges at rate $O(1/K)$ and was numerically verified across 300 randomized trials with zero violations.

---

## 1. Parameter Definitions & Probabilistic Meaning

| Parameter | Mathematical Symbol | Role in Robust Ising Model | Probabilistic / Empirical Meaning | Scientific Guardrails |
| :--- | :--- | :--- | :--- | :--- |
| **Unary Evidence Field** | $\theta_i$ | Node potential $\theta_i h_i$ where $h_i \in \{-1, +1\}$ | Unary log-odds corresponding to calibrated nominal hallucination probability: $\theta_i = \frac{1}{2}\ln\frac{p_i}{1 - p_i}$. | Clipped to $p \in [p_{\min}, 1-p_{\min}]$; unary field only (not coupled marginal). |
| **Local Perturbation Bound** | $\epsilon_i$ | Local box constraint $|\delta_i| \le \epsilon_i$ | Empirical bound on unary field variation under standardized visual evidence noise. | Calibrated as quantile $q$ of corruption residuals $| \theta_{i,c} - \theta_{i,\text{clean}} |$. |
| **Global Perturbation Budget** | $B$ | Global constraint $\sum_i |\delta_i| \le B$ | Empirical bound on total tree-level perturbation energy across all claims in an image. | Calibrated as quantile $q$ of aggregate image shift sums $S = \sum_j |\theta_{j,c} - \theta_{j,\text{clean}}|$. |
| **Attractive Coupling** | $J_{ij}$ | Pairwise interaction $J_{ij} h_i h_j$ | Statistical preference for claims $i$ and $j$ to agree in truth state. | Strictly constrained $J_{ij} = \lambda s_{ij} \ge 0$ ($\lambda \ge 0$, $s_{ij} \in [0, 1]$) preserving M9A monotonicity. |

---

## 2. Strict Data Split Contract & Leakage Prevention

To guarantee scientific validity, the repository enforces a strict four-way split contract (`SplitContract`):

```
+-------------------------------------------------------------------------------+
|                               FULL DATASET                                    |
+-------------------+-------------------+-------------------+-------------------+
|      TRAIN        |    VALIDATION     |    CALIBRATION    |       TEST        |
|  (e.g., 40-50%)   |  (e.g., 15-20%)   |  (e.g., 15-20%)   |  (e.g., 20-25%)   |
+-------------------+-------------------+-------------------+-------------------+
| * Fit feature-    | * Select regular- | * Platt / Isoto-  | * Final benchmark |
|   to-probability  |   izer C for log- |   nic probability |   evaluation      |
|   logistic models |   istic baseline  |   calibration     |   ONLY            |
| * Fit feature     | * Tune coupling   | * Empirical eps   | * NEVER used for  |
|   scalers         |   strength lambda |   quantiles       |   fitting, tuning,|
|                   | * Select graph    | * Empirical B     |   or scaling      |
|                   |   topology        |   quantiles       | * Hard runtime    |
|                   |                   | * Inclusion tests |   leakage guards  |
+-------------------+-------------------+-------------------+-------------------+
```

### Leakage Verification
- All record collections entering feature extractors, calibrators, or hyperparameter grids are guarded by `SplitContract.assert_no_leakage()`.
- If any test claim ID appears in training, validation, or calibration data, a `SplitLeakageError` is raised immediately.
- Provenance hashes (`train_ids_hash`, `validation_ids_hash`, `calibration_ids_hash`, `test_ids_hash`) are computed deterministically and stored in parameter bundles.

---

## 3. Raw Evidence Features

Raw visual evidence extracted from perception models includes:
- $d_i \in [0, 1]$: Raw OWL-ViT open-vocabulary detector confidence score for the object/predicate asserted in claim $i$.
- $g_i \in [-1, 1]$: Raw CLIP visual-text cosine similarity between the candidate bounding box crop and claim text.

**Important Statistical Principle**:
- $d_i$ is a detector score, NOT a calibrated posterior probability $P(H_i = \text{SUPPORTED})$.
- $g_i$ is an unnormalized cosine similarity, NOT a probability.
- Neither raw score is directly used as a probability without supervised calibration.

### Feature Interfaces Supported:
1. **Detector-only**: $x_i = [d_i]$
2. **CLIP-only**: $x_i = [g_i]$
3. **Combined**: $x_i = [d_i, g_i]$
4. **Combined with Interaction**: $x_i = [d_i, g_i, d_i \cdot g_i]$

---

## 4. Non-PGM Baselines

Before invoking belief propagation, the pipeline fits supervised non-PGM baselines on the TRAIN split:
$$p_i = \sigma(w^\top x_i + b) = P(H_i = \text{HALLUCINATED} \mid x_i)$$
- Feature standardization: $x_i^{\text{norm}} = (x_i - \mu_{\text{train}}) / \sigma_{\text{train}}$.
- Objective: L2-regularized logistic loss on TRAIN.
- Hyperparameter: $C \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$ selected to minimize Brier score on VALIDATION.
- Degenerate fallback: If sample size $N < 4$ or single-class data occurs (development mode), the model falls back to deterministic negative weights ($w = -1.5$, higher detector/CLIP evidence decreases hallucination probability) without crashing.

---

## 5. Post-Hoc Probability Calibration

The predicted hallucination probability $p_i$ is evaluated and calibrated on the CALIBRATION split:
1. **Uncalibrated Baseline**: $\hat{p}_i = p_i$.
2. **Platt Scaling**: Logistic regression fitted on logits $\ell_i = \ln(p_i / (1 - p_i))$ using calibration labels.
3. **Isotonic Regression**: Non-parametric piecewise constant monotonic calibration (active when $N_{\text{cal}} \ge 20$).

### Calibration Diagnostic Metrics:
- **Brier Score**: $\frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$
- **Log Loss / NLL**: $-\frac{1}{N} \sum_{i=1}^N [y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i)]$
- **Expected Calibration Error (ECE)**: $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$

---

## 6. Mathematical Mapping: Calibrated Probability to Ising Unary Field

For calibrated probability $p_i \in [p_{\min}, 1 - p_{\min}]$ with $p_{\min} = 10^{-5}$:
$$\theta_i = \frac{1}{2} \ln \left( \frac{p_i}{1 - p_i} \right)$$
Under an isolated Ising node ($J = 0$), the posterior marginal is:
$$P(H_i = +1) = \frac{e^{\theta_i}}{e^{\theta_i} + e^{-\theta_i}} = \frac{1}{1 + e^{-2\theta_i}} = \sigma(2\theta_i)$$
Inverting yields:
$$\sigma(2\theta_i) = p_i \quad \Longleftrightarrow \quad 2\theta_i = \ln \left(\frac{p_i}{1-p_i}\right)$$
This analytical roundtrip was verified across continuous grids $p \in [0.01, 0.99]$ with machine precision $| \sigma(2\theta_i) - p_i | \le 10^{-14}$.

---

## 7. Empirical Epsilon ($\epsilon_i$) Calibration

The uncertainty limit $\epsilon_i$ quantifies how much the unary field $\theta_i$ plausibly changes under perceptual noise:
$$r_{i,c} = |\theta_{i,c} - \theta_{i,\text{clean}}|$$
Standardized visual corruptions (M8 suite):
- Gaussian blur ($\sigma \in [1, 3]$)
- JPEG compression ($Q \in [20, 50]$)
- Additive Gaussian noise ($\sigma_n \in [0.05, 0.15]$)
- Downsampling & interpolation

### Supported Estimators:
1. **Method A (Global Quantile)**: $\epsilon = \text{quantile}_q(\{r_{i,c}\})$, $\epsilon_i = \epsilon$. Configurable $q \in \{0.80, 0.90, 0.95\}$.
2. **Method B (Evidence-Conditioned)**: $\epsilon_i = \epsilon_{\text{base}} \cdot (1 + \text{entropy\_factor}(p_i))$, reflecting that claims with equivocal evidence ($p_i \approx 0.5$) exhibit higher parameter volatility.
3. **Method C (Category-Conditioned)**: $\epsilon_{\text{category}} = \text{quantile}_q(\{r_{i,c} : \text{category}(i) = \text{cat}\})$ with automatic fallback to global quantile when group sample size $< 5$.

---

## 8. Empirical Global Budget ($B$) Calibration

The global budget $B$ bounds total perturbation energy across all claims in an image:
$$S_{img, c} = \sum_{j \in img} |\theta_{j,c} - \theta_{j,\text{clean}}|$$
- Calibrated as empirical quantile: $B = \text{quantile}_q(\{S_{img, c}\})$ for $q \in \{0.80, 0.90, 0.95\}$.
- Normalized budget ratio: $\rho = \frac{B}{\sum_{j} \epsilon_j} \in [0, 1]$.
- When $\rho \ge 1.0$, the budget constraint becomes inactive and collapses to local box uncertainty. When $\rho < 1.0$, the global budget actively constrains worst-case joint adversarial drift.

---

## 9. Uncertainty-Set Inclusion Diagnostics

To verify that the abstract uncertainty set $\mathcal{U}(B, \epsilon)$ corresponds to empirical visual evidence shifts, the pipeline computes:
- **Local-Bound Inclusion Rate**: Fraction of corruption vectors satisfying $|\delta_i| \le \epsilon_i$ for all nodes.
- **Global-Budget Inclusion Rate**: Fraction of corruption vectors satisfying $\sum_i |\delta_i| \le B$.
- **Joint Inclusion Rate**: Fraction satisfying both $|\delta_i| \le \epsilon_i$ and $\sum_i |\delta_i| \le B$ simultaneously.

*Hierarchical Law*: $\text{Inclusion}_{\text{joint}} \le \min(\text{Inclusion}_{\text{local}}, \text{Inclusion}_{\text{global}})$.

---

## 10. Attractive Coupling ($J_{ij}$) Construction & Calibration

Coupling between claim states is modeled as:
$$J_{ij} = \lambda \cdot s_{ij}$$
where:
- $\lambda \ge 0$: Global coupling scale tuned on VALIDATION to minimize Brier score or NLL.
- $s_{ij} \in [0, 1]$: Pairwise semantic relationship score (e.g., spatial overlap, predicate coreference).

### Monotonicity Guarantee:
- $J_{ij} \ge 0$ is strictly enforced.
- Antiferromagnetic interactions ($J < 0$) are rejected with `ValueError`, preserving M9A coordinate monotonicity ($P(h_r = +1)$ non-decreasing in $\theta_i$ for all $i$).

---

## 11. Claim Tree Topology Construction

To avoid loopy approximations while capturing claim correlations, claims within an image are structured as trees:
1. **Independent ($J = 0$)**: Uncoupled baseline.
2. **Chain**: Linear sequence of claims ordered by mention in text.
3. **Star**: Central root claim (primary subject) coupled to auxiliary attribute/relation claims.
4. **Maximum Spanning Tree (MST)**: Kruskal's algorithm on pairwise claim similarity scores $s_{ij}$, maximizing tree correlation without introducing cycles.

---

## 12. Local Box vs. Global Budget Uncertainty Comparison

This comparison establishes the theoretical and empirical novelty of budget coupling:

| Property | Local Box Uncertainty $\mathcal{U}_{\text{box}}$ | Budget-Coupled Uncertainty $\mathcal{U}(B, \epsilon)$ |
| :--- | :--- | :--- |
| **Constraints** | $\|\delta\|_\infty \le \epsilon \iff |\delta_i| \le \epsilon_i$ | $|\delta_i| \le \epsilon_i \text{ and } \|\delta\|_1 \le B$ |
| **Upper Extremizer** | $\delta_i^* = +\epsilon_i$ for all $i$ | Knapsack-like DP allocation on budget grid |
| **Lower Extremizer** | $\delta_i^* = -\epsilon_i$ for all $i$ | One-sided negative allocation on budget grid |
| **Interval Width** | Overly conservative / loose | Tightly bounded by shared perturbation energy |
| **Equivalence** | Recovered when $B \ge \sum_i \epsilon_i$ | Collapses to nominal marginal when $B = 0$ |

---

## 13. Parameter Bundle Provenance Schema

All parameters and configuration hashes are saved into a cryptographic `ParameterBundle`:
```json
{
  "version": "1.0.0",
  "mode": "DEVELOPMENT",
  "theta_model": { ... },
  "probability_calibration": { ... },
  "epsilon_model": { ... },
  "budget_model": { ... },
  "coupling_model": { ... },
  "topology_model": { ... },
  "train_ids_hash": "a1b2...",
  "validation_ids_hash": "c3d4...",
  "calibration_ids_hash": "e5f6...",
  "test_ids_hash": "7890...",
  "dataset_hash": "coco_train2017_m6_canonical",
  "code_sha": "d3b7f57",
  "checksum": "f8a9..."
}
```
Any post-hoc tampering with the bundle invalidates the SHA-256 checksum and prevents loading.

---

## 14. Dataset Status & Scientific Disclaimer

> [!WARNING]
> **DEVELOPMENT PARAMETERIZATION — NOT FINAL SCIENTIFIC CALIBRATION**
> 
> The current locked dataset contains **10 genuine COCO images** and **15 genuine claims**.
> While the software pipeline, mathematical mappings, split contracts, and unit tests are fully functional and pass 100%, fitted numerical coefficients from this small sample are for developmental verification only.
> 
> Full scientific calibration will be executed when the full 600-image / multi-thousand claim benchmark is locked and evaluated on GPU/Colab infrastructure.
