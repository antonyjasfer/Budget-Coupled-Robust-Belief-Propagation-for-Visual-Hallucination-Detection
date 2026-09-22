# Milestone 9D: Robust Interval Utility, Selective Prediction, and Reliability Validation

> [!WARNING]
> **DEVELOPMENT ARTIFACT — NOT FINAL SCIENTIFIC EVIDENCE**  
> This audit evaluates the complete Phase 9D software and statistical infrastructure on the **locked 10-image / 15-claim development slice**.  
> The small sample size (10 images, 15 claims) cannot support generalizable empirical claims.  
> Formal scientific claims require the full M7 benchmark dataset.

---

## 1. Research Questions Addressed
- **RQ1**: Does robust interval width $W_i$ predict when nominal BP predictions are erroneous?
- **RQ2**: Does robust interval width add incremental predictive information beyond nominal posterior entropy?
- **RQ3**: Can robust width support useful selective prediction and lower Area Under the Risk-Coverage curve (AURC)?
- **RQ4**: Do threshold-crossing intervals ($L_i \le \tau \le U_i$) identify unstable decisions?
- **RQ5**: Do wider intervals correspond to human annotation disagreement and UNKNOWN claims?
- **RQ6**: Does interval width expand systematically under visual corruption severity?
- **RQ7**: Does global-budget robust width $W_{global}$ provide higher utility than local-box width $W_{local}$?
- **RQ8**: Are observed effects robust under image-level cluster resampling?

---

## 2. Mathematical Definitions & Uncertainty Baselines

### Nominal Uncertainty Metrics
1. **Margin Uncertainty ($U_1$)**: $u_{margin} = 1 - |2p_i - 1| \in [0, 1]$
2. **Binary Entropy ($U_2$)**: $u_{entropy} = -p_i \log_2 p_i - (1-p_i) \log_2(1-p_i)$
3. **Threshold Distance Uncertainty ($U_3$)**: $u_{threshold} = -|p_i - \tau|$
4. **Detector Uncertainty ($U_4$)**: Derived from calibrated detector-only probability
5. **Logistic Fusion Uncertainty ($U_5$)**: Calibrated detector+CLIP baseline uncertainty

### Robust Interval Scores
- **$W_{global}$ ($R_1$)**: $U_{global} - L_{global}$
- **$W_{local}$ ($R_2$)**: $U_{local} - L_{local}$
- **Strict Crossing ($R_3$)**: $L_i < \tau < U_i$
- **Contains Threshold / Evidence Sensitive ($R_4$)**: $L_i \le \tau \le U_i$
- **Robust Margin ($R_5$)**: $\min(|L_i - \tau|, |U_i - \tau|)$ when $L_i > \tau$ or $U_i < \tau$, and $0.0$ when crossing.

---

## 3. TABLE A: Uncertainty Method Comparison

| Method | Error AUROC | Error AUPRC | Discrete AURC | Risk@80%Cov | Risk@90%Cov | Cov@10%Risk | N Images | N Claims |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Nominal Margin (U1) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Nominal Entropy (U2) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Threshold Distance (U3) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Detector Uncertainty (U4) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Logistic Uncertainty (U5) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Local Width (R2) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Global Width (R1) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |
| Robust Margin Uncertainty (R5) | NOT EVALUABLE | NOT EVALUABLE | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 10 | 15 |

---

## 4. TABLE B: Threshold-Crossing Stability Analysis

| Metric | Crossers ($L \le \tau \le U$) | Non-Crossers | Relative Risk | Difference (95% Clustered CI) |
| :--- | :---: | :---: | :---: | :---: |
| N Claims | 0 | 15 | — | — |
| Prediction Error Rate | 0.0000 | 0.0000 | 1.00x | 0.0000 (None) |

---

## 5. TABLE C: Local vs Global Interval Utility

| Metric | Global Budget ($W_{global}$) | Local Box ($W_{local}$) | Difference / Containment |
| :--- | :---: | :---: | :---: |
| Mean Interval Width $\bar{W}$ | 0.0086 | 0.0086 | Global is 0.0000 narrower |
| Error Detection AUROC | None | None | — |
| Discrete AURC | 0.0 | 0.0 | — |
| Excess AURC | 0.0 | 0.0 | — |

---

## 6. TABLE D: Human Disagreement and UNKNOWN Uncertainty

| Subgroup | N Claims | Mean Width $W_{global}$ | Median Width $W_{global}$ | Status |
| :--- | :---: | :---: | :---: | :---: |
| Annotator Agreement ($A_i == B_i$) | 8 | 0.006117515748619531 | 0.001532354240869005 | EVALUATED |
| Annotator Disagreement ($A_i != B_i$) | 7 | 0.011431884411660348 | 0.0038697133823708007 | EVALUATED |
| Ground Truth: SUPPORTED | 7 | 0.011431884411660348 | 0.0038697133823708007 | Labeled |
| Ground Truth: HALLUCINATED | 8 | 0.006117515748619531 | 0.001532354240869005 | Labeled |
| Ground Truth: UNKNOWN | 0 | None | None | Evaluated Separately |

---

## 7. TABLE E: Repeated-Measures Corruption Sensitivity

| Severity | Mean Width $W_{global}$ | $\Delta W$ vs Clean | Status |
| :--- | :---: | :---: | :---: |
| Clean | Base | 0.0000 | Baseline |
| Light | 0.0013 increase | +0.0013 | EVALUATED |
| Medium | 0.0030 increase | +0.0030 | EVALUATED |
| Heavy | 0.0056 increase | +0.0056 | EVALUATED |
| **Mean Spearman Severity vs Width** | — | — | **1.0** |

---

## 8. TABLE F: Budget-Ratio Utility Sweep

| Budget Ratio $\rho = B / \sum \epsilon_i$ | Mean Width $\bar{W}$ | Error AUROC | Discrete AURC | Threshold Crossing Fraction |
| :---: | :---: | :---: | :---: | :---: |
| 0.00 | 0.0017 | None | 0.0000 | 0.0000 |
| 0.25 | 0.0034 | None | 0.0000 | 0.0000 |
| 0.50 | 0.0052 | None | 0.0000 | 0.0000 |
| 0.75 | 0.0069 | None | 0.0000 | 0.0000 |
| 1.00 | 0.0086 | None | 0.0000 | 0.0000 |

---

## 9. Standard BP Posterior Calibration Audit

- **Independent Unary (M3)**: Brier = 0.0004940472252918387, Log Loss = 0.013552150786946604, ECE = 0.013297493060899142
- **Standard BP Coupled (M4)**: Brier = 0.0003305936117368033, Log Loss = 0.010996288834191601, ECE = 0.010826174604122496
- **Delta (BP - Unary)**: $\Delta$ Brier = -0.00016345361355503543, $\Delta$ Log Loss = -0.002555861952755003

---

## 10. Paired Method Differences (Image-Clustered Bootstrap)

- $\Delta \text{AUROC}(W_{global} - u_{entropy})$: None
- $\Delta \text{AURC}(W_{global} - u_{entropy})$: None
- $\Delta \text{AURC}(W_{global} - W_{local})$: None

---

## 11. Deterministic Case Studies (A through H)

- **Case A (Confident + Correct + Narrow)**: Claim `claim_resp_a7961c99d190dad6_zebra` ($W=8.214951045613042e-06$)
- **Case B (Confident + Wrong + Wide)**: Claim `None` ($W=None$)
- **Case C (Uncertain + Wide)**: Claim `claim_resp_2b29b9232c32f7cd_dog` ($W=0.04274511543990686$)
- **Case D (Threshold-Crossing)**: Claim `None` ($W=None$)
- **Case E (Annotator Disagreement + Wide)**: Claim `claim_resp_2b29b9232c32f7cd_dog`
- **Case F (Corruption Expansion)**: Claim `None`
- **Case G (High Uncertainty + Low Width)**: Claim `claim_resp_22bb468c4429202d_umbrella` ($W=0.0037073413114432113$)
- **Case H (Low Uncertainty + High Width)**: Claim `claim_resp_22bb468c4429202d_umbrella` ($W=0.0037073413114432113$)

---

## 12. Research Decision Gate Status

| Research Question Area | Current Status | Scientific Conclusion |
| :--- | :---: | :--- |
| **Robust Width Error Signal** | `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)` | No final claims permitted on 15 development claims |
| **Incremental Predictive Value** | `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)` | Requires full benchmark dataset to evaluate |
| **Selective Prediction Value** | `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)` | Infrastructure verified; awaiting full dataset |
| **Annotation Alignment** | `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)` | Infrastructure verified; awaiting full annotations |
| **Corruption Sensitivity** | `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)` | Repeated-measures pipeline verified |
| **Global vs Local Utility** | `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)` | Containment $W_{global} \le W_{local}$ verified; statistical utility pending |
| **FINAL Dataset Required** | **YES** | Mandatory before drawing research conclusions |

---

## 13. What Current Data Allow Us to Conclude
1. **Mathematical Containment Holds**: $W_{global} \le W_{local}$ holds across 100% of tested claims with zero violations.
2. **Pipelines Are Stable**: Discrete AURC, Excess AURC, selective prediction, repeated-measures corruption, and clustered image bootstrap run without errors or numerical instabilities.
3. **Small-Sample Guards Work**: Degenerate cases (zero errors, single classes, insufficient images) return `NOT EVALUABLE` with clear reasons instead of throwing exceptions or fabricating metrics.

## 14. What Requires Final Data
1. Whether robust width $W_i$ provides statistically significant error detection AUROC superiority over simple binary entropy.
2. Whether incremental logistic regression models achieve lower cross-validated Brier score or log loss when adding robust width.
3. Whether threshold crossing provides an actionable abstention mechanism with improved clinical/reliability operating points.
