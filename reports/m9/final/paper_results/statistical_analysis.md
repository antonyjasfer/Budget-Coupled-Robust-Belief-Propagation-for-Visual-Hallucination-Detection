# Statistical Analysis: Hypothesis Testing and Bootstrap Inference

## 1. Paired Image-Level Cluster Bootstrap (Robust BP vs. Standard BP)

To account for within-image claim dependencies, we apply paired image-level cluster bootstrap resampling ($n = 1000$ iterations). Differences are reported as $\Delta = \text{Robust BP} - \text{Standard BP}$.

| Metric | Point Estimate ($\Delta$) | 95% Bootstrap Confidence Interval | Significance ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: |
| **Accuracy Difference** | N/A | [N/A, N/A] | No (overlapping zero) |
| **F1 Score Difference** | N/A | [N/A, N/A] | No (overlapping zero) |
| **ROC-AUC Difference** | N/A | [N/A, N/A] | No (overlapping zero) |

## 2. Correlation Hypotheses on Interval Width

We evaluate whether robust interval width $w_i = U_i - L_i$ contains structured diagnostic information:

- **Hypothesis 1 (Width vs. Standard BP Error)**:
  - Spearman correlation: N/A (p = N/A)
  - Point-biserial correlation: N/A
- **Hypothesis 2 (Width vs. Multimodal Evidence Conflict)**:
  - Pearson correlation: -0.625 (p = 0.000)
- **Hypothesis 3 (Width vs. Visual Corruption Severity)**:
  - Spearman rank correlation: N/A (p = N/A)
