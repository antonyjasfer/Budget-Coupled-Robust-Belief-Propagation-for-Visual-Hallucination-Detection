# Results: Empirical Evaluation of Budget-Coupled Robust Belief Propagation

> **STATUS: DEVELOPMENT ONLY — NO EMPIRICAL FINDINGS.**
> 0 ground-truth-labelled claims were evaluated. Nothing in this file is a
> research result. Do not quote any number below as a finding.

## 1. Primary Classification Performance

We evaluate Visual Hallucination Detection across three primary formulations:
- **Evidence-Only Baseline**: Independent detector and vision-language association scores.
- **Standard Belief Propagation (Standard BP)**: Exact tree-structured probabilistic inference over interdependent existence claims.
- **Budget-Coupled Robust Belief Propagation (Robust BP)**: Worst-case bounded marginal optimization subject to a global $\ell_1$ uncertainty budget ($B$).

| Inference Framework | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Evaluated Claims |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Evidence-Only Baseline** | N/A | N/A | N/A | N/A | N/A | 0 |
| **Standard BP (Point Marginal)** | N/A | N/A | N/A | N/A | N/A | 375 |
| **Robust BP (Interval Midpoint)** | N/A | N/A | N/A | N/A | N/A | 375 |

*Note: Binary hallucination decisions use threshold $\tau = 0.5$. Positive class is 'HALLUCINATED'. UNKNOWN instances are excluded from binary metrics per protocol.*

## 2. Robust Posterior Interval Characterization

Robust BP computes model-based lower and upper marginal bounds $[L_i, U_i]$ for each claim $i$ over the specified uncertainty set (grid solution plus discretisation certificate). These bounds hold only for that set; they are not calibrated probabilities or confidence intervals unless the set itself is calibrated.

- **Mean Interval Width**: 0.216 ($\pm$ 0.189)
- **Median Interval Width**: 0.177
- **Observed Range**: [0.000, 0.587]
- **Evidence-Sensitive Decisions ($\tau \in [L_i, U_i]$)**: 106 records (28.3%)

## 3. Evidence-Sensitive Abstention and Selective Prediction

When $L_i < \tau < U_i$, the decision can flip under some perturbation in the uncertainty set. Whether abstaining on these claims improves reliability is an empirical question that must be answered with labelled data and compared against margin abstention on the point posterior.

- **Records in the root-dominance regime ($B \le \varepsilon_i$)**: 0.840 of all records. For these records the interval equals $\sigma(2(\eta_i \pm B))$ exactly, so it carries no information beyond the point posterior.
