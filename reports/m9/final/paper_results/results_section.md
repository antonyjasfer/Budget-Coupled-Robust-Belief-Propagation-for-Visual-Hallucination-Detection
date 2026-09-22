# Results: Empirical Evaluation of Budget-Coupled Robust Belief Propagation

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

Rather than collapsing posterior uncertainty into a single scalar point estimate, Robust BP computes guaranteed upper and lower marginal bounds $[L_i, U_i]$ for each claim $i$. The interval width $w_i = U_i - L_i$ measures epistemic and evidence sensitivity.

- **Mean Interval Width**: 0.216 ($\pm$ 0.189)
- **Median Interval Width**: 0.177
- **Observed Range**: [0.000, 0.587]
- **Evidence-Sensitive Decisions ($\tau \in [L_i, U_i]$)**: 0 claims (0.0%)

## 3. Evidence-Sensitive Abstention and Selective Prediction

When claims exhibit $L_i \le \tau \le U_i$, the classification decision is sensitive to allowable perturbations within budget $B$. Treating this condition as an abstention criterion allows the system to defer judgment on ambiguous claims, trading coverage for increased reliability on non-abstaining predictions.
