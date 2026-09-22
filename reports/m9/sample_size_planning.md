# Phase 9E Sample-Size Precision Planning and Event Adequacy

**Cohort:** MS COCO 600-Image Representative Benchmark  
**Evaluation Partition:** 120 Test Images (~180–240 Test Claims)  
**Methodology:** Precision Planning via Binomial Margin of Error (No Unsubstantiated Power Claims)

---

## 1. Mathematical Formulation

In evaluating hallucination detectors on visual QA and caption claims, the parameter of interest is the true hallucination rate $p \in (0, 1)$ or the model error rate $\mathrm{Err} \in (0, 1)$.

The normal-approximation margin of error at confidence level $(1 - \alpha)$ is:
$$\mathrm{MoE} = z_{1 - \alpha/2} \sqrt{\frac{p(1 - p)}{n}}$$

Where:
- $z_{0.975} = 1.95996$ for a 95% two-sided confidence interval.
- $n$ is the total number of evaluated claims in the test partition.
- $p$ is the nominal baseline hallucination prevalence ($p \approx 0.20$).

---

## 2. Precision Table Across Sample Sizes

| Sample Size ($n$) | Expected $p$ | 95% MoE ($\pm$) | 95% Confidence Interval | Precision Assessment |
|---|---|---|---|---|
| 50 | 0.20 | $\pm 0.1109$ | [0.0891, 0.3109] | Exploratory / Wide |
| 100 | 0.20 | $\pm 0.0784$ | [0.1216, 0.2784] | Moderate Precision |
| 120 (Min Test) | 0.20 | $\pm 0.0716$ | [0.1284, 0.2716] | Target Minimum (1 claim/img) |
| 200 (Expected Test) | 0.20 | $\pm 0.0554$ | [0.1446, 0.2554] | **Standard Research Precision** |
| 300 | 0.20 | $\pm 0.0453$ | [0.1547, 0.2453] | High Precision |

---

## 3. Exclusion of Unsubstantiated Statistical Power Claims

As mandated by scientific approval corrections:
- We do **NOT** claim '80% statistical power' for comparing models.
- Statistical power requires an explicit non-null effect size ($\Delta$), alternative hypothesis distribution, and variance model under clustering.
- In this benchmark, the 120-image test set is designed for **precision bounded interval estimation** ($\mathrm{MoE} \le \pm 0.055$) rather than rejecting arbitrary null hypotheses.

---

## 4. Post-Hoc Test Event Adequacy Policy

When human ground-truth labels are imported for the Test partition:
1. The descriptive event counts ($N_\text{hallucinated}$, $N_\text{supported}$, $N_\text{unknown}$) will be recorded.
2. If minority class count $< 10$, the status is marked **LOW EVENT COUNT / NOT EVALUABLE**.
3. **Strict Freezing Constraint:** Under no circumstances are sampling rules, splits, model hyperparameters, or thresholds adjusted post-hoc in response to low event counts.
