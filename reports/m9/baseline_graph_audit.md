# Baseline Necessity, Graph Validity, and Core-Contribution Ablation Audit

**Repository**: `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  
**Milestone**: M9C (Baseline Necessity & Ablations)  
**Mode**: `DEVELOPMENT (Sample size insufficient for final scientific conclusions)`  

---

## 1. Executive Summary & Research Decision Gate

- **Graph Testability**: `FINAL DATA REQUIRED`
- **Graph Value**: `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)`
- **Global Budget Value**: `NOT EVALUABLE (DEVELOPMENT SIGNAL ONLY)`
- **Simple Baseline Challenge (M2 vs M6)**: `FINAL COMPARISON REQUIRED`
- **Final Data Required**: `True`

---

## 2. Dataset Graph Coverage Audit

- **Total Images**: 10
- **Total Claims**: 15
- **Single-Claim Images ($N=1$)**: 7 (graph coupling inactive by definition)
- **Multi-Claim Images ($N \ge 2$)**: 3
- **Mean Claims / Image**: 1.50
- **Total Candidate Graph Edges**: 0
- **Claims Influenced by at Least One Edge**: 0.0%
- **Status**: `GRAPH-EVALUATION DATA INSUFFICIENT`

---

## 3. Controlled Baseline Ladder (M0 to M6)

### Baseline Ladder Comparison Table (DEVELOPMENT)

| Method | Evidence | Graph | Uncertainty | Global Budget | Acc | Prec | Rec | F1 | ROC-AUC | PR-AUC | Brier | LogLoss | Mean Width | Runtime (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M0_detector_only` | detector_only | NO | NONE | NO | 0.933 | 0.889 | 1.000 | 0.941 | 1.000 | 0.993 | 0.075 | 0.273 | N/A | 0.0 |
| `M1_clip_only` | clip_only | NO | NONE | NO | 0.467 | 0.000 | 0.000 | 0.000 | 0.982 | 0.942 | 0.253 | 0.699 | N/A | 0.1 |
| `M2_logistic_fusion` | combined | NO | NONE | NO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.014 | N/A | 0.0 |
| `M3_unary_isolated` | combined | NO | NONE | NO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.014 | N/A | 0.1 |
| `M4_standard_bp` | combined | YES | NONE | NO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.011 | N/A | 1.2 |
| `M5_local_box_robust` | combined | YES | LOCAL | NO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.011 | 0.011 | 10.1 |
| `M6_budget_coupled_robust` | combined | YES | GLOBAL | YES | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.011 | 0.011 | 23.2 |

---

## 4. Component Contribution Transitions

### Component Contribution Step-by-Step Table

| Transition | Conceptual Component | Delta F1 | Delta Brier | Delta LogLoss | Delta ROC-AUC | Interval Width Change |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `M0 -> M2` | Evidence Fusion (Detector + CLIP) | +0.059 | -0.074 | -0.260 | +0.000 | None |
| `M2 -> M3` | Ising Field Representation (J=0) | +0.000 | +0.000 | +0.000 | +0.000 | None |
| `M3 -> M4` | Graph Coupling Effect (Standard BP) | +0.000 | -0.000 | -0.003 | +0.000 | None |
| `M4 -> M5` | Local Robustness Addition (Box Bounds) | +0.000 | +0.000 | +0.000 | +0.000 | Added Width |
| `M5 -> M6` | Global Budget Coupling (L1 Shared Energy) | +0.000 | +0.000 | +0.000 | +0.000 | -0.000 |

---

## 5. Graph Necessity Test (M3 vs. M4)

Testing whether pairwise coupling ($J > 0$) improves over calibrated independent unaries ($J = 0$):
- **$\Delta$ Brier Score ($M4 - M3$)**: -0.0002
- **$\Delta$ Log Loss ($M4 - M3$)**: -0.0027
- **$\Delta$ F1**: +0.0000
- **$\Delta$ ROC-AUC**: +0.0000
- **Probabilistic Improvement Observed**: `False`

---

## 6. Topology Ablation Summary

| Topology | Tuned $\lambda$ | Total Edges | Acc | F1 | ROC-AUC | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `independent` | 0.00 | 0 | 0.067 | 0.000 | 0.000 | 0.633 |
| `chain` | 0.40 | 5 | 0.067 | 0.000 | 0.000 | 0.659 |
| `star` | 0.40 | 5 | 0.067 | 0.000 | 0.000 | 0.651 |
| `minimum_spanning_tree` | 0.40 | 5 | 0.067 | 0.000 | 0.000 | 0.651 |

---

## 7. Local Box vs. Global Budget Uncertainty

- **Containment Violations ($L_{\text{local}} \le L_{\text{global}} \le U_{\text{global}} \le U_{\text{local}}$)**: 0 (100% verified)
- **Mean Local Box Width ($W_{\text{local}}$)**: 0.011
- **Mean Global Budget Width ($W_{\text{global}}$)**: 0.011
- **Mean Width Reduction ($W_{\text{local}} - W_{\text{global}}$)**: 0.000
- **Local Box Error Detection AUROC**: 0.500
- **Global Budget Error Detection AUROC**: 0.500

---

## 8. Graph x Uncertainty Factorial Matrix (2x3)

| ID | Graph | Uncertainty | Description | F1 | Brier | LogLoss | Mean Width |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `A` | NO | NONE | J=0, No Uncertainty (Independent Unaries) | 1.000 | 0.000 | 0.014 | N/A |
| `B` | YES | NONE | J>0, No Uncertainty (Standard Tree BP) | 1.000 | 0.000 | 0.011 | N/A |
| `C` | NO | LOCAL | J=0, Local Box Uncertainty | 1.000 | 0.000 | 0.014 | 0.013 |
| `D` | YES | LOCAL | J>0, Local Box Uncertainty (Uncoupled Budget) | 1.000 | 0.000 | 0.011 | 0.011 |
| `E` | NO | GLOBAL | J=0, Global Budget Uncertainty (Budgeted Independent) | 1.000 | 0.000 | 0.014 | 0.013 |
| `F` | YES | GLOBAL | J>0, Global Budget Uncertainty (Full Proposed Method) | 1.000 | 0.000 | 0.011 | 0.011 |

---

## 9. Limitations & Requirements for Final Run

1. **Development Scale**: Current findings are derived from the development slice (10 images, 15 claims).
2. **Single-Claim Domination**: Images with only 1 claim cannot evaluate graph coupling benefits.
3. **Next Step**: Execute Phase 9D final consolidation once locked 600-image dataset is annotated.