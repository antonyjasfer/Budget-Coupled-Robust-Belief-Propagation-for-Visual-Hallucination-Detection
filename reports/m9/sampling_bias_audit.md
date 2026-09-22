# Phase 9E Sampling Bias and Pre-Label Isolation Audit

**Timestamp:** 2026-09-22T16:35:28.998599+00:00  
**Target Cohort:** 600 MS COCO 2017 Images  
**Audit Status:** VERIFIED UNBIASED

---

## 1. Compliance Audit Against Pre-Label Independence Rules

| Criterion | Mandated Status | Verified Implementation | Compliance |
|---|---|---|:---:|
| Selection conditioned on claim count? | **STRICTLY PROHIBITED** | Image-level PRNG shuffle over eligible universe | **COMPLIANT** |
| Selection conditioned on graph size? | **STRICTLY PROHIBITED** | Selection ignores PGM topology entirely | **COMPLIANT** |
| One-claim image replacement? | **STRICTLY PROHIBITED** | All sampled images retained regardless of claims | **COMPLIANT** |
| Selection conditioned on model scores? | **STRICTLY PROHIBITED** | No detector or CLIP scores used in sampling | **COMPLIANT** |
| Selection conditioned on human labels? | **STRICTLY PROHIBITED** | Sampling precedes annotation | **COMPLIANT** |
| Frozen split determination? | **FROZEN PRE-LABEL** | Seed 42 frozen before annotation | **COMPLIANT** |

---

## 2. Graph Complexity Subgroup vs. Primary Cohort Separation

- **Primary Cohort:** Exactly 600 images representing the natural MS COCO distribution.
- **Graph-Evaluable Subgroup ($G \ge 2$):** Evaluated strictly as a planned subgroup analysis.
- **Graph-Stress Supplement:** Stored separately in `graph_stress_manifest.json` if needed; never merged with primary representative cohort metrics.
