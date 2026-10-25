# Research Summary: Budget-Coupled Robust Belief Propagation for Visual Hallucination Detection

> **STATUS: DEVELOPMENT ONLY — NO EMPIRICAL FINDINGS.**
> 0 ground-truth-labelled claims were evaluated. Nothing in this file is a
> research result. Do not quote any number below as a finding.

### 1. Research Problem
Large vision-language models generate object claims that the image does not support. We study how stable a claim-level hallucination decision is when the visual evidence behind it shifts.

### 2. Method (as implemented)
- Claims form a tree-structured Ising model; unary fields $\theta_i$ come from detector (and optionally CLIP) evidence.
- Uncertainty set: $|\delta_i| \le \varepsilon_i$ and $\sum_i |\delta_i| \le B$.
- Output: model-based bounds $[L_i, U_i]$ over that set, computed by an exact grid DP with a discretisation certificate.

### 3. Data in this run
- Records: 375 · unique images: 10 · unique claims: 15
- Ground-truth-labelled claims evaluated: 0

### 4. Main Results
Not available: no ground-truth-labelled claims in this run.

### 5. Interval Statistics (descriptive only)
- Mean / median width: 0.216 / 0.177
- Threshold-crossing rate: 28.3%
- Records in the root-dominance regime ($B \le \varepsilon_i$, interval = $\sigma(2(\eta \pm B))$): 0.840

### 6. Ablations, Corruption Study, Statistical Tests
Not available: no ground-truth-labelled claims in this run.

### 7. Known Limitations
Tree topology; grid resolution (certificate gap reported per interval); uncalibrated $\varepsilon$ and $B$; correlated evidence sources (OWL-ViT uses a CLIP backbone); root-dominance regime when $B \le \varepsilon_i$.

### 8. Conclusion
No scientific conclusion is generated automatically. Conclusions require labelled results, the pre-registered tests in docs/PREREG.md, and author review.
