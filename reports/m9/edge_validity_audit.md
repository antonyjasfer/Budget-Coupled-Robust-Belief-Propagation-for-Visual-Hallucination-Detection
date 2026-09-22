# Empirical Edge Validity and Semantic Correlation Audit

> [!WARNING]
> **DEVELOPMENT ONLY — DO NOT TREAT AS FINAL**: Pipeline verification on development data.
> Sample size is insufficient for final scientific conclusions.

**Split Evaluated**: `TRAIN`  
**Total Candidate Edges Evaluated**: 5  
**Statistical Evidence Category**: `little/no evidence`  

## 1. Pairwise Truth-State Dependence Metrics

- **Evaluated Pairs ($n_{\text{pair}}$)**: 5
- **Concordant Pairs ($y_i = y_j$)**: 4
- **Discordant Pairs ($y_i \ne y_j$)**: 1
- **Empirical Agreement Rate**: 0.800
- **Phi Coefficient ($\phi$)**: 0.000
- **Odds Ratio**: 3.000
- **Mutual Information ($I$)**: 0.0000 bits

## 2. Semantic Similarity vs. Empirical Agreement Correlation

- **Spearman Rank Correlation ($\rho$)**: 0.000
- **P-Value**: 1.0000
- **Scientific Conclusion**: Weak or negligible correlation: semantic similarity functions primarily as an intuitive structural heuristic rather than an empirical proxy for truth-state correlation.

## 3. Methodological Implications

> [!IMPORTANT]
> Semantic similarity $s_{ij} \in [0, 1]$ represents conceptual/contextual proximity between claims.
> It DOES NOT automatically establish that hallucination truth-states are positively correlated.
> If empirical correlation is weak, edge weights must be acknowledged as a structural heuristic
> rather than a proven causal or statistical law.

## 4. Evaluated Edge Details (Sample)

| Image ID | Claim $u$ | Claim $v$ | Semantic Similarity $s_{uv}$ | $y_u$ | $y_v$ | Concordant |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `coco_000000000009` | `claim_resp_4a42737a947ef125_bowl` | `claim_resp_4a42737a947ef125_dining table` | 0.990 | 1 | 1 | YES |
| `coco_000000000009` | `claim_resp_4a42737a947ef125_apple` | `claim_resp_4a42737a947ef125_banana` | 0.989 | 1 | 1 | YES |
| `coco_000000000009` | `claim_resp_4a42737a947ef125_apple` | `claim_resp_4a42737a947ef125_bowl` | 0.967 | 1 | 1 | YES |
| `coco_000000000036` | `claim_resp_22bb468c4429202d_person` | `claim_resp_22bb468c4429202d_umbrella` | 0.987 | 1 | 0 | NO |
| `coco_000000000049` | `claim_resp_3abdad6658819b1c_horse` | `claim_resp_3abdad6658819b1c_person` | 0.937 | 1 | 1 | YES |