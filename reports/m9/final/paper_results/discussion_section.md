# Discussion: Interpretation, Evidence, and Boundaries

> **STATUS: DEVELOPMENT ONLY — NO EMPIRICAL FINDINGS.**
> 0 ground-truth-labelled claims were evaluated. Nothing in this file is a
> research result. Do not quote any number below as a finding.

This file separates mathematical facts, measured quantities, and interpretations. Interpretations are written by the authors only after labelled results exist; this generator never writes them.

---

### [FACT: Mathematical Properties (hold by construction)]
1. $0 \le L_i \le U_i \le 1$ and $w_i = U_i - L_i \ge 0$.
2. With $B = 0$ the interval collapses to the point marginal; widths are non-decreasing in $B$ because the uncertainty sets are nested.
3. Root dominance: if $B \le \varepsilon_i$, then $[L_i, U_i] = [\sigma(2(\eta_i - B)), \sigma(2(\eta_i + B))]$ exactly (see docs/theory/root_dominance.md).
4. The grid DP costs $\mathcal{O}(n K^2)$ per target node.

### [FACT: Measured in this run]
- Evaluated (ground-truth-labelled) claims: 0
- Mean / median interval width: 0.216 / 0.177
- Threshold-crossing rate ($L_i < \tau < U_i$): 28.3%
- Fraction of records in the root-dominance regime: 0.840

---

### [INTERPRETATION: Pending]
No interpretation is generated automatically. Questions to answer with labelled data, each against its baseline:
1. Does abstaining when $L_i < \tau < U_i$ beat margin abstention on the point posterior at equal coverage?
2. Does width track real evidence shift (corruption, detector change) beyond what the point posterior already explains?
3. Does claim coupling ($J \neq 0$) improve detection on images with 3 or more claims?

---

### [LIMITATION: Structural Boundaries and Unknowns]
1. **Tree Topology Assumption**: Exact inference requires a tree; cyclic dependencies must be pruned or approximated.
2. **Discretisation**: Grid bounds are within the certificate gap $G = \min(B, \sum_i \gamma_i \min(\varepsilon_i, \Delta))$ of the continuous bounds; report $G$ with every interval.
3. **Uncalibrated Uncertainty Set**: Unless $\varepsilon$ and $B$ are calibrated to measured evidence shifts, the interval has no coverage meaning.
4. **Upstream Evidence**: Detector and CLIP biases bound the quality of the unary potentials; robust inference cannot recover evidence that is missing.
