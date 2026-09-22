# Discussion: Interpretation, Evidence, and Boundaries

To maintain rigorous scientific standards, this discussion explicitly partitions findings into directly measured facts, evidence-supported interpretations, and known limitations.

---

### [FACT: Directly Measured Observations]
1. **Interval Bounds**: Across all evaluated configurations, the mathematical constraint $0 \le L_i \le U_i \le 1$ held without exception, and the calculated width satisfies $w_i = U_i - L_i \ge 0$.
2. **Interval Expansion under Budget**: Setting budget $B = 0$ reduces the robust interval to the point marginal $L_i = U_i = P(H_i = +1 \mid E)$. As budget $B$ increases, interval widths expand non-decreasingly.
3. **Evidence Sensitivity**: A measurable fraction of claim decisions span the nominal decision threshold ($L_i < \tau < U_i$), identifying claims whose classification depends directly on evidence uncertainty.
4. **Discretization Complexity**: Inference runtime per claim scales linearly with the numerical grid resolution $K$.

---

### [INTERPRETATION: Evidence-Based Scientific Analysis]
1. **Uncertainty Calibration**: Wide posterior intervals correlate with evidence conflict between the object detector and the vision-language projection (CLIP), indicating that interval width serves as a valid indicator of multimodal ambiguity.
2. **Selective Risk Reduction**: Filtering out evidence-sensitive claims before automated downstream actions reduces error rates on the retained set, supporting the utility of robust intervals for safety-critical deployment.
3. **Graph Coupling vs. Independent Bounds**: Incorporating pairwise spatial/co-occurrence potentials enforces consistency across related objects, preventing isolated false positives that contradict connected detections.

---

### [LIMITATION: Structural Boundaries and Unknowns]
1. **Tree Topology Assumption**: Exact polynomial-time belief propagation relies on tree-structured factor graphs. Applying the method to dense cyclic graphs requires either tree-decomposition or loopy approximations.
2. **Discretization Approximation**: The grid-based numerical solver operates at resolution $K$. While empirically stable, continuous analytical bounds may differ by $\mathcal{O}(1/K)$.
3. **VLM and Evidence Bias**: Upstream detector miss rates or CLIP representation biases bound the quality of initial potentials; robust BP models uncertainty over given potentials but cannot recover completely missing visual primitives.
