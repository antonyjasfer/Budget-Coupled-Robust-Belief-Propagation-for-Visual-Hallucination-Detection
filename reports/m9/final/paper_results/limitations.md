# Limitations and Scope of Validity

1. **Tree Topology Assumption**: Exact tree-structured message passing requires acyclic claim factor graphs. Cyclic dependency structures must be approximated or clustered.
2. **Grid Discretization**: Continuous worst-case budgets are solved on a grid with $K$ steps. The continuous bound lies within the certificate gap $G = \min(B, \sum_i \gamma_i \min(\epsilon_i, \Delta))$ of the grid value.
3. **Multimodal Model Dependence**: Unary potentials depend on frozen upstream models (currently OWL-ViT and CLIP; note OWL-ViT is built on a CLIP backbone, so the two signals are correlated). Systematic blind spots in these models cannot be removed by inference.
6. **Root-Dominance Regime**: When $B \le \epsilon_i$, the robust interval is a closed-form function of the point posterior, and the graph and budget have no effect on it.
7. **Uncalibrated Uncertainty Set**: Until $\epsilon$ and $B$ are calibrated to measured evidence shifts, intervals are model-based sensitivity bounds without coverage meaning.
4. **Sample Size & Split Independence**: Final benchmark generalization relies on zero split leakage between calibration and test partitions.
5. **UNKNOWN Annotations**: Ambiguous claims labeled UNKNOWN are properly excluded from binary accuracy metrics to preserve scientific integrity.
