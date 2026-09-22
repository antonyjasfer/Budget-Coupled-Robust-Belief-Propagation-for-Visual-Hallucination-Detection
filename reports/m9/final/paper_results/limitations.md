# Limitations and Scope of Validity

1. **Tree Topology Assumption**: Exact tree-structured message passing requires acyclic claim factor graphs. Cyclic dependency structures must be approximated or clustered.
2. **Grid Discretization**: Continuous worst-case budgets are solved numerically via grid discretization $K$. Grid resolution introduces a truncation error bounded by $\mathcal{O}(1/K)$.
3. **Multimodal Model Dependence**: Detection potentials depend upon frozen upstream models (e.g. Grounding DINO, CLIP). Systematic blind spots in the upstream vision models cannot be eliminated through inference alone.
4. **Sample Size & Split Independence**: Final benchmark generalization relies on zero split leakage between calibration and test partitions.
5. **UNKNOWN Annotations**: Ambiguous claims labeled UNKNOWN are properly excluded from binary accuracy metrics to preserve scientific integrity.
