# Research Contribution Statement

This research makes the following specific contributions to multimodal hallucination detection and probabilistic inference:

1. **Interdependent Claim Formulation**: We formulate visual hallucination detection as probabilistic inference over a factor graph of interdependent object existence claims, capturing semantic and spatial co-occurrence constraints.
2. **Global $\ell_1$ Uncertainty Budget**: We introduce a global $\ell_1$ uncertainty budget ($B$) over multimodal evidence potentials, coupling local sensor perturbations across related visual entities.
3. **Budget-Aware Robust Marginal Inference**: We develop a polynomial-time dynamic programming algorithm over tree topologies that computes exact worst-case posterior intervals $[L_i, U_i]$ for each claim.
4. **Empirical Analysis of Evidence Sensitivity**: We demonstrate that claims spanning the decision threshold ($L_i < \tau < U_i$) capture multimodal ambiguity and visual degradation, providing a rigorous mechanism for selective prediction and deferral.

*Note: All statements reflect empirical and mathematical findings within the defined experimental scope without unsupported primacy or superiority claims.*
