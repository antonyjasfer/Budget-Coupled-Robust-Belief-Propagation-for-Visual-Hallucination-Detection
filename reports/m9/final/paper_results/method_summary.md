# Method Summary: Budget-Coupled Robust Belief Propagation

## 1. Problem Formulation
Given an image $I$ and candidate visual existence claims $\mathcal{H} = \{H_1, \dots, H_n\}$ extracted from a Vision-Language Model (VLM) generation, we represent claim existence as binary random variables $H_i \in \{-1, +1\}$, where $+1$ denotes a hallucination (object absent in $I$) and $-1$ denotes supported (object verified).

The joint distribution is defined over a factor graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
$$P(H \mid E) \propto \prod_{i \in \mathcal{V}} \psi_i(H_i, E_i) \prod_{(i, j) \in \mathcal{E}} \psi_{ij}(H_i, H_j)$$

where:
- $\psi_i(H_i, E_i) = \exp(\theta_i H_i)$ denotes unary evidence potentials from object detectors and CLIP similarity.
- $\psi_{ij}(H_i, H_j) = \exp(J_{ij} H_i H_j)$ denotes pairwise coupling potentials encoding spatial or semantic relations.

## 2. Global $\ell_1$ Uncertainty Budget
To account for sensor noise, visual degradation, and calibration inaccuracies, we model perturbations to the unary evidence potentials:
$$\tilde{\theta}_i = \theta_i + \delta_i, \quad \text{subject to } |\delta_i| \le \epsilon_i \text{ and } \sum_{i=1}^n |\delta_i| \le B$$

Here, $B \ge 0$ enforces a **global uncertainty budget** across the entire claim graph, coupling local potential variations.

## 3. Robust Posterior Intervals
Rather than evaluating a single point posterior, we compute guaranteed lower and upper bounds on the marginal hallucination probability:
$$L_i = \min_{\|\delta\|_1 \le B, |\delta_k| \le \epsilon_k} P_{\theta + \delta}(H_i = +1 \mid E)$$
$$U_i = \max_{\|\delta\|_1 \le B, |\delta_k| \le \epsilon_k} P_{\theta + \delta}(H_i = +1 \mid E)$$

The numerical solver discretizes the allowable budget allocation via dynamic programming over the tree structure with resolution $K$, yielding polynomial complexity $\mathcal{O}(n K^2)$.
