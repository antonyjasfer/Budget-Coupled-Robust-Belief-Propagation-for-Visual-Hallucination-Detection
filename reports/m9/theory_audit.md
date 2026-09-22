# Milestone 9A: Mathematical Hardening and Theoretical Audit Report

**Repository**: `antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection`  
**Project**: Budget-Coupled Robust Belief Propagation for Visual Hallucination Detection  
**Phase**: Phase 9A — Mathematical Hardening and Verification  
**Date**: September 2026  

---

## 1. Mathematical Model

### 1.1. Model Variables and State Space
The system models a collection of $n$ atomic visual claims for an image. Each claim $i \in \mathcal{V} = \{0, 1, \dots, n-1\}$ is associated with a binary spin random variable:
$$h_i \in \{-1, +1\}$$
under the fixed semantic mapping:
- $h_i = -1$: **SUPPORTED** (The claimed object is visually grounded in the image).
- $h_i = +1$: **HALLUCINATED** (The claimed object is absent, false, or unsupported).

### 1.2. Graph Topology and Joint Probability
The dependency structure across claims is defined by an undirected, acyclic tree graph $\mathcal{T} = (\mathcal{V}, \mathcal{E})$ with $|\mathcal{E}| = n - 1$.
The joint probability distribution conditioned on visual evidence $E$ is parameterised as an attractive binary Ising Markov Random Field:
$$P_\delta(h \mid E) = \frac{1}{Z(\theta + \delta)} \exp\left( \sum_{i \in \mathcal{V}} (\theta_i + \delta_i) h_i + \sum_{(i,j) \in \mathcal{E}} J_{ij} h_i h_j \right)$$
where:
- $\theta_i \in \mathbb{R}$ is the nominal unary field derived from visual feature evidence (e.g., fusion of object detector score and CLIP alignment score).
- $\delta_i \in \mathbb{R}$ is an adversarial / uncertain perturbation to the unary evidence field at node $i$.
- $J_{ij} \ge 0$ is the symmetric attractive pairwise coupling between claims $i$ and $j$, derived from semantic co-occurrence and visual overlap.
- $Z(\theta + \delta) = \sum_{h \in \{-1, +1\}^n} \exp\left( \sum_{i} (\theta_i + \delta_i) h_i + \sum_{(i,j)} J_{ij} h_i h_j \right)$ is the partition function.

### 1.3. Budget-Coupled Uncertainty Set
Perturbation vectors $\delta = (\delta_1, \dots, \delta_n)$ are constrained by a shared, coupled uncertainty set $\mathcal{U}(B, \epsilon)$:
$$\mathcal{U}(B, \epsilon) = \left\{ \delta \in \mathbb{R}^n : |\delta_i| \le \epsilon_i \text{ for all } i \in \mathcal{V}, \quad \sum_{i \in \mathcal{V}} |\delta_i| \le B \right\}$$
where:
- $\epsilon_i \ge 0$ is the local perturbation limit for claim $i$ (reflecting evidence sensor noise and detector calibration error).
- $B \ge 0$ is the total shared budget (enforcing global coupling and preventing worst-case independent adversarial saturation across all nodes simultaneously).

### 1.4. Target Quantities
For a specified claim of interest $r \in \mathcal{V}$, the robust inference targets are the lower and upper bounds of the posterior hallucination probability:
$$L_r = \inf_{\delta \in \mathcal{U}(B, \epsilon)} P_{\theta + \delta}(H_r = +1 \mid E)$$
$$U_r = \sup_{\delta \in \mathcal{U}(B, \epsilon)} P_{\theta + \delta}(H_r = +1 \mid E)$$
The interval width $w_r = U_r - L_r \ge 0$ measures the sensitivity of claim $r$'s hallucination status to evidence perturbations.

---

## 2. Assumptions

The theoretical guarantees established in this audit rest on the following explicit conditions:
1. **Tree Topology**: $\mathcal{T} = (\mathcal{V}, \mathcal{E})$ is a connected acyclic graph.
2. **Attractive Interactions**: $J_{ij} \ge 0$ for all $(i,j) \in \mathcal{E}$ (strictly non-negative couplings; no negative, frustrated, or competitive interactions).
3. **Additive Unary Uncertainty**: Perturbations $\delta_i$ act additively on the unary fields $\theta_i$. Pairwise couplings $J_{ij}$ are assumed fixed and unperturbed.
4. **Convex Budget Set**: The continuous uncertainty set $\mathcal{U}(B, \epsilon)$ is an intersection of hyper-rectangles with an $L_1$ ball.
5. **Finite Energies**: $|\theta_i| < \infty$ and $0 \le J_{ij} < \infty$.

---

## 3. Exact Standard Belief Propagation Result

### 3.1. Message Parameterization
For tree models, Belief Propagation (BP) is mathematically exact. The cavity message $\nu_{u \to v}$ sent from node $u$ to adjacent node $v$ represents the log-likelihood ratio contribution of the subtree $T_{u \setminus v}$:
$$\nu_{u \to v} = \frac{1}{2} \ln \left( \frac{\mu_{u \to v}(+1)}{\mu_{u \to v}(-1)} \right)$$
The message update equation is given by the transfer function:
$$\nu_{u \to v} = f_{J_{uv}}(x_u) = \operatorname{atanh}\left( \tanh(J_{uv}) \tanh(x_u) \right)$$
where $x_u = \theta_u + \delta_u + \sum_{w \in \operatorname{adj}(u) \setminus \{v\}} \nu_{w \to u}$ is the incoming cavity field.

### 3.2. Numerical Stability
In the implementation ([`src/pgm/standard_bp.py`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/src/pgm/standard_bp.py)), the transfer function is computed via the identity:
$$f_J(x) = \frac{1}{2} \left( \operatorname{logaddexp}(J + x, -(J + x)) - \operatorname{logaddexp}(J - x, -(J - x)) \right)$$
which guarantees zero division, avoids overflow for large $|x|$ or large $J$, and bounds $|\nu_{u \to v}| \le J_{uv}$.

### 3.3. Exact Marginals
At root $r$, the total effective field is $\eta_r = \theta_r + \delta_r + \sum_{c \in \operatorname{adj}(r)} \nu_{c \to r}$.
The exact marginal probability is:
$$P(H_r = +1 \mid E) = \sigma(2 \eta_r) = \frac{1}{1 + \exp(-2 \eta_r)}$$
Two-pass scheduling computes exact marginals for all $n$ nodes in $O(n)$ time.

---

## 4. Monotonicity Result

**Theorem (Monotonicity of Marginals, Griffiths-Kelly-Sherman / GKS):**  
For any attractive Ising tree model ($J_{ij} \ge 0$), the marginal probability $P(H_r = +1)$ is strictly non-decreasing with respect to every unary field $\theta_i$:
$$\frac{\partial P(H_r = +1)}{\partial \theta_i} = \frac{1}{2} \operatorname{Cov}(h_r, h_i) \ge 0$$

### Derivation Summary
1. $\frac{\partial \mathbb{E}[h_r]}{\partial \theta_i} = \mathbb{E}[h_r h_i] - \mathbb{E}[h_r] \mathbb{E}[h_i] = \operatorname{Cov}(h_r, h_i)$ by differentiation of the partition function.
2. Under tree BP, $P(H_r = +1) = \sigma(2\eta_r)$. By the chain rule along the unique path from node $i$ to root $r$:
   $$\frac{\partial \eta_r}{\partial \theta_i} = \prod_{e \in \operatorname{path}(i \to r)} f_{J_e}'(x_e)$$
   Since $f_J'(x) = \frac{\tanh(J)(1 - \tanh^2(x))}{1 - \tanh^2(J)\tanh^2(x)} \ge 0$ for all $J \ge 0$, the derivative is everywhere non-negative.
3. Therefore $\operatorname{Cov}(h_r, h_i) \ge 0$ for all $i, r \in \mathcal{V}$, and $P(H_r = +1)$ is coordinate-wise monotonic.
4. Detailed proof documented in [`docs/theory/monotonicity.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/docs/theory/monotonicity.md).

---

## 5. One-Sided Extremizer Result

**Proposition (One-Sided Extremizers):**  
Under coordinate-wise monotonicity, the global optimization over the full symmetric $L_1$ uncertainty set $\mathcal{U}(B, \epsilon)$ reduces without loss of optimality to:
- **Upper bound ($U_r$)**: $\delta_i \ge 0$ for all $i \in \mathcal{V}$, $\sum_i \delta_i \le B$, $0 \le \delta_i \le \epsilon_i$.
- **Lower bound ($L_r$)**: $\delta_i \le 0$ for all $i \in \mathcal{V}$, $\sum_i (-\delta_i) \le B$, $0 \le -\delta_i \le \epsilon_i$.

### Proof Summary
For any feasible $\delta \in \mathcal{U}(B, \epsilon)$ with negative coordinates, the projected vector $\delta^+ = \max(0, \delta)$ is feasible ($\sum_i \delta^+_i \le \sum_i |\delta_i| \le B$) and satisfies $\delta^+_i \ge \delta_i$. By monotonicity, $P_{\theta + \delta^+}(H_r = +1) \ge P_{\theta + \delta}(H_r = +1)$.
Hence, the supremum must be attained in the non-negative orthant $\mathcal{U}^+$.
Detailed proof documented in [`docs/theory/one_sided_extrema.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/docs/theory/one_sided_extrema.md).

---

## 6. Discrete Robust DP Correctness

**Theorem (Exactness over Discretized Feasible Set):**  
Let grid step $\Delta = B / K$. The dynamic programming algorithm implemented in [`src/robust_bp/solver.py`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/src/robust_bp/solver.py):
$$x_u[m] = \theta_u + \max_{0 \le k \le \min(\lfloor \epsilon_u / \Delta \rfloor, m)} \left( k \Delta + v_{\text{children}}[m - k] \right)$$
where $v_{\text{children}} = \bigoplus_{c \in \operatorname{children}(u)} \nu_{c \to u}$ is evaluated via 1D max-plus convolutions, computes the **exact global optimum** over the discretized uncertainty set:
$$\mathcal{U}_{\text{grid}}^+(B, \epsilon, K) = \left\{ \delta : \delta_i = a_i \Delta, \, a_i \in \mathbb{N}_0, \, a_i \le \lfloor \epsilon_i / \Delta \rfloor, \, \sum_{i=1}^n a_i \le K \right\}$$

### Proof Summary
Because vertex-disjoint subtrees interact additively through the cavity field and transfer functions $f_J$ are strictly monotonic, the Bellman Principle of Optimality holds at every node. Every valid integer partition of the discrete budget across children is evaluated exhaustively by the convolutions, guaranteeing zero suboptimality on the grid lattice.
Detailed proof documented in [`docs/theory/discrete_dp_correctness.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/docs/theory/discrete_dp_correctness.md).

---

## 7. Continuous-vs-Discrete Limitation

> [!IMPORTANT]
> **Key Distinction: Grid Bounds vs. Continuous Bounds**
> - The grid bounds $[L_{\text{grid}}, U_{\text{grid}}]$ are exact **only over the discrete grid lattice** $\mathcal{U}_{\text{grid}}$.
> - Because $\mathcal{U}_{\text{grid}} \subset \mathcal{U}$, the grid bounds form an **inner approximation**:
>   $$L_{\text{continuous}} \le L_{\text{grid}} \le U_{\text{grid}} \le U_{\text{continuous}}$$
> - Continuous outer bounds $[L_{\text{certified}}, U_{\text{certified}}]$ are obtained by expanding the effective field by the rigorous continuous certificate gap $G_{\text{cert}} = \min(B, \sum_{i \in \mathcal{V}} \gamma_i \min(\epsilon_i, \Delta))$ (Theorem 3, accounting for sub-quantum residual accumulation across coordinates):
>   $$L_{\text{certified}} = \sigma(2(\eta_{\text{lower, grid}} - G_{\text{cert}})) \le L_{\text{continuous}}$$
>   $$U_{\text{certified}} = \sigma(2(\eta_{\text{upper, grid}} + G_{\text{cert}})) \ge U_{\text{continuous}}$$
> - As $K \to \infty$ ($\Delta \to 0$), both inner and outer bounds converge uniformly to the continuous optimum. Detailed proof, counterexample to naive $\Delta$ bound, and verification documented in [`docs/theory/continuous_grid_certificate.md`](file:///c:/Users/anton/OneDrive/Desktop/Mths%20project/docs/theory/continuous_grid_certificate.md).

---

## 8. Complexity Audit

Let $n$ be the number of nodes, $K$ the number of grid intervals, and $d_u$ the number of children of node $u$ in the rooted tree.

### 8.1. Single-Target Complexity
1. **Children Convolutions**:
   Node $u$ with $d_u$ children performs $\max(0, d_u - 1)$ pairwise 1D max-plus convolutions.
   Total convolutions across the tree: $\sum_{u \in \mathcal{V}} \max(0, d_u - 1) = \text{leaves} - 1 < n$.
   Each convolution of two arrays of length $K+1$ takes $\sum_{m=0}^K (m + 1) = \frac{(K+1)(K+2)}{2} = \frac{1}{2} K^2 + O(K)$ operations.
2. **Local Optimization**:
   For each node $u$, scanning $0 \le k \le m$ for all $m \in \{0, \dots, K\}$ takes $\sum_{m=0}^K (m + 1) = \frac{1}{2} K^2 + O(K)$ operations.
3. **Message Evaluation**:
   Evaluating $f_J(x)$ takes $(n - 1)(K + 1)$ operations.
4. **Total Single-Target Time**:
   $$T_1(n, K) = \frac{1}{2} n K^2 + (\text{leaves} - 1) \frac{1}{2} K^2 + O(n K) = \Theta(n K^2)$$

### 8.2. All-Target Complexity
Because the budget allocation $B$ is rooted at the target node, computing bounds for all $n$ nodes requires re-rooting:
$$T_{\text{all}}(n, K) = n \cdot \Theta(n K^2) = \Theta(n^2 K^2)$$

### 8.3. Space Complexity
Storing budget profiles and convolution backpointers for witness tracing requires:
$$S(n, K) = O(n K)$$
which scales linearly in tree size and grid resolution.

---

## 9. Verified Invariants

The following invariants have been strictly verified across all unit tests and brute-force oracle benchmarks:
1. **Zero Budget Collapse**: $B = 0 \implies L_i = \text{nominal} = U_i$ to machine precision ($10^{-12}$).
2. **Zero Perturbation Limit Collapse**: $\epsilon = 0 \implies L_i = U_i = \text{nominal}$ for arbitrary $B \ge 0$.
3. **Sandwich Validity**: $0 \le L_{\text{cert}} \le L_{\text{grid}} \le \text{nominal} \le U_{\text{grid}} \le U_{\text{cert}} \le 1$ everywhere.
4. **Budget Monotonicity**: For $B_2 \ge B_1$:
   $$L(B_2) \le L(B_1) \quad \text{and} \quad U(B_2) \ge U(B_1)$$
5. **Field Monotonicity**: Increasing any unary field $\theta_i$ never decreases target hallucination probability ($d P / d \theta_i \ge 0$).
6. **Independent Decoupling**: For $J = 0$, nodes decouple, and bounds match the independent analytical optimum.
7. **Budget Saturation**: For $B \ge \sum_i \epsilon_i$, budget coupling is non-binding, and grid bounds match independent box bounds (with certified bounds bracketing unaligned cases).
8. **Small Tree Oracle Match**: For all $n \le 6$, the DP matches exhaustive $2^n \times \text{all-grid-vectors}$ brute-force enumeration to within $10^{-6}$.
9. **Numerical Robustness**: Extreme inputs ($\theta = \pm 30$, $J = 30$, $\epsilon = 10^{-8}$) execute without NaN, inf, or domain errors.

---

## 10. Claims That Are Safe to Make

1. **"The DP solver returns the exact global optimum over the discretized budget uncertainty set $\mathcal{U}_{\text{grid}}(B, \epsilon, K)$."** (Proved by Theorem 2, verified by brute-force oracle).
2. **"Marginal probabilities in attractive Ising trees are coordinate-wise non-decreasing in all unary fields."** (Proved by Theorem 1, verified by covariance analysis).
3. **"The robust upper and lower optimization problems admit one-sided non-negative and non-positive extremizers, respectively."** (Proved by Proposition 1).
4. **"The continuous Lipschitz certificate rigorously bounds the true continuous robust posterior interval from the outside."** (Proved by Lipschitz continuity of tree BP messages with constant $L \le 1$).
5. **"Single-target robust inference runs in $O(n K^2)$ time and $O(n K)$ memory."** (Proved by convolution loop analysis).

---

## 11. Claims That Are Unsafe and Must Not Be Made

1. **DO NOT CLAIM**: *"The DP solver computes the exact continuous robust bounds."*  
   **Correction**: The DP is exact over the **grid lattice** $\mathcal{U}_{\text{grid}}$. The true continuous optimum may lie between grid points.
2. **DO NOT CLAIM**: *"The interval $[L_{\text{grid}}, U_{\text{grid}}]$ is a certified outer bound on the continuous problem."*  
   **Correction**: $[L_{\text{grid}}, U_{\text{grid}}]$ is an **inner bound**. The certified outer bound is $[L_{\text{certified}}, U_{\text{certified}}]$.
3. **DO NOT CLAIM**: *"Monotonicity holds for arbitrary Ising models."*  
   **Correction**: Monotonicity holds strictly for **attractive models** ($J_{ij} \ge 0$). Antiferromagnetic or frustrated models violate monotonicity.
4. **DO NOT CLAIM**: *"The robust interval is a Bayesian posterior credible interval or frequentist confidence interval."*  
   **Correction**: The robust interval is an **uncertainty envelope** (infimum/supremum over an uncertainty set), not a coverage interval over stochastic parameter draws.

---

## 12. Open Mathematical Issues

1. **Pairwise Coupling Uncertainty**:  
   The current formulation perturbs only unary fields $\theta_i + \delta_i$. Incorporating perturbation of edge couplings $J_{ij} \pm \delta J_{ij}$ subject to $J_{ij} \ge 0$ would require multilinear message convolutions, as $f_J(x)$ is non-linear in $J$.
2. **All-Target 2-Pass Robust Dynamic Programming**:  
   Currently, all-target evaluation re-roots the tree for each node in $O(n^2 K^2)$. Devising a 2-pass message-passing scheme that computes robust envelopes for all nodes simultaneously in $O(n K^2)$ remains an open algorithmic question because budget allocation is globally constrained toward a specific root.
3. **Continuous Off-Grid Convex Reformulation**:  
   Determining whether the continuous minimax problem $\sup_{\delta \in \mathcal{U}} \eta_r(\delta)$ can be solved to arbitrary precision via continuous convex duality or Frank-Wolfe coordinate ascent without grid discretization.
