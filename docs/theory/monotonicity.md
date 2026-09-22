# Monotonicity of Marginals in Attractive Ising Tree Models

## 1. Theorem Statement

**Theorem 1 (Coordinate-wise Monotonicity of Ising Marginals).**
Let $(\mathcal{V}, \mathcal{E})$ be an undirected tree with $|\mathcal{V}| = n$ nodes. Consider a binary Ising model on spin configurations $h = (h_1, \dots, h_n) \in \{-1, +1\}^n$ with probability distribution:
$$P_\theta(h) = \frac{1}{Z(\theta)} \exp\left( \sum_{i=1}^n \theta_i h_i + \sum_{(i,j) \in \mathcal{E}} J_{ij} h_i h_j \right)$$
where $Z(\theta) = \sum_{h \in \{-1, +1\}^n} \exp\left( \sum_{i=1}^n \theta_i h_i + \sum_{(i,j) \in \mathcal{E}} J_{ij} h_i h_j \right)$ is the partition function, and $J_{ij} \ge 0$ for all $(i,j) \in \mathcal{E}$ (attractive / ferromagnetic couplings).

Then for any target node $r \in \mathcal{V}$ and any node $i \in \mathcal{V}$:
1. The marginal probability of the hallucinated state $P_\theta(H_r = +1)$ is strictly non-decreasing in every unary field $\theta_i$:
   $$\frac{\partial P_\theta(H_r = +1)}{\partial \theta_i} = \frac{1}{2} \operatorname{Cov}_\theta(h_r, h_i) \ge 0$$
2. If the graph is connected and all couplings are finite and strictly positive ($0 < J_{jk} < \infty$), the derivative is strictly positive:
   $$\frac{\partial P_\theta(H_r = +1)}{\partial \theta_i} > 0$$

---

## 2. Assumptions

The monotonicity theorem holds strictly under the following assumptions:
1. **Binary Spins**: $h_i \in \{-1, +1\}$ where $-1$ corresponds to `SUPPORTED` and $+1$ corresponds to `HALLUCINATED`.
2. **Attractive Couplings**: $J_{ij} \ge 0$ for all edges $(i, j) \in \mathcal{E}$. No antiferromagnetic or frustrated interactions are present.
3. **Graph Topology**: $(\mathcal{V}, \mathcal{E})$ is an acyclic connected tree. (Note: On general graphs, $\operatorname{Cov}(h_r, h_i) \ge 0$ also holds for ferromagnetic interactions by Griffiths-Kelly-Sherman / Fortuin-Kasteleyn-Ginibre inequalities; on trees, it additionally admits an elementary message-passing proof holding for arbitrary mixed-sign unary fields $\theta \in \mathbb{R}^n$).
4. **Finite Parameters**: $|\theta_i| < \infty$ and $J_{ij} < \infty$ to ensure $Z(\theta) > 0$ and non-degenerate probabilities.

---

## 3. Derivation

### 3.1. Covariance Identity
The expected magnetization at target node $r$ is:
$$\mathbb{E}_\theta[h_r] = \sum_{h \in \{-1, +1\}^n} h_r P_\theta(h) = \frac{1}{Z(\theta)} \sum_{h} h_r \exp(H_\theta(h))$$
where $H_\theta(h) = \sum_{k} \theta_k h_k + \sum_{(j,k)} J_{jk} h_j h_k$.

Differentiating the partition function with respect to $\theta_i$:
$$\frac{\partial Z(\theta)}{\partial \theta_i} = \sum_{h} h_i \exp(H_\theta(h)) = Z(\theta) \mathbb{E}_\theta[h_i]$$

Differentiating the unnormalized expectation:
$$\frac{\partial}{\partial \theta_i} \left[ Z(\theta) \mathbb{E}_\theta[h_r] \right] = \sum_{h} h_r h_i \exp(H_\theta(h)) = Z(\theta) \mathbb{E}_\theta[h_r h_i]$$

Applying the quotient rule to $\mathbb{E}_\theta[h_r] = \frac{Z(\theta) \mathbb{E}_\theta[h_r]}{Z(\theta)}$:
$$\frac{\partial \mathbb{E}_\theta[h_r]}{\partial \theta_i} = \frac{Z(\theta) \frac{\partial}{\partial \theta_i}[Z(\theta)\mathbb{E}_\theta[h_r]] - [Z(\theta)\mathbb{E}_\theta[h_r]] \frac{\partial Z(\theta)}{\partial \theta_i}}{Z(\theta)^2}$$
$$= \frac{Z(\theta)^2 \mathbb{E}_\theta[h_r h_i] - Z(\theta)^2 \mathbb{E}_\theta[h_r] \mathbb{E}_\theta[h_i]}{Z(\theta)^2} = \mathbb{E}_\theta[h_r h_i] - \mathbb{E}_\theta[h_r] \mathbb{E}_\theta[h_i] = \operatorname{Cov}_\theta(h_r, h_i)$$

Since $h_r \in \{-1, +1\}$, the marginal probability of $H_r = +1$ is related to the expectation by:
$$P_\theta(H_r = +1) = \frac{1 + \mathbb{E}_\theta[h_r]}{2}$$
Therefore:
$$\frac{\partial P_\theta(H_r = +1)}{\partial \theta_i} = \frac{1}{2} \frac{\partial \mathbb{E}_\theta[h_r]}{\partial \theta_i} = \frac{1}{2} \operatorname{Cov}_\theta(h_r, h_i)$$

### 3.2. Proof of Non-Negative Covariance on Trees via Belief Propagation
Because $(\mathcal{V}, \mathcal{E})$ is a tree, belief propagation is exact. Root the tree at target node $r$.
The marginal probability at node $r$ is:
$$P_\theta(H_r = +1) = \sigma(2 \eta_r) = \frac{1}{1 + \exp(-2 \eta_r)}$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the logistic sigmoid function, and $\eta_r$ is the total effective belief field:
$$\eta_r = \theta_r + \sum_{c \in \operatorname{children}(r)} \nu_{c \to r}$$

For any directed edge $u \to v$ in the rooted tree, the cavity message is given by the exact transfer function:
$$\nu_{u \to v} = f_{J_{uv}}(x_u) = \operatorname{atanh}(\tanh(J_{uv}) \tanh(x_u))$$
where $x_u$ is the cavity field at node $u$ excluding parent $v$:
$$x_u = \theta_u + \sum_{w \in \operatorname{children}(u)} \nu_{w \to u}$$

Differentiating the transfer function $f_J(x)$ with respect to $x$:
$$\frac{d f_J(x)}{dx} = \frac{1}{1 - \tanh^2(J) \tanh^2(x)} \cdot \tanh(J) (1 - \tanh^2(x))$$
Because $J \ge 0$, $\tanh(J) \ge 0$. Since $\tanh^2(t) < 1$ for all real $t$:
$$\frac{d f_J(x)}{dx} \ge 0 \quad \text{for all } x \in \mathbb{R}, \, J \ge 0$$
with $\frac{d f_J(x)}{dx} > 0$ whenever $J > 0$.

Now, let $i \in \mathcal{V}$. There is a unique simple path from node $i$ to root $r$ in the tree:
$$i = v_0 \to v_1 \to v_2 \to \dots \to v_m = r$$
By the chain rule:
$$\frac{\partial \eta_r}{\partial \theta_i} = \prod_{k=0}^{m-1} \frac{\partial \nu_{v_k \to v_{k+1}}}{\partial x_{v_k}} \cdot \frac{\partial x_{v_k}}{\partial \nu_{v_{k-1} \to v_k}} = \prod_{k=0}^{m-1} f_{J_{v_k v_{k+1}}}'(x_{v_k})$$
Each factor in the product satisfies $f_{J_{v_k v_{k+1}}}'(x_{v_k}) \ge 0$. Therefore:
$$\frac{\partial \eta_r}{\partial \theta_i} \ge 0$$
Since $\sigma'(2 \eta_r) = 2 \sigma(2 \eta_r) (1 - \sigma(2 \eta_r)) > 0$ for all finite $\eta_r$:
$$\frac{\partial P_\theta(H_r = +1)}{\partial \theta_i} = 2 \sigma(2 \eta_r)(1 - \sigma(2 \eta_r)) \frac{\partial \eta_r}{\partial \theta_i} \ge 0$$
Equating this with $\frac{1}{2} \operatorname{Cov}_\theta(h_r, h_i)$ establishes:
$$\operatorname{Cov}_\theta(h_r, h_i) \ge 0 \quad \text{for all } i, r \in \mathcal{V}$$
for arbitrary unary field configurations $\theta \in \mathbb{R}^n$. $\blacksquare$

---

## 4. Limitations and Boundary Conditions

1. **Frustration / Antiferromagnetic Interactions ($J_{ij} < 0$)**:
   If any $J_{ij} < 0$, $\tanh(J_{ij}) < 0$, which causes $f_J'(x) < 0$. In that case, increasing $\theta_i$ can decrease the marginal at $r$. Monotonicity strictly fails for non-attractive models.
2. **Infinite Couplings ($J \to \infty$)**:
   In the deterministic coupling limit ($J \to \infty$), $f_J(x) = x$, and spins become rigidly clamped ($h_u = h_v$ almost surely). Monotonicity remains non-decreasing, though derivatives may saturate at the boundary.
3. **Continuous vs Discrete Feasible Sets**:
   Monotonicity guarantees that coordinates move in predictable directions. However, this derivative property holds in continuous $\theta$; on a discrete grid, step quantizations must be verified separately.

---

## 5. References

1. **Griffiths, R. B. (1967)**. *Correlations in Ising Ferromagnets. I, II*. Journal of Mathematical Physics, 8(3):478-489.
2. **Kelly, D. G., & Sherman, S. (1968)**. *General Griffiths' Inequalities on Correlation in Ising Ferromagnets*. Journal of Mathematical Physics, 9(3):466-484.
3. **Fortuin, C. M., Kasteleyn, P. W., & Ginibre, J. (1971)**. *Correlation inequalities on partially ordered sets*. Communications in Mathematical Physics, 22(2):89-103. (The FKG Inequality).

---

## 6. Implication for Robust Optimization

Because $\frac{\partial P_\theta(H_r = +1)}{\partial \theta_i} \ge 0$ everywhere on the domain:
- To **maximize** $P(H_r = +1)$, every coordinate perturbation $\delta_i$ should be made as **positive** as feasible ($\delta_i \ge 0$).
- To **minimize** $P(H_r = +1)$, every coordinate perturbation $\delta_i$ should be made as **negative** as feasible ($\delta_i \le 0$).

This completely justifies decomposing the symmetric $L_1$ ball uncertainty set into one-sided subproblems.
