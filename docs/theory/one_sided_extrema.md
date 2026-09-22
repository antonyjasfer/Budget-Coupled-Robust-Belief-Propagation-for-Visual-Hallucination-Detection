# One-Sided Extremizer Reduction for Symmetric Uncertainty Sets

## 1. Proposition Statement

**Proposition 1 (One-Sided Extremizers of Budget-Constrained Robust Marginals).**
Let $(\mathcal{V}, \mathcal{E})$ be an attractive Ising tree model ($J_{ij} \ge 0$) with base unary fields $\theta \in \mathbb{R}^n$. Consider the continuous symmetric uncertainty set:
$$\mathcal{U}(B, \epsilon) = \left\{ \delta \in \mathbb{R}^n : |\delta_i| \le \epsilon_i \text{ for all } i \in \mathcal{V}, \quad \sum_{i \in \mathcal{V}} |\delta_i| \le B \right\}$$
where $\epsilon_i \ge 0$ and $B \ge 0$. Define the upper and lower robust marginal targets at root node $r \in \mathcal{V}$:
$$U_r = \sup_{\delta \in \mathcal{U}(B, \epsilon)} P_{\theta + \delta}(H_r = +1)$$
$$L_r = \inf_{\delta \in \mathcal{U}(B, \epsilon)} P_{\theta + \delta}(H_r = +1)$$

Define the one-sided positive and negative sub-uncertainty sets:
$$\mathcal{U}^+(B, \epsilon) = \left\{ \delta \in \mathbb{R}^n : 0 \le \delta_i \le \epsilon_i \text{ for all } i \in \mathcal{V}, \quad \sum_{i \in \mathcal{V}} \delta_i \le B \right\}$$
$$\mathcal{U}^-(B, \epsilon) = \left\{ \delta \in \mathbb{R}^n : -\epsilon_i \le \delta_i \le 0 \text{ for all } i \in \mathcal{V}, \quad \sum_{i \in \mathcal{V}} (-\delta_i) \le B \right\}$$

Then:
1. The supremum over the full symmetric set equals the supremum over the positive set:
   $$U_r = \sup_{\delta \in \mathcal{U}^+(B, \epsilon)} P_{\theta + \delta}(H_r = +1)$$
2. The infimum over the full symmetric set equals the infimum over the negative set:
   $$L_r = \inf_{\delta \in \mathcal{U}^-(B, \epsilon)} P_{\theta + \delta}(H_r = +1)$$
3. The same reduction holds exactly over the grid-discretized counterpart $\mathcal{U}_{\text{grid}}(B, \epsilon)$.

---

## 2. Mathematical Proof

### 2.1. Upper Bound Reduction
Let $\delta \in \mathcal{U}(B, \epsilon)$ be any feasible perturbation vector.
Construct a modified perturbation vector $\delta^+ \in \mathbb{R}^n$ defined componentwise by:
$$\delta^+_i = \max(0, \delta_i) \ge 0$$

We verify feasibility of $\delta^+$:
1. **Box constraint**: Since $|\delta_i| \le \epsilon_i$ and $\epsilon_i \ge 0$, we have $0 \le \delta^+_i = \max(0, \delta_i) \le |\delta_i| \le \epsilon_i$.
2. **Budget constraint**: Since $|\delta^+_i| = \delta^+_i \le |\delta_i|$ for every coordinate $i$:
   $$\sum_{i \in \mathcal{V}} \delta^+_i \le \sum_{i \in \mathcal{V}} |\delta_i| \le B$$
Hence, $\delta^+ \in \mathcal{U}^+(B, \epsilon) \subseteq \mathcal{U}(B, \epsilon)$.

Now, compare coordinates:
$$\delta^+_i = \max(0, \delta_i) \ge \delta_i \quad \text{for all } i \in \mathcal{V}$$
By Theorem 1 (Monotonicity), $P_{\theta + \delta}(H_r = +1)$ is coordinate-wise non-decreasing in each unary field perturbation $\delta_i$. Because $\delta^+_i \ge \delta_i$ for all $i \in \mathcal{V}$, applying monotonicity coordinate-by-coordinate yields:
$$P_{\theta + \delta^+}(H_r = +1) \ge P_{\theta + \delta}(H_r = +1)$$

Taking the supremum over all $\delta \in \mathcal{U}(B, \epsilon)$:
$$\sup_{\delta \in \mathcal{U}(B, \epsilon)} P_{\theta + \delta}(H_r = +1) \le \sup_{\delta^+ \in \mathcal{U}^+(B, \epsilon)} P_{\theta + \delta^+}(H_r = +1)$$
Conversely, since $\mathcal{U}^+(B, \epsilon) \subseteq \mathcal{U}(B, \epsilon)$, the reverse inequality trivially holds:
$$\sup_{\delta \in \mathcal{U}^+(B, \epsilon)} P_{\theta + \delta}(H_r = +1) \le \sup_{\delta \in \mathcal{U}(B, \epsilon)} P_{\theta + \delta}(H_r = +1)$$
Combining both inequalities gives:
$$U_r = \sup_{\delta \in \mathcal{U}^+(B, \epsilon)} P_{\theta + \delta}(H_r = +1) \quad \blacksquare$$

### 2.2. Lower Bound Reduction
Let $\delta \in \mathcal{U}(B, \epsilon)$ be any feasible perturbation vector.
Construct $\delta^- \in \mathbb{R}^n$ defined componentwise by:
$$\delta^-_i = \min(0, \delta_i) \le 0$$

We verify feasibility of $\delta^-$:
1. **Box constraint**: Since $|\delta_i| \le \epsilon_i$, $-\epsilon_i \le -|\delta_i| \le \delta^-_i \le 0$.
2. **Budget constraint**: $|\delta^-_i| = -\delta^-_i \le |\delta_i|$, so:
   $$\sum_{i \in \mathcal{V}} (-\delta^-_i) \le \sum_{i \in \mathcal{V}} |\delta_i| \le B$$
Hence, $\delta^- \in \mathcal{U}^-(B, \epsilon) \subseteq \mathcal{U}(B, \epsilon)$.

Since $\delta^-_i \le \delta_i$ for all $i \in \mathcal{V}$, by Theorem 1:
$$P_{\theta + \delta^-}(H_r = +1) \le P_{\theta + \delta}(H_r = +1)$$
Taking the infimum over all $\delta \in \mathcal{U}(B, \epsilon)$ establishes:
$$L_r = \inf_{\delta \in \mathcal{U}^-(B, \epsilon)} P_{\theta + \delta}(H_r = +1) \quad \blacksquare$$

---

## 3. Precise Assumptions

1. **Attractive Couplings ($J_{ij} \ge 0$)**: If couplings were negative or mixed, monotonicity fails, and an adversary could increase $P(H_r = +1)$ by decreasing an opposing node's field. Under frustration, the one-sided reduction is FALSE.
2. **Independent Unary Uncertainty**: The uncertainty set constraints decouple coordinates except through the monotonic budget sum $\sum_i |\delta_i| \le B$. No pairwise coupling perturbations $(\delta J_{ij})$ are considered in this model.
3. **Tree Topology**: Assumed for exact belief propagation and derivative non-negativity.

---

## 4. Implications for Implementation

The one-sided extremizer proposition provides massive computational savings without any loss of optimality:
1. **Search Space Reduction**: Instead of exploring $3^n$ or $(2K+1)^n$ signed grid allocations, the solver only needs to explore $2^n$ or $(K+1)^n$ one-sided non-negative allocations.
2. **Decoupled Solvers**: The upper bound optimization is solved by setting sign $= +1$ with max-plus convolutions, while the lower bound is solved independently by setting sign $= -1$ with min-plus convolutions.
3. **No Mixed Sign Vectors**: There is zero loss in restricting $\delta_i \ge 0$ for the upper problem and $\delta_i \le 0$ for the lower problem. This is an exact equivalence, not an approximation or heuristic.
