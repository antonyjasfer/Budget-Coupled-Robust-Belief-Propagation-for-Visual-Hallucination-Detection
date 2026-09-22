# Correctness of Budget-Coupled Dynamic Programming over Discretized Uncertainty

## 1. The Discretized Feasible Set

Let $(\mathcal{V}, \mathcal{E})$ be an attractive Ising tree with $n = |\mathcal{V}|$ nodes, base fields $\theta \in \mathbb{R}^n$, coupling $J_{ij} \ge 0$, local perturbation limits $\epsilon_i \ge 0$, and shared budget $B \ge 0$.
Let $K \in \mathbb{N}_{\ge 1}$ be the number of grid intervals, defining the discretization quantum:
$$\Delta = \frac{B}{K}$$

For the upper problem ($U_r$), by Proposition 1 (One-Sided Extremizers), perturbations satisfy $\delta_i \ge 0$. We discretize the perturbations onto integer multiples of $\Delta$:
$$\delta_i = a_i \Delta, \quad a_i \in \{0, 1, \dots, \bar{a}_i\}$$
where $\bar{a}_i = \left\lfloor \frac{\epsilon_i}{\Delta} \right\rfloor$ is the maximum discrete quantum permissible by the box constraint.

The **discrete feasible set** for the upper problem is:
$$\mathcal{U}_{\text{grid}}^+(B, \epsilon, K) = \left\{ \delta \in \mathbb{R}^n : \delta_i = a_i \Delta, \, a_i \in \mathbb{N}_0, \, 0 \le a_i \le \left\lfloor \frac{\epsilon_i}{\Delta} \right\rfloor, \, \sum_{i \in \mathcal{V}} a_i \le K \right\}$$

Similarly, the discrete feasible set for the lower problem is:
$$\mathcal{U}_{\text{grid}}^-(B, \epsilon, K) = \left\{ \delta \in \mathbb{R}^n : \delta_i = -a_i \Delta, \, a_i \in \mathbb{N}_0, \, 0 \le a_i \le \left\lfloor \frac{\epsilon_i}{\Delta} \right\rfloor, \, \sum_{i \in \mathcal{V}} a_i \le K \right\}$$

---

## 2. Dynamic Programming Formulation

Root the tree at target node $r$. For any node $u \in \mathcal{V}$, let:
- $T_u \subseteq \mathcal{V}$ denote the subtree rooted at $u$.
- $p(u)$ denote the unique parent of $u$ (with $p(r) = \text{None}$).
- $\mathcal{C}(u) = \{c_1, \dots, c_{d_u}\}$ denote the set of children of $u$ in the rooted tree.

### 2.1. DP State Definition
For each node $u \in \mathcal{V}$ and budget quantum index $m \in \{0, 1, \dots, K\}$, define:
- $x_u[m]$: The maximum reachable cavity field at node $u$ (toward parent $p(u)$) using at most $m$ quanta of budget in the subtree $T_u$.
- $\nu_{u \to p}[m]$: The maximum message transmitted from $u$ to parent $p(u)$ using at most $m$ quanta in $T_u$:
  $$\nu_{u \to p}[m] = f_{J_{u, p(u)}}(x_u[m])$$
  where $f_J(x) = \operatorname{atanh}(\tanh(J) \tanh(x))$.

### 2.2. Child-Budget Convolution
Subtrees rooted at distinct children $c \in \mathcal{C}(u)$ are mutually vertex-disjoint:
$$T_{c_j} \cap T_{c_k} = \emptyset \quad \text{for all } j \ne k$$
Because messages from different children enter the cavity field of $u$ additively:
$$x_u = \theta_u + \delta_u + \sum_{c \in \mathcal{C}(u)} \nu_{c \to u}$$
the joint optimization over the budget allocations to children decomposes into a sequence of 1D max-plus convolutions:
$$(A \oplus B)[m] = \max_{0 \le j \le m} (A[m - j] + B[j])$$

For children $\mathcal{C}(u) = \{c_1, \dots, c_d\}$:
- If $d = 0$ (leaf node): $v_{\text{children}}[m] = 0$ for all $m \in \{0, \dots, K\}$.
- If $d = 1$: $v_{\text{children}}[m] = \nu_{c_1 \to u}[m]$.
- If $d > 1$: Initialize $v^{(1)} = \nu_{c_1 \to u}$. For $l = 2, \dots, d$:
  $$v^{(l)} = v^{(l-1)} \oplus \nu_{c_l \to u}$$
  Set $v_{\text{children}} = v^{(d)}$.

### 2.3. Local Optimization Recurrence
Node $u$ itself may consume $a_u \in \{0, \dots, \min(\bar{a}_u, m)\}$ quanta:
$$x_u[m] = \theta_u + \max_{0 \le k \le \min(\bar{a}_u, m)} \left( k \Delta + v_{\text{children}}[m - k] \right)$$

### 2.4. Root Aggregation
At the root $r$, the total effective field using at most $K$ budget quanta across the entire tree $T_r = \mathcal{V}$ is:
$$\eta_r^* = x_r[K] = \theta_r + \max_{0 \le k \le \min(\bar{a}_r, K)} \left( k \Delta + v_{\text{children}}[K - k] \right)$$
The discrete robust upper marginal is:
$$U_{r, \text{grid}} = \sigma(2 \eta_r^*) = \frac{1}{1 + \exp(-2 \eta_r^*)}$$

The lower problem $L_{r, \text{grid}}$ follows the identical recurrence replacing $\max$ with $\min$, $+$ with $-$, and using min-plus convolutions $(A \oplus_{\min} B)[m] = \min_{0 \le j \le m}(A[m-j] + B[j])$.

---

## 3. Proof of Exactness over the Discretized Set

**Theorem 2 (Exactness of DP on the Discretized Uncertainty Set).**
The DP algorithm computes the exact global optimum over the discrete set $\mathcal{U}_{\text{grid}}^+(B, \epsilon, K)$:
$$x_r[K] = \max_{\delta \in \mathcal{U}_{\text{grid}}^+(B, \epsilon, K)} \eta_r(\theta + \delta)$$
and consequently:
$$U_{r, \text{grid}} = \max_{\delta \in \mathcal{U}_{\text{grid}}^+(B, \epsilon, K)} P_{\theta + \delta}(H_r = +1)$$

### Proof
We proceed by structural induction on the tree topology from leaves to root:
1. **Base Case (Leaves)**:
   Let $u$ be a leaf ($\mathcal{C}(u) = \emptyset$). Then $T_u = \{u\}$. Any budget allocation $m$ to $T_u$ can only be allocated to $u$ itself: $a_u \in \{0, \dots, \min(\bar{a}_u, m)\}$.
   Since $v_{\text{children}}[m] = 0$, $x_u[m] = \theta_u + \max_{0 \le k \le \min(\bar{a}_u, m)} k \Delta$, which is the exact maximum perturbation on the leaf with budget $m$.
2. **Inductive Step (Internal Nodes)**:
   Assume inductively that for every child $c \in \mathcal{C}(u)$, $\nu_{c \to u}[m_c]$ is the exact maximum incoming message from subtree $T_c$ given integer budget $m_c$.
   Since $f_J(x) = \operatorname{atanh}(\tanh(J)\tanh(x))$ is strictly increasing in $x$ for $J \ge 0$, maximizing the message $\nu_{c \to u}$ is strictly equivalent to maximizing the cavity field $x_c$:
   $$\max f_J(x_c) = f_J(\max x_c)$$
   By the associative and commutative properties of addition, partitioning total budget $m - a_u$ among disjoint subtrees $T_{c_1}, \dots, T_{c_d}$ decomposes into:
   $$\max_{\sum_{l=1}^d m_l = m - a_u} \sum_{l=1}^d \nu_{c_l \to u}[m_l] = v_{\text{children}}[m - a_u]$$
   The 1D max-plus convolutions evaluate every integer partition of $m - a_u$ across children, guaranteeing zero loss of optimality.
   Finally, the local optimization step exhaustively checks all valid discrete allocations $a_u \in \{0, \dots, \min(\bar{a}_u, m)\}$.
   By the Principle of Optimality (Bellman), $x_u[m]$ is the exact maximum cavity field for $T_u$ with budget $m$.
3. **Conclusion at Root**:
   At root $r$, $x_r[K]$ computes the exact maximum field $\eta_r$ over all discrete vectors $\delta \in \mathcal{U}_{\text{grid}}^+(B, \epsilon, K)$.
   Since $\sigma(2\eta_r)$ is strictly monotonic, applying sigmoid preserves the exact discrete supremum. $\blacksquare$

---

## 4. Distinction: Continuous vs. Discretized Bounds

> [!WARNING]
> **Exactness over $\mathcal{U}_{\text{grid}}$ DOES NOT imply exactness over continuous $\mathcal{U}$.**

The true continuous problem optimizes over all real vectors $\delta \in \mathcal{U}(B, \epsilon)$.
Since $\mathcal{U}_{\text{grid}}(B, \epsilon, K) \subset \mathcal{U}(B, \epsilon)$:
$$U_{r, \text{grid}} \le U_{r, \text{continuous}} \quad \text{and} \quad L_{r, \text{grid}} \ge L_{r, \text{continuous}}$$
The grid interval $[L_{\text{grid}}, U_{\text{grid}}]$ is an **inner approximation** (sub-interval) of the true continuous robust interval.

To guarantee continuous containment, the solver applies a Lipschitz discretization gap $\Delta \cdot L_{\text{lip}}$:
$$L_{\text{certified}} = \sigma(2(\text{field}_{\text{lower, grid}} - \Delta)) \le L_{\text{continuous}}$$
$$U_{\text{certified}} = \sigma(2(\text{field}_{\text{upper, grid}} + \Delta)) \ge U_{\text{continuous}}$$
producing a rigorous **outer approximation** $[L_{\text{certified}}, U_{\text{certified}}]$.
