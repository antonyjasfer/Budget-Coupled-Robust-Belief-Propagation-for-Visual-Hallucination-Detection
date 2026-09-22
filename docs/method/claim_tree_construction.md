# Claim Dependency Graph and Tree Construction

## 1. Overview and Problem Definition

In visual hallucination detection, an image response generates a set of atomic object-existence claims:
$$\mathcal{V} = \{c_1, c_2, \dots, c_n\}$$
Each claim asserts that an object category $o_i$ exists in the image.

To model collective visual hallucination dependencies while preserving exact polynomial-time belief propagation and robust dynamic programming, we construct an **acyclic connected tree** $\mathcal{T} = (\mathcal{V}, \mathcal{E})$.

---

## 2. Node Definition and Candidate Relationships

- **Node $i \in \mathcal{V}$**: Corresponds uniquely to an atomic object claim $c_i = (o_i, \text{span}_i)$, endowed with unary evidence features $(d_i, g_i)$ (detector score and CLIP similarity).
- **Candidate Edges**: Any pair of claims $(i, j)$ in the same image can have a candidate attractive dependency.
- **Relationship Score $s_{ij} \in [0, 1]$**:
  Measures the strength of pairwise co-occurrence and visual-semantic association. In the repository, $s_{ij}$ is derived from:
  1. **Semantic Similarity**: WordNet / cosine embedding similarity between category labels $o_i$ and $o_j$.
  2. **Co-occurrence Frequency**: Normalized frequency of categories $o_i$ and $o_j$ appearing together in the training annotations:
     $$s_{ij} = \frac{N(o_i, o_j)}{\sqrt{N(o_i) N(o_j)}}$$
  3. **Spatial Overlap / Proximity**: When bounding boxes are predicted, the normalized intersection-over-union (IoU) or centroid proximity.

---

## 3. Tree Construction Strategies

We support four explicit topologies:

### 3.1. INDEPENDENT (No Graph)
- $\mathcal{E} = \emptyset$, all $J_{ij} = 0$.
- Serves as the ablation baseline to measure the marginal contribution of graph coupling.

### 3.2. CHAIN Topology
- Edges connect adjacent claims in the order they appear in the model response text:
  $$\mathcal{E} = \{(i, i+1) : i = 0, \dots, n-2\}$$
- Reflects sequential discourse order in the vision-language model generation.

### 3.3. STAR Topology
- The claim with highest evidence certainty (or designated dominant object) acts as the root/hub node $0$:
  $$\mathcal{E} = \{(0, i) : i = 1, \dots, n-1\}$$
- Models contextual conditioning where auxiliary claims depend primarily on the main scene subject.

### 3.4. MAXIMUM SPANNING TREE (MST) Topology
- On the complete candidate graph with edge weights $w_{ij} = s_{ij}$, compute the Maximum Spanning Tree via Kruskal's or Prim's algorithm.
- Maximizes total captured dependency $\sum_{(i,j) \in \mathcal{E}} s_{ij}$ without introducing cycles:
  $$\mathcal{E}_{\text{MST}} = \arg\max_{\mathcal{T} \text{ is a tree}} \sum_{(i,j) \in \mathcal{E}(\mathcal{T})} s_{ij}$$

---

## 4. Coupling Strength Parameterization

Given tree $\mathcal{T} = (\mathcal{V}, \mathcal{E})$ and edge weights $s_{ij} \in [0, 1]$:
$$J_{ij} = \lambda \cdot s_{ij}$$
where $\lambda \ge 0$ is the global coupling multiplier calibrated on the `VALIDATION` split.
Because $\lambda \ge 0$ and $s_{ij} \ge 0$:
$$J_{ij} \ge 0 \quad \text{for all } (i, j) \in \mathcal{E}$$
This strictly satisfies the attractive Ising requirement, guaranteeing coordinate-wise monotonicity (Theorem 1), one-sided extremizer reduction (Proposition 1), and DP optimality (Theorem 2).

---

## 5. Why a Tree?

1. **Exact Inference**: Standard Belief Propagation on trees is exact in 2 passes ($O(n)$ operations), avoiding loopy BP convergence failures or approximation artifacts.
2. **Exact Discretized Robust Optimization**: The budget-coupled robust dynamic programming solver operates in $O(n K^2)$ time by decomposing budget allocations across disjoint subtrees through 1D max-plus convolutions.
3. **No Frustration**: Trees have no cycles, completely eliminating cycle-based frustration and non-convex energy landscapes.

---

## 6. Limitations of the Tree Approximation

1. **Cycle Pruning**: In real scenes, co-occurrence relationships frequently contain cycles (e.g., triangle cliques: *plate*, *fork*, *knife*). Pruning cycles to a spanning tree inevitably discards secondary dependencies.
2. **Root Asymmetry in Re-Rooting**: The optimal tree depends on the choice of objective and may not reflect higher-order (hypergraph) contextual relations.
3. **MST Sensitivity to Weight Noise**: When relationship scores $s_{ij}$ are noisy, small changes in evidence can alter the selected spanning tree edges.
