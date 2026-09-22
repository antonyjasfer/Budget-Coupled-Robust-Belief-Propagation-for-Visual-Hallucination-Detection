# Spanning Tree Objective and Semantics Audit

**Milestone**: Phase 9D-0 (Integrity Gate A)  
**Date**: September 2026  
**Status**: AUDITED & VERIFIED  

---

## 1. Context and Problem Statement

The M9C report listed `minimum_spanning_tree` as one of the graph topologies. In visual hallucination detection, pairwise edge weights $s_{ij} \in [0, 1]$ represent **semantic relationship strength** (e.g. CLIP/WordNet similarity or normalized co-occurrence frequency).

Higher $s_{ij}$ indicates **stronger semantic association**, meaning claims $i$ and $j$ are more likely to share consistent hallucination states.

The mathematical goal of tree-based structured prediction is to retain the strongest relationships without introducing cycles:
$$\mathcal{T}^* = \arg\max_{\mathcal{T} \text{ is a tree}} \sum_{(i,j) \in \mathcal{E}(\mathcal{T})} s_{ij}$$

This audit resolves whether the repository implementation builds a **Maximum Spanning Tree** (maximizing relationship strength) or inadvertently prunes strong edges by minimizing similarity.

---

## 2. Code Inspection: `compute_kruskal_mst`

In `src/calibration/coupling_calibrator.py` (lines 58–99):

```python
def compute_kruskal_mst(num_nodes: int, weighted_edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """Compute Maximum Spanning Tree (MST) using Kruskal's algorithm on similarity weights.

    Weighted edges sorted in descending order of weight.
    """
    if num_nodes <= 1:
        return []

    sorted_edges = sorted(weighted_edges, key=lambda x: x[2], reverse=True)
    parent = list(range(num_nodes))
    ...
```

Notice the critical sort parameter:
$$\text{sorted\_edges} = \text{sorted}(\text{weighted\_edges}, \text{key}=\lambda x: x[2], \mathbf{reverse=True})$$

Kruskal's algorithm greedily evaluates candidate edges in descending order of similarity weight $w = s_{ij}$. An edge is accepted if and only if it connects two previously disjoint components. By the cut/cycle optimality properties of Kruskal's greedy matroid algorithm:
$$\sum_{(u, v) \in \mathcal{E}} s_{uv} = \max_{\mathcal{T}} \sum_{(u, v) \in \mathcal{T}} s_{uv}$$

---

## 3. Resolution of Central Audit Questions

1. **Does $s_{ij}$ increase with semantic similarity?**  
   **YES**. As documented in `docs/method/claim_tree_construction.md`, $s_{ij} \in [0, 1]$ measures WordNet category similarity, co-occurrence frequency, or spatial overlap. Higher values correspond directly to stronger semantic relationships.

2. **Does the tree maximize total similarity?**  
   **YES**. Because edges are sorted in descending order (`reverse=True`), Kruskal's algorithm selects candidate edges with maximal similarity first.

3. **Is edge cost transformed (e.g., $\text{cost}_{ij} = 1 - s_{ij}$ or $\text{cost}_{ij} = -s_{ij}$)?**  
   **Direct descending weight selection is used instead of sign transformation**. Sorting weights descending with Kruskal's algorithm is mathematically isomorphic to minimizing $\text{cost}_{ij} = -s_{ij}$ or $1 - s_{ij}$.

4. **Is only the function name / enum misleading?**  
   **YES**. The enum `GraphTopologyType.MST = "minimum_spanning_tree"` and the abbreviation "MST" are common shorthand for "spanning tree via Kruskal's algorithm", but the mathematical objective implemented in code is strictly the **Maximum Spanning Tree (MaxST)** on similarity $s_{ij}$.

5. **Is the implementation genuinely choosing the weakest semantic edges?**  
   **NO**. The implementation does NOT choose the weakest edges; it explicitly chooses the strongest semantic edges.

---

## 4. Conclusion and Verification

- The existing topology code correctly maximizes semantic dependency $\sum_{(i,j)} s_{ij}$.
- No functional code alteration is required in `compute_kruskal_mst` because its behavior is mathematically optimal.
- Regression test `tests/test_mst_semantics.py` verifies that for any complete graph with varying weights, the total weight of the selected tree equals the true Maximum Spanning Tree weight, and strictly exceeds the Minimum Spanning Tree weight.
