"""
Binary attractive tree probabilistic graphical model.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np


@dataclass
class TreeModel:
    """
    Represents a binary attractive tree Markov Random Field (Ising model).

    Spins are binary: h_i in {-1, +1}.
    Unary potentials: psi_i(h_i) = exp((theta_i + delta_i) * h_i)
    Pairwise potentials: psi_ij(h_i, h_j) = exp(J_ij * h_i * h_j), where J_ij >= 0 (attractive).

    Attributes:
        num_nodes: Number of nodes n in the tree, indexed 0..n-1.
        theta: Array of shape (n,) containing unary field parameters.
        edges: List of undirected edges (u, v) with u < v.
        coupling: Dictionary mapping edge tuple (min(u,v), max(u,v)) -> J >= 0.
        epsilon: Optional array of shape (n,) containing perturbation bounds |delta_i| <= eps_i.
    """
    num_nodes: int
    theta: np.ndarray
    edges: List[Tuple[int, int]] = field(default_factory=list)
    coupling: Dict[Tuple[int, int], float] = field(default_factory=dict)
    epsilon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    adj: Dict[int, List[int]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.theta = np.asarray(self.theta, dtype=np.float64)
        if self.theta.shape != (self.num_nodes,):
            raise ValueError(f"theta must have shape ({self.num_nodes},), got {self.theta.shape}")

        if len(self.epsilon) == 0:
            self.epsilon = np.zeros(self.num_nodes, dtype=np.float64)
        else:
            self.epsilon = np.asarray(self.epsilon, dtype=np.float64)
            if self.epsilon.shape != (self.num_nodes,):
                raise ValueError(f"epsilon must have shape ({self.num_nodes},), got {self.epsilon.shape}")
            if np.any(self.epsilon < -1e-12):
                raise ValueError("epsilon bounds must be non-negative")

        # Build adjacency list
        self.adj = {i: [] for i in range(self.num_nodes)}
        for u, v in self.edges:
            if u == v or u < 0 or u >= self.num_nodes or v < 0 or v >= self.num_nodes:
                raise ValueError(f"Invalid edge: ({u}, {v})")
            edge_key = (min(u, v), max(u, v))
            j_val = self.coupling.get(edge_key, 0.0)
            if j_val < -1e-12:
                raise ValueError(f"Coupling J for edge {edge_key} must be attractive (J >= 0), got {j_val}")
            self.adj[u].append(v)
            self.adj[v].append(u)

        self._validate_tree_topology()

    def get_coupling(self, u: int, v: int) -> float:
        """Get the attractive coupling J_uv >= 0 between nodes u and v."""
        edge_key = (min(u, v), max(u, v))
        return float(self.coupling.get(edge_key, 0.0))

    def _validate_tree_topology(self):
        """Validate that the graph is a valid connected acyclic graph (tree)."""
        if self.num_nodes == 0:
            return
        if self.num_nodes == 1:
            if len(self.edges) != 0:
                raise ValueError("Single node tree cannot have edges")
            return

        if len(self.edges) != self.num_nodes - 1:
            raise ValueError(
                f"Tree with {self.num_nodes} nodes must have {self.num_nodes - 1} edges, got {len(self.edges)}"
            )

        # Check connectivity and cycle absence using BFS/DFS
        visited: Set[int] = set()
        queue = [0]
        visited.add(0)

        while queue:
            curr = queue.pop(0)
            for neighbor in self.adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(visited) != self.num_nodes:
            raise ValueError("Graph is disconnected; must be a connected tree")

    def get_rooted_tree(self, root: int) -> Tuple[List[int], Dict[int, Optional[int]], Dict[int, List[int]]]:
        """
        Produce a topological ordering (leaves to root), parent map, and children map.

        Args:
            root: Target node index to act as root of the tree.

        Returns:
            post_order: List of node indices ordered from leaves to root.
            parent: Dict mapping node -> parent node (root has parent None).
            children: Dict mapping node -> list of children nodes.
        """
        if root < 0 or root >= self.num_nodes:
            raise ValueError(f"Invalid root node {root}")

        parent: Dict[int, Optional[int]] = {root: None}
        children: Dict[int, List[int]] = {i: [] for i in range(self.num_nodes)}
        bfs_order = [root]
        queue = [root]

        while queue:
            curr = queue.pop(0)
            for neighbor in self.adj[curr]:
                if neighbor != parent[curr]:
                    parent[neighbor] = curr
                    children[curr].append(neighbor)
                    bfs_order.append(neighbor)
                    queue.append(neighbor)

        # Reverse BFS order gives bottom-up (leaves to root) post-order
        post_order = bfs_order[::-1]
        return post_order, parent, children


def create_chain_tree(
    num_nodes: int,
    theta: np.ndarray,
    couplings: np.ndarray,
    epsilon: Optional[np.ndarray] = None
) -> TreeModel:
    """
    Helper to construct a chain tree: 0 - 1 - 2 - ... - (n-1).

    Args:
        num_nodes: Number of nodes.
        theta: Unary field array of shape (num_nodes,).
        couplings: Edge coupling array of shape (num_nodes - 1,).
        epsilon: Optional perturbation bounds array of shape (num_nodes,).
    """
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    coupling_dict = {(i, i + 1): float(couplings[i]) for i in range(num_nodes - 1)}
    eps = epsilon if epsilon is not None else np.zeros(num_nodes)
    return TreeModel(
        num_nodes=num_nodes,
        theta=theta,
        edges=edges,
        coupling=coupling_dict,
        epsilon=eps
    )


def create_star_tree(
    num_leaves: int,
    center_theta: float,
    leaf_thetas: np.ndarray,
    center_leaf_couplings: np.ndarray,
    epsilon: Optional[np.ndarray] = None
) -> TreeModel:
    """
    Helper to construct a star tree with center node 0 and leaf nodes 1..num_leaves.
    """
    num_nodes = num_leaves + 1
    theta = np.zeros(num_nodes)
    theta[0] = center_theta
    theta[1:] = leaf_thetas

    edges = [(0, i) for i in range(1, num_nodes)]
    coupling_dict = {(0, i): float(center_leaf_couplings[i - 1]) for i in range(1, num_nodes)}
    eps = epsilon if epsilon is not None else np.zeros(num_nodes)
    return TreeModel(
        num_nodes=num_nodes,
        theta=theta,
        edges=edges,
        coupling=coupling_dict,
        epsilon=eps
    )
