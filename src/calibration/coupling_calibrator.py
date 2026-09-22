"""Coupling J_ij calibration and graph topology construction.

Supports:
1. Coupling strategies:
   - J0: J_ij = 0 (independent baseline)
   - JCONST: J_ij = lambda (uniform attractive)
   - JWEIGHTED: J_ij = lambda * s_ij with s_ij in [0, 1]
2. Validation tuning of lambda >= 0 (strictly non-negative to preserve M9A monotonicity).
3. Graph topologies:
   - INDEPENDENT
   - CHAIN
   - STAR
   - MAXIMUM SPANNING TREE (MST via Kruskal's algorithm)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.calibration.evidence_models import compute_calibration_diagnostics
from src.pgm.standard_bp import run_standard_bp
from src.pgm.tree_model import TreeModel


class CouplingStrategy(str, Enum):
    J0 = "J0"
    JCONST = "JCONST"
    JWEIGHTED = "JWEIGHTED"


class GraphTopologyType(str, Enum):
    INDEPENDENT = "independent"
    CHAIN = "chain"
    STAR = "star"
    MST = "minimum_spanning_tree"


@dataclass
class CouplingCalibrationResult:
    """Provenance and results of coupling strength calibration."""

    strategy: str
    topology: str
    lambda_param: float
    best_val_score: float
    val_objective: str
    grid_scores: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_kruskal_mst(num_nodes: int, weighted_edges: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """Compute Maximum Spanning Tree (MST) using Kruskal's algorithm on similarity weights.

    Weighted edges sorted in descending order of weight.
    """
    if num_nodes <= 1:
        return []

    sorted_edges = sorted(weighted_edges, key=lambda x: x[2], reverse=True)
    parent = list(range(num_nodes))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> bool:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    mst_edges: List[Tuple[int, int, float]] = []
    for u, v, w in sorted_edges:
        if union(u, v):
            mst_edges.append((min(u, v), max(u, v), w))
            if len(mst_edges) == num_nodes - 1:
                break

    # Connect any remaining disconnected components
    for i in range(num_nodes - 1):
        if len(mst_edges) == num_nodes - 1:
            break
        if union(i, i + 1):
            mst_edges.append((min(i, i + 1), max(i, i + 1), 0.5))

    return mst_edges


def build_candidate_tree_edges(
    n_nodes: int,
    topology: str = "chain",
    similarity_matrix: Optional[Union[List[List[float]], np.ndarray]] = None,
) -> List[Tuple[int, int, float]]:
    """Build candidate tree edges for a claim tree of size n_nodes.

    Returns list of (u, v, score) where score is in [0, 1].
    """
    if n_nodes <= 1:
        return []

    topo = topology.lower().replace("-", "_")
    if topo in ("independent", "none"):
        return []

    mat = np.array(similarity_matrix, dtype=np.float64) if similarity_matrix is not None else None

    if topo == "chain":
        edges = []
        for i in range(n_nodes - 1):
            s = float(mat[i, i + 1]) if mat is not None else 1.0
            edges.append((i, i + 1, float(np.clip(s, 0.0, 1.0))))
        return edges

    elif topo == "star":
        edges = []
        for i in range(1, n_nodes):
            s = float(mat[0, i]) if mat is not None else 1.0
            edges.append((0, i, float(np.clip(s, 0.0, 1.0))))
        return edges

    elif topo in ("minimum_spanning_tree", "mst"):
        all_pairs: List[Tuple[int, int, float]] = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                w = float(mat[i, j]) if mat is not None else 1.0
                all_pairs.append((i, j, float(np.clip(w, 0.0, 1.0))))
        return compute_kruskal_mst(n_nodes, all_pairs)

    else:
        raise ValueError(f"Unknown topology: {topology}")


def construct_tree_edges(
    num_nodes: int,
    topology: Union[str, GraphTopologyType] = GraphTopologyType.CHAIN,
    relationship_matrix: Optional[np.ndarray] = None,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Construct (edges, edge_scores) dict for tree."""
    topo_str = topology.value if isinstance(topology, GraphTopologyType) else str(topology)
    raw = build_candidate_tree_edges(num_nodes, topology=topo_str, similarity_matrix=relationship_matrix)
    edges = [(u, v) for u, v, _ in raw]
    scores = {(u, v): s for u, v, s in raw}
    return edges, scores


class CouplingCalibrator:
    """Calibrates non-negative attractive coupling lambda >= 0 using VALIDATION data."""

    def __init__(
        self,
        strategy: Union[str, CouplingStrategy] = CouplingStrategy.JWEIGHTED,
        default_lambda: float = 0.30,
        objective: str = "nll",
        lambda_grid: Optional[List[float]] = None,
        topology: Union[str, GraphTopologyType] = GraphTopologyType.MST,
    ) -> None:
        if default_lambda < 0.0:
            raise ValueError(f"lambda must be non-negative to preserve M9A monotonicity, got: {default_lambda}")

        self.strategy = strategy.value if isinstance(strategy, CouplingStrategy) else str(strategy).upper()
        self.default_lambda = float(default_lambda)
        self.objective = objective.lower()
        self.lambda_grid = lambda_grid or [0.0, 0.1, 0.25, 0.5, 0.8, 1.2]
        self.topology = topology.value if isinstance(topology, GraphTopologyType) else str(topology).lower()
        self.best_lambda: float = self.default_lambda
        self.provenance: Dict[str, Any] = {}

    def get_couplings(
        self,
        edges: List[Tuple[int, int, float]],
        strategy: Optional[str] = None,
        lambda_val: Optional[float] = None,
    ) -> Dict[Tuple[int, int], float]:
        """Convert edges with similarity scores to attractive couplings J_ij >= 0."""
        strat = (strategy or self.strategy).upper()
        lam = self.best_lambda if lambda_val is None else float(lambda_val)
        if lam < 0.0:
            raise ValueError(f"lambda must be non-negative, got: {lam}")

        j_map: Dict[Tuple[int, int], float] = {}
        for edge in edges:
            u, v = int(edge[0]), int(edge[1])
            s = float(edge[2]) if len(edge) > 2 else 1.0
            if s < 0.0 or s > 1.0:
                raise ValueError(f"Similarity score must be in [0, 1], got: {s}")

            if strat == "J0":
                j = 0.0
            elif strat == "JCONST":
                j = lam
            elif strat == "JWEIGHTED":
                j = lam * s
            else:
                raise ValueError(f"Unknown coupling strategy: {strat}")

            j_map[(min(u, v), max(u, v))] = float(max(0.0, j))
        return j_map

    def fit(
        self,
        val_thetas: np.ndarray,
        val_labels: np.ndarray,
        candidate_edges: List[Tuple[int, int, float]],
        root_index: int = 0,
        lambda_grid: Optional[List[float]] = None,
    ) -> float:
        """Fit optimal non-negative lambda on validation data by evaluating NLL or Brier score."""
        grid = lambda_grid or self.lambda_grid
        for lam in grid:
            if lam < 0.0:
                raise ValueError(f"lambda must be non-negative, got: {lam}")

        thetas = np.asarray(val_thetas, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.float64)
        n = len(thetas)

        if n <= 1 or not candidate_edges or self.strategy == "J0":
            self.best_lambda = 0.0
            self.provenance = {"status": "zero_coupling_fit"}
            return 0.0

        # Ground truth in binary 0/1: labels may be {-1, +1} or {0, 1}
        y_binary = np.where(labels > 0.0, 1.0, 0.0)

        best_score = np.inf
        best_lam = 0.0
        grid_scores = {}

        for lam in grid:
            j_map = self.get_couplings(candidate_edges, lambda_val=lam)
            edges_list = list(j_map.keys())
            model = TreeModel(
                num_nodes=n,
                theta=thetas.tolist(),
                edges=edges_list,
                coupling=j_map,
            )
            res = run_standard_bp(model)
            marginals = res.marginals  # P(h_i = +1)

            diag = compute_calibration_diagnostics(y_binary, marginals)
            score = diag.log_loss if self.objective == "nll" else diag.brier_score

            grid_scores[str(lam)] = float(score)
            if score < best_score:
                best_score = score
                best_lam = lam

        self.best_lambda = float(best_lam)
        self.provenance = {
            "strategy": self.strategy,
            "objective": self.objective,
            "best_lambda": self.best_lambda,
            "best_score": float(best_score),
            "grid_scores": grid_scores,
        }
        return self.best_lambda

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "default_lambda": self.default_lambda,
            "best_lambda": self.best_lambda,
            "objective": self.objective,
            "topology": self.topology,
            "provenance": self.provenance,
        }
