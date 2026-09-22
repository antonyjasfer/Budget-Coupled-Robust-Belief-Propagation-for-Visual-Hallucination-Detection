"""Tests for Spanning Tree Semantics and Kruskal Objective.

Verifies that compute_kruskal_mst correctly computes the Maximum Spanning Tree
(maximizing total relationship strength) when given semantic similarities.
"""

import pytest
import numpy as np

from src.calibration.coupling_calibrator import (
    compute_kruskal_mst,
    build_candidate_tree_edges,
    construct_tree_edges,
    GraphTopologyType,
)


def test_kruskal_maximizes_total_similarity():
    """Verify that compute_kruskal_mst selects the Maximum Spanning Tree."""
    # Graph with 4 nodes:
    # Edges:
    # (0, 1): 0.9  (strong)
    # (1, 2): 0.8  (strong)
    # (2, 3): 0.7  (strong)
    # (0, 2): 0.3  (weak)
    # (0, 3): 0.2  (weak)
    # (1, 3): 0.1  (weak)
    # The true MaxST uses edges (0, 1) w=0.9, (1, 2) w=0.8, (2, 3) w=0.7. Total = 2.4.
    # The MinST would use edges (1, 3) w=0.1, (0, 3) w=0.2, (0, 2) w=0.3. Total = 0.6.
    num_nodes = 4
    edges = [
        (0, 1, 0.9),
        (1, 2, 0.8),
        (2, 3, 0.7),
        (0, 2, 0.3),
        (0, 3, 0.2),
        (1, 3, 0.1),
    ]

    mst = compute_kruskal_mst(num_nodes, edges)
    assert len(mst) == num_nodes - 1

    selected_weights = [w for _, _, w in mst]
    total_similarity = sum(selected_weights)

    # Must equal MaxST total
    assert np.isclose(total_similarity, 2.4)
    # Must strictly exceed MinST total
    assert total_similarity > 1.5

    # Verify specific edges selected
    selected_pairs = {(min(u, v), max(u, v)) for u, v, _ in mst}
    assert selected_pairs == {(0, 1), (1, 2), (2, 3)}


def test_kruskal_mst_triangle_picks_two_heaviest():
    """On a triangle graph, Kruskal must pick the two edges with highest similarity."""
    num_nodes = 3
    edges = [
        (0, 1, 0.2),
        (1, 2, 0.6),
        (0, 2, 0.9),
    ]
    mst = compute_kruskal_mst(num_nodes, edges)
    selected_pairs = {(min(u, v), max(u, v)) for u, v, _ in mst}
    assert selected_pairs == {(0, 2), (1, 2)}
    total_w = sum(w for _, _, w in mst)
    assert np.isclose(total_w, 1.5)


def test_build_candidate_tree_edges_mst_topology():
    """Verify build_candidate_tree_edges uses MaxST for topology='mst' or 'minimum_spanning_tree'."""
    sim_matrix = np.array([
        [1.0, 0.95, 0.10],
        [0.95, 1.0, 0.85],
        [0.10, 0.85, 1.0],
    ])
    edges = build_candidate_tree_edges(3, topology="mst", similarity_matrix=sim_matrix)
    assert len(edges) == 2
    pairs = {(u, v) for u, v, _ in edges}
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) not in pairs
