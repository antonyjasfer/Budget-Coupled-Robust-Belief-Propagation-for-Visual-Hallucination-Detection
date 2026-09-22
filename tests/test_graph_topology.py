"""Unit tests for claim graph topology construction and tree guarantees."""

import pytest

from src.calibration.coupling_calibrator import build_candidate_tree_edges


def test_independent_topology():
    """Verify independent topology produces 0 edges."""
    edges = build_candidate_tree_edges(n_nodes=4, topology="independent")
    assert len(edges) == 0


def test_chain_topology():
    """Verify chain topology produces exactly N-1 sequential edges."""
    edges = build_candidate_tree_edges(n_nodes=5, topology="chain")
    assert len(edges) == 4
    for i, (u, v, s) in enumerate(edges):
        assert u == i
        assert v == i + 1
        assert 0.0 <= s <= 1.0


def test_star_topology():
    """Verify star topology connects root (node 0) to all other nodes."""
    edges = build_candidate_tree_edges(n_nodes=5, topology="star")
    assert len(edges) == 4
    for u, v, s in edges:
        assert u == 0
        assert v > 0
        assert 0.0 <= s <= 1.0


def test_mst_topology_maximum_similarity():
    """Verify minimum/maximum spanning tree algorithm selects highest-weight tree."""
    # 3 nodes: complete graph with pairwise similarities:
    # (0, 1): 0.2
    # (1, 2): 0.8
    # (0, 2): 0.9
    # Maximum spanning tree must choose (0, 2) [0.9] and (1, 2) [0.8], rejecting (0, 1) [0.2]
    sim_matrix = [
        [1.0, 0.2, 0.9],
        [0.2, 1.0, 0.8],
        [0.9, 0.8, 1.0],
    ]
    edges = build_candidate_tree_edges(
        n_nodes=3,
        topology="minimum_spanning_tree",
        similarity_matrix=sim_matrix,
    )
    assert len(edges) == 2
    # Check that highest similarity edges were selected
    selected_pairs = {(min(u, v), max(u, v)) for u, v, _ in edges}
    assert (0, 2) in selected_pairs
    assert (1, 2) in selected_pairs
    assert (0, 1) not in selected_pairs


def test_single_node_tree():
    """Verify 1-node claim trees return 0 edges for any topology."""
    for topo in ["independent", "chain", "star", "minimum_spanning_tree"]:
        edges = build_candidate_tree_edges(n_nodes=1, topology=topo)
        assert len(edges) == 0
