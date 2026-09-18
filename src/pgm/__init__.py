"""
Probabilistic Graphical Models (PGM) package for binary attractive tree models.
"""

from src.pgm.tree_model import TreeModel, create_chain_tree, create_star_tree
from src.pgm.standard_bp import (
    stable_f_J,
    run_standard_bp,
    compute_rooted_total_field,
    StandardBPResult,
)
from src.pgm.brute_force import (
    run_brute_force_inference,
    run_brute_force_robust_grid,
    ExactInferenceResult,
)

__all__ = [
    "TreeModel",
    "create_chain_tree",
    "create_star_tree",
    "stable_f_J",
    "run_standard_bp",
    "compute_rooted_total_field",
    "StandardBPResult",
    "run_brute_force_inference",
    "run_brute_force_robust_grid",
    "ExactInferenceResult",
]
