"""
Budget-Coupled Robust Belief Propagation package.
"""

from src.robust_bp.budget_convolution import max_plus_convolve, min_plus_convolve
from src.robust_bp.certification import (
    compute_continuous_certificate,
    compute_one_node_continuous_bounds,
    CertificateResult,
)
from src.robust_bp.witness import (
    trace_witness_perturbation,
    validate_witness,
    WitnessResult,
)
from src.robust_bp.solver import (
    solve_robust_bp,
    solve_independent_box_bounds,
    RobustBPResult,
)

__all__ = [
    "max_plus_convolve",
    "min_plus_convolve",
    "compute_continuous_certificate",
    "compute_one_node_continuous_bounds",
    "CertificateResult",
    "trace_witness_perturbation",
    "validate_witness",
    "WitnessResult",
    "solve_robust_bp",
    "solve_independent_box_bounds",
    "RobustBPResult",
]
