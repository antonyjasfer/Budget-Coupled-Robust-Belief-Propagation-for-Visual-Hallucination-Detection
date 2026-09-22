"""
Specialized robust inference runner for profile generation and witness extraction.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.data.schemas import GroundTruthStatus, DecisionStatus
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds, RobustBPResult
from src.annotation.schemas import M7ClaimRecord
from src.experiments.parameterization import ImagePGMContext, build_image_pgm_context
from src.experiments.configs import PGMParameterConfig


@dataclass
class RobustClaimProfile:
    """
    Detailed robust interval profile across discrete budget points for a target claim.
    """
    image_id: str
    claim_id: str
    object_category: str
    ground_truth: Optional[str]
    budget: float
    num_grid_steps: int
    lower_grid: float
    upper_grid: float
    lower_certified: float
    upper_certified: float
    interval_width: float
    nominal_marginal: float
    field_nominal: float
    field_lower_grid: float
    field_upper_grid: float
    field_cert_gap: float
    budget_grid: List[float]
    upper_field_profile: List[float]
    lower_field_profile: List[float]
    witness_upper_delta: List[float]
    witness_lower_delta: List[float]
    witness_upper_norm: float
    witness_lower_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_claim_robust_profile(
    ctx: ImagePGMContext,
    target_claim_id: str,
    ground_truth: Optional[GroundTruthStatus] = None,
) -> RobustClaimProfile:
    """
    Compute full budget profile and witness perturbation vectors for a specific claim.
    """
    if target_claim_id not in ctx.claim_to_node:
        raise KeyError(f"Claim ID '{target_claim_id}' not found in image context.")

    target_node = ctx.claim_to_node[target_claim_id]
    cfg = ctx.config

    res: RobustBPResult = solve_robust_bp(
        model=ctx.model,
        target_node=target_node,
        budget=ctx.budget,
        num_grid_steps=cfg.grid_steps,
        lipschitz_const=cfg.lipschitz_const,
    )

    width = float(res.upper_grid - res.lower_grid)
    gt_str = ground_truth.value if ground_truth is not None else None

    wit_up_delta = res.witness_upper.delta.tolist() if hasattr(res.witness_upper.delta, "tolist") else list(res.witness_upper.delta)
    wit_low_delta = res.witness_lower.delta.tolist() if hasattr(res.witness_lower.delta, "tolist") else list(res.witness_lower.delta)

    return RobustClaimProfile(
        image_id=ctx.image_id,
        claim_id=target_claim_id,
        object_category=ctx.claim_ids[target_node].split("_")[-1] if "_" in ctx.claim_ids[target_node] else "unknown",
        ground_truth=gt_str,
        budget=float(ctx.budget),
        num_grid_steps=int(res.num_grid_steps),
        lower_grid=float(res.lower_grid),
        upper_grid=float(res.upper_grid),
        lower_certified=float(res.lower_certified),
        upper_certified=float(res.upper_certified),
        interval_width=width,
        nominal_marginal=float(res.nominal_marginal),
        field_nominal=float(res.field_nominal),
        field_lower_grid=float(res.field_lower_grid),
        field_upper_grid=float(res.field_upper_grid),
        field_cert_gap=float(res.field_cert_gap),
        budget_grid=[float(b) for b in res.budget_grid],
        upper_field_profile=[float(x) for x in res.upper_field_profile],
        lower_field_profile=[float(x) for x in res.lower_field_profile],
        witness_upper_delta=wit_up_delta,
        witness_lower_delta=wit_low_delta,
        witness_upper_norm=float(res.witness_upper.total_budget_spent),
        witness_lower_norm=float(res.witness_lower.total_budget_spent),
    )
