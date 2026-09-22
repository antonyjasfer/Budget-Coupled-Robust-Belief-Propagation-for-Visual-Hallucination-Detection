"""
Baseline methods and inference wrappers for visual hallucination detection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.special import expit

from src.data.schemas import GroundTruthStatus, DecisionStatus
from src.pgm.tree_model import TreeModel
from src.pgm.standard_bp import run_standard_bp, StandardBPResult
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds, RobustBPResult
from src.annotation.schemas import M7ClaimRecord
from src.experiments.parameterization import ImagePGMContext, build_image_pgm_context
from src.experiments.configs import PGMParameterConfig, TreeTopology


@dataclass
class BaselineOutput:
    """
    Standardized inference output for any evaluated baseline method.
    """
    method_name: str
    image_id: str
    claim_id: str
    split: str
    ground_truth: Optional[GroundTruthStatus]
    hallucination_score: float  # continuous score / posterior in [0, 1]
    predicted_decision: DecisionStatus  # SUPPORTED, HALLUCINATED, ABSTAIN
    is_evidence_sensitive: bool
    interval_lower: Optional[float] = None
    interval_upper: Optional[float] = None
    interval_width: Optional[float] = None
    certified_lower: Optional[float] = None
    certified_upper: Optional[float] = None
    field_nominal: Optional[float] = None
    runtime_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_evidence_point_baseline(
    claim: M7ClaimRecord,
    threshold: float = 0.5,
    ground_truth: Optional[GroundTruthStatus] = None,
) -> BaselineOutput:
    """
    Method A: Raw visual evidence baseline without PGM / MRF coupling.
    Uses object detector score d_i directly:
    hallucination_score = 1.0 - d_i
    """
    ev = claim.evidence
    det_score = ev.detector_score if ev.detector_available else 0.5
    if det_score is None:
        det_score = 0.5

    halluc_score = float(1.0 - np.clip(det_score, 0.0, 1.0))
    pred = DecisionStatus.HALLUCINATED if halluc_score >= threshold else DecisionStatus.SUPPORTED

    return BaselineOutput(
        method_name="evidence_point_baseline",
        image_id=claim.image_id,
        claim_id=claim.claim_id,
        split=claim.split,
        ground_truth=ground_truth,
        hallucination_score=halluc_score,
        predicted_decision=pred,
        is_evidence_sensitive=False,
        metadata={"raw_detector_score": det_score, "evidence_source": "owlvit_presence_max"},
    )


def run_standard_bp_baseline(
    ctx: ImagePGMContext,
    ground_truth_map: Optional[Dict[str, Optional[GroundTruthStatus]]] = None,
) -> Dict[str, BaselineOutput]:
    """
    Method B: Standard Belief Propagation (exact 2-pass point posterior inference).
    """
    import time
    t0 = time.perf_counter()
    bp_res: StandardBPResult = run_standard_bp(ctx.model)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    threshold = ctx.config.decision_threshold
    outputs: Dict[str, BaselineOutput] = {}

    for node_idx, cid in ctx.node_to_claim.items():
        p_halluc = float(bp_res.marginals[node_idx])
        eta_field = float(bp_res.effective_fields[node_idx])
        pred = DecisionStatus.HALLUCINATED if p_halluc >= threshold else DecisionStatus.SUPPORTED
        gt = ground_truth_map.get(cid) if ground_truth_map else None

        outputs[cid] = BaselineOutput(
            method_name="standard_bp",
            image_id=ctx.image_id,
            claim_id=cid,
            split=ctx.claims_by_id[cid].split if hasattr(ctx, "claims_by_id") and cid in ctx.claims_by_id else "unknown",
            ground_truth=gt,
            hallucination_score=p_halluc,
            predicted_decision=pred,
            is_evidence_sensitive=False,
            field_nominal=eta_field,
            runtime_ms=elapsed_ms / ctx.model.num_nodes,
            metadata={"effective_field": eta_field, "theta": float(ctx.model.theta[node_idx])},
        )

    return outputs


def run_robust_bp_proposed(
    ctx: ImagePGMContext,
    ground_truth_map: Optional[Dict[str, Optional[GroundTruthStatus]]] = None,
) -> Dict[str, BaselineOutput]:
    """
    Method C: Proposed Budget-Coupled Robust Belief Propagation.
    """
    import time
    threshold = ctx.config.decision_threshold
    outputs: Dict[str, BaselineOutput] = {}

    for node_idx, cid in ctx.node_to_claim.items():
        t0 = time.perf_counter()
        res: RobustBPResult = solve_robust_bp(
            model=ctx.model,
            target_node=node_idx,
            budget=ctx.budget,
            num_grid_steps=ctx.config.grid_steps,
            lipschitz_const=ctx.config.lipschitz_const,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        low = float(res.lower_grid)
        up = float(res.upper_grid)
        width = float(up - low)
        mid = float(0.5 * (low + up))

        # Robust decision logic:
        # If upper bound < threshold -> definitely SUPPORTED
        # If lower bound > threshold -> definitely HALLUCINATED
        # If lower <= threshold <= upper -> ABSTAIN (evidence-sensitive)
        evidence_sensitive = (low <= threshold <= up)
        if up < threshold:
            pred = DecisionStatus.SUPPORTED
        elif low > threshold:
            pred = DecisionStatus.HALLUCINATED
        else:
            pred = DecisionStatus.ABSTAIN

        gt = ground_truth_map.get(cid) if ground_truth_map else None

        outputs[cid] = BaselineOutput(
            method_name="robust_bp_budget_coupled",
            image_id=ctx.image_id,
            claim_id=cid,
            split=ctx.claims_by_id[cid].split if hasattr(ctx, "claims_by_id") and cid in ctx.claims_by_id else "unknown",
            ground_truth=gt,
            hallucination_score=mid,
            predicted_decision=pred,
            is_evidence_sensitive=evidence_sensitive,
            interval_lower=low,
            interval_upper=up,
            interval_width=width,
            certified_lower=float(res.lower_certified),
            certified_upper=float(res.upper_certified),
            field_nominal=float(res.field_nominal),
            runtime_ms=elapsed_ms,
            metadata={
                "budget": ctx.budget,
                "grid_steps": ctx.config.grid_steps,
                "field_cert_gap": float(res.field_cert_gap),
                "nominal_marginal": float(res.nominal_marginal),
            },
        )

    return outputs


def run_robust_bp_no_coupling(
    ctx: ImagePGMContext,
    ground_truth_map: Optional[Dict[str, Optional[GroundTruthStatus]]] = None,
) -> Dict[str, BaselineOutput]:
    """
    Method D: Robust BP ablation without graph coupling (J_ij = 0).
    """
    # Create uncoupled tree model copy (preserve tree edges with zero coupling J=0)
    uncoupled_coupling = {e: 0.0 for e in ctx.model.edges}
    uncoupled_model = TreeModel(
        num_nodes=ctx.model.num_nodes,
        theta=ctx.model.theta.copy(),
        edges=list(ctx.model.edges),
        coupling=uncoupled_coupling,
        epsilon=ctx.model.epsilon.copy(),
    )
    uncoupled_ctx = ImagePGMContext(
        image_id=ctx.image_id,
        model=uncoupled_model,
        claim_ids=ctx.claim_ids,
        claim_to_node=ctx.claim_to_node,
        node_to_claim=ctx.node_to_claim,
        budget=ctx.budget,
        parameter_hash=ctx.parameter_hash + "_uncoupled",
        config=ctx.config,
    )
    outputs = run_robust_bp_proposed(uncoupled_ctx, ground_truth_map=ground_truth_map)
    for out in outputs.values():
        out.method_name = "robust_bp_no_coupling"
    return outputs


def run_robust_bp_independent_box(
    ctx: ImagePGMContext,
    ground_truth_map: Optional[Dict[str, Optional[GroundTruthStatus]]] = None,
) -> Dict[str, BaselineOutput]:
    """
    Method E: Robust inference without global budget coupling (Independent Box bounds).
    """
    import time
    threshold = ctx.config.decision_threshold
    outputs: Dict[str, BaselineOutput] = {}

    for node_idx, cid in ctx.node_to_claim.items():
        t0 = time.perf_counter()
        p_low, p_up, f_low, f_up = solve_independent_box_bounds(ctx.model, target_node=node_idx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        low = float(p_low)
        up = float(p_up)
        width = float(up - low)
        mid = float(0.5 * (low + up))

        evidence_sensitive = (low <= threshold <= up)
        if up < threshold:
            pred = DecisionStatus.SUPPORTED
        elif low > threshold:
            pred = DecisionStatus.HALLUCINATED
        else:
            pred = DecisionStatus.ABSTAIN

        gt = ground_truth_map.get(cid) if ground_truth_map else None

        outputs[cid] = BaselineOutput(
            method_name="robust_bp_independent_box",
            image_id=ctx.image_id,
            claim_id=cid,
            split=ctx.claims_by_id[cid].split if hasattr(ctx, "claims_by_id") and cid in ctx.claims_by_id else "unknown",
            ground_truth=gt,
            hallucination_score=mid,
            predicted_decision=pred,
            is_evidence_sensitive=evidence_sensitive,
            interval_lower=low,
            interval_upper=up,
            interval_width=width,
            runtime_ms=elapsed_ms,
            metadata={"unbudgeted": True},
        )

    return outputs


EXTERNAL_METHODS_CONCEPTUAL_NOTE: str = """
RELATED-WORK COMPARISON & METHODOLOGICAL DISTINCTION:
1. BTProp (Binary Tree Propagation): Proposed for discrete belief propagation under local constraints,
   differing from our budget-coupled max-plus convolution in its treatment of shared L1 perturbation budgets
   and Lipschitz certified discretization gaps.
2. Standard Point-Posteriors: Standard graphical models compute only point expectations P(H_i = +1 | E),
   masking evidence fragility and high-uncertainty boundary crossings.
"""
