"""
PGM Parameterization and evidence-to-graph mapping for M8.
"""

from dataclasses import dataclass, field
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.pgm.tree_model import TreeModel
from src.annotation.schemas import M7ClaimRecord
from src.experiments.configs import PGMParameterConfig, TreeTopology


@dataclass
class ImagePGMContext:
    """
    Constructed PGM tree model and node mappings for an image.
    """
    image_id: str
    model: TreeModel
    claim_ids: List[str]
    claim_to_node: Dict[str, int]
    node_to_claim: Dict[int, str]
    budget: float
    parameter_hash: str
    config: PGMParameterConfig
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_unary_theta(
    detector_score: Optional[float],
    clip_score: Optional[float],
    w_det: float = 1.0,
    w_clip: float = 0.0,
    theta_max: float = 5.0,
    eps_num: float = 1e-4,
) -> float:
    """
    Map raw multimodal evidence into unary Ising potential theta_i.

    Mathematical Mapping:
    - High visual support (high detector score / high CLIP similarity) -> SUPPORTED (h_i = -1) -> theta_i < 0.
    - Low visual support (low detector score / low CLIP similarity) -> HALLUCINATED (h_i = +1) -> theta_i > 0.
    - P(h_i = +1 | theta_i) = sigmoid(2 * theta_i).
    """
    # 1. Normalized visual support score from detector in [eps_num, 1 - eps_num]
    d_val = 0.5 if detector_score is None else float(detector_score)
    s_det = float(np.clip(d_val, eps_num, 1.0 - eps_num))

    # 2. Normalized visual support score from CLIP in [eps_num, 1 - eps_num]
    if clip_score is not None:
        c_norm = (float(clip_score) + 1.0) / 2.0  # map [-1, 1] to [0, 1]
        s_clip = float(np.clip(c_norm, eps_num, 1.0 - eps_num))
    else:
        s_clip = s_det

    # 3. Weighted visual support combination
    total_w = w_det + w_clip
    if total_w <= 1e-12:
        v_support = 0.5
    else:
        v_support = (w_det * s_det + w_clip * s_clip) / total_w
    v_support = float(np.clip(v_support, eps_num, 1.0 - eps_num))

    # 4. Ising field: theta = 0.5 * ln((1 - v) / v)
    theta = 0.5 * np.log((1.0 - v_support) / v_support)
    theta_clipped = float(np.clip(theta, -theta_max, theta_max))
    return theta_clipped


def compute_uncertainty_epsilon(
    detector_score: Optional[float],
    clip_score: Optional[float],
    base_epsilon: float = 0.5493061443340549,
    epsilon_scale: float = 1.0,
    discrepancy_weight: float = 0.0,
    min_epsilon: float = 0.01,
) -> float:
    """
    Compute local perturbation bound epsilon_i for a claim.
    """
    eps = base_epsilon * epsilon_scale
    if discrepancy_weight > 1e-12 and detector_score is not None and clip_score is not None:
        s_det = float(np.clip(detector_score, 0.0, 1.0))
        s_clip = float(np.clip((clip_score + 1.0) / 2.0, 0.0, 1.0))
        discrepancy = abs(s_det - s_clip)
        eps += discrepancy_weight * discrepancy

    return float(max(min_epsilon, eps))


def build_image_pgm_context(
    image_id: str,
    claims: Optional[List[M7ClaimRecord]] = None,
    config: Optional[PGMParameterConfig] = None,
    *,
    claim_records: Optional[List[M7ClaimRecord]] = None,
    cfg: Optional[PGMParameterConfig] = None,
) -> ImagePGMContext:
    """
    Construct a TreeModel for all claims associated with a given image.
    """
    actual_claims = claims if claims is not None else (claim_records or [])
    actual_config = config or cfg or PGMParameterConfig()
    
    n = len(actual_claims)
    if n == 0:
        raise ValueError(f"Cannot build PGM model for image '{image_id}' with 0 claims.")

    cfg = actual_config
    sorted_claims = sorted(actual_claims, key=lambda c: c.claim_id)
    claim_ids = [c.claim_id for c in sorted_claims]
    claim_to_node = {cid: idx for idx, cid in enumerate(claim_ids)}
    node_to_claim = {idx: cid for idx, cid in enumerate(claim_ids)}

    theta = np.zeros(n, dtype=np.float64)
    epsilon = np.zeros(n, dtype=np.float64)

    for idx, claim in enumerate(sorted_claims):
        ev = claim.evidence
        det_score = ev.detector_score if ev.detector_available else None
        clip_score = ev.clip_score if ev.similarity_available else None

        th = compute_unary_theta(
            detector_score=det_score,
            clip_score=clip_score,
            w_det=cfg.unary_detector_weight,
            w_clip=cfg.unary_clip_weight,
            theta_max=cfg.theta_clip_max,
        )
        eps = compute_uncertainty_epsilon(
            detector_score=det_score,
            clip_score=clip_score,
            base_epsilon=cfg.base_epsilon,
            epsilon_scale=cfg.epsilon_scale,
            discrepancy_weight=cfg.uncertainty_discrepancy_weight,
            min_epsilon=cfg.min_epsilon,
        )
        theta[idx] = th
        epsilon[idx] = eps

    edges: List[Tuple[int, int]] = []
    coupling: Dict[Tuple[int, int], float] = {}

    j_val = float(max(0.0, cfg.base_coupling_j))

    if n > 1:
        if cfg.topology == TreeTopology.STAR:
            # Center node 0, edges (0, 1), (0, 2), ..., (0, n-1)
            for i in range(1, n):
                edge_key = (0, i)
                edges.append(edge_key)
                coupling[edge_key] = j_val if cfg.topology != TreeTopology.INDEPENDENT else 0.0
        else:
            # Default CHAIN topology: (0, 1), (1, 2), ..., (n-2, n-1)
            for i in range(n - 1):
                edge_key = (i, i + 1)
                edges.append(edge_key)
                coupling[edge_key] = j_val if cfg.topology != TreeTopology.INDEPENDENT else 0.0

    model = TreeModel(
        num_nodes=n,
        theta=theta,
        edges=edges,
        coupling=coupling,
        epsilon=epsilon,
    )

    # Compute budget B
    if cfg.fixed_budget is not None:
        budget = float(max(0.0, cfg.fixed_budget))
    else:
        # Scale budget by budget_ratio * base_epsilon (or budget_ratio * sum(epsilon))
        budget = float(max(0.0, cfg.budget_ratio * cfg.base_epsilon))

    # Parameter hash for reproducibility provenance
    param_dict = {
        "image_id": image_id,
        "num_nodes": n,
        "theta": theta.tolist(),
        "epsilon": epsilon.tolist(),
        "edges": edges,
        "coupling": {f"{u}_{v}": float(val) for (u, v), val in coupling.items()},
        "budget": budget,
        "config": cfg.to_dict(),
    }
    param_hash = hashlib.sha256(json.dumps(param_dict, sort_keys=True).encode("utf-8")).hexdigest()

    return ImagePGMContext(
        image_id=image_id,
        model=model,
        claim_ids=claim_ids,
        claim_to_node=claim_to_node,
        node_to_claim=node_to_claim,
        budget=budget,
        parameter_hash=param_hash,
        config=cfg,
        metadata={"total_claims": n},
    )
