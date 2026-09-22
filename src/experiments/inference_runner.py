"""
Core execution engine for standard and robust inference across datasets and ablations.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Any, Iterator
import numpy as np

from src.data.schemas import GroundTruthStatus, DecisionStatus
from src.pgm.tree_model import TreeModel
from src.pgm.standard_bp import run_standard_bp, StandardBPResult
from src.robust_bp.solver import solve_robust_bp, solve_independent_box_bounds, RobustBPResult
from src.annotation.schemas import M7ClaimRecord, FinalGroundTruthRecord
from src.experiments.configs import M8ExperimentConfig, PGMParameterConfig, ExecutionMode
from src.experiments.loaders import M8DatasetBundle
from src.experiments.parameterization import build_image_pgm_context, ImagePGMContext


@dataclass
class ClaimEvaluationResult:
    """
    Standardized, self-contained record for raw claim-level evaluation results.
    """
    run_id: str
    experiment_name: str
    condition: str
    image_id: str
    claim_id: str
    object_category: str
    text_span: Optional[str]
    split: str
    ground_truth: Optional[str]  # "supported", "hallucinated", "unknown", None
    detector_score: Optional[float]
    clip_score: Optional[float]
    theta_i: float
    epsilon_i: float
    coupling_j: float
    budget: float
    epsilon_scale: float
    decision_threshold: float
    grid_steps: int
    standard_posterior: float
    standard_prediction: str  # "supported", "hallucinated"
    robust_lower: float
    robust_upper: float
    robust_midpoint: float
    interval_width: float
    certified_lower: float
    certified_upper: float
    field_cert_gap: float
    robust_prediction: str  # "supported", "hallucinated", "abstain"
    evidence_sensitive: bool
    abstained: bool
    corruption_type: str = "none"
    corruption_severity: str = "clean"
    runtime_ms: float = 0.0
    seed: int = 42
    parameter_hash: str = ""
    dataset_hash: str = ""
    code_revision: str = ""
    is_synthetic: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimEvaluationResult":
        return cls(**data)


def compute_cache_key(
    image_id: str,
    condition: str,
    parameter_hash: str,
    dataset_hash: str,
) -> str:
    """Compute unique cache key for an image condition."""
    raw = f"{image_id}_{condition}_{parameter_hash}_{dataset_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class InferenceRunner:
    """
    Executes standard and robust BP over dataset bundles with caching and resume capability.
    """
    def __init__(
        self,
        config: M8ExperimentConfig,
        bundle: Optional[M8DatasetBundle] = None,
        run_id: Optional[str] = None,
        code_revision: str = "unknown",
        cache_dir: Optional[Path] = None,
    ):
        self.config = config
        self.bundle = bundle
        self.run_id = run_id or f"run_{int(time.time())}"
        self.code_revision = code_revision
        self.cache_dir = cache_dir or Path(config.output.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run_dataset(
        self,
        bundle: M8DatasetBundle,
        condition_tag: str = "clean",
        splits: Optional[List[str]] = None,
        pgm_config: Optional[PGMParameterConfig] = None,
    ) -> List[ClaimEvaluationResult]:
        """
        Execute inference over all claims in the provided dataset bundle.
        """
        self.bundle = bundle
        return self.run_all(
            splits=splits,
            condition=condition_tag,
            pgm_config=pgm_config,
        )

    def run_image(
        self,
        image_id: str,
        claims: List[M7ClaimRecord],
        condition: str = "clean",
        pgm_config: Optional[PGMParameterConfig] = None,
        corruption_type: str = "none",
        corruption_severity: str = "clean",
    ) -> List[ClaimEvaluationResult]:
        """
        Execute standard and robust BP for all claims in a single image.
        """
        cfg = pgm_config or self.config.pgm
        ctx = build_image_pgm_context(image_id, claims, config=cfg)

        # Check cache if resume is enabled
        dataset_hash = self.bundle.dataset_hash if self.bundle else "adhoc"
        cache_key = compute_cache_key(image_id, condition, ctx.parameter_hash, dataset_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if self.config.resume and not self.config.force and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    return [ClaimEvaluationResult.from_dict(d) for d in cached_data]
            except Exception:
                pass  # Fall back to recomputation if cache corrupted

        # 1. Standard BP inference
        t0_std = time.perf_counter()
        std_bp_res: StandardBPResult = run_standard_bp(ctx.model)
        std_runtime_ms = (time.perf_counter() - t0_std) * 1000.0

        results: List[ClaimEvaluationResult] = []
        tau = cfg.decision_threshold

        for node_idx, cid in ctx.node_to_claim.items():
            claim_rec = next((c for c in claims if c.claim_id == cid), None)
            split_val = claim_rec.split if claim_rec else "train"
            gt_enum = self.bundle.get_ground_truth_for_claim(cid)
            gt_str = gt_enum.value if gt_enum is not None else None

            # Standard BP values
            p_std = float(std_bp_res.marginals[node_idx])
            std_pred = DecisionStatus.HALLUCINATED.value if p_std >= tau else DecisionStatus.SUPPORTED.value

            # 2. Robust BP inference
            t0_rob = time.perf_counter()
            rob_res: RobustBPResult = solve_robust_bp(
                model=ctx.model,
                target_node=node_idx,
                budget=ctx.budget,
                num_grid_steps=cfg.grid_steps,
                lipschitz_const=cfg.lipschitz_const,
            )
            rob_runtime_ms = (time.perf_counter() - t0_rob) * 1000.0

            l_i = float(rob_res.lower_grid)
            u_i = float(rob_res.upper_grid)
            w_i = float(u_i - l_i)
            mid_i = float(0.5 * (l_i + u_i))

            # Evidence sensitive condition: interval strictly brackets decision threshold
            ev_sensitive = bool(l_i <= tau <= u_i)

            if u_i < tau:
                rob_pred = DecisionStatus.SUPPORTED.value
                abstained = False
            elif l_i > tau:
                rob_pred = DecisionStatus.HALLUCINATED.value
                abstained = False
            else:
                rob_pred = DecisionStatus.ABSTAIN.value
                abstained = True

            ev = claim_rec.evidence if claim_rec else None
            det_score = ev.detector_score if (ev and ev.detector_available) else None
            clip_score = ev.clip_score if (ev and ev.similarity_available) else None

            row = ClaimEvaluationResult(
                run_id=self.run_id,
                experiment_name=self.config.experiment_id,
                condition=condition,
                image_id=image_id,
                claim_id=cid,
                object_category=claim_rec.object_category if claim_rec else "unknown",
                text_span=claim_rec.text_span if claim_rec else None,
                split=split_val,
                ground_truth=gt_str,
                detector_score=det_score,
                clip_score=clip_score,
                theta_i=float(ctx.model.theta[node_idx]),
                epsilon_i=float(ctx.model.epsilon[node_idx]),
                coupling_j=float(cfg.base_coupling_j),
                budget=float(ctx.budget),
                epsilon_scale=float(cfg.epsilon_scale),
                decision_threshold=float(tau),
                grid_steps=int(cfg.grid_steps),
                standard_posterior=p_std,
                standard_prediction=std_pred,
                robust_lower=l_i,
                robust_upper=u_i,
                robust_midpoint=mid_i,
                interval_width=w_i,
                certified_lower=float(rob_res.lower_certified),
                certified_upper=float(rob_res.upper_certified),
                field_cert_gap=float(rob_res.field_cert_gap),
                robust_prediction=rob_pred,
                evidence_sensitive=ev_sensitive,
                abstained=abstained,
                corruption_type=corruption_type,
                corruption_severity=corruption_severity,
                runtime_ms=float(rob_runtime_ms + (std_runtime_ms / ctx.model.num_nodes)),
                seed=self.config.seed,
                parameter_hash=ctx.parameter_hash,
                dataset_hash=self.bundle.dataset_hash,
                code_revision=self.code_revision,
                is_synthetic=self.bundle.is_synthetic,
                metadata={
                    "field_nominal": float(rob_res.field_nominal),
                    "nominal_marginal": float(rob_res.nominal_marginal),
                    "total_image_claims": len(claims),
                },
            )
            results.append(row)

        # Atomic cache write
        try:
            tmp_cache = cache_file.with_suffix(".tmp")
            with open(tmp_cache, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            tmp_cache.replace(cache_file)
        except Exception:
            pass

        return results

    def run_all(
        self,
        splits: Optional[List[str]] = None,
        condition: str = "clean",
        pgm_config: Optional[PGMParameterConfig] = None,
        corruption_type: str = "none",
        corruption_severity: str = "clean",
    ) -> List[ClaimEvaluationResult]:
        if splits is not None:
            if any(s.strip().lower() == "all" for s in splits):
                target_splits = None
            else:
                target_splits = [s.strip().lower() for s in splits]
        else:
            if any(s.strip().lower() == "all" for s in self.config.eval_splits):
                target_splits = None
            else:
                target_splits = [s.strip().lower() for s in self.config.eval_splits]

        # Group relevant claims by image
        all_eval_claims = self.bundle.get_eval_claims(target_splits)
        if not all_eval_claims:
            return []

        images_to_eval: Dict[str, List[M7ClaimRecord]] = {}
        for c in all_eval_claims:
            if c.image_id not in images_to_eval:
                images_to_eval[c.image_id] = []
            images_to_eval[c.image_id].append(c)

        all_results: List[ClaimEvaluationResult] = []
        for img_id, claims in sorted(images_to_eval.items()):
            img_res = self.run_image(
                image_id=img_id,
                claims=claims,
                condition=condition,
                pgm_config=pgm_config,
                corruption_type=corruption_type,
                corruption_severity=corruption_severity,
            )
            all_results.extend(img_res)

        return all_results
