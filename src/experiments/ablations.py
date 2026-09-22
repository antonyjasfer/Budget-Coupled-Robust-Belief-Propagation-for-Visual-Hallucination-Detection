"""
Ablation matrix execution and parameter sensitivity sweeps for M8.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import copy
import time

from src.experiments.configs import M8ExperimentConfig, PGMParameterConfig, TreeTopology
from src.experiments.loaders import M8DatasetBundle
from src.experiments.inference_runner import InferenceRunner, ClaimEvaluationResult
from src.experiments.corruption import create_corrupted_claim_record
from src.experiments.evaluation import evaluate_claim_results, ClassificationMetrics
from src.experiments.interval_metrics import compute_interval_statistics, RobustIntervalStatistics


@dataclass
class AblationRunSummary:
    """
    Summary metrics for a specific ablation condition.
    """
    ablation_name: str
    condition: str
    parameter_description: str
    classification_metrics: ClassificationMetrics
    interval_statistics: RobustIntervalStatistics
    raw_results_count: int
    mean_runtime_ms_per_claim: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ablation_name": self.ablation_name,
            "condition": self.condition,
            "parameter_description": self.parameter_description,
            "classification_metrics": self.classification_metrics.to_dict(),
            "interval_statistics": self.interval_statistics.to_dict(),
            "raw_results_count": self.raw_results_count,
            "mean_runtime_ms_per_claim": self.mean_runtime_ms_per_claim,
        }


class AblationMatrixRunner:
    """
    Orchestrates the 8 required ablation experiments.
    """
    def __init__(
        self,
        runner_or_config: Union[InferenceRunner, M8ExperimentConfig],
        config: Optional[M8ExperimentConfig] = None,
        bundle: Optional[M8DatasetBundle] = None,
    ):
        if isinstance(runner_or_config, InferenceRunner):
            self.runner = runner_or_config
            self.config = config or self.runner.config
            self.bundle = bundle or self.runner.bundle
        else:
            self.config = runner_or_config
            self.bundle = bundle
            self.runner = InferenceRunner(self.config, bundle=self.bundle)

    def run_all_ablations(
        self,
        bundle_or_splits: Optional[Union[M8DatasetBundle, List[str]]] = None,
    ) -> Dict[str, List[ClaimEvaluationResult]]:
        """
        Execute full ablation matrix and return mapping of ablation_name -> claim results.
        """
        if isinstance(bundle_or_splits, M8DatasetBundle):
            self.bundle = bundle_or_splits
            self.runner.bundle = bundle_or_splits
            eval_splits = ["all"]
        else:
            eval_splits = bundle_or_splits or self.config.eval_splits

        all_ablation_results: Dict[str, List[ClaimEvaluationResult]] = {}

        # 1. Main Proposed Method: Robust BP with global budget B
        if self.config.ablations.run_robust_bp:
            res_main = self.runner.run_all(
                splits=eval_splits,
                condition="robust_bp_proposed",
                pgm_config=self.config.pgm,
            )
            all_ablation_results["robust_bp_proposed"] = res_main

        # 2. Ablation: Standard BP (point posterior)
        if self.config.ablations.run_standard_bp:
            # We can re-use the standard BP fields computed during inference runner
            res_std = copy.deepcopy(all_ablation_results.get("robust_bp_proposed") or self.runner.run_all(splits=eval_splits, condition="standard_bp"))
            for r in res_std:
                r.condition = "standard_bp"
            all_ablation_results["standard_bp"] = res_std

        # 3. Ablation: Robust BP with B = 0 (recovers nominal)
        if self.config.ablations.run_robust_b0:
            cfg_b0 = copy.deepcopy(self.config.pgm)
            cfg_b0.fixed_budget = 0.0
            cfg_b0.budget_ratio = 0.0
            res_b0 = self.runner.run_all(
                splits=eval_splits,
                condition="robust_bp_b0",
                pgm_config=cfg_b0,
            )
            all_ablation_results["robust_bp_b0"] = res_b0

        # 4. Ablation: Robust BP with J = 0 (no graph coupling)
        if self.config.ablations.run_no_coupling:
            cfg_j0 = copy.deepcopy(self.config.pgm)
            cfg_j0.base_coupling_j = 0.0
            cfg_j0.topology = TreeTopology.INDEPENDENT
            res_j0 = self.runner.run_all(
                splits=eval_splits,
                condition="robust_bp_no_coupling",
                pgm_config=cfg_j0,
            )
            all_ablation_results["robust_bp_no_coupling"] = res_j0

        # 5. Ablation: Budget sweep B in {B1, B2, ...}
        if self.config.ablations.run_budget_sweep:
            for mult in self.config.ablations.budget_sweep_multipliers:
                cfg_b = copy.deepcopy(self.config.pgm)
                cfg_b.budget_ratio = mult
                cfg_b.fixed_budget = mult * self.config.pgm.base_epsilon
                cond_name = f"budget_sweep_b_{mult:.2f}"
                res_b = self.runner.run_all(
                    splits=eval_splits,
                    condition=cond_name,
                    pgm_config=cfg_b,
                )
                all_ablation_results[cond_name] = res_b

        # 6. Ablation: Epsilon scaling sweep alpha in {0.0, 0.5, 1.0, 1.5, 2.0}
        if self.config.ablations.run_epsilon_sweep:
            for alpha in self.config.ablations.epsilon_scale_values:
                cfg_eps = copy.deepcopy(self.config.pgm)
                cfg_eps.epsilon_scale = alpha
                cond_name = f"epsilon_sweep_alpha_{alpha:.2f}"
                res_eps = self.runner.run_all(
                    splits=eval_splits,
                    condition=cond_name,
                    pgm_config=cfg_eps,
                )
                all_ablation_results[cond_name] = res_eps

        # 7. Ablation: Discrete grid resolution sweep K in {10, 25, 50, 100, 200}
        if self.config.ablations.run_grid_sweep:
            for k_steps in self.config.ablations.grid_step_values:
                cfg_k = copy.deepcopy(self.config.pgm)
                cfg_k.grid_steps = k_steps
                cond_name = f"grid_sweep_k_{k_steps}"
                res_k = self.runner.run_all(
                    splits=eval_splits,
                    condition=cond_name,
                    pgm_config=cfg_k,
                )
                all_ablation_results[cond_name] = res_k

        # 8. Ablation: Clean vs Corrupted visual evidence
        if self.config.ablations.run_corruption_sweep or self.config.corruption.enabled:
            for c_type in self.config.corruption.corruption_types:
                for sev in self.config.corruption.severities:
                    if sev == "clean":
                        continue
                    cond_name = f"corrupt_{c_type}_{sev}"
                    # Prepare corrupted claims
                    target_claims = self.bundle.get_eval_claims(eval_splits)
                    corrupted_claims_by_img: Dict[str, List[Any]] = {}
                    for c in target_claims:
                        c_corrupt = create_corrupted_claim_record(
                            clean_claim=c,
                            corruption_type=c_type,
                            severity=sev,
                            seed=self.config.seed,
                        )
                        if c.image_id not in corrupted_claims_by_img:
                            corrupted_claims_by_img[c.image_id] = []
                        corrupted_claims_by_img[c.image_id].append(c_corrupt)

                    corrupt_results: List[ClaimEvaluationResult] = []
                    for img_id, c_list in sorted(corrupted_claims_by_img.items()):
                        c_res = self.runner.run_image(
                            image_id=img_id,
                            claims=c_list,
                            condition=cond_name,
                            pgm_config=self.config.pgm,
                            corruption_type=c_type,
                            corruption_severity=sev,
                        )
                        corrupt_results.extend(c_res)

                    all_ablation_results[cond_name] = corrupt_results

        return all_ablation_results
