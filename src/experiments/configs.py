"""
Typed experiment configurations and schema definitions for Milestone 8 (M8).
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json
import yaml


class ExecutionMode(str, Enum):
    """Execution status and safety mode."""
    AUTO = "auto"
    SMOKE = "smoke"
    DEVELOPMENT = "development"
    FINAL = "final"


class TreeTopology(str, Enum):
    """Coupling topology for multi-claim graphical models."""
    CHAIN = "chain"
    STAR = "star"
    INDEPENDENT = "independent"


@dataclass
class PGMParameterConfig:
    """
    Configuration for evidence mapping and PGM parameters.
    """
    unary_detector_weight: float = 1.0
    unary_clip_weight: float = 0.0
    theta_clip_max: float = 5.0
    base_coupling_j: float = 0.5493061443340549  # atanh(0.5)
    base_epsilon: float = 0.5493061443340549     # atanh(0.5)
    epsilon_scale: float = 1.0
    uncertainty_discrepancy_weight: float = 0.0
    min_epsilon: float = 0.01
    budget_ratio: float = 1.0  # B = budget_ratio * base_epsilon or ratio * sum_i eps_i
    fixed_budget: Optional[float] = None
    decision_threshold: float = 0.5
    grid_steps: int = 100
    lipschitz_const: float = 1.0
    topology: TreeTopology = TreeTopology.CHAIN

    def validate(self) -> None:
        """Validate parameter ranges."""
        if self.grid_steps < 2:
            raise ValueError(f"grid_steps must be >= 2, got {self.grid_steps}")
        if not (0.0 <= self.decision_threshold <= 1.0):
            raise ValueError(f"decision_threshold must be in [0, 1], got {self.decision_threshold}")
        if self.base_coupling_j < 0:
            raise ValueError(f"base_coupling_j must be >= 0, got {self.base_coupling_j}")

    def __post_init__(self) -> None:
        self.validate()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["topology"] = self.topology.value if hasattr(self.topology, "value") else str(self.topology)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PGMParameterConfig":
        data_copy = dict(data)
        if "topology" in data_copy and isinstance(data_copy["topology"], str):
            data_copy["topology"] = TreeTopology(data_copy["topology"])
        return cls(**data_copy)


@dataclass
class CorruptionConfig:
    """
    Configuration for visual corruption experiments.
    """
    enabled: bool = False
    corruption_types: List[str] = field(
        default_factory=lambda: [
            "gaussian_blur",
            "jpeg_compression",
            "additive_noise",
            "downsampling",
            "center_occlusion",
        ]
    )
    severities: List[str] = field(
        default_factory=lambda: ["clean", "light", "medium", "heavy"]
    )
    severity_params: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "gaussian_blur": {"light": 1.5, "medium": 3.0, "heavy": 6.0},
            "jpeg_compression": {"light": 70, "medium": 40, "heavy": 15},
            "additive_noise": {"light": 0.05, "medium": 0.15, "heavy": 0.30},
            "downsampling": {"light": 0.5, "medium": 0.25, "heavy": 0.125},
            "center_occlusion": {"light": 0.15, "medium": 0.30, "heavy": 0.50},
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorruptionConfig":
        return cls(**data)


@dataclass
class AblationConfig:
    """
    Configuration for systematic ablation experiments.
    """
    run_standard_bp: bool = True
    run_robust_bp: bool = True
    run_robust_b0: bool = True
    run_no_coupling: bool = True
    run_independent_box: bool = True
    run_evidence_point_baseline: bool = True
    run_budget_sweep: bool = True
    run_epsilon_sweep: bool = True
    run_grid_sweep: bool = True
    run_corruption_sweep: bool = False

    budget_sweep_multipliers: List[float] = field(
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    )
    epsilon_scale_values: List[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0]
    )
    grid_step_values: List[int] = field(
        default_factory=lambda: [10, 25, 50, 100, 200]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AblationConfig":
        return cls(**data)


@dataclass
class BootstrapConfig:
    """
    Configuration for non-parametric bootstrap statistical analysis.
    """
    enabled: bool = True
    n_resamples: int = 500
    confidence_level: float = 0.95
    resampling_unit: str = "image"  # Cluster bootstrap by image ID
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BootstrapConfig":
        return cls(**data)


@dataclass
class OutputConfig:
    """
    Output destinations and artifact formats.
    """
    output_dir: str = "reports/m8"
    cache_dir: str = "data/cache/m8"
    save_plots: bool = True
    save_tables: bool = True
    plot_formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    raw_results_filename: str = "raw_claim_results.jsonl"
    raw_results_csv: str = "raw_claim_results.csv"
    manifest_filename: str = "experiment_manifest.json"
    summary_md_filename: str = "summary.md"
    summary_json_filename: str = "summary.json"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        return cls(**data)


@dataclass
class M8ExperimentConfig:
    """
    Root configuration for M8 experiment execution.
    """
    experiment_id: str = "m8_robust_hallucination_evaluation"
    description: str = "Budget-Coupled Robust Belief Propagation Evaluation Framework"
    mode: ExecutionMode = ExecutionMode.AUTO
    eval_splits: List[str] = field(default_factory=lambda: ["test"])
    seed: int = 42
    
    # Dataset file paths
    manifest_path: str = "data/manifests/m7_image_manifest.json"
    claims_path: str = "data/exports/m7_claims.jsonl"
    ground_truth_path: str = "data/exports/m7_ground_truth.jsonl"
    evidence_path: str = "data/exports/claim_level_evidence.jsonl"
    image_base_dir: str = "data/real_images"
    dataset_lock_path: str = "data/manifests/m7_dataset_lock.json"

    # Execution controls
    resume: bool = True
    force: bool = False
    subset_size: Optional[int] = None
    
    # Component sub-configs
    pgm: PGMParameterConfig = field(default_factory=PGMParameterConfig)
    ablations: AblationConfig = field(default_factory=AblationConfig)
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "mode": self.mode.value if hasattr(self.mode, "value") else str(self.mode),
            "eval_splits": list(self.eval_splits),
            "seed": self.seed,
            "manifest_path": self.manifest_path,
            "claims_path": self.claims_path,
            "ground_truth_path": self.ground_truth_path,
            "evidence_path": self.evidence_path,
            "image_base_dir": self.image_base_dir,
            "dataset_lock_path": self.dataset_lock_path,
            "resume": self.resume,
            "force": self.force,
            "subset_size": self.subset_size,
            "pgm": self.pgm.to_dict(),
            "ablations": self.ablations.to_dict(),
            "corruption": self.corruption.to_dict(),
            "bootstrap": self.bootstrap.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "M8ExperimentConfig":
        data_copy = dict(data)
        if "mode" in data_copy and isinstance(data_copy["mode"], str):
            data_copy["mode"] = ExecutionMode(data_copy["mode"])
        if "pgm" in data_copy and isinstance(data_copy["pgm"], dict):
            data_copy["pgm"] = PGMParameterConfig.from_dict(data_copy["pgm"])
        if "ablations" in data_copy and isinstance(data_copy["ablations"], dict):
            data_copy["ablations"] = AblationConfig.from_dict(data_copy["ablations"])
        if "corruption" in data_copy and isinstance(data_copy["corruption"], dict):
            data_copy["corruption"] = CorruptionConfig.from_dict(data_copy["corruption"])
        if "bootstrap" in data_copy and isinstance(data_copy["bootstrap"], dict):
            data_copy["bootstrap"] = BootstrapConfig.from_dict(data_copy["bootstrap"])
        if "output" in data_copy and isinstance(data_copy["output"], dict):
            data_copy["output"] = OutputConfig.from_dict(data_copy["output"])
        return cls(**data_copy)

    def to_yaml(self, path: Union[str, Path]) -> Path:
        return save_m8_config(self, path)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "M8ExperimentConfig":
        return load_m8_config(path)


def load_m8_config(path_or_dict: Union[str, Path, Dict[str, Any]]) -> M8ExperimentConfig:
    """Load and validate an M8 configuration from YAML, JSON, or dictionary."""
    if isinstance(path_or_dict, dict):
        return M8ExperimentConfig.from_dict(path_or_dict)

    path = Path(path_or_dict)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in [".yaml", ".yml"]:
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    return M8ExperimentConfig.from_dict(raw or {})


def save_m8_config(config: M8ExperimentConfig, path: Union[str, Path]) -> Path:
    """Save an M8 configuration to YAML or JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_dict()

    if out_path.suffix.lower() in [".yaml", ".yml"]:
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return out_path


def get_default_m8_config() -> M8ExperimentConfig:
    """Return default production configuration."""
    return M8ExperimentConfig()


def get_smoke_m8_config() -> M8ExperimentConfig:
    """Return fast synthetic smoke test configuration."""
    return M8ExperimentConfig(
        experiment_id="m8_smoke_test",
        mode=ExecutionMode.SMOKE,
        eval_splits=["test", "train"],
        subset_size=10,
        pgm=PGMParameterConfig(grid_steps=20),
        ablations=AblationConfig(
            budget_sweep_multipliers=[0.0, 0.5, 1.0],
            epsilon_scale_values=[0.5, 1.0],
            grid_step_values=[10, 20],
            run_corruption_sweep=True,
        ),
        corruption=CorruptionConfig(
            enabled=True,
            severities=["clean", "light"],
            corruption_types=["gaussian_blur", "jpeg_compression"],
        ),
        bootstrap=BootstrapConfig(n_resamples=50),
        output=OutputConfig(
            output_dir="reports/m8_smoke",
            cache_dir="data/cache/m8_smoke",
            plot_formats=["png"],
        ),
    )
