"""
Reproducibility manifest generation and environment provenance tracking for M8.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any

from src.experiments.configs import M8ExperimentConfig
from src.experiments.loaders import M8DatasetBundle


@dataclass
class HardwareInfo:
    cpu_count: int
    processor: str
    system: str
    machine: str
    cuda_available: bool
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentManifest:
    """
    Complete reproducibility manifest for an M8 experiment run.
    """
    experiment_id: str
    run_id: str
    execution_mode: str  # "SMOKE", "DEVELOPMENT", "FINAL"
    status: str  # "COMPLETED", "INCOMPLETE", "FAILED"
    git_commit: str
    git_dirty: bool
    python_version: str
    platform_info: str
    dependency_versions: Dict[str, str]
    seed: int
    dataset_hash: str
    evidence_hash: str
    is_dataset_locked: bool
    splits_evaluated: List[str]
    total_images_evaluated: int
    total_claims_evaluated: int
    annotated_claims_evaluated: int
    start_timestamp: str
    end_timestamp: str
    elapsed_seconds: float
    hardware_info: HardwareInfo
    config: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["hardware_info"] = self.hardware_info.to_dict()
        return d


def get_git_info() -> Tuple[str, bool]:
    """Inspect current git commit SHA and working tree status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit = "unknown_commit"

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status_output) > 0
    except Exception:
        is_dirty = False

    return commit, is_dirty


def get_dependency_versions() -> Dict[str, str]:
    """Retrieve installed versions of critical runtime packages."""
    deps: Dict[str, str] = {}
    packages = ["numpy", "scipy", "sklearn", "PIL", "yaml", "torch", "transformers", "pytest"]
    for pkg in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            deps[pkg] = str(ver)
        except ImportError:
            deps[pkg] = "not_installed"
    return deps


def get_hardware_info() -> HardwareInfo:
    """Inspect CPU and CUDA hardware properties."""
    import os
    cpu_count = os.cpu_count() or 1
    processor = platform.processor() or "unknown"
    sys_name = platform.system()
    machine = platform.machine()

    cuda_avail = False
    cuda_count = 0
    cuda_name = None

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            cuda_count = torch.cuda.device_count()
            cuda_name = torch.cuda.get_device_name(0) if cuda_count > 0 else None
    except Exception:
        pass

    return HardwareInfo(
        cpu_count=cpu_count,
        processor=processor,
        system=sys_name,
        machine=machine,
        cuda_available=cuda_avail,
        cuda_device_count=cuda_count,
        cuda_device_name=cuda_name,
    )


def create_experiment_manifest(
    config: M8ExperimentConfig,
    bundle: M8DatasetBundle,
    run_id: str,
    start_time: float,
    end_time: float,
    splits_evaluated: List[str],
    total_images: int,
    total_claims: int,
    annotated_claims: int,
    status: str = "COMPLETED",
) -> ExperimentManifest:
    """
    Construct a complete experiment manifest.
    """
    git_sha, git_dirty = get_git_info()
    deps = get_dependency_versions()
    hw = get_hardware_info()

    # Determine execution mode label
    if bundle.is_synthetic:
        exec_mode = "SMOKE"
    elif bundle.is_locked and bundle.metadata.get("total_images", 0) >= 600:
        exec_mode = "FINAL"
    else:
        exec_mode = "DEVELOPMENT"

    start_iso = datetime.fromtimestamp(start_time, timezone.utc).isoformat()
    end_iso = datetime.fromtimestamp(end_time, timezone.utc).isoformat()
    elapsed = float(end_time - start_time)

    return ExperimentManifest(
        experiment_id=config.experiment_id,
        run_id=run_id,
        execution_mode=exec_mode,
        status=status,
        git_commit=git_sha,
        git_dirty=git_dirty,
        python_version=sys.version.split()[0],
        platform_info=platform.platform(),
        dependency_versions=deps,
        seed=config.seed,
        dataset_hash=bundle.dataset_hash,
        evidence_hash=bundle.evidence_hash,
        is_dataset_locked=bundle.is_locked,
        splits_evaluated=splits_evaluated,
        total_images_evaluated=total_images,
        total_claims_evaluated=total_claims,
        annotated_claims_evaluated=annotated_claims,
        start_timestamp=start_iso,
        end_timestamp=end_iso,
        elapsed_seconds=elapsed,
        hardware_info=hw,
        config=config.to_dict(),
        metadata={
            "execution_state": bundle.execution_state,
            "target_benchmark_size": 600,
        },
    )


def save_experiment_manifest(manifest: ExperimentManifest, path: Path) -> Path:
    """Save experiment manifest to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return path
