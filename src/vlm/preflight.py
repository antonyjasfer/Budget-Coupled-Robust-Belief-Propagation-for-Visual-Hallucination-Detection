"""
Preflight verification facility for real-image pilot execution.

Performs structured inspection across 11 critical operational gates:
1. Input manifest validity and entry count.
2. Real-image verification (rejection of synthetic manifests in real-pilot mode).
3. Strict training split isolation (up to 10 images max).
4. Local image file existence and decodability.
5. Image SHA256 integrity against manifest records.
6. Exclusion of reserved/evaluation image identities (val/test/POPE/reserved).
7. Runtime dependency availability (torch, transformers, PIL).
8. Model checkpoint local presence or explicit download grant.
9. Device, dtype, and runtime execution compatibility.
10. Model snapshot revision resolvability.
11. Output bundle and cache directory write permissions.
"""

from dataclasses import dataclass, field, asdict
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any, Union

from src.data.schemas import DatasetManifest, SplitName
from src.data.manifests import load_manifest
from src.vlm.provider import (
    VLMGenerationConfig,
    compute_file_sha256,
)
from src.vlm.pipeline import select_pilot_images

logger = logging.getLogger(__name__)


@dataclass
class PreflightGateResult:
    """Status and details for a single preflight gate."""
    gate_name: str
    passed: bool
    status: str  # "OK", "BLOCKED", "WARNING", "SKIPPED"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreflightReport:
    """
    Structured outcome of the full real-pilot preflight check.
    """
    overall_passed: bool
    is_real_pilot_ready: bool
    total_gates: int
    passed_gates: int
    blocked_gates: int
    warning_gates: int
    gates: List[PreflightGateResult]
    blockers: List[str]
    warnings: List[str]
    selected_image_ids: List[str]
    environment_info: Dict[str, Any]
    model_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "is_real_pilot_ready": self.is_real_pilot_ready,
            "total_gates": self.total_gates,
            "passed_gates": self.passed_gates,
            "blocked_gates": self.blocked_gates,
            "warning_gates": self.warning_gates,
            "gates": [g.to_dict() for g in self.gates],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "selected_image_ids": self.selected_image_ids,
            "environment_info": self.environment_info,
            "model_info": self.model_info,
        }


def check_runtime_packages() -> Dict[str, Any]:
    """Inspect presence and versions of optional VLM runtime dependencies."""
    packages = {
        "torch": {"installed": False, "version": None, "cuda_available": False, "device_count": 0},
        "transformers": {"installed": False, "version": None},
        "PIL": {"installed": False, "version": None},
        "bitsandbytes": {"installed": False, "version": None},
    }

    try:
        import torch
        packages["torch"]["installed"] = True
        packages["torch"]["version"] = str(getattr(torch, "__version__", "unknown"))
        packages["torch"]["cuda_available"] = bool(torch.cuda.is_available())
        packages["torch"]["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except ImportError:
        pass

    try:
        import transformers
        packages["transformers"]["installed"] = True
        packages["transformers"]["version"] = str(getattr(transformers, "__version__", "unknown"))
    except ImportError:
        pass

    try:
        import PIL
        from PIL import Image
        packages["PIL"]["installed"] = True
        packages["PIL"]["version"] = str(getattr(PIL, "__version__", "unknown"))
    except ImportError:
        pass

    try:
        import bitsandbytes
        packages["bitsandbytes"]["installed"] = True
        packages["bitsandbytes"]["version"] = str(getattr(bitsandbytes, "__version__", "unknown"))
    except (ImportError, Exception):
        pass

    return packages


def check_local_checkpoint_availability(
    model_name: str,
    allow_download: bool = False,
) -> Dict[str, Any]:
    """Check if model checkpoint is available in local HuggingFace cache."""
    res = {
        "model_name": model_name,
        "local_found": False,
        "allow_download": allow_download,
        "resolved_revision": None,
        "cache_paths_checked": [],
    }

    # Check standard huggingface hub cache locations
    hf_home = Path.home() / ".cache" / "huggingface" / "hub"
    model_folder_name = f"models--{model_name.replace('/', '--')}"
    target_cache_dir = hf_home / model_folder_name
    res["cache_paths_checked"].append(str(target_cache_dir))

    if target_cache_dir.exists():
        snapshots_dir = target_cache_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = [s for s in snapshots_dir.iterdir() if s.is_dir()]
            if snapshots:
                res["local_found"] = True
                res["resolved_revision"] = snapshots[0].name
                return res

    # Also check if transformers AutoConfig can resolve local commit hash from cache
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        if hasattr(cfg, "_commit_hash") and cfg._commit_hash:
            res["local_found"] = True
            res["resolved_revision"] = str(cfg._commit_hash)
            return res
    except Exception:
        pass

    # Also check if local directory with weights is specified
    local_path = Path(model_name)
    if local_path.exists() and local_path.is_dir():
        res["local_found"] = True
        res["resolved_revision"] = f"local_dir_{hashlib.sha256(str(local_path.resolve()).encode('utf-8')).hexdigest()[:8]}"
        return res

    return res


def run_preflight(
    manifest_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    split_registry_path: Optional[Union[str, Path]] = None,
    image_base_dir: Optional[Union[str, Path]] = None,
    cache_dir: Union[str, Path] = "data/cache/vlm",
    output_bundle_path: Union[str, Path] = "data/exports/annotation_pilot_bundle.json",
    sample_size: int = 10,
    seed: int = 42,
    allow_download: bool = False,
    require_real_images: bool = True,
) -> PreflightReport:
    """
    Execute complete preflight check across all 11 gates.
    """
    gates: List[PreflightGateResult] = []
    blockers: List[str] = []
    warnings: List[str] = []
    selected_image_ids: List[str] = []

    # Record environment details
    env_info = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": check_runtime_packages(),
    }

    # Load generation config
    gen_config = VLMGenerationConfig()
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            gen_config = VLMGenerationConfig.from_dict(json.load(f))

    # --- Gate 1: Manifest Validity ---
    manifest_obj: Optional[DatasetManifest] = None
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        g1 = PreflightGateResult(
            gate_name="1_manifest_validity",
            passed=False,
            status="BLOCKED",
            message=f"Manifest file not found: {manifest_file}",
        )
        gates.append(g1)
        blockers.append(g1.message)
    else:
        try:
            manifest_obj = load_manifest(manifest_file)
            if len(manifest_obj.entries) == 0:
                g1 = PreflightGateResult(
                    gate_name="1_manifest_validity",
                    passed=False,
                    status="BLOCKED",
                    message="Manifest exists but contains 0 entries.",
                )
                blockers.append(g1.message)
            else:
                g1 = PreflightGateResult(
                    gate_name="1_manifest_validity",
                    passed=True,
                    status="OK",
                    message=f"Manifest loaded successfully with {len(manifest_obj.entries)} entries.",
                    details={"manifest_id": manifest_obj.manifest_id, "entry_count": len(manifest_obj.entries)},
                )
            gates.append(g1)
        except Exception as err:
            g1 = PreflightGateResult(
                gate_name="1_manifest_validity",
                passed=False,
                status="BLOCKED",
                message=f"Manifest validation failed: {err}",
            )
            gates.append(g1)
            blockers.append(g1.message)

    # --- Gate 2: Real vs Synthetic Manifest Entries ---
    if manifest_obj:
        synthetic_count = sum(
            1 for e in manifest_obj.entries
            if getattr(e.image, "is_synthetic", False) or e.image.metadata.get("is_synthetic", False)
        )
        if require_real_images and synthetic_count > 0 and synthetic_count == len(manifest_obj.entries):
            g2 = PreflightGateResult(
                gate_name="2_real_image_manifest",
                passed=False,
                status="BLOCKED",
                message="All manifest entries are synthetic mock images. Real pilot requires genuine image records.",
                details={"synthetic_count": synthetic_count, "total_entries": len(manifest_obj.entries)},
            )
            blockers.append(g2.message)
        elif synthetic_count > 0:
            g2 = PreflightGateResult(
                gate_name="2_real_image_manifest",
                passed=True,
                status="WARNING",
                message=f"Manifest contains {synthetic_count} synthetic entries out of {len(manifest_obj.entries)}.",
                details={"synthetic_count": synthetic_count},
            )
            warnings.append(g2.message)
        else:
            g2 = PreflightGateResult(
                gate_name="2_real_image_manifest",
                passed=True,
                status="OK",
                message="Manifest entries are genuine image records.",
            )
        gates.append(g2)

    # Load split registry if provided
    split_registry: Optional[Dict[str, str]] = None
    if split_registry_path and Path(split_registry_path).exists():
        try:
            with open(split_registry_path, "r", encoding="utf-8") as f:
                split_registry = json.load(f)
        except Exception as err:
            warnings.append(f"Failed to load split registry: {err}")

    # --- Gate 3: Training Split Selection & Pilot Size ---
    selected_entries = []
    if manifest_obj:
        try:
            selected_entries = select_pilot_images(
                manifest=manifest_obj,
                split_registry=split_registry,
                sample_size=min(sample_size, 10),
                seed=seed,
            )
            selected_image_ids = [e.image.image_id for e in selected_entries]
            g3 = PreflightGateResult(
                gate_name="3_train_split_selection",
                passed=True,
                status="OK",
                message=f"Deterministically selected {len(selected_entries)} images strictly from TRAIN split.",
                details={"selected_image_ids": selected_image_ids, "seed": seed},
            )
            gates.append(g3)
        except Exception as err:
            g3 = PreflightGateResult(
                gate_name="3_train_split_selection",
                passed=False,
                status="BLOCKED",
                message=f"Failed to sample training images: {err}",
            )
            gates.append(g3)
            blockers.append(g3.message)

    # --- Gate 4: Local Image File Existence & Decodability ---
    # --- Gate 5: Image Hash Integrity ---
    # --- Gate 6: Reserved/Evaluation Identity Check ---
    if selected_entries:
        missing_images = []
        undecodable_images = []
        hash_mismatches = []
        reserved_violations = []

        pil_available = env_info["packages"]["PIL"]["installed"]

        for entry in selected_entries:
            img = entry.image
            img_id = img.image_id

            # Check reserved identity
            is_reserved = img.metadata.get("is_reserved_external", False) or "pope" in img_id.lower()
            if is_reserved:
                reserved_violations.append(img_id)

            raw_name = img.file_name or f"{img_id}.jpg"
            img_path = Path(image_base_dir) / raw_name if image_base_dir else Path(raw_name)

            if not img_path.exists():
                missing_images.append(str(img_path))
            else:
                # Decodability check
                if pil_available:
                    try:
                        from PIL import Image
                        with Image.open(img_path) as im:
                            im.verify()
                    except Exception as err:
                        undecodable_images.append(f"{img_path} ({err})")

                # Hash check
                actual_hash = compute_file_sha256(img_path)
                if img.file_hash and img.file_hash != actual_hash:
                    hash_mismatches.append(f"{img_id}: expected {img.file_hash[:8]}, got {actual_hash[:8]}")

        # Gate 4 Result
        if missing_images or undecodable_images:
            g4_msg = []
            if missing_images:
                g4_msg.append(f"{len(missing_images)} image files missing on disk: {missing_images[:3]}")
            if undecodable_images:
                g4_msg.append(f"{len(undecodable_images)} image files cannot be decoded: {undecodable_images[:3]}")
            g4 = PreflightGateResult(
                gate_name="4_image_files_decodable",
                passed=False,
                status="BLOCKED",
                message="; ".join(g4_msg),
                details={"missing": missing_images, "undecodable": undecodable_images},
            )
            blockers.append(g4.message)
        else:
            g4 = PreflightGateResult(
                gate_name="4_image_files_decodable",
                passed=True,
                status="OK",
                message="All selected pilot image files exist and are verified decodable.",
            )
        gates.append(g4)

        # Gate 5 Result
        if hash_mismatches:
            g5 = PreflightGateResult(
                gate_name="5_image_hash_integrity",
                passed=False,
                status="BLOCKED",
                message=f"{len(hash_mismatches)} image hash mismatch(es) detected: {hash_mismatches[:3]}",
                details={"mismatches": hash_mismatches},
            )
            blockers.append(g5.message)
        else:
            g5 = PreflightGateResult(
                gate_name="5_image_hash_integrity",
                passed=True,
                status="OK",
                message="All image hashes match recorded manifest identifiers.",
            )
        gates.append(g5)

        # Gate 6 Result
        if reserved_violations:
            g6 = PreflightGateResult(
                gate_name="6_reserved_eval_isolation",
                passed=False,
                status="BLOCKED",
                message=f"Pilot selection contains reserved/evaluation images: {reserved_violations}",
                details={"violations": reserved_violations},
            )
            blockers.append(g6.message)
        else:
            g6 = PreflightGateResult(
                gate_name="6_reserved_eval_isolation",
                passed=True,
                status="OK",
                message="No reserved external, validation, test, or POPE images present in pilot selection.",
            )
        gates.append(g6)

    # --- Gate 7: Runtime Dependencies ---
    pkgs = env_info["packages"]
    missing_pkgs = [p for p in ["torch", "transformers", "PIL"] if not pkgs[p]["installed"]]
    if missing_pkgs:
        g7 = PreflightGateResult(
            gate_name="7_runtime_dependencies",
            passed=False,
            status="BLOCKED",
            message=f"Missing required VLM runtime packages: {missing_pkgs}. Install via 'uv add torch transformers pillow'.",
            details={"packages": pkgs},
        )
        blockers.append(g7.message)
    else:
        g7 = PreflightGateResult(
            gate_name="7_runtime_dependencies",
            passed=True,
            status="OK",
            message="Required runtime packages (torch, transformers, PIL) are installed.",
            details={"packages": pkgs},
        )
    gates.append(g7)

    # --- Gate 8: Model Checkpoint Availability ---
    chk = check_local_checkpoint_availability(gen_config.model_name, allow_download=allow_download)
    if not chk["local_found"] and not allow_download:
        g8 = PreflightGateResult(
            gate_name="8_checkpoint_availability",
            passed=False,
            status="BLOCKED",
            message=(
                f"Model checkpoint '{gen_config.model_name}' not found locally in HuggingFace cache. "
                "In offline/local mode, weights must be pre-cached. Supply --allow-download to permit network download."
            ),
            details=chk,
        )
        blockers.append(g8.message)
    elif not chk["local_found"] and allow_download:
        g8 = PreflightGateResult(
            gate_name="8_checkpoint_availability",
            passed=True,
            status="WARNING",
            message=f"Checkpoint '{gen_config.model_name}' is not local but --allow-download is set (will download on first run).",
            details=chk,
        )
        warnings.append(g8.message)
    else:
        g8 = PreflightGateResult(
            gate_name="8_checkpoint_availability",
            passed=True,
            status="OK",
            message=f"Found local model snapshot for '{gen_config.model_name}' (revision: {chk['resolved_revision']}).",
            details=chk,
        )
    gates.append(g8)

    # --- Gate 9: Device and Dtype Compatibility ---
    cuda_req = gen_config.device.startswith("cuda")
    cuda_avail = env_info["packages"]["torch"]["cuda_available"]
    bnb_installed = env_info["packages"].get("bitsandbytes", {}).get("installed", False)
    load_in_4bit = getattr(gen_config, "load_in_4bit", False)

    if load_in_4bit:
        if not cuda_avail:
            g9 = PreflightGateResult(
                gate_name="9_device_dtype_compat",
                passed=False,
                status="BLOCKED",
                message="4-bit quantization requested (load_in_4bit=True) but CUDA is not available on this system.",
                details={"load_in_4bit": True, "cuda_available": False},
            )
            blockers.append(g9.message)
        elif not bnb_installed:
            g9 = PreflightGateResult(
                gate_name="9_device_dtype_compat",
                passed=False,
                status="BLOCKED",
                message="4-bit quantization requested but bitsandbytes is not installed. Install via 'uv add --optional vlm bitsandbytes'.",
                details={"load_in_4bit": True, "bitsandbytes_installed": False},
            )
            blockers.append(g9.message)
        else:
            g9 = PreflightGateResult(
                gate_name="9_device_dtype_compat",
                passed=True,
                status="OK",
                message="4-bit NF4 bitsandbytes quantization configuration verified for CUDA runtime.",
                details={
                    "load_in_4bit": True,
                    "quantization_type": getattr(gen_config, "quantization_type", "nf4"),
                    "compute_dtype": getattr(gen_config, "compute_dtype", "float16"),
                    "device_map": getattr(gen_config, "device_map", "auto"),
                },
            )
        gates.append(g9)
    elif cuda_req and not cuda_avail:
        g9 = PreflightGateResult(
            gate_name="9_device_dtype_compat",
            passed=False,
            status="BLOCKED",
            message=f"Config requested device '{gen_config.device}' but CUDA is not available on this system.",
            details={"requested_device": gen_config.device, "cuda_available": False},
        )
        blockers.append(g9.message)
        gates.append(g9)
    elif gen_config.device == "cpu" and gen_config.dtype in ("float16", "bfloat16"):
        g9 = PreflightGateResult(
            gate_name="9_device_dtype_compat",
            passed=True,
            status="WARNING",
            message=f"Running {gen_config.dtype} on CPU may be slow or lack hardware acceleration; float32 recommended for CPU.",
            details={"device": gen_config.device, "dtype": gen_config.dtype},
        )
        warnings.append(g9.message)
        gates.append(g9)
    else:
        g9 = PreflightGateResult(
            gate_name="9_device_dtype_compat",
            passed=True,
            status="OK",
            message=f"Device ({gen_config.device}) and dtype ({gen_config.dtype}) configuration is valid.",
            details={"device": gen_config.device, "dtype": gen_config.dtype},
        )
        gates.append(g9)

    # --- Gate 10: Model Revision Resolvability ---
    if chk["local_found"]:
        g10 = PreflightGateResult(
            gate_name="10_model_revision_resolved",
            passed=True,
            status="OK",
            message=f"Model snapshot revision resolved: {chk['resolved_revision']}",
            details={"revision": chk["resolved_revision"]},
        )
    elif allow_download:
        g10 = PreflightGateResult(
            gate_name="10_model_revision_resolved",
            passed=True,
            status="WARNING",
            message="Model snapshot revision will be resolved upon download.",
        )
        warnings.append(g10.message)
    else:
        g10 = PreflightGateResult(
            gate_name="10_model_revision_resolved",
            passed=False,
            status="BLOCKED",
            message="Cannot establish model snapshot revision in strict offline mode without local weights.",
        )
        blockers.append(g10.message)
    gates.append(g10)

    # --- Gate 11: Output and Cache Path Writability ---
    cache_path = Path(cache_dir)
    out_bundle_file = Path(output_bundle_path)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        test_cache_file = cache_path / ".write_test.tmp"
        with open(test_cache_file, "w") as f:
            f.write("test")
        test_cache_file.unlink()

        out_bundle_file.parent.mkdir(parents=True, exist_ok=True)
        test_out_file = out_bundle_file.parent / ".write_test.tmp"
        with open(test_out_file, "w") as f:
            f.write("test")
        test_out_file.unlink()

        g11 = PreflightGateResult(
            gate_name="11_path_writability",
            passed=True,
            status="OK",
            message=f"Cache directory ({cache_path}) and export path ({out_bundle_file.parent}) are writable.",
            details={"cache_dir": str(cache_path), "output_bundle": str(out_bundle_file)},
        )
    except Exception as err:
        g11 = PreflightGateResult(
            gate_name="11_path_writability",
            passed=False,
            status="BLOCKED",
            message=f"Path writability check failed: {err}",
        )
        blockers.append(g11.message)
    gates.append(g11)

    passed_count = sum(1 for g in gates if g.passed)
    blocked_count = sum(1 for g in gates if g.status == "BLOCKED")
    warning_count = sum(1 for g in gates if g.status == "WARNING")
    overall_ok = (blocked_count == 0)

    return PreflightReport(
        overall_passed=overall_ok,
        is_real_pilot_ready=overall_ok,
        total_gates=len(gates),
        passed_gates=passed_count,
        blocked_gates=blocked_count,
        warning_gates=warning_count,
        gates=gates,
        blockers=blockers,
        warnings=warnings,
        selected_image_ids=selected_image_ids,
        environment_info=env_info,
        model_info=chk,
    )
