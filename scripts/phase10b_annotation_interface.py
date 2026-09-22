"""
Phase 10B Human Annotation Interface CLI.

Provides ready-to-use operational commands for Phase 10B:
  - export-tasks: Export blank masked tasks for Annotator A or B (label=null)
  - import-labels: Import and validate completed annotation files (supported/hallucinated/unknown)
  - validate-claims: Verify claim ID consistency between annotators
  - compute-agreement: Calculate multiclass inter-annotator agreement (Cohen's Kappa / 3x3 matrix)
  - create-adjudication-queue: Extract disagreements for expert adjudication

Canonical ground truth vocabulary from src.data.schemas.GroundTruthStatus:
  - supported
  - hallucinated
  - unknown

NOTE: This script does NOT generate synthetic labels or simulate annotators.
It is an operational protocol interface for real human annotators.
"""

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import GroundTruthStatus
from src.annotation.agreement import compute_cohens_kappa

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase10b_interface")

CANONICAL_LABELS = {status.value for status in GroundTruthStatus}
REJECTED_LABELS_HELP = ["refuted", "abstain", "yes", "no", "0", "1", "null"]


def export_tasks(annotator_id: str, output_path: Optional[Path] = None) -> Path:
    """Export blank task file for Annotator A or B (labels masked as null)."""
    annotator_id = annotator_id.upper()
    if annotator_id not in ("A", "B"):
        raise ValueError(f"Annotator ID must be 'A' or 'B', got '{annotator_id}'")

    source_path = PROJECT_ROOT / "data" / "annotations" / f"annotator_{annotator_id}_tasks_v2.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Source task package not found at {source_path}")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "annotations" / f"export_annotator_{annotator_id}_blank.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported {data.get('tasks_count', 0)} blank tasks to {output_path}")
    return output_path


def validate_imported_labels(labels_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate structure and label values of a completed imported annotation file.
    
    Permitted labels are strictly: 'supported', 'hallucinated', 'unknown'.
    Rejects 'refuted', 'abstain', 'yes', 'no', numeric labels, and nulls.
    """
    issues = []
    tasks = labels_data.get("tasks", [])
    if not tasks:
        issues.append("Imported file contains 0 tasks.")
        return False, issues

    canonical_list = sorted(list(CANONICAL_LABELS))

    for idx, t in enumerate(tasks):
        claim_id = t.get("claim_id")
        if not claim_id:
            issues.append(f"Task {idx} is missing 'claim_id'.")
        
        label = t.get("label")
        if label is None:
            issues.append(
                f"Task {idx} ({claim_id}) has null label. "
                f"Completed annotation imports require a non-null label from {canonical_list}."
            )
        elif not isinstance(label, str) or label.strip().lower() not in CANONICAL_LABELS:
            issues.append(
                f"Task {idx} ({claim_id}) has invalid label '{label}'. "
                f"Canonical labels are strictly {canonical_list}. "
                f"Labels such as 'refuted', 'abstain', 'yes'/'no', or numbers are rejected."
            )

    return len(issues) == 0, issues


def validate_claims_match(data_a: Dict[str, Any], data_b: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Verify that both annotator packages have identical claim sets."""
    issues = []
    tasks_a = {t["claim_id"]: t for t in data_a.get("tasks", []) if "claim_id" in t}
    tasks_b = {t["claim_id"]: t for t in data_b.get("tasks", []) if "claim_id" in t}

    if len(tasks_a) != len(tasks_b):
        issues.append(f"Task count mismatch: Annotator A has {len(tasks_a)}, B has {len(tasks_b)}")

    diff_ab = set(tasks_a.keys()) - set(tasks_b.keys())
    if diff_ab:
        issues.append(f"Claims present in A but missing in B: {len(diff_ab)} (e.g., {list(diff_ab)[:3]})")

    diff_ba = set(tasks_b.keys()) - set(tasks_a.keys())
    if diff_ba:
        issues.append(f"Claims present in B but missing in A: {len(diff_ba)} (e.g., {list(diff_ba)[:3]})")

    return len(issues) == 0, issues


def compute_inter_annotator_agreement(data_a: Dict[str, Any], data_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute multiclass inter-annotator agreement and Cohen's Kappa using
    canonical project agreement infrastructure across SUPPORTED, HALLUCINATED, UNKNOWN.
    
    Includes 3x3 confusion matrix and handles Pe == 1 boundary (cohens_kappa = None).
    """
    tasks_a = {t["claim_id"]: t.get("label") for t in data_a.get("tasks", []) if "claim_id" in t}
    tasks_b = {t["claim_id"]: t.get("label") for t in data_b.get("tasks", []) if "claim_id" in t}

    shared_ids = sorted(set(tasks_a.keys()) & set(tasks_b.keys()))
    if not shared_ids:
        return {"error": "No shared claim IDs between annotator packages."}

    statuses_a = []
    statuses_b = []
    invalid_labels = []

    for cid in shared_ids:
        la = tasks_a[cid]
        lb = tasks_b[cid]

        try:
            sa = GroundTruthStatus(str(la).strip().lower())
        except ValueError:
            invalid_labels.append(f"Annotator A claim {cid} has non-canonical label '{la}'")
            continue

        try:
            sb = GroundTruthStatus(str(lb).strip().lower())
        except ValueError:
            invalid_labels.append(f"Annotator B claim {cid} has non-canonical label '{lb}'")
            continue

        statuses_a.append(sa)
        statuses_b.append(sb)

    if invalid_labels:
        return {
            "error": "Non-canonical labels encountered during agreement computation.",
            "issues": invalid_labels[:10],
        }

    # Delegate to canonical multiclass agreement module
    kappa_result = compute_cohens_kappa(statuses_a, statuses_b)

    return {
        "total_claims": kappa_result.joint_count,
        "agreement_count": kappa_result.agreement_count,
        "disagreement_count": kappa_result.joint_count - kappa_result.agreement_count,
        "raw_agreement": round(kappa_result.observed_agreement, 4),
        "chance_agreement": round(kappa_result.chance_agreement, 4),
        "cohens_kappa": round(kappa_result.cohens_kappa, 4) if kappa_result.cohens_kappa is not None else None,
        "kappa_defined": kappa_result.kappa_defined,
        "status_message": kappa_result.status_message,
        "confusion_matrix": kappa_result.confusion_matrix,
        "marginals_a": kappa_result.marginals_a,
        "marginals_b": kappa_result.marginals_b,
    }


def create_adjudication_queue(data_a: Dict[str, Any], data_b: Dict[str, Any], output_path: Path) -> List[Dict[str, Any]]:
    """
    Extract claims where Annotator A and B disagree for expert adjudication.
    
    Preserves original A and B labels permanently.
    Adjudicated label is initialized to null; senior adjudicator may assign
    SUPPORTED, HALLUCINATED, or UNKNOWN.
    """
    tasks_a = {t["claim_id"]: t for t in data_a.get("tasks", []) if "claim_id" in t}
    tasks_b = {t["claim_id"]: t for t in data_b.get("tasks", []) if "claim_id" in t}

    shared_ids = sorted(set(tasks_a.keys()) & set(tasks_b.keys()))
    queue = []

    for cid in shared_ids:
        ta = tasks_a[cid]
        tb = tasks_b[cid]
        la = ta.get("label")
        lb = tb.get("label")
        if la != lb:
            queue.append({
                "claim_id": cid,
                "image_id": ta.get("image_id"),
                "claim_surface": ta.get("claim_surface"),
                "image_path": ta.get("image_path"),
                "annotator_A_label": la,
                "annotator_B_label": lb,
                "adjudicated_label": None,
                "adjudicator_notes": None,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "2.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disagreement_count": len(queue),
            "allowed_adjudicated_labels": sorted(list(CANONICAL_LABELS)),
            "adjudication_queue": queue,
        }, f, indent=2)

    logger.info(f"Saved adjudication queue ({len(queue)} items) to {output_path}")
    return queue


def main():
    parser = argparse.ArgumentParser(description="Phase 10B Annotation Protocol CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # export-tasks
    p_exp = subparsers.add_parser("export-tasks", help="Export blank tasks for Annotator A or B")
    p_exp.add_argument("--annotator", required=True, choices=["A", "B", "a", "b"])
    p_exp.add_argument("--output", type=str, default=None)

    # import-labels
    p_imp = subparsers.add_parser("import-labels", help="Validate and import completed annotations")
    p_imp.add_argument("--input", required=True, type=str)

    # validate-claims
    p_val = subparsers.add_parser("validate-claims", help="Validate matching claims between A and B")
    p_val.add_argument("--labels-a", required=True, type=str)
    p_val.add_argument("--labels-b", required=True, type=str)

    # compute-agreement
    p_agr = subparsers.add_parser("compute-agreement", help="Compute Cohen's Kappa and raw agreement")
    p_agr.add_argument("--labels-a", required=True, type=str)
    p_agr.add_argument("--labels-b", required=True, type=str)

    # create-adjudication-queue
    p_adj = subparsers.add_parser("create-adjudication-queue", help="Generate queue of disagreements")
    p_adj.add_argument("--labels-a", required=True, type=str)
    p_adj.add_argument("--labels-b", required=True, type=str)
    p_adj.add_argument("--output", required=True, type=str)

    args = parser.parse_args()

    if args.command == "export-tasks":
        out = Path(args.output) if args.output else None
        export_tasks(args.annotator, out)
    elif args.command == "import-labels":
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid, issues = validate_imported_labels(data)
        if valid:
            print("VALIDATION PASSED: All labels valid and canonical.")
            sys.exit(0)
        else:
            print(f"VALIDATION FAILED: {len(issues)} issues found:")
            for iss in issues[:10]:
                print(f"  [!] {iss}")
            sys.exit(1)
    elif args.command in ("validate-claims", "compute-agreement", "create-adjudication-queue"):
        with open(args.labels_a, "r", encoding="utf-8") as f:
            da = json.load(f)
        with open(args.labels_b, "r", encoding="utf-8") as f:
            db = json.load(f)

        if args.command == "validate-claims":
            v, iss = validate_claims_match(da, db)
            if v:
                print("VALIDATION PASSED: Identical claim sets.")
                sys.exit(0)
            else:
                print("VALIDATION FAILED:")
                for i in iss:
                    print(f"  [!] {i}")
                sys.exit(1)
        elif args.command == "compute-agreement":
            res = compute_inter_annotator_agreement(da, db)
            print(json.dumps(res, indent=2))
        elif args.command == "create-adjudication-queue":
            create_adjudication_queue(da, db, Path(args.output))


if __name__ == "__main__":
    main()
