"""
Phase 10B Human Annotation Interface CLI.

Provides ready-to-use operational commands for the upcoming Phase 10B:
  - export-tasks: Export blank masked tasks for Annotator A or B
  - import-labels: Import and validate completed annotation files
  - validate-claims: Verify claim ID consistency between annotators
  - compute-agreement: Calculate inter-annotator agreement (Cohen's Kappa / Raw)
  - create-adjudication-queue: Extract disagreements for expert adjudication

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase10b_interface")

VALID_LABELS = {"supported", "refuted"}


def export_tasks(annotator_id: str, output_path: Optional[Path] = None) -> Path:
    """Export blank task file for Annotator A or B."""
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
    """Validate structure and label values of an imported annotation file."""
    issues = []
    tasks = labels_data.get("tasks", [])
    if not tasks:
        issues.append("Imported file contains 0 tasks.")
        return False, issues

    for idx, t in enumerate(tasks):
        claim_id = t.get("claim_id")
        if not claim_id:
            issues.append(f"Task {idx} is missing 'claim_id'.")
        
        label = t.get("label")
        if label is None:
            issues.append(f"Task {idx} ({claim_id}) has null label. Must be 'supported' or 'refuted'.")
        elif label not in VALID_LABELS:
            issues.append(f"Task {idx} ({claim_id}) has invalid label '{label}'. Must be in {VALID_LABELS}.")

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
    """Compute raw agreement and Cohen's Kappa between Annotators A and B."""
    tasks_a = {t["claim_id"]: t.get("label") for t in data_a.get("tasks", []) if "claim_id" in t}
    tasks_b = {t["claim_id"]: t.get("label") for t in data_b.get("tasks", []) if "claim_id" in t}

    shared_ids = sorted(set(tasks_a.keys()) & set(tasks_b.keys()))
    if not shared_ids:
        return {"error": "No shared claim IDs between annotator packages."}

    agreements = 0
    disagreements = 0
    contingency = {"supp_supp": 0, "supp_ref": 0, "ref_supp": 0, "ref_ref": 0}

    for cid in shared_ids:
        la = tasks_a[cid]
        lb = tasks_b[cid]
        if la == lb:
            agreements += 1
        else:
            disagreements += 1

        if la == "supported" and lb == "supported":
            contingency["supp_supp"] += 1
        elif la == "supported" and lb == "refuted":
            contingency["supp_ref"] += 1
        elif la == "refuted" and lb == "supported":
            contingency["ref_supp"] += 1
        elif la == "refuted" and lb == "refuted":
            contingency["ref_ref"] += 1

    total = len(shared_ids)
    p_o = agreements / total if total > 0 else 0.0

    # Cohen's Kappa calculation
    # p_e = marginal chance agreement
    p_a_supp = (contingency["supp_supp"] + contingency["supp_ref"]) / total
    p_a_ref = 1.0 - p_a_supp
    p_b_supp = (contingency["supp_supp"] + contingency["ref_supp"]) / total
    p_b_ref = 1.0 - p_b_supp

    p_e = (p_a_supp * p_b_supp) + (p_a_ref * p_b_ref)
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) != 0 else 1.0

    return {
        "total_shared_claims": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "raw_agreement_rate": round(p_o, 4),
        "cohens_kappa": round(kappa, 4),
        "contingency_table": contingency,
    }


def create_adjudication_queue(data_a: Dict[str, Any], data_b: Dict[str, Any], output_path: Path) -> List[Dict[str, Any]]:
    """Extract claims where Annotator A and B disagree for senior adjudication."""
    tasks_a = {t["claim_id"]: t for t in data_a.get("tasks", []) if "claim_id" in t}
    tasks_b = {t["claim_id"]: t for t in data_b.get("tasks", []) if "claim_id" in t}

    shared_ids = sorted(set(tasks_a.keys()) & set(tasks_b.keys()))
    queue = []

    for cid in shared_ids:
        ta = tasks_a[cid]
        tb = tasks_b[cid]
        if ta.get("label") != tb.get("label"):
            queue.append({
                "claim_id": cid,
                "image_id": ta.get("image_id"),
                "claim_surface": ta.get("claim_surface"),
                "image_path": ta.get("image_path"),
                "annotator_A_label": ta.get("label"),
                "annotator_B_label": tb.get("label"),
                "adjudicated_label": None,
                "adjudicator_notes": None,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": "2.0.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "disagreement_count": len(queue),
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
            print("VALIDATION PASSED: All labels valid.")
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
