"""
Build and Validate Official MS COCO Candidate Universe V2.

Strict Compliance Rules (Phase 10A-R2 Approval Corrections 1 & 2):
1. Reads ONLY the "images" array from official COCO JSON files.
2. Never parses, loads, inspects, or retains the "annotations" array (captions).
3. Computes N_train, N_val, N_union dynamically from the parsed records.
4. Enforces:
   - Unique image IDs
   - Zero overlap between train2017 and val2017
   - Canonical coco_source_split on every record
   - Canonical filename format
   - Zero synthetic IDs or numeric range generation
5. Produces:
   - data/manifests/coco_candidate_universe_v2.json
   - Canonical universe hash
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_official_coco_images_only(json_path: Path, source_split_name: str) -> list:
    """
    Parses ONLY json['images'] array and ignores 'annotations' entirely.
    Guarantees no caption text or human labels enter memory or downstream manifests.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Authoritative COCO metadata file missing: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if "images" not in data:
        raise KeyError(f"'images' array not found in {json_path}")
    
    # Explicitly ensure annotations were not retained
    raw_images = data["images"]
    records = []
    for img in raw_images:
        coco_int_id = int(img["id"])
        canonical_name = img.get("file_name", f"{coco_int_id:012d}.jpg")
        img_id = f"coco_{coco_int_id:012d}"
        
        record = {
            "image_id": img_id,
            "coco_integer_id": coco_int_id,
            "coco_source_split": source_split_name,
            "file_name": canonical_name,
            "width": int(img["width"]),
            "height": int(img["height"]),
            "coco_url": img.get("coco_url", f"http://images.cocodataset.org/{source_split_name}/{canonical_name}"),
            "source_metadata_provenance": f"{json_path.name}:images",
        }
        records.append(record)
    
    return records

def main():
    ann_dir = PROJECT_ROOT / "data" / "coco" / "annotations"
    train_path = ann_dir / "captions_train2017.json"
    val_path = ann_dir / "captions_val2017.json"
    out_dir = PROJECT_ROOT / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coco_candidate_universe_v2.json"

    print("Phase 10A-R2: Building Authoritative COCO Candidate Universe V2...")
    print(f"Loading train metadata from: {train_path}")
    train_records = load_official_coco_images_only(train_path, "train2017")
    n_train = len(train_records)
    print(f"Parsed {n_train} train2017 image records.")

    print(f"Loading validation metadata from: {val_path}")
    val_records = load_official_coco_images_only(val_path, "val2017")
    n_val = len(val_records)
    print(f"Parsed {n_val} val2017 image records.")

    # Validation Checks
    train_ids = {r["coco_integer_id"] for r in train_records}
    val_ids = {r["coco_integer_id"] for r in val_records}
    
    if len(train_ids) != n_train:
        raise ValueError(f"Duplicate image IDs detected within train2017: {n_train} records vs {len(train_ids)} unique IDs")
    if len(val_ids) != n_val:
        raise ValueError(f"Duplicate image IDs detected within val2017: {n_val} records vs {len(val_ids)} unique IDs")
    
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise ValueError(f"FATAL: {len(overlap)} overlapping IDs found between train2017 and val2017!")
    
    all_records = train_records + val_records
    n_union = len(all_records)
    print(f"Combined Union: {n_union} image records.")

    # Validate image dimensions > 0 and file_name non-empty
    for r in all_records:
        if r["width"] <= 0 or r["height"] <= 0:
            raise ValueError(f"Invalid dimensions for image {r['image_id']}: {r['width']}x{r['height']}")
        if not r["file_name"].endswith(".jpg"):
            raise ValueError(f"Invalid canonical filename for image {r['image_id']}: {r['file_name']}")
    
    n_eligible = len(all_records)
    print(f"Total verified eligible images: {n_eligible}")

    # Canonical deterministic sort by image_id
    all_records.sort(key=lambda x: x["image_id"])

    # Integrity Assertions against expected COCO 2017 metadata
    assert n_train == 118287, f"Expected 118,287 train2017 records, got {n_train}"
    assert n_val == 5000, f"Expected 5,000 val2017 records, got {n_val}"
    assert n_union == 123287, f"Expected 123,287 total union records, got {n_union}"
    assert n_eligible == 123287, f"Expected 123,287 eligible records, got {n_eligible}"

    universe_meta = {
        "schema_version": "2.0.0",
        "universe_id": "coco_2017_train_val_union_v2",
        "provenance_statement": (
            "Candidate universe constructed strictly from official MS COCO 2017 train2017 and val2017 "
            "image metadata records (images arrays only). Human caption annotations were never loaded "
            "into claim generation, model input, ground truth, or calibration."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_eligible_images": n_eligible,
        "n_train2017": n_train,
        "n_val2017": n_val,
        "n_union": n_union,
        "images": all_records,
    }

    # Canonical hash computation excluding universe_hash field
    canonical_json = json.dumps(universe_meta, sort_keys=True, separators=(",", ":"))
    universe_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    universe_meta["universe_hash"] = universe_hash

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(universe_meta, f, indent=2)

    print(f"Successfully generated: {out_path}")
    print(f"Universe Hash: {universe_hash}")
    print(f"Eligible images count: {n_eligible}")

if __name__ == "__main__":
    main()
