"""
Phase 10A-R2-5: Source Image Acquisition Gate.

Downloads and cryptographically verifies all 600 primary representative cohort images.
Strict Compliance Rules:
1. Enforces requested=600, valid=600, missing=0, corrupt=0.
2. Validates:
   - PIL image decoding
   - Exact match with COCO metadata width and height
   - Canonical filename
   - Exact SHA-256 recording
3. Outputs:
   - reports/m10a_recovery2/source_image_audit_v2.json
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import urllib.request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("source_acquisition_v2")

def download_image(url: str, dest_path: Path, max_retries: int = 3, timeout: int = 30) -> bool:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "MthsProject-Research-Acquisition/2.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content = response.read()
            # Write to temp file then atomic rename
            tmp_path = dest_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                f.write(content)
            tmp_path.replace(dest_path)
            return True
        except Exception as e:
            time.sleep(1.0 * (attempt + 1))
            if attempt == max_retries - 1:
                logger.error(f"Failed to download {url} after {max_retries} attempts: {e}")
                return False
    return False

def verify_image(dest_path: Path, expected_w: int, expected_h: int) -> tuple:
    """Verifies PIL decode, dimensions, and computes SHA-256."""
    if not dest_path.exists():
        return False, "MISSING", 0, 0, ""
    try:
        with open(dest_path, "rb") as f:
            data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()
        
        with Image.open(dest_path) as img:
            img.verify()
        
        # Re-open for size check because verify closes stream in PIL
        with Image.open(dest_path) as img:
            w, h = img.size
        
        if w != expected_w or h != expected_h:
            return False, f"DIMENSION_MISMATCH (got {w}x{h}, expected {expected_w}x{expected_h})", w, h, sha256
        
        return True, "VALID", w, h, sha256
    except Exception as e:
        return False, f"CORRUPT_{e.__class__.__name__}", 0, 0, ""

def main():
    manifest_p = PROJECT_ROOT / "data" / "manifests" / "final_sampling_manifest_v2.json"
    img_dir = PROJECT_ROOT / "data" / "coco" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    report_dir = PROJECT_ROOT / "reports" / "m10a_recovery2"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_audit_p = report_dir / "source_image_audit_v2.json"

    with open(manifest_p, "r", encoding="utf-8") as f:
        sampling_data = json.load(f)

    images = sampling_data.get("images", [])
    total_requested = len(images)
    logger.info(f"Target Primary Cohort: {total_requested} images.")
    assert total_requested == 600, f"Expected 600 images, got {total_requested}"

    # Determine which need downloading
    to_download = []
    for img in images:
        f_name = img["file_name"]
        f_path = img_dir / f_name
        is_valid, status, _, _, _ = verify_image(f_path, img["width"], img["height"])
        if not is_valid:
            to_download.append((img, f_path))

    logger.info(f"{len(to_download)} images require download / re-verification.")

    if to_download:
        logger.info("Starting concurrent downloads (max_workers=16)...")
        completed = 0
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_img = {
                executor.submit(download_image, item[0]["coco_url"], item[1]): item
                for item in to_download
            }
            for future in as_completed(future_to_img):
                res = future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(to_download):
                    logger.info(f"Downloaded {completed}/{len(to_download)} images.")

    # Full verification pass
    logger.info("Verifying all 600 images...")
    audit_records = []
    valid_count = 0
    missing_count = 0
    corrupt_count = 0

    for img in images:
        f_name = img["file_name"]
        f_path = img_dir / f_name
        is_valid, status, w, h, sha = verify_image(f_path, img["width"], img["height"])
        
        record = {
            "image_id": img["image_id"],
            "coco_integer_id": img["coco_integer_id"],
            "file_name": f_name,
            "coco_source_split": img["coco_source_split"],
            "research_split": img["research_split"],
            "expected_width": img["width"],
            "expected_height": img["height"],
            "actual_width": w,
            "actual_height": h,
            "sha256": sha,
            "status": status,
            "is_valid": is_valid,
        }
        audit_records.append(record)
        
        if is_valid:
            valid_count += 1
        elif status == "MISSING":
            missing_count += 1
        else:
            corrupt_count += 1

    audit_summary = {
        "schema_version": "2.0.0",
        "dataset_version": "v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sampling_manifest_hash": sampling_data.get("manifest_hash"),
        "total_requested": total_requested,
        "valid_count": valid_count,
        "missing_count": missing_count,
        "corrupt_count": corrupt_count,
        "records": audit_records,
    }

    with open(out_audit_p, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Audit report saved to: {out_audit_p}")
    logger.info(f"Results: Valid={valid_count}, Missing={missing_count}, Corrupt={corrupt_count}")

    if valid_count != 600 or missing_count > 0 or corrupt_count > 0:
        raise RuntimeError(
            f"SOURCE IMAGE ACQUISITION GATE FAILED! "
            f"Valid={valid_count}/600, Missing={missing_count}, Corrupt={corrupt_count}. "
            f"STOPPING per Phase 10A-R2-5 protocol."
        )

    logger.info("SOURCE IMAGE ACQUISITION GATE PASSED (600/600 valid, 0 missing, 0 corrupt).")

if __name__ == "__main__":
    main()
