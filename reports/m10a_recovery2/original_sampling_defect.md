# Phase 10A-R2: Forensic Report on Original Sampling Defect and Resampling Protocol

**Document Version:** 1.0.0  
**Phase:** 10A-R2-0  
**Date:** 2026-09-23  
**Status:** ARCHIVED & PERMANENTLY SUPERSEDED  

---

## 1. Original Sampling Algorithm and Defect Specification

During Milestone 9E (`scripts/generate_m9e_manifests.py`), the candidate MS COCO universe generator was implemented as follows:

```python
def generate_candidate_coco_universe(n_universe: int = 5000, seed: int = 123) -> list:
    rng = np.random.RandomState(seed)
    candidates = []
    # Defect: Real COCO IDs are sparse, but the function sampled uniform integers:
    base_ids = rng.choice(np.arange(1000, 580000), size=n_universe, replace=False)
    for b_id in sorted(base_ids):
        w = int(rng.choice([640, 500, 480, 428, 375]))
        h = int(rng.choice([480, 375, 428, 640, 500]))
        img_id = f"coco_{b_id:012d}"
        candidates.append({
            "id": img_id,
            "image_id": img_id,
            "file_name": f"{b_id:012d}.jpg",
            "width": w,
            "height": h,
            "license": 3,
            "coco_url": f"http://images.cocodataset.org/val2017/{b_id:012d}.jpg",
        })
    return candidates
```

### Mechanism of Failure
1. **Numeric Range Assumption**: The function drew candidate IDs from a dense integer interval `np.arange(1000, 580000)`.
2. **Sparsity of Official COCO Identifiers**: In reality, MS COCO image IDs are not contiguous integers. Across all partitions (train, val, test, unlabeled), only ~50% of the integers in that range correspond to actual image records.
3. **Synthetic Image Dimension Assignment**: Image dimensions (`width`, `height`) were randomly assigned from a discrete set rather than extracted from official metadata records.

---

## 2. Discovery Timeline and Number of Nonexistent IDs

- **Discovery Milestone**: Phase 10A-R (Milestone 10A Recovery).
- **Forensic Verification**: During the Phase 10A source image download and verification gate, 507 of the 600 requested images were initially reported missing. A comprehensive audit across all official COCO archives (`train2017`, `val2017`, `test2017`, `test2015`, `test2014`, `unlabeled2017`) revealed:
  - **331 genuine COCO images** existed in the sampled cohort across various partitions.
  - **269 IDs were completely non-existent** (`NON_EXISTENT_COCO_INDEX`) across every public COCO partition and AWS S3 bucket.
- **Confirmation**: The 269 phantom IDs returned HTTP 404 from `images.cocodataset.org` and were absent from all official JSON annotation indices.

---

## 3. Scientific Necessity of Cohort Resampling

Continuing with the flawed cohort was strictly rejected for the following scientific reasons:
1. **Sample Truncation Bias**: Reducing the cohort from 600 to 331 images would severely compromise statistical power, precision planning, and the pre-registered split sizes (300 Train, 90 Val, 90 Cal, 120 Test).
2. **Selective Replacement Invalidity**: Replacing only the 269 missing IDs would mix two distinct sampling mechanisms (random range for the first 331, metadata-conditioned for the 269), introducing unknown selection artifacts.
3. **Split Re-allocation Imbalance**: Partition assignments in v1 were computed over 600 IDs. With 269 IDs missing, the realized counts across Train, Val, Cal, and Test became severely distorted (e.g., Test had only 64 real images instead of 120).
4. **Authoritative Population Definition**: A valid scientific benchmark must represent an explicit, well-defined candidate population: the union of official MS COCO 2017 `train2017` and `val2017` images.

---

## 4. Integrity and Non-Contamination Confirmation

It is formally confirmed that at the time this defect was identified and the resampling decision made:
1. **Zero Human Annotations**: No human labels, bounding boxes, or adjudication had been performed on any claim or image.
2. **Zero Outcome Inspection**: No test set accuracy, robust interval metrics, AUROC, F1 scores, or parameter calibrations had been computed.
3. **Pre-Pre-Annotation State**: The research was strictly at the data acquisition / pre-annotation freeze stage.
4. **No Outcome-Informed Resampling**: The resampling of the 600-image cohort is performed purely to repair source image existence from authoritative metadata before any model claims or human labels are established.

---

## 5. Supersession and Archival Notice

- **Original Manifest**: `data/manifests/final_sampling_manifest.json` is preserved in `data/manifests/archived_v1/final_sampling_manifest_v1.json`.
- **Classification**: `INVALID_SAMPLING_UNIVERSE` / `SUPERSEDED`.
- **Active Replacement**: `data/manifests/final_sampling_manifest_v2.json` constructed under Phase 10A-R2.
