# MS COCO 600-Image Evidence Acquisition & Human Annotation Runbook

**Milestone:** Phase 9E  
**Target Environment:** Google Colab Pro / Cloud GPU (T4 / A100) + Local Human Annotation  
**Target Cohort:** 600 MS COCO 2017 Images (300 Train, 90 Val, 90 Cal, 120 Test)

---

## 1. Overview & Operational Separation

This runbook guides the complete execution of real multi-modal evidence extraction and independent double human annotation.

```
+-------------------------------------------------------------------------------+
| STAGE 1: COLAB GPU PIPELINE                                                  |
| 1. Download MS COCO val2017 images for the 600 sampled IDs                   |
| 2. Run LLaVA-1.5-7B captioning (4-bit NF4)                                   |
| 3. Extract atomic claims conservatively via NLP pipeline                      |
| 4. Probe open-vocabulary evidence (OWL-ViT) and cross-modal features (CLIP)  |
| 5. Export masked human annotation tasks                                       |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| STAGE 2: INDEPENDENT HUMAN DOUBLE ANNOTATION (OUTSIDE PYTHON)                |
| 1. Deliver masked tasks to ANNOTATOR_A (no model scores, no captions)        |
| 2. Deliver masked tasks to ANNOTATOR_B independently                          |
| 3. Annotators independently record: SUPPORTED, HALLUCINATED, or UNKNOWN       |
| 4. Privacy: Opaque/pseudonymous IDs only; zero PII stored                     |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| STAGE 3: IMPORT, AGREEMENT AUDIT, AND ADJUDICATION                            |
| 1. Import completed human annotation JSON files                               |
| 2. Compute inter-annotator agreement (Cohen's Kappa, raw agreement)          |
| 3. Queue all disagreements and UNKNOWN claims for ADJUDICATOR_1               |
| 4. Adjudicate remaining ambiguous claims                                      |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
| STAGE 4: TWO-STEP CRYPTOGRAPHIC DATASET LOCK                                 |
| 1. Pre-lock validation: verify zero mock/pseudo labels & complete coverage   |
| 2. Create dataset_lock.json hashing Sampling, Evidence & Annotation manifests |
| 3. Independently re-read files from disk and verify SHA-256 integrity        |
| 4. Final experiment readiness certified                                       |
+-------------------------------------------------------------------------------+
```

---

## 2. Stage 1: Colab Setup & Evidence Acquisition

### 2.1 Colab Environment Initialization
Open a new Google Colab notebook with a GPU runtime (T4 or A100). Run the setup cell:

```bash
# Clone repository
!git clone https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection.git
%cd Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection

# Install exact requirements
!pip install -r requirements.txt
!pip install -e .
!python -m spacy download en_core_web_sm
```

### 2.2 Download MS COCO Images
Download only the 600 sampled images using the pre-frozen sampling manifest:

```python
import json
from pathlib import Path
import urllib.request

with open("data/manifests/final_sampling_manifest.json") as f:
    manifest = json.load(f)

img_dir = Path("data/coco/val2017")
img_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading {len(manifest['selected_image_ids'])} images from COCO repository...")
for img_id in manifest["selected_image_ids"]:
    # Format: coco_000000xxxxxx -> xxxxxx.jpg
    raw_num = img_id.replace("coco_", "").lstrip("0")
    file_name = f"{int(raw_num):012d}.jpg"
    url = f"http://images.cocodataset.org/val2017/{file_name}"
    dst = img_dir / file_name
    if not dst.exists():
        urllib.request.urlretrieve(url, dst)
print("COCO image download complete.")
```

### 2.3 Run Evidence Acquisition
Execute the atomic, resumable acquisition script:

```bash
python scripts/acquire_final_evidence.py \
  --sampling-manifest data/manifests/final_sampling_manifest.json \
  --coco-dir data/coco/val2017 \
  --output-evidence data/manifests/evidence_manifest.json \
  --failure-log data/manifests/evidence_failures.json \
  --device cuda
```

### 2.4 Predeclared Evidence Failure Handling
If an evidence provider fails or is unavailable for an image:
- The error is appended to `evidence_failures.json` with `reason_code`, `provider`, `attempt_count`, and `last_error_class`.
- The claim remains in the dataset for human annotation.
- Missing evidence is **NEVER** replaced with numeric zero.

---

## 3. Stage 2: Independent Human Double Annotation

> [!IMPORTANT]
> Python code does **NOT** generate human ground-truth labels. Real human annotators must inspect images and judge whether claimed objects exist.

### 3.1 Export Masked Annotation Tasks
Generate the masked task files:

```bash
python -m src.annotation.workflow export \
  --input-evidence data/manifests/evidence_manifest.json \
  --output-tasks data/annotations/masked_tasks.json
```

**Masking Verification:**  
Inspect `data/annotations/masked_tasks.json` to confirm:
- NO detector confidence scores,
- NO CLIP similarity scores,
- NO model predictions or posteriors,
- NO Ising parameters ($\theta, \epsilon, J$),
- NO full generated caption (prevents linguistic confirmation bias).

### 3.2 Human Annotation Execution
1. Send `data/annotations/masked_tasks.json` to **`ANNOTATOR_A`**.
2. Send the exact same file to **`ANNOTATOR_B`** independently.
3. Both annotators judge visual object support:
   - `SUPPORTED`: Object is clearly visible in the image.
   - `HALLUCINATED`: Object is absent from the image.
   - `UNKNOWN`: Occluded, ambiguous, or impossible to determine.
4. Pseudonymous IDs: Annotators are identified strictly as `ANNOTATOR_A` and `ANNOTATOR_B` to preserve privacy. Zero PII is stored.

---

## 4. Stage 3: Import, Agreement Audit, and Adjudication

### 4.1 Import Annotations
Once completed task JSON files are received:

```bash
python -m src.annotation.workflow import \
  --annotator-a data/annotations/annotator_a_results.json \
  --annotator-b data/annotations/annotator_b_results.json \
  --output data/annotations/double_annotations.json
```

### 4.2 Compute Inter-Annotator Agreement
Audit the imported annotations:
- Cohen's Kappa ($\kappa \ge 0.70$ recommended).
- Raw agreement percentage ($\ge 85\%$).
- Identify all disagreement claims ($A \neq B$) and claims marked `UNKNOWN`.

### 4.3 Adjudication of Disagreements
Export the adjudication queue:

```bash
python -m src.annotation.workflow queue-adjudication \
  --annotations data/annotations/double_annotations.json \
  --output data/annotations/adjudication_queue.json
```

Deliver the queue to an independent adjudicator (`ADJUDICATOR_1`). Import the adjudicated decisions:

```bash
python -m src.annotation.workflow apply-adjudication \
  --adjudication-results data/annotations/adjudication_results.json \
  --output-manifest data/manifests/annotation_manifest.json
```

---

## 5. Stage 4: Two-Step Cryptographic Dataset Lock

### 5.1 Pre-Lock Validation
Verify dataset cleanliness in `DEVELOPMENT` mode before locking:

```bash
python scripts/validate_final_dataset.py --mode DEVELOPMENT
```

### 5.2 Execute Two-Step Lock
Run the lock generator:

```python
from src.data.dataset_lock import create_and_verify_dataset_lock

lock = create_and_verify_dataset_lock(
    sampling_manifest_path="data/manifests/final_sampling_manifest.json",
    evidence_manifest_path="data/manifests/evidence_manifest.json",
    annotation_manifest_path="data/manifests/annotation_manifest.json",
    output_lock_path="data/manifests/dataset_lock.json",
    code_sha="<git commit sha>",
    cohort_counts={
        "total_images": 600,
        "train_images": 300,
        "validation_images": 90,
        "calibration_images": 90,
        "test_images": 120,
    }
)
print(f"Dataset successfully locked! Lock Hash: {lock.lock_hash}")
```

### 5.3 Final Mode Certification
Run the final dataset validator:

```bash
python scripts/validate_final_dataset.py --mode FINAL
```

When this command exits with code 0, the dataset is certified as **LOCKED AND READY FOR FINAL EXPERIMENTS**.
