# Milestone 7 Colab Runbook: 600-Image Real Evidence Acquisition & Ground-Truth Locking

This runbook specifies the exact procedure to run the full 600-image real neural evidence acquisition pipeline and execute human annotation tasks on Google Colab GPU (Tesla T4 or higher), locking Milestone 7 before Milestone 8 begins.

---

## 1. Environment Setup

Select a GPU runtime in Google Colab:
- **Runtime** -> **Change runtime type** -> **Hardware accelerator**: `T4 GPU` (or A100 / V100).
- Python 3.10+ is standard on Colab.

Clone the repository and mount Google Drive (optional, for persistent caching):
```bash
!git clone https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection.git
%cd Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection
```

---

## 2. Package Installation

Install required dependencies with `accelerate`, `bitsandbytes`, `transformers`, and `torch`:
```bash
!pip install --upgrade pip
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install transformers accelerate bitsandbytes pillow pydantic numpy scipy pytest
```

Verify GPU availability and CUDA:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

---

## 3. Hugging Face Authentication

If needed for gated checkpoints or higher rate limits:
```python
from huggingface_hub import login
# Provide HuggingFace token if prompted
login()
```

---

## 4. Hardware & GPU Memory Configuration

The pipeline enforces dual-device memory safety:
- **VLM Device**: `cuda:0` with 4-bit NormalFloat4 (NF4) quantization via bitsandbytes (`compute_dtype=float16`).
- **Evidence Device**: `cpu` (or secondary GPU/offload) to ensure zero OOM crashes when OWL-ViT and CLIP operate.

---

## 5. Model Checkpoints & Specifications

- **VLM**: `llava-hf/llava-1.5-7b-hf` (Revision: resolved commit SHA on HuggingFace Hub)
- **Zero-Shot Detector**: `google/owlvit-base-patch32`
- **Vision-Language Alignment**: `openai/clip-vit-base-patch32`
- **Detection Prompt**: `"a photo of a {category}"`
- **Score Semantics**:
  - Detector score $d_i \in [0, 1]$ (calibrated confidence)
  - CLIP score $g_i \in [-1, 1]$ (cosine similarity)

---

## 6. Full 600-Image Evidence Acquisition Execution

Run the scalable acquisition CLI across the 600-image manifest with automatic checkpointing and resumption:
```bash
python experiments/run_m7_evidence_acquisition.py \
    --manifest data/manifests/m7_image_manifest.json \
    --image-base-dir data/real_images \
    --cache-dir data/cache/vlm \
    --output-claims data/exports/m7_claims.jsonl \
    --device cuda:0 \
    --evidence-device cpu \
    --allow-download \
    --load-in-4bit \
    --resume
```

### Checkpoint & Resume Behavior:
- The script appends to `data/exports/m7_claims.jsonl` image-by-image.
- If interrupted (e.g. Colab timeout), re-running the exact same command will detect already processed `image_id` entries and resume seamlessly.

---

## 7. Export Masked Human Annotation Tasks

Generate independent, double-blind annotation templates without exposing detector scores, CLIP scores, or split information:
```bash
python -m experiments.run_m7_dataset_pipeline export-templates \
    --evidence-file data/exports/m7_claims.jsonl \
    --output-dir data/annotations
```

Outputs generated:
- `data/annotations/m7_template_annotator_A.jsonl`
- `data/annotations/m7_template_annotator_B.jsonl`

---

## 8. Human Annotation Procedure

Two independent annotators (Annotator A and Annotator B) inspect the images and assign ground-truth labels for each claim in their respective template:
- `supported`: Object physically exists and is visually verifiable.
- `hallucinated`: Claimed object is not present in the image.
- `unknown`: Occlusion, blur, or severe ambiguity prevents definitive visual verification.

Completed annotations are saved as:
- `data/annotations/m7_annotator_A.jsonl`
- `data/annotations/m7_annotator_B.jsonl`

---

## 9. Dataset Merge, Inter-Annotator Agreement & Adjudication

Merge completed annotations, compute multiclass Cohen's kappa, and adjudicate disagreements:
```bash
python -m experiments.run_m7_dataset_pipeline merge \
    --annotator-a data/annotations/m7_annotator_A.jsonl \
    --annotator-b data/annotations/m7_annotator_B.jsonl \
    --adjudicated data/annotations/m7_adjudications.jsonl \
    --output data/exports/m7_ground_truth.jsonl
```

Generate the comprehensive quality audit report:
```bash
python -m experiments.run_m7_dataset_pipeline audit \
    --manifest data/manifests/m7_image_manifest.json \
    --annotator-a data/annotations/m7_annotator_A.jsonl \
    --annotator-b data/annotations/m7_annotator_B.jsonl \
    --adjudicated data/annotations/m7_adjudications.jsonl \
    --report-json reports/m7_dataset_quality_report.json \
    --report-md reports/m7_dataset_quality_report.md
```

---

## 10. Dataset Lock Checksum Generation

Once 100% of claims are annotated and adjudicated:
```bash
python -m experiments.run_m7_dataset_pipeline lock \
    --manifest data/manifests/m7_image_manifest.json \
    --claims-out data/exports/m7_claims.jsonl \
    --output data/exports/m7_ground_truth.jsonl \
    --annotator-a data/annotations/m7_annotator_A.jsonl \
    --annotator-b data/annotations/m7_annotator_B.jsonl \
    --adjudicated data/annotations/m7_adjudications.jsonl \
    --target-size 600 \
    --lock-out data/manifests/m7_dataset_lock.json
```

When all 600 images and annotations are verified:
- `status`: `"LOCKED"`
- All SHA-256 checksums recorded in `data/manifests/m7_dataset_lock.json`.

---

## 11. Strict Milestone Boundary Notice

**DO NOT execute Milestone 8 in this session.**
Milestone 8 ($\theta$, $\epsilon$, $J$ parameter fitting and robust BP) begins strictly in a subsequent phase after the M7 dataset lock is verified and committed.
