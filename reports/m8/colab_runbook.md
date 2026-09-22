# Milestone 8: Google Colab Execution Runbook

This runbook provides step-by-step instructions for executing the full Milestone 8 experiment on Google Colab when the complete 600-image M7 dataset is available.

---

## 1. Prerequisites & Environment Setup

### 1.1 Select Colab Runtime
- Go to **Runtime > Change runtime type**.
- Select **T4 GPU** (or V100/A100 if available) with High-RAM.

### 1.2 Clone Repository & Install Dependencies
```bash
# Clone the repository
!git clone https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection.git
%cd Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection

# Install project in editable mode with test dependencies
!pip install -e .
!pip install pyyaml matplotlib scikit-learn scipy pytest
```

### 1.3 Verify Environment & GPU
```bash
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

---

## 2. Step-by-Step Execution Workflow

### Step 1: Run Quick Smoke Test (Verify Math & Pipelines)
```bash
python scripts/run_m8_experiments.py --smoke --output-dir reports/m8_smoke
```
*Expected duration: < 5 seconds.*

### Step 2: Validate Artifacts from Smoke Run
```bash
python scripts/validate_m8_artifacts.py --results-dir reports/m8_smoke
```

### Step 3: Run Full Evaluation on Locked M7 Dataset
```bash
python scripts/run_m8_experiments.py \
    --config configs/m8_colab.yaml \
    --output-dir reports/m8 \
    --resume
```

### Step 4: Validate Final Output Artifacts
```bash
python scripts/validate_m8_artifacts.py --results-dir reports/m8
```

---

## 3. Resuming After Disconnection

The M8 runner features atomic file writing and automatic caching in `reports/m8/raw/`.

If a Colab session disconnects or is preempted:
1. Re-mount Google Drive or reconnect to the session.
2. Re-run Step 3 with the `--resume` flag:
```bash
python scripts/run_m8_experiments.py --config configs/m8_colab.yaml --output-dir reports/m8 --resume
```
All completed claim evaluations and cached inference results will be reused without redundant computation.

---

## 4. Archiving and Exporting Results

To download the publication tables, figures, and manifest:
```python
import shutil
shutil.make_archive('m8_results_archive', 'zip', 'reports/m8')
from google.colab import files
files.download('m8_results_archive.zip')
```

---

## 5. Troubleshooting & Resource Management

- **Out of Memory (OOM)**: The PGM inference layer runs entirely in CPU/RAM and consumes < 200MB. If evidence generation runs out of GPU memory, ensure batch size is 1.
- **Single-class Test Split**: If the test split contains only one ground-truth class, ROC-AUC is automatically reported as `None` with a descriptive message rather than crashing.
- **Integrity Validation Errors**: If `validate_m8_artifacts.py` fails, check that the dataset split files have not been modified or corrupted.
