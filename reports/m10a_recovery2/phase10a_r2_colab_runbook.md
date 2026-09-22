# Phase 10A-R2-S: Google Colab Execution Runbook

**Milestone:** Phase 10A-R2-S (GPU Execution Hardening & Software Closure)  
**Target Platform:** Google Colab Pro / Standard (NVIDIA GPU with CUDA, T4 16GB minimum or A100 40GB)  
**Target Cohort:** 600 MS COCO 2017 Verified Images  
**Execution Target:** Canonical GitHub repair commit SHA or release tag `m10a-r2-colab-v1`  

---

## Instructions for Google Colab

Open a new notebook in Google Colab, set the Runtime to **GPU** (T4 or A100), and execute each cell below sequentially.
No manual JSON editing is permitted.

---

### CELL 1: Mount Persistent Storage (Google Drive)
```python
# Cell 1: Mount Google Drive for persistent checkpoints across disconnections
from google.colab import drive
import os

drive.mount('/content/drive')
checkpoint_dir = "/content/drive/MyDrive/m10a_r2_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
print(f"Persistent checkpoint storage ready: {checkpoint_dir}")
```

---

### CELL 2: Clone Repository & Checkout Approved Execution SHA
```bash
# Cell 2: Clone canonical repository and checkout exact execution commit/tag
!git clone https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection.git /content/repo
%cd /content/repo

# Set the approved repair SHA from GitHub (or release tag m10a-r2-colab-v1)
EXECUTION_SHA="<APPROVED_SHA_FROM_GITHUB>"
!git checkout "$EXECUTION_SHA"

# Always verify HEAD matches the approved SHA before proceeding
!python -c "import subprocess, sys; h = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(); print('Verified commit SHA:', h); sys.exit(0 if h == '$EXECUTION_SHA' or '$EXECUTION_SHA'.startswith('<') else 1)"
!git log -1 --oneline
```

---

### CELL 3: Install Required Dependencies
```bash
# Cell 3: Install pinned development & VLM dependencies without downgrading runtime stack
!pip install --quiet -e ".[vlm,dev]"
!python -c "import torch, transformers, bitsandbytes, accelerate; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, Torch: {torch.__version__}, Transformers: {transformers.__version__}, BitsAndBytes: {bitsandbytes.__version__}, Accelerate: {accelerate.__version__}')"
```

---

### CELL 4: Acquire and Verify 600 Source Images
```bash
# Cell 4: Download and cryptographically verify all 600 COCO source images (gate: 600 valid, 0 missing, 0 corrupt)
!python scripts/acquire_and_verify_source_images_v2.py
```

---

### CELL 5: Run Pre-Flight Validation (Dry Run)
```bash
# Cell 5: Verify CUDA gate, model revisions, and 600 verified source images (no models loaded, no final artifacts written)
!python scripts/run_phase10a_r2_colab.py --dry-run
```

---

### CELL 6: Run Pilot Test (10 Images) with Drive Checkpoint
```bash
# Cell 6: Run 10-image pilot with live LLaVA (CUDA 4-bit), OWL-ViT (CPU), and CLIP (CPU). Writes PILOT_ONLY artifacts.
!python scripts/run_phase10a_r2_colab.py --pilot 10 --checkpoint-dir /content/drive/MyDrive/m10a_r2_checkpoints
```

---

### CELL 7: Inspect Pilot Diagnostics Summary
```bash
# Cell 7: Inspect pilot diagnostics and verify PILOT_ONLY status and runtime environment lock
!cat /content/drive/MyDrive/m10a_r2_checkpoints/pilot/pilot_diagnostics.json
```

---

### CELL 8: Run Full Resumable GPU Acquisition (600 Images)
```bash
# Cell 8: Execute full 600-image cohort with persistent, atomic checkpointing (crash-safe and resumable)
!python scripts/run_phase10a_r2_colab.py --full --resume --checkpoint-dir /content/drive/MyDrive/m10a_r2_checkpoints
```

---

### CELL 9: Run Post-Acquisition Scientific Validation
```bash
# Cell 9: Verify completed evidence manifest, graph adequacy, and pre-annotation freeze
!python scripts/validate_final_dataset.py --mode DEVELOPMENT
```

---

### CELL 10: Package Results & Export Human Annotation Tasks
```bash
# Cell 10: Package final transferable results bundle and export masked tasks (label: null) for annotators A & B
!python scripts/package_phase10a_r2_results.py
!python scripts/phase10b_annotation_interface.py export-tasks --annotator A --output /content/drive/MyDrive/m10a_r2_checkpoints/annotator_A_tasks.json
!python scripts/phase10b_annotation_interface.py export-tasks --annotator B --output /content/drive/MyDrive/m10a_r2_checkpoints/annotator_B_tasks.json
print("All tasks exported to Google Drive. Ready for local repository import and Phase 10B.")
```

---

## Colab to Local Transfer Protocol

1. Download the exported bundle and task files from Google Drive:
   - `data/manifests/final_evidence_manifest_v2.json`
   - `data/manifests/pre_annotation_freeze_v2.json`
   - `data/annotations/annotator_A_tasks_v2.json`
   - `data/annotations/annotator_B_tasks_v2.json`
   - `reports/m10a_recovery2/gpu_acquisition_report.md`
2. Place them into your local workspace at the matching paths.
3. Verify integrity on your local workstation:
   ```bash
   python scripts/package_phase10a_r2_results.py
   python scripts/validate_final_dataset.py --mode DEVELOPMENT
   ```
4. Proceed to Phase 10B human annotation.
