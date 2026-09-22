# Phase 10A-R2-S: Google Colab Execution Runbook

**Milestone:** Phase 10A-R2-S (GPU Execution Hardening & Software Closure)  
**Target Platform:** Google Colab Pro / Standard (T4 16GB or A100 40GB)  
**Target Cohort:** 600 MS COCO 2017 Verified Images  
**Repository SHA:** `6510402d106eb1b8e0716d7d648a3a6fa1b03896` (or latest closure commit)  

---

## Instructions for Google Colab

Open a new notebook in Google Colab, set the Runtime to **GPU** (T4 or A100), and execute each cell below sequentially.

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

### CELL 2: Clone Repository & Checkout Exact SHA
```bash
# Cell 2: Clone canonical repository and checkout exact execution commit
!git clone https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection.git /content/repo
%cd /content/repo
!git checkout 6510402d106eb1b8e0716d7d648a3a6fa1b03896
!git log -1 --oneline
```

---

### CELL 3: Install Required Dependencies
```bash
# Cell 3: Install pinned dependencies
!pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip install --quiet transformers==4.38.2 accelerate==0.28.0 bitsandbytes==0.43.0 scipy pillow pytest
!python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

### CELL 4: Run Pre-Flight Validation
```bash
# Cell 4: Verify CUDA gate, model revisions, and 600 verified source images
!python scripts/run_phase10a_r2_colab.py --dry-run
```

---

### CELL 5: Run Pilot Test (10 Images)
```bash
# Cell 5: Validate 10-image pilot with live LLaVA, OWL-ViT, and CLIP inference
!python scripts/run_phase10a_r2_colab.py --pilot 10 --checkpoint-dir /content/drive/MyDrive/m10a_r2_checkpoints
```

---

### CELL 6: Run Full Resumable GPU Acquisition (600 Images)
```bash
# Cell 6: Execute full cohort with persistent, atomic checkpointing (resumable)
!python scripts/run_phase10a_r2_colab.py --full --resume --checkpoint-dir /content/drive/MyDrive/m10a_r2_checkpoints
```

---

### CELL 7: Run Post-Acquisition Scientific Validation
```bash
# Cell 7: Verify completed evidence manifest, graph adequacy, and pre-annotation freeze
!python scripts/validate_final_dataset.py --mode DEVELOPMENT
```

---

### CELL 8: Package & Export Human Annotation Tasks
```bash
# Cell 8: Package final transferable results bundle and export masked tasks for annotators A & B
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
   - `data/manifests/annotation_task_claimset_v2.json`
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
