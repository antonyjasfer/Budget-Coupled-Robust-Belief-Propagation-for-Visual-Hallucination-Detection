# Milestone 9 Google Colab Final Execution Runbook

This runbook guides running the complete Milestone 9 scientific pipeline on Google Colab with GPU acceleration and locked datasets.

---

### Step 1: Environment Setup
```bash
!git clone https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection.git repo
%cd repo
!pip install -r requirements.txt
!pip install pytest
```

### Step 2: GPU and System Verification
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Step 3: Verify Dataset and Checksums
```bash
python scripts/verify_dataset_checksums.py
```

### Step 4: Run Milestone 9 Final Gate
```python
from src.scientific.gate import FinalDatasetGate
gate = FinalDatasetGate()
res = gate.verify()
print(f"Gate Status: {res.status.value}")
print(f"Passed: {res.passed}")
if not res.passed:
    for diag in res.diagnostics:
        print(f"  - {diag}")
```

### Step 5: Execute Final Scientific Pipeline
```bash
python scripts/run_m9_evaluation.py --mode final --output-dir reports/m9/final
```

### Step 6: Validate All Scientific Artifacts
```bash
python scripts/validate_m9_final.py --results-dir reports/m9/final
```

### Step 7: Archive Results
```bash
!zip -r m9_final_results.zip reports/m9/final/
```
