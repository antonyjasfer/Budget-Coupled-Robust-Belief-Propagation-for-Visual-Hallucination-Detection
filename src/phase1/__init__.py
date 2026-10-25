"""Phase 1 falsification pilot (see docs/PREREG.md and docs/phase1/RUN_GUIDE.md).

Pure-Python modules (splits, chair, analysis) run on CPU and are unit-tested.
GPU work lives in src/phase1/gpu_pipeline.py and imports torch lazily.
"""
