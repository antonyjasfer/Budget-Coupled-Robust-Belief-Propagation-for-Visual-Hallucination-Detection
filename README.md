# Budget-Coupled Robust Belief Propagation for Visual Hallucination Detection

**Milestone 1: Mathematical Inference Core**  
**Milestone 2: Dataset Manifests, Label-Safe Claim Schemas, and Leakage-Safe Splits**  
**Milestone 3: Deterministic Object-Existence Claim Extraction and Raw Evidence Contracts**  
**Milestone 4: Frozen-VLM Caption Acquisition, Caching, and Pilot Pipeline**  
**Milestone 5: Real-Image Pilot Readiness, 11-Gate Preflight, and Portable Execution**

A pure mathematical inference engine implementing exact Belief Propagation, brute-force ground-truth verification, and budget-coupled robust dynamic programming over binary attractive tree graphical models, coupled with a typed data foundation, conservative object-existence claim extraction, raw visual evidence contracts, atomic disk caching with synthetic/real provenance isolation, structured preflight verification, and portable real-image pilot packaging.

---

## 1. Mathematical Formulation

### 1.1 Binary Attractive Tree Graphical Model
Let $T = (V, E)$ be a tree with $n$ nodes ($h_i \in \{-1, +1\}$).
- **Unary potentials:**
  $$\psi_i(h_i) = \exp((\theta_i + \delta_i) h_i)$$
- **Pairwise potentials:**
  $$\psi_{ij}(h_i, h_j) = \exp(J_{ij} h_i h_j), \quad J_{ij} \ge 0 \ (\text{ferromagnetic/attractive})$$

### 1.2 Standard Belief Propagation & Stable $f_J(x)$
In log-cavity field domain, the tree message passing equation is:
$$f_J(x) = \text{atanh}(\tanh(J) \tanh(x))$$

Numerically stabilized using $\text{logaddexp}$ / $\ln\cosh$:
$$f_J(x) = \frac{1}{2} \left[ \text{logaddexp}(J + x, -(J + x)) - \text{logaddexp}(J - x, -(J - x)) \right]$$

Total belief field at node $i$:
$$\eta_i = \theta_i + \delta_i + \sum_{k \in N(i)} \nu_{k \to i}$$
$$P(h_i = +1) = \sigma(2 \eta_i) = \frac{1}{1 + e^{-2 \eta_i}}$$

### 1.3 Budget-Coupled Uncertainty Model
The adversary allocates perturbations $\boldsymbol{\delta} = (\delta_1, \dots, \delta_n)$ from the uncertainty set:
$$\mathcal{U} = \left\{ \boldsymbol{\delta} \in \mathbb{R}^n : |\delta_i| \le \varepsilon_i \ \forall i, \ \sum_{i=1}^n |\delta_i| \le B \right\}$$

Because the model is attractive ($J_{ij} \ge 0$), the marginal $P(h_{\text{target}} = +1)$ is strictly monotonic in all local fields $\theta_i + \delta_i$.
- **Upper Bound:** $\delta_i \ge 0$, solved via **max-plus tree convolution dynamic programming**.
- **Lower Bound:** $\delta_i \le 0$, solved via **min-plus tree convolution dynamic programming**.

### 1.4 Continuous Certification
Given discretization grid step $\Delta = B / K$, we maintain separate variables for:
- `lower_grid`, `upper_grid`: Discretized grid bounds.
- `lower_certified`, `upper_certified`: Rigorous continuous bounds satisfying:
  $$\text{lower\_certified} \le \text{lower\_grid} \le P(h_r = +1) \le \text{upper\_grid} \le \text{upper\_certified}$$

---

## 2. Claim Extraction & Raw Evidence Contracts

1. **Object-Existence Scope**: Strictly restricted to atomic object-existence claims (no spatial relations, attributes, or unvalidated graph topologies).
2. **Conservative Contextual Filtering**:
   - Mentions under negation ("no dog", "without a car"), uncertainty ("might be a dog"), questions ("Is there a dog?"), hypotheticals, non-visual references, or depictions are rejected with explicit reason codes.
   - Ambiguous words (e.g. color "orange" vs fruit, verb "train" vs vehicle) are filtered.
   - Mentions are deduplicated within a response while preserving all supporting character offsets.
3. **Raw Visual Evidence Contracts**:
   - Bounded object detector presence score $d_i \in [0, 1]$ ($\max_r s_r$, with explicit missing flags).
   - Global image-claim cosine similarity $g_i \in [-1, 1]$.
   - Raw features are NOT calibrated probabilities and are not converted to PGM fields or radii in this milestone.

---

## 3. Frozen-VLM Provider, Caching, and Pilot Readiness (Milestones 4 & 5)

1. **Target Checkpoint**: `llava-hf/llava-1.5-7b-hf` with greedy generation (`temperature=0.0`, `do_sample=False`, `max_new_tokens=64`).
2. **Deterministic Prompt**:
   ```
   Describe the visible physical objects in two short sentences.
   Do not speculate about objects outside the image.
   ```
3. **Cache & Provenance Isolation (`VLMCache`)**:
   - Keying distinguishes image SHA-256 hash, provider kind (`synthetic_fixture` vs `llava_15_hf`), synthetic/real provenance, model snapshot revision, prompt text hash, and generation hyperparameters.
   - Real inference queries cannot hit synthetic cache entries.
   - Response IDs incorporate caption text hashes to prevent collision.
4. **11-Gate Preflight Verification (`src/vlm/preflight.py`)**:
   - Checks manifest validity, real image flags, training split isolation, image decodability & SHA-256 hashes, exclusion of reserved/evaluation IDs, runtime dependencies (`torch`, `transformers`, `PIL`), local checkpoint presence or download permissions, device/dtype compatibility, model snapshot revision, and path permissions.
5. **Portable Pilot Bundle (`src/vlm/pilot_bundle.py`)**:
   - Packages up to 10 selected real training images, portable manifest with relative paths, generation config, provenance run manifest, and run documentation for execution in GPU environments.

---

## 4. Directory Structure

```
.
├── configs/
│   ├── data_pipeline.json         # Pipeline configuration and split parameters
│   ├── claim_extraction.json      # Claim extractor and evidence rules
│   └── vlm_generation.json        # Frozen VLM generation hyperparameters
├── pyproject.toml
├── README.md
├── src/
│   ├── __init__.py
│   ├── claims/
│   │   ├── __init__.py
│   │   ├── vocabulary.py          # CategoryRegistry, versioned alias rules, longest-match
│   │   ├── extraction.py          # Conservative rule-based claim extractor & diagnostics
│   │   └── cli.py                 # CLI for claim extraction and evidence attachment
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Typed dataclasses, enums, PGM spin mappings
│   │   ├── coco.py                # Local COCO metadata adapter & evidence index
│   │   ├── pope.py                # Local POPE benchmark adapter
│   │   ├── image_registry.py      # Image identity groups & overlap auditing
│   │   ├── manifests.py           # Manifest creation, validation, save/load
│   │   ├── splits.py              # Seeded, deterministic image-group splitting
│   │   └── cli.py                 # CLI for manifest validation, splitting, overlap audit
│   ├── evidence/
│   │   ├── __init__.py
│   │   ├── schemas.py             # RawEvidenceRecord contracts & validation
│   │   ├── provider.py            # EvidenceProvider protocol
│   │   └── fixture_provider.py    # Offline fixture evidence provider
│   ├── pgm/
│   │   ├── __init__.py
│   │   ├── tree_model.py          # TreeModel, graph validation, tree rooting
│   │   ├── standard_bp.py         # Numerically stable standard tree BP
│   │   └── brute_force.py         # Exact 2^N state inference & grid search
│   ├── robust_bp/
│   │   ├── __init__.py
│   │   ├── budget_convolution.py  # 1D max-plus / min-plus convolutions
│   │   ├── solver.py              # Budget-coupled robust BP tree DP solver
│   │   ├── certification.py       # Continuous certification & 1-node bounds
│   │   └── witness.py             # Witness backpointer reconstruction & verification
│   └── vlm/
│       ├── __init__.py
│       ├── provider.py            # VLMProvider protocol, VLMGenerationConfig, SyntheticVLMProvider
│       ├── llava_provider.py      # Production adapter for llava-hf/llava-1.5-7b-hf
│       ├── cache.py               # Atomic, collision-resistant VLMCache with provenance isolation
│       ├── pipeline.py            # Pilot execution pipeline & AnnotationBundle export
│       ├── preflight.py           # 11-gate preflight validation facility
│       ├── pilot_bundle.py        # Portable pilot bundle packaging & validation
│       └── cli.py                 # Unified CLI for preflight, bundle export/validation, and pilot runs
├── tests/
│   ├── __init__.py
│   ├── fixtures/                  # Synthetic offline test fixtures (COCO, POPE, M3, M4)
│   ├── test_vlm_preflight.py      # Preflight checks, blockers, and warnings
│   ├── test_pilot_bundle.py       # Portable bundle export, relocation, and hash validation
│   ├── test_vlm_cache.py          # Cache determinism, hit/miss, synthetic/real isolation
│   ├── test_vlm_pipeline.py       # Train-split sampling, pilot run, bundle serialization
│   ├── test_vlm_provenance.py     # Provenance tracking, response ID collision, dependency isolation
│   ├── test_vlm_cli.py            # CLI subcommand integration tests
│   ├── test_claim_vocabulary.py   # Vocabulary and alias resolution tests
│   ├── test_claim_extraction.py   # Rule-based claim extractor and offset tests
│   ├── test_evidence_schemas.py   # Raw evidence record validation tests
│   ├── test_fixture_evidence_provider.py # Fixture provider tests
│   ├── test_claim_evidence_pipeline.py # End-to-end integration tests
│   ├── test_data_schemas.py       # Schema validation and round trips
│   ├── test_coco_adapter.py       # COCO parsing and non-definitive absence checks
│   ├── test_pope_adapter.py       # POPE benchmark parsing and validation
│   ├── test_image_registry.py     # Identity grouping and overlap detection
│   ├── test_data_splits.py        # Deterministic splitting and reservation checks
│   ├── test_data_manifests.py     # Manifest integrity validation
│   ├── test_data_cli.py           # CLI subcommand testing
│   ├── test_math_regressions.py   # Mathematical edge cases and continuous checks
│   ├── test_standard_bp.py        # BP vs brute force on trees, numerical stability
│   ├── test_robust_bp.py          # Nominal recovery (B=0, eps=0), monotonicity
│   ├── test_brute_force_grid.py   # Robust DP vs exhaustive grid brute force
│   ├── test_certification.py      # Analytical 1-node bounds and certificate sandwich
│   └── test_witness.py            # Witness constraint and field verification
└── experiments/
    ├── run_3node_chain.py         # Benchmark 3-node chain reproduction
    ├── run_milestone2_demo.py     # Milestone 2 data splitting demonstration
    ├── run_milestone3_demo.py     # Milestone 3 extraction and evidence demonstration
    ├── run_milestone4_demo.py     # Milestone 4 frozen-VLM acquisition and pilot demonstration
    └── run_milestone5_demo.py     # Milestone 5 preflight, portable bundle, and cache isolation demonstration
```

---

## 5. Running Tests, Experiments, and CLI

### Run the complete pytest test suite:
```bash
uv run pytest -v
```

### Run Benchmark Experiments & Demos:
```bash
uv run python experiments/run_3node_chain.py
uv run python experiments/run_milestone2_demo.py
uv run python experiments/run_milestone3_demo.py
uv run python experiments/run_milestone4_demo.py
uv run python experiments/run_milestone5_demo.py
```

### Run Frozen-VLM & Pilot CLI:
```bash
# 1. Run Preflight verification
uv run python -m src.vlm.cli preflight --manifest tests/fixtures/coco_instances_synthetic.json

# 2. Export portable pilot bundle
uv run python -m src.vlm.cli export-bundle --manifest tests/fixtures/coco_instances_synthetic.json --output-dir data/portable_pilot

# 3. Validate portable pilot bundle
uv run python -m src.vlm.cli validate-bundle --bundle-dir data/portable_pilot

# 4. Run offline synthetic pilot demonstration
uv run python -m src.vlm.cli run-pilot --manifest tests/fixtures/coco_instances_synthetic.json --synthetic --sample-size 3

# 5. Run real-model caption acquisition (in GPU environment with installed dependencies)
uv run python -m src.vlm.cli generate-caption --image-path path/to/image.jpg --config configs/vlm_generation.json
```

---

## 6. Milestone 6: Real Visual Evidence Pipeline & Memory-Safe 4-Bit Inference

The visual evidence extraction pipeline connects genuine COCO training images to real neural models:
$$\text{10 Real COCO Images} \xrightarrow{\text{LLaVA-1.5-7B}} \text{Captions} \xrightarrow{\text{Conservative Extractor}} \text{Claims} \xrightarrow{\text{OWL-ViT + CLIP}} \text{Claim-Level JSONL}$$

### Memory-Safe 4-Bit Execution (Tesla T4 / Google Colab 15 GB)
On GPUs with $\le 16\text{ GB}$ VRAM (e.g., Google Colab Tesla T4), loading the unquantized FP16 checkpoint onto `cuda:0` risks OOM runtime termination during weight allocation. The pipeline provides an optional 4-bit inference mode via `bitsandbytes` NF4 quantization with automatic multi-device dispatch (`device_map="auto"`).

#### Exact Colab Command for T4 4-Bit Run:
```bash
python experiments/run_full_evidence_pipeline.py --load-in-4bit --dtype float16 --allow-download
```

#### Standard FP16 Run (For GPUs with $\ge 24\text{ GB}$ VRAM):
```bash
python experiments/run_full_evidence_pipeline.py --device cuda:0 --dtype float16 --allow-download
```

### Reproducibility & Provenance Tracking
Every response and model record explicitly logs the runtime execution provenance:
- `quantization_enabled`: `True` in 4-bit mode, `False` in FP16 mode.
- `quantization_type`: `"nf4"` (Normalized Float 4).
- `compute_dtype`: `"float16"` (`torch.float16` for matrix multiplication).
- `device_map`: `"auto"` (multi-layer auto-dispatch).
- Exact model checkpoint: `llava-hf/llava-1.5-7b-hf`.
- Greedy decoding: `do_sample=False`, `temperature=0.0`, `max_new_tokens=64`.
- Strict split isolation: 10 images strictly drawn from `TRAIN` split.
- Raw uncalibrated evidence: $d_i \in [0, 1]$, $g_i \in [-1, 1]$. No PGM fitting ($\theta_i$, $\epsilon_i$, $J_{ij}$, posteriors) is introduced at this stage.
