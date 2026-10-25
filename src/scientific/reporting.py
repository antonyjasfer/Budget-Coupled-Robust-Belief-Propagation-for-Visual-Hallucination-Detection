"""
Milestone 9 Paper-Ready Results Package and Scientific Summary Generator.

Generates:
- reports/m9/final/paper_results/results_section.md
- reports/m9/final/paper_results/discussion_section.md
- reports/m9/final/paper_results/method_summary.md
- reports/m9/final/paper_results/statistical_analysis.md
- reports/m9/final/paper_results/limitations.md
- reports/m9/final/paper_results/reproducibility.md
- reports/m9/final/paper_results/case_studies/case_studies.md
- reports/m9/final/research_summary.md
- reports/m9/final/contribution_statement.md
- reports/m9/final_colab_runbook.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.scientific.consolidator import M9ConsolidatedData

logger = logging.getLogger("m9_reporting")


def generate_m9_research_package(data: M9ConsolidatedData, paper_results_dir: Path) -> None:
    """Generate complete research documentation and paper-ready package."""
    paper_results_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = paper_results_dir / "case_studies"
    cases_dir.mkdir(parents=True, exist_ok=True)

    out_dir = data.output_dir

    # 1. Paper Results: Results Section
    _write_results_section(data, paper_results_dir / "results_section.md")

    # 2. Paper Results: Discussion Section (FACT / INTERPRETATION / LIMITATION)
    _write_discussion_section(data, paper_results_dir / "discussion_section.md")

    # 3. Paper Results: Method Summary
    _write_method_summary(paper_results_dir / "method_summary.md")

    # 4. Paper Results: Statistical Analysis
    _write_statistical_analysis(data, paper_results_dir / "statistical_analysis.md")

    # 5. Paper Results: Limitations
    _write_limitations(paper_results_dir / "limitations.md")

    # 6. Paper Results: Reproducibility
    _write_reproducibility(data, paper_results_dir / "reproducibility.md")

    # 7. Paper Results: Case Studies Markdown
    _write_case_studies(data, cases_dir / "case_studies.md")

    # 8. Research Summary
    _write_research_summary(data, out_dir / "research_summary.md")

    # 9. Contribution Statement
    _write_contribution_statement(out_dir / "contribution_statement.md")

    # 10. Colab Runbook
    _write_colab_runbook(out_dir.parent / "final_colab_runbook.md")


def _fmt(val: Optional[float], decimals: int = 3) -> str:
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Evidence status helpers (Phase 0 integrity rule: no claim without data)
# ---------------------------------------------------------------------------

def _crossing_rate(istats: Dict[str, Any]) -> float:
    """Threshold-crossing rate (L < tau < U); older stats store it as evidence_sensitive_fraction."""
    for key in ("threshold_crossing_rate", "evidence_sensitive_fraction"):
        val = istats.get(key)
        if val is not None:
            return float(val)
    return float("nan")


def _evaluated_claim_count(data: "M9ConsolidatedData") -> int:
    """Largest number of ground-truth-labelled claims evaluated by any method."""
    counts = [
        int(m.get("evaluated_claims", 0) or 0)
        for m in data.metrics.values()
        if isinstance(m, dict)
    ]
    return max(counts) if counts else 0


def _root_dominance_fraction(data: "M9ConsolidatedData") -> Optional[float]:
    """Fraction of claim records with B <= eps_i.

    In that regime the robust interval equals sigma(2(eta +/- B)) exactly
    (root-dominance lemma, docs/theory/root_dominance.md), so the graph,
    the global budget and the DP do not change the interval.
    """
    rows = [c for c in data.claim_results if getattr(c, "budget", None) is not None]
    if not rows:
        return None
    hits = sum(1 for c in rows if float(c.budget) <= float(c.epsilon_i) + 1e-12)
    return hits / len(rows)


def _evidence_banner(data: "M9ConsolidatedData") -> str:
    n_eval = _evaluated_claim_count(data)
    if n_eval == 0:
        return (
            "> **STATUS: DEVELOPMENT ONLY — NO EMPIRICAL FINDINGS.**\n"
            "> 0 ground-truth-labelled claims were evaluated. Nothing in this file is a\n"
            "> research result. Do not quote any number below as a finding.\n"
        )
    return (
        f"> **STATUS: {n_eval} ground-truth-labelled claims evaluated.** Every statement\n"
        "> below is computed from the result files; interpretation is left to the authors.\n"
    )


def _write_results_section(data: M9ConsolidatedData, target_path: Path) -> None:
    m_std = data.metrics.get("standard_bp", {})
    m_rob = data.metrics.get("robust_bp", {})
    m_ev = data.metrics.get("evidence_only", {})
    istats = data.interval_stats

    content = f"""# Results: Empirical Evaluation of Budget-Coupled Robust Belief Propagation

{_evidence_banner(data)}
## 1. Primary Classification Performance

We evaluate Visual Hallucination Detection across three primary formulations:
- **Evidence-Only Baseline**: Independent detector and vision-language association scores.
- **Standard Belief Propagation (Standard BP)**: Exact tree-structured probabilistic inference over interdependent existence claims.
- **Budget-Coupled Robust Belief Propagation (Robust BP)**: Worst-case bounded marginal optimization subject to a global $\\ell_1$ uncertainty budget ($B$).

| Inference Framework | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Evaluated Claims |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Evidence-Only Baseline** | {_fmt(m_ev.get('accuracy'))} | {_fmt(m_ev.get('precision'))} | {_fmt(m_ev.get('recall'))} | {_fmt(m_ev.get('f1'))} | {_fmt(m_ev.get('roc_auc'))} | {m_ev.get('total_claims', 0)} |
| **Standard BP (Point Marginal)** | {_fmt(m_std.get('accuracy'))} | {_fmt(m_std.get('precision'))} | {_fmt(m_std.get('recall'))} | {_fmt(m_std.get('f1'))} | {_fmt(m_std.get('roc_auc'))} | {m_std.get('total_claims', 0)} |
| **Robust BP (Interval Midpoint)** | {_fmt(m_rob.get('accuracy'))} | {_fmt(m_rob.get('precision'))} | {_fmt(m_rob.get('recall'))} | {_fmt(m_rob.get('f1'))} | {_fmt(m_rob.get('roc_auc'))} | {m_rob.get('total_claims', 0)} |

*Note: Binary hallucination decisions use threshold $\\tau = 0.5$. Positive class is 'HALLUCINATED'. UNKNOWN instances are excluded from binary metrics per protocol.*

## 2. Robust Posterior Interval Characterization

Robust BP computes model-based lower and upper marginal bounds $[L_i, U_i]$ for each claim $i$ over the specified uncertainty set (grid solution plus discretisation certificate). These bounds hold only for that set; they are not calibrated probabilities or confidence intervals unless the set itself is calibrated.

- **Mean Interval Width**: {_fmt(istats.get('mean_width'))} ($\\pm$ {_fmt(istats.get('std_width'))})
- **Median Interval Width**: {_fmt(istats.get('median_width'))}
- **Observed Range**: [{_fmt(istats.get('min_width'))}, {_fmt(istats.get('max_width'))}]
- **Evidence-Sensitive Decisions ($\\tau \\in [L_i, U_i]$)**: {sum(1 for c in data.claim_results if getattr(c, 'evidence_sensitive', False))} records ({_crossing_rate(istats) * 100:.1f}%)

## 3. Evidence-Sensitive Abstention and Selective Prediction

When $L_i < \\tau < U_i$, the decision can flip under some perturbation in the uncertainty set. Whether abstaining on these claims improves reliability is an empirical question that must be answered with labelled data and compared against margin abstention on the point posterior.

- **Records in the root-dominance regime ($B \\le \\varepsilon_i$)**: {_fmt(_root_dominance_fraction(data) if _root_dominance_fraction(data) is not None else None, 3)} of all records. For these records the interval equals $\\sigma(2(\\eta_i \\pm B))$ exactly, so it carries no information beyond the point posterior.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_discussion_section(data: M9ConsolidatedData, target_path: Path) -> None:
    istats = data.interval_stats
    rd = _root_dominance_fraction(data)
    content = f"""# Discussion: Interpretation, Evidence, and Boundaries

{_evidence_banner(data)}
This file separates mathematical facts, measured quantities, and interpretations. Interpretations are written by the authors only after labelled results exist; this generator never writes them.

---

### [FACT: Mathematical Properties (hold by construction)]
1. $0 \\le L_i \\le U_i \\le 1$ and $w_i = U_i - L_i \\ge 0$.
2. With $B = 0$ the interval collapses to the point marginal; widths are non-decreasing in $B$ because the uncertainty sets are nested.
3. Root dominance: if $B \\le \\varepsilon_i$, then $[L_i, U_i] = [\\sigma(2(\\eta_i - B)), \\sigma(2(\\eta_i + B))]$ exactly (see docs/theory/root_dominance.md).
4. The grid DP costs $\\mathcal{{O}}(n K^2)$ per target node.

### [FACT: Measured in this run]
- Evaluated (ground-truth-labelled) claims: {_evaluated_claim_count(data)}
- Mean / median interval width: {_fmt(istats.get('mean_width'))} / {_fmt(istats.get('median_width'))}
- Threshold-crossing rate ($L_i < \\tau < U_i$): {_crossing_rate(istats) * 100:.1f}%
- Fraction of records in the root-dominance regime: {_fmt(rd)}

---

### [INTERPRETATION: Pending]
No interpretation is generated automatically. Questions to answer with labelled data, each against its baseline:
1. Does abstaining when $L_i < \\tau < U_i$ beat margin abstention on the point posterior at equal coverage?
2. Does width track real evidence shift (corruption, detector change) beyond what the point posterior already explains?
3. Does claim coupling ($J \\neq 0$) improve detection on images with 3 or more claims?

---

### [LIMITATION: Structural Boundaries and Unknowns]
1. **Tree Topology Assumption**: Exact inference requires a tree; cyclic dependencies must be pruned or approximated.
2. **Discretisation**: Grid bounds are within the certificate gap $G = \\min(B, \\sum_i \\gamma_i \\min(\\varepsilon_i, \\Delta))$ of the continuous bounds; report $G$ with every interval.
3. **Uncalibrated Uncertainty Set**: Unless $\\varepsilon$ and $B$ are calibrated to measured evidence shifts, the interval has no coverage meaning.
4. **Upstream Evidence**: Detector and CLIP biases bound the quality of the unary potentials; robust inference cannot recover evidence that is missing.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_method_summary(target_path: Path) -> None:
    content = r"""# Method Summary: Budget-Coupled Robust Belief Propagation

## 1. Problem Formulation
Given an image $I$ and candidate visual existence claims $\mathcal{H} = \{H_1, \dots, H_n\}$ extracted from a Vision-Language Model (VLM) generation, we represent claim existence as binary random variables $H_i \in \{-1, +1\}$, where $+1$ denotes a hallucination (object absent in $I$) and $-1$ denotes supported (object verified).

The joint distribution is defined over a factor graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
$$P(H \mid E) \propto \prod_{i \in \mathcal{V}} \psi_i(H_i, E_i) \prod_{(i, j) \in \mathcal{E}} \psi_{ij}(H_i, H_j)$$

where:
- $\psi_i(H_i, E_i) = \exp(\theta_i H_i)$ denotes unary evidence potentials from object detectors and CLIP similarity.
- $\psi_{ij}(H_i, H_j) = \exp(J_{ij} H_i H_j)$ denotes pairwise coupling potentials encoding spatial or semantic relations.

## 2. Global $\ell_1$ Uncertainty Budget
To account for sensor noise, visual degradation, and calibration inaccuracies, we model perturbations to the unary evidence potentials:
$$\tilde{\theta}_i = \theta_i + \delta_i, \quad \text{subject to } |\delta_i| \le \epsilon_i \text{ and } \sum_{i=1}^n |\delta_i| \le B$$

Here, $B \ge 0$ enforces a **global uncertainty budget** across the entire claim graph, coupling local potential variations.

## 3. Robust Posterior Intervals
Rather than evaluating a single point posterior, we compute the lower and upper marginal hallucination probability over the uncertainty set (model-based bounds; their real-world meaning depends on how $\epsilon$ and $B$ are calibrated):
$$L_i = \min_{\|\delta\|_1 \le B, |\delta_k| \le \epsilon_k} P_{\theta + \delta}(H_i = +1 \mid E)$$
$$U_i = \max_{\|\delta\|_1 \le B, |\delta_k| \le \epsilon_k} P_{\theta + \delta}(H_i = +1 \mid E)$$

The numerical solver discretizes the allowable budget allocation via dynamic programming over the tree structure with resolution $K$, yielding polynomial complexity $\mathcal{O}(n K^2)$.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_statistical_analysis(data: M9ConsolidatedData, target_path: Path) -> None:
    bcomp = data.bootstrap_comparison
    corrs = data.correlations

    acc_diff = bcomp.get("diff_accuracy", {})
    f1_diff = bcomp.get("diff_f1", {})
    roc_diff = bcomp.get("diff_roc_auc", {})

    content = f"""# Statistical Analysis: Hypothesis Testing and Bootstrap Inference

## 1. Paired Image-Level Cluster Bootstrap (Robust BP vs. Standard BP)

To account for within-image claim dependencies, we apply paired image-level cluster bootstrap resampling ($n = 1000$ iterations). Differences are reported as $\\Delta = \\text{{Robust BP}} - \\text{{Standard BP}}$.

| Metric | Point Estimate ($\\Delta$) | 95% Bootstrap Confidence Interval | Significance ($\\alpha = 0.05$) |
| :--- | :---: | :---: | :---: |
| **Accuracy Difference** | {_fmt(acc_diff.get('mean'))} | [{_fmt(acc_diff.get('ci_lower'))}, {_fmt(acc_diff.get('ci_upper'))}] | {'Yes' if acc_diff.get('p_value', 1.0) < 0.05 else 'No (overlapping zero)'} |
| **F1 Score Difference** | {_fmt(f1_diff.get('mean'))} | [{_fmt(f1_diff.get('ci_lower'))}, {_fmt(f1_diff.get('ci_upper'))}] | {'Yes' if f1_diff.get('p_value', 1.0) < 0.05 else 'No (overlapping zero)'} |
| **ROC-AUC Difference** | {_fmt(roc_diff.get('mean'))} | [{_fmt(roc_diff.get('ci_lower'))}, {_fmt(roc_diff.get('ci_upper'))}] | {'Yes' if roc_diff.get('p_value', 1.0) < 0.05 else 'No (overlapping zero)'} |
"""

    # 2. Correlation Hypotheses on Interval Width
    c_err = corrs.get('width_vs_standard_bp_error', corrs.get('width_vs_error', {}))
    c_conf = corrs.get('width_vs_detector_clip_conflict', corrs.get('width_vs_conflict', {}))
    c_corr = corrs.get('width_vs_corruption_severity', corrs.get('width_vs_corruption', {}))

    content += f"""
## 2. Correlation Hypotheses on Interval Width

We evaluate whether robust interval width $w_i = U_i - L_i$ contains structured diagnostic information:

- **Hypothesis 1 (Width vs. Standard BP Error)**:
  - Spearman correlation: {_fmt(c_err.get('spearman_rho'))} (p = {_fmt(c_err.get('spearman_p'))})
  - Point-biserial correlation: {_fmt(c_err.get('pearson_r'))}
- **Hypothesis 2 (Width vs. Multimodal Evidence Conflict)**:
  - Pearson correlation: {_fmt(c_conf.get('pearson_r'))} (p = {_fmt(c_conf.get('pearson_p'))})
- **Hypothesis 3 (Width vs. Visual Corruption Severity)**:
  - Spearman rank correlation: {_fmt(c_corr.get('spearman_rho'))} (p = {_fmt(c_corr.get('spearman_p'))})
"""
    target_path.write_text(content, encoding="utf-8")


def _write_limitations(target_path: Path) -> None:
    content = r"""# Limitations and Scope of Validity

1. **Tree Topology Assumption**: Exact tree-structured message passing requires acyclic claim factor graphs. Cyclic dependency structures must be approximated or clustered.
2. **Grid Discretization**: Continuous worst-case budgets are solved on a grid with $K$ steps. The continuous bound lies within the certificate gap $G = \min(B, \sum_i \gamma_i \min(\epsilon_i, \Delta))$ of the grid value.
3. **Multimodal Model Dependence**: Unary potentials depend on frozen upstream models (currently OWL-ViT and CLIP; note OWL-ViT is built on a CLIP backbone, so the two signals are correlated). Systematic blind spots in these models cannot be removed by inference.
6. **Root-Dominance Regime**: When $B \le \epsilon_i$, the robust interval is a closed-form function of the point posterior, and the graph and budget have no effect on it.
7. **Uncalibrated Uncertainty Set**: Until $\epsilon$ and $B$ are calibrated to measured evidence shifts, intervals are model-based sensitivity bounds without coverage meaning.
4. **Sample Size & Split Independence**: Final benchmark generalization relies on zero split leakage between calibration and test partitions.
5. **UNKNOWN Annotations**: Ambiguous claims labeled UNKNOWN are properly excluded from binary accuracy metrics to preserve scientific integrity.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_reproducibility(data: M9ConsolidatedData, target_path: Path) -> None:
    m = data.manifest
    content = f"""# Reproducibility Protocol and Experiment Provenance

- **Git Commit SHA**: `{m.get('git', {}).get('commit_sha')}`
- **Working Tree Clean**: `{m.get('git', {}).get('is_clean')}`
- **Execution Mode**: `{m.get('execution_mode')}`
- **Timestamp**: `{m.get('timestamp')}`
- **Random Seed**: `{m.get('config', {}).get('random_seed')}`
- **Python Version**: `{m.get('environment', {}).get('python_version', '').split()[0]}`
- **Platform**: `{m.get('environment', {}).get('platform')}`
- **Total Evaluated Records**: `{m.get('dataset_summary', {}).get('total_records')}`
- **Unique Images**: `{m.get('dataset_summary', {}).get('unique_images')}`
- **Unique Claims**: `{m.get('dataset_summary', {}).get('unique_claims')}`
- **Deterministic Result SHA256**: `{m.get('result_checksum')}`

All experiments were executed with deterministic pseudo-random seeds and cached evidence extraction.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_case_studies(data: M9ConsolidatedData, target_path: Path) -> None:
    lines = [
        "# Milestone 9 Representative Case Studies\n",
        "Deterministic case studies illustrating key operational regimes of Robust BP:\n",
    ]
    for case in data.case_studies:
        cid = case.get('category_id') or case.get('case_id', 'CASE')
        cname = case.get('category_name') or case.get('description', '')
        lines.append(f"## {cid}: {cname}")
        lines.append(f"- **Image ID**: `{case.get('image_id')}`")
        lines.append(f"- **Claim ID**: `{case.get('claim_id')}`")
        lines.append(f"- **Claim Text**: \"{case.get('text_span') or case.get('claim_text', '')}\"")
        lines.append(f"- **Ground Truth**: `{case.get('ground_truth')}`")
        lines.append(f"- **Detector Score**: {_fmt(case.get('detector_score') or case.get('detector_prob'))}")
        lines.append(f"- **CLIP Score**: {_fmt(case.get('clip_score') or case.get('clip_prob'))}")
        lines.append(f"- **Standard BP Posterior**: {_fmt(case.get('standard_posterior') or case.get('standard_bp_prob'))}")
        lines.append(f"- **Robust Interval**: [{_fmt(case.get('robust_lower'))}, {_fmt(case.get('robust_upper'))}] (Width: {_fmt(case.get('interval_width') or case.get('robust_width'))})")
        lines.append(f"- **Decision Regime**: `{case.get('explanation') or case.get('condition')}`\n")
    target_path.write_text("\n".join(lines), encoding="utf-8")


def _write_research_summary(data: M9ConsolidatedData, target_path: Path) -> None:
    m_std = data.metrics.get("standard_bp", {})
    m_rob = data.metrics.get("robust_bp", {})
    istats = data.interval_stats
    n_eval = _evaluated_claim_count(data)
    ds = data.manifest.get("dataset_summary", {}) if isinstance(data.manifest, dict) else {}
    rd = _root_dominance_fraction(data)
    pending = "Not available: no ground-truth-labelled claims in this run." if n_eval == 0 else None

    content = f"""# Research Summary: Budget-Coupled Robust Belief Propagation for Visual Hallucination Detection

{_evidence_banner(data)}
### 1. Research Problem
Large vision-language models generate object claims that the image does not support. We study how stable a claim-level hallucination decision is when the visual evidence behind it shifts.

### 2. Method (as implemented)
- Claims form a tree-structured Ising model; unary fields $\\theta_i$ come from detector (and optionally CLIP) evidence.
- Uncertainty set: $|\\delta_i| \\le \\varepsilon_i$ and $\\sum_i |\\delta_i| \\le B$.
- Output: model-based bounds $[L_i, U_i]$ over that set, computed by an exact grid DP with a discretisation certificate.

### 3. Data in this run
- Records: {ds.get('total_records', 'N/A')} · unique images: {ds.get('unique_images', 'N/A')} · unique claims: {ds.get('unique_claims', 'N/A')}
- Ground-truth-labelled claims evaluated: {n_eval}

### 4. Main Results
{pending or f"- Standard BP accuracy {_fmt(m_std.get('accuracy'))}, F1 {_fmt(m_std.get('f1'))}, ROC-AUC {_fmt(m_std.get('roc_auc'))}" + chr(10) + f"- Robust BP accuracy {_fmt(m_rob.get('accuracy'))}, F1 {_fmt(m_rob.get('f1'))}, ROC-AUC {_fmt(m_rob.get('roc_auc'))}"}

### 5. Interval Statistics (descriptive only)
- Mean / median width: {_fmt(istats.get('mean_width'))} / {_fmt(istats.get('median_width'))}
- Threshold-crossing rate: {_crossing_rate(istats) * 100:.1f}%
- Records in the root-dominance regime ($B \\le \\varepsilon_i$, interval = $\\sigma(2(\\eta \\pm B))$): {_fmt(rd)}

### 6. Ablations, Corruption Study, Statistical Tests
{pending or "See final_ablation_results.json, final_corruption_results.json and final_statistics.json. Conclusions are written by the authors, not by this generator."}

### 7. Known Limitations
Tree topology; grid resolution (certificate gap reported per interval); uncalibrated $\\varepsilon$ and $B$; correlated evidence sources (OWL-ViT uses a CLIP backbone); root-dominance regime when $B \\le \\varepsilon_i$.

### 8. Conclusion
No scientific conclusion is generated automatically. Conclusions require labelled results, the pre-registered tests in docs/PREREG.md, and author review.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_contribution_statement(target_path: Path) -> None:
    content = """# Contribution Statement (DRAFT — candidate contributions, not yet supported by results)

Status of each candidate contribution. Only items marked ESTABLISHED may be claimed in a paper today.

1. **Exact budget DP on trees** — ESTABLISHED (mathematical). An $O(nK^2)$ dynamic program computes exact bounds over the discretised budget set on attractive trees, with a continuous certificate gap. Related prior work (credal networks, graph certification with local and global budgets) must be cited.
2. **Root-dominance lemma** — ESTABLISHED (mathematical, numerically verified). If $B \\le \\varepsilon_r$, the interval is $\\sigma(2(\\eta_r \\pm B))$; in general the graph adds at most $\\gamma_{\\max}(B - \\varepsilon_r)_+$ to the target field.
3. **Calibrated global budget uncertainty set with a coverage guarantee** — PLANNED (Phase 3/5). Requires the conformal construction and measured evidence shifts.
4. **Claim-graph CRF with learned couplings for hallucination detection** — PLANNED (Phase 4). Requires labelled data and comparison with J = 0.
5. **Empirical utility of robust intervals for selective prediction** — NOT YET SUPPORTED. Requires labelled results against margin-abstention and conformal-filter baselines.

Claims we must not make: "first", "guaranteed" or "certified" without stating the uncertainty set and assumptions, "multimodal" unless several evidence types are actually used in the reported run.
"""
    target_path.write_text(content, encoding="utf-8")


def _write_colab_runbook(target_path: Path) -> None:
    content = """# Milestone 9 Google Colab Final Execution Runbook

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
"""
    target_path.write_text(content, encoding="utf-8")
