"""Empirical edge validity audit and semantic vs label dependence analysis.

Analyzes candidate claim pairs within the same image on TRAIN and VALIDATION splits only.
Tests whether semantic similarity s_ij predicts empirical hallucination state dependence
or if edges merely serve as structural heuristics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr

from src.calibration.splits import SplitLeakageError, SplitRole


@dataclass
class PairwiseDependenceMetrics:
    """Statistical summary of empirical truth-state association between claim pairs."""

    n_pairs: int
    n_concordant: int  # (0, 0) or (1, 1)
    n_discordant: int  # (0, 1) or (1, 0)
    agreement_rate: float
    phi_coefficient: float
    odds_ratio: float
    mutual_information: float
    evidence_category: str  # "positive dependence observed", "little/no evidence", "insufficient samples", "contradictory tendency"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EdgeValidityAuditReport:
    """Comprehensive edge audit comparing semantic similarity against empirical state dependence."""

    split_evaluated: str
    total_candidate_edges: int
    evaluated_edges_count: int
    pairwise_metrics: PairwiseDependenceMetrics
    spearman_rho: float
    spearman_pvalue: float
    semantic_predictive_conclusion: str
    edges_detail: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_pairwise_state_dependence(
    candidate_pairs: List[Tuple[int, int, float]],
    labels: List[int],
    min_samples: int = 5,
) -> PairwiseDependenceMetrics:
    """Compute empirical agreement, phi coefficient, odds ratio, and mutual information.

    Args:
        candidate_pairs: List of (node_i, node_j, similarity_s).
        labels: Binary labels for nodes (0 = SUPPORTED, 1 = HALLUCINATED).
        min_samples: Minimum required pairs for statistical reliability.
    """
    n_pairs = len(candidate_pairs)
    if n_pairs < min_samples:
        return PairwiseDependenceMetrics(
            n_pairs=n_pairs,
            n_concordant=0,
            n_discordant=0,
            agreement_rate=0.5,
            phi_coefficient=0.0,
            odds_ratio=1.0,
            mutual_information=0.0,
            evidence_category="insufficient samples",
        )

    # Contingency table: n_00, n_01, n_10, n_11
    n_00 = n_01 = n_10 = n_11 = 0
    for u, v, _ in candidate_pairs:
        y_u = labels[u]
        y_v = labels[v]
        if y_u == 0 and y_v == 0:
            n_00 += 1
        elif y_u == 0 and y_v == 1:
            n_01 += 1
        elif y_u == 1 and y_v == 0:
            n_10 += 1
        else:
            n_11 += 1

    n_concordant = n_00 + n_11
    n_discordant = n_01 + n_10
    agreement_rate = float(n_concordant / n_pairs) if n_pairs > 0 else 0.5

    # Phi coefficient = (n_11 * n_00 - n_10 * n_01) / sqrt((n_11+n_10)(n_01+n_00)(n_11+n_01)(n_10+n_00))
    n_dot_1 = n_11 + n_01
    n_dot_0 = n_10 + n_00
    n_1_dot = n_11 + n_10
    n_0_dot = n_01 + n_00
    denom = float(np.sqrt(float(n_dot_1 * n_dot_0 * n_1_dot * n_0_dot)))

    if denom > 1e-9:
        phi = float((n_11 * n_00 - n_10 * n_01) / denom)
    else:
        phi = 0.0

    # Odds ratio with Haldane-Anscombe 0.5 correction
    odds_ratio = float(((n_11 + 0.5) * (n_00 + 0.5)) / ((n_10 + 0.5) * (n_01 + 0.5)))

    # Mutual information
    mi = 0.0
    for n_xy, p_x, p_y in [
        (n_00, n_0_dot / n_pairs, n_dot_0 / n_pairs),
        (n_01, n_0_dot / n_pairs, n_dot_1 / n_pairs),
        (n_10, n_1_dot / n_pairs, n_dot_0 / n_pairs),
        (n_11, n_1_dot / n_pairs, n_dot_1 / n_pairs),
    ]:
        if n_xy > 0 and p_x > 0 and p_y > 0:
            p_xy = n_xy / n_pairs
            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    # Evidence category classification
    if phi >= 0.20 and agreement_rate >= 0.60:
        ev_cat = "positive dependence observed"
    elif phi <= -0.15:
        ev_cat = "contradictory tendency"
    elif abs(phi) < 0.20:
        ev_cat = "little/no evidence"
    else:
        ev_cat = "little/no evidence"

    return PairwiseDependenceMetrics(
        n_pairs=n_pairs,
        n_concordant=n_concordant,
        n_discordant=n_discordant,
        agreement_rate=agreement_rate,
        phi_coefficient=phi,
        odds_ratio=odds_ratio,
        mutual_information=float(max(0.0, mi)),
        evidence_category=ev_cat,
    )


def audit_edge_validity(
    records: List[Dict[str, Any]],
    candidate_edges_by_image: Dict[str, List[Tuple[int, int, float]]],
    split_role: Union[str, SplitRole] = SplitRole.TRAIN,
    min_samples: int = 5,
) -> EdgeValidityAuditReport:
    """Perform audit of candidate edges using TRAIN or VALIDATION records only.

    CRITICAL LEAKAGE GUARD: TEST records are strictly forbidden.
    """
    role_str = split_role.value if isinstance(split_role, SplitRole) else str(split_role).upper()
    if role_str == "TEST":
        raise SplitLeakageError("Edge validity audit cannot use TEST data! Use TRAIN or VALIDATION only.")

    # Group records by image_id
    img_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        sp = str(r.get("split", "train")).upper()
        if sp == "TEST":
            raise SplitLeakageError("Encountered TEST record during edge validity audit!")
        img_id = str(r.get("image_id", "default"))
        img_groups.setdefault(img_id, []).append(r)

    all_pairs: List[Tuple[int, int, float]] = []
    all_pair_labels: List[Tuple[int, int]] = []
    semantic_scores: List[float] = []
    pair_agreements: List[float] = []
    edge_details: List[Dict[str, Any]] = []

    for img_id, img_recs in img_groups.items():
        edges = candidate_edges_by_image.get(img_id, [])
        n_nodes = len(img_recs)
        labels = [int(r.get("label", 0)) for r in img_recs]

        for u, v, s in edges:
            if u < n_nodes and v < n_nodes:
                all_pairs.append((len(semantic_scores), len(semantic_scores), s))
                y_u = labels[u]
                y_v = labels[v]
                agree = 1.0 if y_u == y_v else 0.0
                semantic_scores.append(float(s))
                pair_agreements.append(agree)
                edge_details.append({
                    "image_id": img_id,
                    "claim_u": img_recs[u].get("claim_id", f"c_{u}"),
                    "claim_v": img_recs[v].get("claim_id", f"c_{v}"),
                    "semantic_similarity": float(s),
                    "label_u": y_u,
                    "label_v": y_v,
                    "concordant": bool(agree == 1.0),
                })

    # Global pairwise metrics
    pair_labels_flat = []
    pair_nodes = []
    for idx, d in enumerate(edge_details):
        pair_nodes.append((2 * idx, 2 * idx + 1, d["semantic_similarity"]))
        pair_labels_flat.extend([d["label_u"], d["label_v"]])

    pairwise_metrics = compute_pairwise_state_dependence(
        pair_nodes, pair_labels_flat, min_samples=min_samples
    )

    # Spearman rank correlation between semantic similarity and agreement
    if len(semantic_scores) >= min_samples and len(set(semantic_scores)) > 1:
        corr, pval = spearmanr(semantic_scores, pair_agreements)
        spearman_rho = float(corr) if not np.isnan(corr) else 0.0
        spearman_pval = float(pval) if not np.isnan(pval) else 1.0
    else:
        spearman_rho = 0.0
        spearman_pval = 1.0

    if pairwise_metrics.evidence_category == "insufficient samples":
        conclusion = "Sample size too small to evaluate semantic-to-label correlation; treated as structural heuristic."
    elif spearman_rho > 0.30 and spearman_pval < 0.05:
        conclusion = "Moderate positive correlation: higher semantic similarity significantly predicts state agreement."
    else:
        conclusion = (
            "Weak or negligible correlation: semantic similarity functions primarily as an intuitive "
            "structural heuristic rather than an empirical proxy for truth-state correlation."
        )

    return EdgeValidityAuditReport(
        split_evaluated=role_str,
        total_candidate_edges=len(semantic_scores),
        evaluated_edges_count=len(semantic_scores),
        pairwise_metrics=pairwise_metrics,
        spearman_rho=spearman_rho,
        spearman_pvalue=spearman_pval,
        semantic_predictive_conclusion=conclusion,
        edges_detail=edge_details,
    )


def generate_edge_validity_report(
    report: EdgeValidityAuditReport,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Generate Markdown report for reports/m9/edge_validity_audit.md."""
    lines = [
        "# Empirical Edge Validity and Semantic Correlation Audit",
        "",
        "> [!WARNING]",
        "> **DEVELOPMENT ONLY — DO NOT TREAT AS FINAL**: Pipeline verification on development data.",
        "> Sample size is insufficient for final scientific conclusions.",
        "",
        f"**Split Evaluated**: `{report.split_evaluated}`  ",
        f"**Total Candidate Edges Evaluated**: {report.evaluated_edges_count}  ",
        f"**Statistical Evidence Category**: `{report.pairwise_metrics.evidence_category}`  ",
        "",
        "## 1. Pairwise Truth-State Dependence Metrics",
        "",
        f"- **Evaluated Pairs ($n_{{\\text{{pair}}}}$)**: {report.pairwise_metrics.n_pairs}",
        f"- **Concordant Pairs ($y_i = y_j$)**: {report.pairwise_metrics.n_concordant}",
        f"- **Discordant Pairs ($y_i \\ne y_j$)**: {report.pairwise_metrics.n_discordant}",
        f"- **Empirical Agreement Rate**: {report.pairwise_metrics.agreement_rate:.3f}",
        f"- **Phi Coefficient ($\\phi$)**: {report.pairwise_metrics.phi_coefficient:.3f}",
        f"- **Odds Ratio**: {report.pairwise_metrics.odds_ratio:.3f}",
        f"- **Mutual Information ($I$)**: {report.pairwise_metrics.mutual_information:.4f} bits",
        "",
        "## 2. Semantic Similarity vs. Empirical Agreement Correlation",
        "",
        f"- **Spearman Rank Correlation ($\\rho$)**: {report.spearman_rho:.3f}",
        f"- **P-Value**: {report.spearman_pvalue:.4f}",
        f"- **Scientific Conclusion**: {report.semantic_predictive_conclusion}",
        "",
        "## 3. Methodological Implications",
        "",
        "> [!IMPORTANT]",
        "> Semantic similarity $s_{ij} \\in [0, 1]$ represents conceptual/contextual proximity between claims.",
        "> It DOES NOT automatically establish that hallucination truth-states are positively correlated.",
        "> If empirical correlation is weak, edge weights must be acknowledged as a structural heuristic",
        "> rather than a proven causal or statistical law.",
        "",
        "## 4. Evaluated Edge Details (Sample)",
        "",
        "| Image ID | Claim $u$ | Claim $v$ | Semantic Similarity $s_{uv}$ | $y_u$ | $y_v$ | Concordant |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for d in report.edges_detail[:15]:
        c_str = "YES" if d["concordant"] else "NO"
        lines.append(
            f"| `{d['image_id']}` | `{d['claim_u']}` | `{d['claim_v']}` | {d['semantic_similarity']:.3f} | "
            f"{d['label_u']} | {d['label_v']} | {c_str} |"
        )

    content = "\n".join(lines)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

    return content
