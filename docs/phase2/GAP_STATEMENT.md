# Phase 2 — Novelty position and draft gap statement (v1, 2026-10-26)

Status: draft from the first literature pass (40 papers, each opened at its official page; see `literature_matrix.csv`).
It must be revised after the team reads the assigned papers.

## What the literature already covers

| Idea | Already done by | Implication |
|---|---|---|
| Claim-level guarantees for LVLM outputs (filter claims to reach a factuality level) | ConfLVLM (EMNLP 2025); text precursors Mohri & Hashimoto 2024, Cherian et al. 2024 | "Conformal claim filtering for VLMs" is **not** new. H3 must be compared against it directly. |
| Conformal abstention on COCO existence claims, with re-examining the image under a budget | BCEA (arXiv 2606.16667, 2026) | Our use of "budget" must be clearly distinguished (theirs = compute). |
| Image blur as an uncertainty probe for hallucination | VL-Uncertainty (2024) | Our shift families are not new as a *signal*; what we add must be the *guarantee*. |
| Belief-function / evidential uncertainty for VLM hallucination | Huang et al., IJAR 2025 | Uncertainty-theoretic detection is published, in a venue we listed. |
| Tree graphical model over claims | BTProp (NAACL 2025, text LLMs) | Claim trees are not new; Gate 1 also found no within-image dependence in our data. |
| Exact interval bounds on tree posteriors | 2U (credal polytrees), imprecise Markov trees | Our robust DP is related prior art; cite it and do not claim it as new. |
| Local + global perturbation budgets with certificates | Bojchevski & Günnemann 2019; Zügner & Günnemann 2019; Bertsimas & Sim 2004 | The budgeted set structure is not new. |

## What still appears open (to be confirmed by the team's reading)

1. **Guarantees on the posterior under measured evidence shift.** The existing conformal work guarantees claim correctness under exchangeable data. None of the papers read gives a finite-sample guarantee that a claim's hallucination posterior stays within reported bounds when the visual evidence shifts (corruption, detector change, crop, prompt), using a set calibrated on measured shifts (our T5).
2. **When does structure help?** No paper read gives conditions under which claim coupling changes robust bounds. Our root-dominance lemma and graph-contribution bound answer this, and the Gate 1 data (no measurable coupling) is consistent with them.
3. **Image-level (joint) robustness.** Whether a calibrated global budget gives tighter guarantees for a caption-level quantity (e.g. "this caption contains no hallucination") than independent per-claim sets is untested here (Phase 3/5).

## Draft gap statement (for the introduction; revise after reading)

> Recent work provides statistical guarantees for the factuality of vision-language model outputs under exchangeable data, and separately uses image perturbations as heuristic uncertainty signals. What is missing is a guarantee that a claim-level hallucination assessment remains valid when the visual evidence behind it shifts. We give such a guarantee by calibrating uncertainty sets on measured evidence shifts and propagating them exactly through a claim-level probabilistic model, and we characterise when dependence between claims can tighten the resulting bounds.

## Claims we must not make

- "First conformal / guaranteed hallucination detection for VLMs" (ConfLVLM, BCEA).
- "First uncertainty-theoretic VLM hallucination detection" (Huang et al. 2025).
- "Novel robust inference on trees" or "novel budgeted uncertainty sets" (2U, imprecise Markov trees, Bertsimas & Sim, graph certificates).
- Anything about claim coupling improving results, unless Phase 7 shows it.
