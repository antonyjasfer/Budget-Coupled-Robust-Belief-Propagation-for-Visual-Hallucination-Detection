# Decisions Log

One row per research decision, newest first. Record what was decided, why, and who decided. Never delete a row; add a new one that supersedes it.

| Date | Decision | Reason | Who |
|---|---|---|---|
| 2026-10-25 | Phase 1 adds a direct H2 measure: mean interval width under the calibrated combined set vs the box set (reported for both grid and certified widths). | By root dominance, when calibrated B > ε the target's own radius is unchanged, so a smaller budget-vs-box total does not by itself mean narrower intervals; with J = 0 the width ratio is exactly 1. H2 therefore depends on claim coupling. | Phase 1 |
| 2026-10-25 | Phase 1 pilot: 200 images drawn only from the PREREG train split (100 pilot_cal / 100 pilot_eval). PREREG splits materialised over all COCO val2014 ids, seed 42. | Keeps the locked test split untouched while testing Gate 1. | Phase 1 |
| 2026-10-25 | CHAIR labels reimplemented from the official code; deviations: regex tokeniser instead of nltk, `inflect` singulariser instead of Python-2 `pattern.en`. Synonyms file downloaded at run time (not vendored: no upstream licence) and pinned by SHA-256. | Python-3 compatibility; reproducibility. Report the deviations in the paper. | Phase 1 |
| 2026-10-25 | **Engineering freeze.** No new infrastructure (state machines, preflight gates, packaging layers) unless an experiment is blocked without it. | Software is ahead of the science: 0 labelled claims evaluated so far. Time goes to Phases 1–7. | Proposed in Phase 0; team to confirm |
| 2026-10-25 | Auto-generated reports may not state findings. With 0 labelled claims they carry a DEVELOPMENT ONLY banner; conclusions are written by authors. | `research_summary.md` previously stated corruption and bootstrap "findings" that had no data behind them. | Phase 0 |
| 2026-10-25 | "No global budget" is represented as an inactive budget (B = Σ ε_i), never B = 0. B = 0 means zero perturbation only. | The independent-box ablation was recorded as B = 0, which contradicted the B = 0 condition. | Phase 0 |
| 2026-10-25 | Ablation rows report the nominal posterior of the same model the interval comes from, and a threshold decision on it. | `ablation_j0` stored the interval midpoint and robust decision in the "standard" columns. | Phase 0 |
| 2026-10-25 | Theorem 3 restated for the supremum only. | The previous "for any δ" statement is false. | Phase 0 |
| 2026-10-25 | Root-dominance lemma adopted; the default B = ε = J = atanh(0.5) configuration is degenerate (interval = σ(2(η ± B))). Margin abstention becomes a mandatory baseline. | Proved in `docs/theory/root_dominance.md`; matches all 19 B ≤ ε conditions in the M9 dev results to ~1e-16. | Phase 0 |
| 2026-10-25 | Research direction: calibrated (conformal) uncertainty sets with a coverage guarantee; learned tree CRF; COCO-CHAIR + AMBER + POPE; 2–3 VLMs. | Q1 research plan, Phases 1–9. | Team |
