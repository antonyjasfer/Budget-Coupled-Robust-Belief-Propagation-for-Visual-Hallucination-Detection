# Contribution Statement (DRAFT — candidate contributions, not yet supported by results)

Status of each candidate contribution. Only items marked ESTABLISHED may be claimed in a paper today.

1. **Exact budget DP on trees** — ESTABLISHED (mathematical). An $O(nK^2)$ dynamic program computes exact bounds over the discretised budget set on attractive trees, with a continuous certificate gap. Related prior work (credal networks, graph certification with local and global budgets) must be cited.
2. **Root-dominance lemma** — ESTABLISHED (mathematical, numerically verified). If $B \le \varepsilon_r$, the interval is $\sigma(2(\eta_r \pm B))$; in general the graph adds at most $\gamma_{\max}(B - \varepsilon_r)_+$ to the target field.
3. **Calibrated global budget uncertainty set with a coverage guarantee** — PLANNED (Phase 3/5). Requires the conformal construction and measured evidence shifts.
4. **Claim-graph CRF with learned couplings for hallucination detection** — PLANNED (Phase 4). Requires labelled data and comparison with J = 0.
5. **Empirical utility of robust intervals for selective prediction** — NOT YET SUPPORTED. Requires labelled results against margin-abstention and conformal-filter baselines.

Claims we must not make: "first", "guaranteed" or "certified" without stating the uncertainty set and assumptions, "multimodal" unless several evidence types are actually used in the reported run.
