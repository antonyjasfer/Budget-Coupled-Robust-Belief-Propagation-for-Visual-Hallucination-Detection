# Deviations and additions after the prereg-v3 freeze

`prereg-v3.md` is frozen at tag `prereg-v3` and is not edited. As it states, any further
deviation is listed here and in the paper's deviations table, with its date and reason.

| # | Date | Deviation | Reason | Effect |
|---|---|---|---|---|
| D6 | 2026-11-26, before any human label exists | Audit label wording and an error-cause tag. Annotators choose **supported / hallucinated / unclear** (mapped to the audit module's present / absent / ambiguous); the former `synonym_mismatch` label becomes an optional `error_type` tag: `color_or_homonym`, `synonym_miss`, `ambiguous`, `none`. Annotators judge whether the COCO category in the sheet is visible in the photo; when the caption used the word in another sense (e.g. "orange" as a colour) the row is marked hallucinated and tagged. Cohen's kappa is reported over the three labels and, binary, with "unclear" excluded. Rows whose adjudicated tag is `color_or_homonym` or `synonym_miss` are treated as parser artefacts and excluded from the audited re-analysis, not counted as VLM hallucinations. | The pre-registered labels did not separate a wrong word sense from a real hallucination. | The sample (600 claims, seed 2026) is unchanged. The exclusion rule was chosen after the exploratory "orange" sensitivity analysis had been seen, so the audited re-analysis is reported **both with and without** the exclusion. |

## Exploratory additions (not pre-registered)

| # | Date | Analysis | Script |
|---|---|---|---|
| E2 to E6 | 2026-11-26 | Joint calibration + test bootstrap; truncation vs prevalence; clustering and the label budget (1,000 repeats; m = 170, 200, 250); detector score medians and thresholds; operating points of the warning W | `scripts/phase5_exploratory.py` |
| S2 | 2026-11-26 | Frozen analysis re-run without the CHAIR category "orange" | `scripts/phase5_drop_category.py --drop orange` |
| E7 | 2026-11-26 | "Orange" mini-audit: 100 CHAIR-hallucinated "orange" claims from the COCO test split (Qwen3-VL-4B 34, Qwen2.5-VL-3B 33, InternVL3-2B 33; seed 2029; none from the frozen 600), labelled fruit / color_adjective / unclear | `scripts/phase5_orange_audit.py` |
