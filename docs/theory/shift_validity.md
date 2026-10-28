# Proposition P1: when a clean conformal hallucination guarantee survives a shift

**Status:** proved below (PREREG v2, Section 5). Tested empirically as H1/H2 (`src/phase3/shift_validity.py`; confirmatory analysis in Phase 4). Added 2026-10-28.

## 1. Setting

A claim has an evidence score $S \in \mathbb{R}$ (logit of the detector score; larger = more evidence that the object is present) and a label $Y \in \{0, 1\}$ ($Y = 1$: hallucinated). Calibration uses $n$ images drawn from the clean source distribution $P$.

**Claim-level filter (CEBC-style).** Let $S_1, \dots, S_m$ be the clean scores of the hallucinated calibration claims and
$$\hat\tau = S_{(k)}, \qquad k = \lceil (m+1)(1-\delta) \rceil \quad (\hat\tau = +\infty \text{ if } k > m).$$
A claim is kept iff its score exceeds $\hat\tau$. The **risk** is $R = \Pr(\text{a hallucinated test claim is kept}) = \Pr(S_{\text{test}} > \hat\tau \mid Y_{\text{test}} = 1)$.

**Image-level filter (ConfLVLM-style, $\lambda = 0$).** For image $j$ let $M_j = \max\{S : \text{hallucinated claims of image } j\}$ ($-\infty$ if none), $\hat\tau = M_{(k)}$ with $k = \lceil (n+1)(1-\alpha) \rceil$. The risk is $\Pr(M_{\text{test}} > \hat\tau)$, the probability that a test image keeps at least one hallucinated claim.

**Clean guarantee.** If the test claim (image) is exchangeable with the calibration claims (images), the standard conformal argument gives $\Pr(S_{\text{test}} > \hat\tau) \le \delta$ (resp. $\Pr(M_{\text{test}} > \hat\tau) \le \alpha$).

**Shift.** At test time the image (or detector, VLM, domain) changes, so the hallucinated test claim has score $\tilde S$ instead of $S$ (image level: $\tilde M$ instead of $M$). Calibration data are unchanged, so $\hat\tau$ is independent of the shifted test point.

## 2. Proposition P1

**(a) Claim level.** If the shifted score of a hallucinated claim is stochastically no larger than the clean score,
$$\Pr(\tilde S > t) \le \Pr(S > t) \quad \text{for all } t \in \mathbb{R}, \tag{D}$$
where both probabilities are for a hallucinated test claim, then the clean-calibrated filter keeps its guarantee under the shift:
$$\Pr(\tilde S > \hat\tau) \le \Pr(S > \hat\tau) \le \delta .$$

**(b) Image level.** If $\Pr(\tilde M > t) \le \Pr(M > t)$ for all $t$, then $\Pr(\tilde M > \hat\tau) \le \alpha$.

**(c) Sufficient pointwise condition.** If the shift never raises the score of a hallucinated claim ($\tilde S \le S$ almost surely, for the same claim on the same image), then (D) holds, and so does the image-level condition, because the maximum of pointwise smaller numbers is smaller.

## 3. Proof

(a) The test point is independent of the calibration sample, hence of $\hat\tau$. Condition on $\hat\tau = t$:
$$\Pr(\tilde S > \hat\tau) = \mathbb{E}\big[\Pr(\tilde S > t)\big|_{t = \hat\tau}\big] \le \mathbb{E}\big[\Pr(S > t)\big|_{t = \hat\tau}\big] = \Pr(S > \hat\tau) \le \delta,$$
where the inequality is (D) applied for each value $t$, and the last step is the clean guarantee (the clean test score $S$ is exchangeable with the calibration scores). If $\hat\tau = +\infty$, both sides are 0.

(b) The same argument with $M, \tilde M, \alpha$.

(c) If $\tilde S \le S$ a.s., then $\{\tilde S > t\} \subseteq \{S > t\}$ for every $t$, which gives (D). For images, $\tilde M = \max_i \tilde S_i \le \max_i S_i = M$. $\blacksquare$

## 4. Remarks

1. **No condition on supported claims.** Both guaranteed quantities are false-acceptance probabilities for hallucinated claims. How the shift changes the scores of supported claims does not enter.
2. **Precision is not covered.** The share of kept claims that are hallucinated can *increase* under a shift satisfying (D), if supported claims lose more score than hallucinated ones. Example: a shift that lowers supported scores by 3 logits and hallucinated scores by 0.1 keeps the risk below δ but can leave almost only hallucinated claims among the few that are kept. PREREG v2 therefore reports precision as a secondary, unguaranteed metric, together with the share of supported claims kept (a filter that keeps nothing is trivially "safe").
3. **Only the region of $\hat\tau$ matters.** The proof uses (D) only at values $t$ that $\hat\tau$ can take. Condition (D) on an interval containing the support of $\hat\tau$ is enough. This is why PREREG v2 also reports the difference $\Pr(\tilde S > \hat\tau) - \Pr(S > \hat\tau)$ at the calibrated threshold.
4. **Testing (D).** H2 estimates $D^+ = \sup_t [\hat\Pr(\tilde S > t) - \hat\Pr(S > t)]$ on hallucinated test claims (one-sided Kolmogorov–Smirnov statistic) with an image-level paired bootstrap.
5. **When it fails, recalibration repairs it (H4).** If the shifted pipeline can be run on the calibration images (e.g. a detector swap), recomputing $\hat\tau$ on the shifted calibration scores restores exchangeability between calibration and test, and the clean guarantee applies again. The claim labels do not depend on the detector, so no new annotation is needed.
6. **Relation to robust conformal prediction.** Gendler et al. (2022) inflate the threshold to cover any perturbation within a known norm ball, and Cauchois et al. (2024) cover any distribution in an $f$-divergence ball. P1 is a one-sided, assumption-light condition: it needs no radius, and it explains why typical image degradations are harmless for detector-based hallucination filters.

## 5. Evidence so far (exploratory pilot, not confirmatory)

| Shift | Share of hallucinated claims whose score went up | Claim-level risk (target 0.10) |
|---|---|---|
| Blur (r = 3) | 0.29 | 0.070 |
| JPEG (q = 10) | 0.48 | 0.074 |
| Downsample ×4 | 0.28 | 0.066 |
| Detector swap → Grounding DINO | 0.94 | 0.528 |

Sources: `reports/phase3/e1_report.md`, `reports/phase3b/phase3b_report.md`.
