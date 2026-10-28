# Root Dominance in Budget-Coupled Robust BP

**Status:** proved (below) and verified numerically (`tests/test_root_dominance.py`, `scripts/check_root_dominance.py`). Added in Phase 0, 2026-10-25.

## 1. Setting

Attractive Ising tree $(\mathcal{V}, \mathcal{E})$ with $J_e \ge 0$, fields $\theta$, local radii $\varepsilon_i \ge 0$, global budget $B \ge 0$, and uncertainty set

$$\mathcal{U}(B, \varepsilon) = \{\delta : |\delta_i| \le \varepsilon_i,\ \textstyle\sum_i |\delta_i| \le B\}.$$

Root the tree at the target $r$. The effective field is

$$\eta_r(\delta) = \theta_r + \delta_r + \sum_{c \in \mathrm{ch}(r)} \nu_{c \to r}(\delta_{-r}),$$

so the messages into $r$ do not depend on $\delta_r$. Define the path gain $\gamma_i = \prod_{e \in \mathrm{path}(i \to r)} \tanh(J_e)$ for $i \ne r$, and $\gamma_{\max} = \max_{i \ne r} \gamma_i$ ($\gamma_{\max} = 0$ if $n = 1$). Note $\gamma_{\max} < 1$ for finite couplings.

## 2. Lemma (root dominance)

Let $a = \min(\varepsilon_r, B)$ and $\eta_r = \eta_r(0)$. Then

$$\eta_r + a \;\le\; \sup_{\delta \in \mathcal{U}} \eta_r(\delta) \;\le\; \eta_r + a + \gamma_{\max}\,(B - \varepsilon_r)_+ ,$$

and symmetrically for the infimum. In particular, **if $B \le \varepsilon_r$, then**

$$[L_r, U_r] = \big[\sigma(2(\eta_r - B)),\ \sigma(2(\eta_r + B))\big]$$

**exactly**: the tree, the couplings, the other nodes' radii and the DP have no effect on the interval.

## 3. Proof

*Lower bound.* $\delta = a\, e_r$ is feasible and gives $\eta_r(\delta) = \eta_r + a$.

*Upper bound.* By monotonicity (Proposition 1, `one_sided_extrema.md`) it suffices to take $\delta \ge 0$. Since $\delta_r$ enters additively,

$$\eta_r(\delta) - \eta_r = \delta_r + \big[\eta_r(0, \delta_{-r}) - \eta_r(0)\big].$$

By the mean value theorem along the segment from $0$ to $(0, \delta_{-r})$ and the path-derivative bound $0 \le \partial \eta_r / \partial \delta_i \le \gamma_i$ (Theorem 3, step 3, `continuous_grid_certificate.md`):

$$\eta_r(0, \delta_{-r}) - \eta_r(0) \le \sum_{i \ne r} \gamma_i \delta_i \le \gamma_{\max} \sum_{i \ne r} \delta_i \le \gamma_{\max} (B - \delta_r).$$

Hence $\eta_r(\delta) - \eta_r \le \delta_r + \gamma_{\max}(B - \delta_r) = \gamma_{\max} B + (1 - \gamma_{\max})\delta_r$. This is increasing in $\delta_r$, and $\delta_r \le a$, so

$$\eta_r(\delta) - \eta_r \le a + \gamma_{\max}(B - a) = a + \gamma_{\max}(B - \varepsilon_r)_+ .$$

If $B \le \varepsilon_r$, then $a = B$ and the two bounds coincide, so the supremum equals $\eta_r + B$ and is attained at $\delta = B e_r$. The infimum follows by symmetry with $\delta \le 0$. Applying the increasing map $\eta \mapsto \sigma(2\eta)$ gives the interval. $\blacksquare$

## 4. Consequences for the project

1. **Default configuration is degenerate.** With $B = \varepsilon_i = J = \operatorname{atanh}(0.5)$, every claim is in the regime $B \le \varepsilon_r$. The M9 development results match the closed form to about $10^{-16}$ in all 19 such conditions.
2. **"Evidence-sensitive" collapses to a margin rule.** In this regime $L_r < \tau < U_r$ (with $\tau = 0.5$) iff $|\eta_r| < B$, i.e. iff the point posterior lies in $(\sigma(-2B), \sigma(2B))$. For the default $B$ that is $(0.25, 0.75)$.
3. **Where the graph can matter.** Only when $B > \varepsilon_r$, and then the graph adds at most $\gamma_{\max}(B - \varepsilon_r)$ to the target field. Experiments must report the fraction of claims in each regime.
4. **Mandatory baseline.** Margin abstention on the point posterior must be compared against robust abstention at equal coverage.

## 5. Relation to prior work (added 2026-10-28; PREREG v2 Proposition P2)

The "gain ≤ 1" ingredient is known. Chan & Darwiche (2002, *JAIR* 17, Theorem 3.3, pp. 275–276) prove, for any Bayesian network and a binary variable $X$, that changing one parameter $\theta_{x|u}$ changes the log-odds of any query by at most the log-odds change of that parameter, $|\ln O'(y\mid e) - \ln O(y\mid e)| \le |\ln O'(x\mid u) - \ln O(x\mid u)|$, with no assumption on the network structure. Our contribution is the part their structure-free bound cannot give: on an attractive Ising tree, a non-target node's gain is at most $\gamma_i = \prod_{e} \tanh J_e < 1$ (single-edge contraction; cf. the message-error contraction of Ihler et al. 2005), and under a shared $\ell_1$ budget the worst case puts the whole budget on the target when $B \le \varepsilon_r$, with the graph adding at most $\gamma_{\max}(B - \varepsilon_r)_+$ otherwise. In the paper this is presented as a proposition that explains why claim coupling does not change the robust bounds in the measured regime (pilot within-image label correlation 0.031), not as a headline theorem.
