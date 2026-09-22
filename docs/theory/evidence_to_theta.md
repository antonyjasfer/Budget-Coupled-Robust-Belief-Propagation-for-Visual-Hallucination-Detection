# Mapping Calibrated Probabilities to Ising Unary Fields

## 1. Overview and Core Principle

In the binary Ising Markov Random Field:
$$h_i \in \{-1, +1\}$$
where:
- $h_i = -1$: **SUPPORTED**
- $h_i = +1$: **HALLUCINATED**

The joint distribution is:
$$P(h) \propto \exp\left( \sum_{i \in \mathcal{V}} \theta_i h_i + \sum_{(i,j) \in \mathcal{E}} J_{ij} h_i h_j \right)$$

For an isolated node $i$ (or when all couplings $J_{ij} = 0$), the marginal probability that claim $i$ is hallucinated is given by:
$$P(H_i = +1) = \frac{\exp(\theta_i \cdot (+1))}{\exp(\theta_i \cdot (+1)) + \exp(\theta_i \cdot (-1))} = \frac{e^{\theta_i}}{e^{\theta_i} + e^{-\theta_i}} = \frac{1}{1 + e^{-2\theta_i}} = \sigma(2\theta_i)$$
where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the standard logistic sigmoid function.

---

## 2. The Analytical Inverse Mapping

Given a calibrated hallucination probability $p_i = P(H_i = \text{HALLUCINATED} \mid \text{evidence})$:
$$p_i = \sigma(2\theta_i) \iff 2\theta_i = \operatorname{logit}(p_i) = \ln \left( \frac{p_i}{1 - p_i} \right)$$
Therefore, the unique unary Ising field $\theta_i$ corresponding to probability $p_i$ is:
$$\theta_i = \frac{1}{2} \ln \left( \frac{p_i}{1 - p_i} \right)$$

### Sign Convention
- **$p_i > 0.5$** (likely hallucinated): $\frac{p_i}{1 - p_i} > 1 \implies \theta_i > 0$.
- **$p_i < 0.5$** (likely supported): $\frac{p_i}{1 - p_i} < 1 \implies \theta_i < 0$.
- **$p_i = 0.5$** (uninformative evidence): $\frac{p_i}{1 - p_i} = 1 \implies \theta_i = 0$.

This exactly aligns with the Ising Hamiltonian energy:
$$E_i(h_i) = -\theta_i h_i$$
When $h_i = +1$ (hallucinated) and $\theta_i > 0$, the energy is negative (favorable state).
When $h_i = -1$ (supported) and $\theta_i < 0$, $-\theta_i h_i = -(-\theta_i)(-1) < 0$ (favorable state).

---

## 3. Numerical Stabilization and Probability Clipping

To prevent singularity at $p_i \in \{0, 1\}$ and avoid floating-point overflow in downstream belief propagation messages:
1. Probability values are clipped strictly to $[p_{\text{min}}, 1 - p_{\text{min}}]$:
   $$\tilde{p}_i = \operatorname{clip}(p_i, \, p_{\text{min}}, \, 1 - p_{\text{min}})$$
   where default $p_{\text{min}} = 10^{-5}$.
2. Maximum unary field magnitude is bounded:
   $$\theta_{\text{max}} = \frac{1}{2} \ln \left( \frac{1 - p_{\text{min}}}{p_{\text{min}}} \right) \approx 5.756$$
   For $p_{\text{min}} = 10^{-4}$, $\theta_{\text{max}} \approx 4.605$.
3. Stable logit computation:
   $$\theta_i = \frac{1}{2} (\ln(\tilde{p}_i) - \ln(1 - \tilde{p}_i))$$

---

## 4. Invertibility Guarantee

For any clipped probability $\tilde{p}_i \in (0, 1)$:
$$\sigma(2\theta_i) = \frac{1}{1 + \exp\left(-2 \cdot \frac{1}{2} \ln \frac{\tilde{p}_i}{1 - \tilde{p}_i}\right)} = \frac{1}{1 + \frac{1 - \tilde{p}_i}{\tilde{p}_i}} = \tilde{p}_i$$
The mapping is a continuous bijection between $(0, 1)$ and $\mathbb{R}$.

---

## 5. Critical Distinction: Unary Field vs. Final Marginal

> [!IMPORTANT]
> **Unary Prior vs. Coupled Posterior**:
> - $\theta_i$ represents the **local evidence log-odds** of hallucination before graph interactions.
> - Once attractive couplings $J_{ij} > 0$ are active, the true marginal probability $P(H_i = +1 \mid E)$ is computed by belief propagation:
>   $$P(H_i = +1 \mid E) = \sigma(2 \eta_i) \quad \text{where } \eta_i = \theta_i + \sum_{j \in \operatorname{adj}(i)} \nu_{j \to i}$$
> - Therefore, $\theta_i$ is **not** the final output probability. It is the unary evidence potential injected into the graphical model.
