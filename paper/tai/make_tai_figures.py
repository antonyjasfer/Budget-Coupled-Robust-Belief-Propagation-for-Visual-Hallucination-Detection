"""Figures and number macros for the journal (IEEE TAI) version.

Reads reports/phase4/phase4_results.json and reports/phase4/exploratory_vlm.json and writes
paper/tai/tai_numbers.tex and paper/tai/figures/*.pdf. Nothing here is typed by hand except
the target levels (delta = 0.10, margin 0.03) that the pre-registration fixed.

Run:  python paper/tai/make_tai_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import beta  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = json.loads((ROOT / "reports/phase4/phase4_results.json").read_text())
EXP = json.loads((ROOT / "reports/phase4/exploratory_vlm.json").read_text())
LV = RES["levels"]["0.1"]
H2 = RES["hypotheses"]["H2"]["per_condition"]
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

DELTA, EPS = 0.10, 0.03
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})
W1 = 3.45  # one IEEE column, inches

macros: dict[str, str] = {}


def mac(name: str, value: str) -> None:
    macros[name] = value


# ---------------------------------------------------------------- Theorem 2: prevalence
# pi = share of test images with at least one hallucinated claim. CHAIR_s is the share of the
# 500 captions with a hallucination; images without any claim cannot have one, so on the
# n_images images that produced a claim, pi = CHAIR_s * 500 / n_images.
VLMS = [("llava", "clean", "LLaVA-1.5-7B"), ("qwen", "vlm_qwen", "Qwen2.5-VL-3B"),
        ("internvl", "vlm_internvl", "InternVL3-2B")]
prev = {}
for key, cond, name in VLMS:
    ch = RES["chair"][key]
    n_img = LV[cond]["n_images"]
    pi = ch["chair_s"] * ch["captions"] / n_img
    r_img = LV[cond]["image_risk"]["point"]
    prev[key] = dict(name=name, pi=pi, r=r_img, rho=r_img / pi,
                     rc=LV[cond]["claim_risk"]["point"], dec_c=LV[cond]["claim_risk"]["decision"],
                     dec_i=LV[cond]["image_risk"]["decision"])
tag = {"llava": "Llava", "qwen": "Qwen", "internvl": "Intern"}
for k, v in prev.items():
    mac(f"Pi{tag[k]}", f"{v['pi']:.3f}")
    mac(f"Rho{tag[k]}", f"{v['rho']:.3f}")
mac("RhoInternAtLlavaPi", f"{prev['internvl']['rho'] * prev['llava']['pi']:.3f}")
mac("AlphaOverPiIntern", f"{DELTA / prev['internvl']['pi']:.3f}")
mac("AlphaOverPiLlava", f"{DELTA / prev['llava']['pi']:.3f}")

fig, ax = plt.subplots(figsize=(W1, 2.35))
pis = np.linspace(0.15, 0.6, 200)
for a, ls, xl in ((0.10, "-", 0.585), (0.05, ":", 0.315), (0.20, "--", 0.585)):
    ax.plot(pis, a / pis, color="0.45", lw=0.8, ls=ls)
    ax.text(xl, a / xl + 0.004, rf"$\pi\rho={a:.2f}$", fontsize=6.5, color="0.35", va="bottom", ha="right")
for k, v in prev.items():
    col = "#c0392b" if v["dec_c"] == "violated" else ("#d68910" if v["dec_c"] == "inconclusive" else "0.2")
    ax.plot(v["pi"], v["rho"], "o", ms=5, color=col)
    off = {"llava": (5, 3), "qwen": (-4, -11), "internvl": (-4, 6)}[k]
    ax.annotate(v["name"], (v["pi"], v["rho"]), textcoords="offset points", xytext=off,
                ha="left" if k == "llava" else "right", fontsize=6.8)
ax.annotate("", xy=(prev["llava"]["pi"], prev["internvl"]["rho"]), xytext=(prev["internvl"]["pi"], prev["internvl"]["rho"]),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7, ls="--"))
ax.text(prev["llava"]["pi"] + 0.008, prev["internvl"]["rho"],
        f"same tail at LLaVA\nprevalence: {prev['internvl']['rho'] * prev['llava']['pi']:.3f}",
        fontsize=6.3, color="#c0392b", ha="left", va="center")
ax.set_xlim(0.15, 0.6)
ax.set_ylim(0.15, 0.40)
ax.set_xlabel(r"prevalence $\pi$ = P(caption has a hallucination)")
ax.set_ylabel(r"$\rho$ = P(survives $\mid$ has one)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_prevalence.pdf")
fig.savefig(FIG / "fig_prevalence.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- Theorem 1: locality
tau = EXP["tau_full_cal"]
src = np.asarray(EXP["vlms"]["LLaVA-1.5-7B (test)"]["hallucinated_scores"])


def sf(x, t):
    return float(np.mean(x > t))


def dplus(q, p):
    grid = np.unique(np.concatenate([p, q]))
    return float(max(0.0, max(sf(q, t) - sf(p, t) for t in grid)))


pts = []  # (kind, label, global D+, local Delta(tau), realised excess claim risk)
clean_rc = LV["clean"]["claim_risk"]["point"]
for c, v in H2.items():
    pts.append(("S1", c, v["d_plus"], v["diff_at_tau"], LV[c]["claim_risk"]["point"] - clean_rc))
for name, cond in (("Qwen2.5-VL-3B", "vlm_qwen"), ("InternVL3-2B", "vlm_internvl")):
    q = np.asarray(EXP["vlms"][name]["hallucinated_scores"])
    pts.append(("S3", name, dplus(q, src), sf(q, tau) - sf(src, tau), LV[cond]["claim_risk"]["point"] - clean_rc))
P = np.array([[p[2], p[3], p[4]] for p in pts])
mac("CorrLocal", f"{np.corrcoef(P[:, 1], P[:, 2])[0, 1]:.2f}")
mac("CorrGlobal", f"{np.corrcoef(P[:, 0], P[:, 2])[0, 1]:.2f}")
mac("MaxAbsLocalGap", f"{np.max(np.abs(P[:, 1] - P[:, 2])):.3f}")
mac("NSoneGlobalFail", str(sum(1 for p in pts if p[0] == "S1" and p[2] > EPS)))
mac("NLocalPoints", str(len(pts)))
for name, key in (("Qwen2.5-VL-3B", "Qwen"), ("InternVL3-2B", "Intern")):
    p = next(x for x in pts if x[1] == name)
    mac(f"LocalDelta{key}", f"{p[3]:+.3f}")
    mac(f"Excess{key}", f"{p[4]:+.3f}")
    mac(f"Dplus{key}", f"{p[2]:.3f}")

fig, axs = plt.subplots(1, 2, figsize=(W1, 1.85), sharey=True)
lim = (-0.08, 0.12)
for ax, col, ttl in ((axs[0], 0, r"global gap $D^+$"), (axs[1], 1, r"local gap $\hat\Delta(\hat\tau)$")):
    for kind, mk, cl in (("S1", "o", "0.35"), ("S3", "s", "#c0392b")):
        sel = np.array([p[0] == kind for p in pts])
        ax.plot(P[sel, col], P[sel, 2], mk, ms=3.2, color=cl, alpha=0.85,
                label="degradations (S1)" if kind == "S1" else "VLM swaps (S3)")
    ax.plot(lim, lim, color="0.6", lw=0.6, ls="--")
    ax.axhline(0, color="0.8", lw=0.5)
    ax.axvline(0, color="0.8", lw=0.5)
    ax.set_xlim(-0.08, 0.16 if col == 0 else 0.12)
    ax.set_ylim(*lim)
    ax.set_title(ttl, pad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axs[0].set_ylabel("realised excess claim risk")
axs[1].legend(loc="upper left", frameon=False, handletextpad=0.2, borderaxespad=0.1)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(FIG / "fig_local.pdf")
fig.savefig(FIG / "fig_local.png", dpi=220)
plt.close(fig)

# ---------------------------------------------------------------- Theorem 3: label budget


def tail(m: int, eps: float = EPS, d: float = DELTA) -> float:
    k = math.ceil((m + 1) * (1 - d))
    if k > m:
        return 0.0
    return float(beta.sf(d + eps, m + 1 - k, k))


def budget(b: float, eps: float = EPS, horizon: int = 4000) -> int:
    ok = [tail(m, eps) <= b for m in range(10, horizon + 1)]
    for i in range(len(ok)):
        if all(ok[i:]):
            return i + 10
    raise ValueError("no stable budget below horizon")


mac("BudgetTen", str(budget(0.10)))
mac("BudgetFive", str(budget(0.05)))
mac("BudgetTenEpsFive", str(budget(0.10, 0.05)))
for m, word in ((50, "Fifty"), (100, "Hundred"), (200, "TwoHundred"), (500, "FiveHundred")):
    mac(f"TailM{word}", f"{tail(m):.3f}")
mac("NormalApproxBudget", f"{1.2815515655446004 ** 2 * DELTA * (1 - DELTA) / EPS ** 2:.0f}")

ms = np.arange(20, 1001)
fig, ax = plt.subplots(figsize=(W1, 1.9))
for eps, ls in ((0.03, "-"), (0.05, "--")):
    ax.plot(ms, [tail(int(m), eps) for m in ms], color="0.2", lw=0.8, ls=ls,
            label=rf"$\varepsilon={eps:.2f}$")
ax.axhline(0.10, color="0.6", lw=0.6, ls=":")
ax.axvline(budget(0.10), color="#c0392b", lw=0.6, ls=":")
ax.text(budget(0.10) + 12, 0.32, f"m = {budget(0.10)}", color="#c0392b", fontsize=6.8)
ax.set_xscale("log")
ax.set_xlabel(r"labelled hallucinated claims used to recalibrate, $m$")
ax.set_ylabel(r"P(realised risk $>0.10+\varepsilon$)")
ax.set_ylim(0, 0.45)
ax.legend(frameon=False, loc="upper right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(FIG / "fig_budget.pdf")
fig.savefig(FIG / "fig_budget.png", dpi=220)
plt.close(fig)

(HERE / "tai_numbers.tex").write_text(
    "% generated by paper/tai/make_tai_figures.py - do not edit\n"
    + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
print(json.dumps(macros, indent=1))
