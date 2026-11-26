"""Column-width figures (3.45 in) for the IEEE two-column Word manuscript.

    python paper/make_ieee_figures.py

Same data and colours as make_figures.py; panels are stacked vertically so that
text stays legible at single-column width. Output: paper/figures/col_*.png (600 dpi).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "reports/phase4/phase4_results.json").read_text())
VLM = json.loads((ROOT / "reports/phase4/exploratory_vlm.json").read_text())
FIG = ROOT / "paper/figures"
LV = RES["levels"]["0.1"]
TARGET, MARGIN = 0.10, 0.03
W = 3.45
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e1"
CORR = [("gaussian_blur", "Gaussian blur"), ("jpeg", "JPEG"), ("pixelate", "Pixelate"),
        ("gaussian_noise", "Gaussian noise"), ("contrast", "Contrast")]
OTHER = [("clean", "Clean (reference)"),
         ("swap_owlvit_v1", "Detector: OWL-ViT v1"), ("swap_owlvit_v1+recalibrated", "OWL-ViT v1, recal."),
         ("swap_grounding_dino", "Detector: Grounding DINO"), ("swap_grounding_dino+recalibrated", "Grounding DINO, recal."),
         ("vlm_qwen", "VLM: Qwen2.5-VL-3B"), ("vlm_internvl", "VLM: InternVL3-2B"),
         ("domain_amber", "Domain: AMBER")]
LEVEL = [("claim_risk", "(a) Claim-level filter"), ("image_risk", "(b) Image-level filter")]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "legend.frameon": False, "axes.linewidth": 0.6,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def band(ax, horizontal=True):
    if horizontal:
        ax.axhspan(TARGET, TARGET + MARGIN, color="#efefeb", zorder=0)
        ax.axhline(TARGET, color=INK, lw=0.8, ls="--", zorder=1)
    else:
        ax.axvspan(TARGET, TARGET + MARGIN, color="#efefeb", zorder=0)
        ax.axvline(TARGET, color=INK, lw=0.8, ls="--", zorder=1)


# ------------------------------------------------------------ Fig. 1: pipeline and shifts
fig, ax = plt.subplots(figsize=(W, 2.15))
ax.set_xlim(-0.15, 10.15)
ax.set_ylim(0.0, 5.0)
ax.axis("off")


def box(x, y, w, h, text, fc="#ffffff", ec=INK, fs=5.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.15)


def arrow(x0, y0, x1, y1, color=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, shrinkA=0, shrinkB=0, mutation_scale=7))


y = 2.45
for x, w, txt in [(0.0, 1.5, "Image"), (1.95, 1.75, "VLM\ncaption"), (4.15, 1.75, "Object\nclaims"),
                  (6.35, 1.6, "Detector\nscore $S$")]:
    box(x, y, w, 1.0, txt)
box(8.4, y, 1.6, 1.0, "Keep if\n$S>\\hat\\tau$", fc="#eaf2fc", ec=COLORS[0])
for x0, x1 in [(1.5, 1.95), (3.7, 4.15), (5.9, 6.35), (7.95, 8.4)]:
    arrow(x0, y + 0.5, x1, y + 0.5)
box(3.55, 4.0, 5.0, 0.75, "Labels from COCO / AMBER annotations", fc="#f6f6f3", fs=5.4)
arrow(5.05, 4.0, 5.05, 3.45)
box(3.9, 0.65, 6.1, 1.15, "Threshold $\\hat\\tau$: conformal quantile of the scores\nof hallucinated claims in clean calibration data",
    fc="#eaf2fc", ec=COLORS[0], fs=5.2)
arrow(9.2, 1.8, 9.2, 2.45)
for cx, txt in [(0.75, "S1 degradation\nS4 domain\nS5 object removal"), (2.83, "S3 VLM swap"), (7.15, "S2 detector\nswap")]:
    ax.text(cx, 2.27, txt, ha="center", va="top", fontsize=5.2, color=COLORS[1], linespacing=1.1)
    arrow(cx, 2.3, cx, 2.45, color=COLORS[1])
ax.text(-0.1, 0.0, "Calibrated once on clean COCO captions (LLaVA-1.5-7B, OWLv2); each shift applied alone at test time.",
        fontsize=5.0, color=MUTED, ha="left", va="bottom")
save(fig, "col_pipeline")

# ------------------------------------------------------------ Fig. 2: S1 severity
fig, axes = plt.subplots(2, 1, figsize=(W, 4.1), sharex=True)
for ax, (key, title) in zip(axes, LEVEL):
    band(ax)
    ax.axhline(LV["clean"][key]["point"], color=MUTED, lw=0.7, ls=":")
    for i, (c, name) in enumerate(CORR):
        pts = [LV[f"{c}_{s}"][key] for s in range(1, 6)]
        x = np.arange(1, 6) + (i - 2) * 0.07
        ax.errorbar(x, [p["point"] for p in pts],
                    yerr=[[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]],
                    color=COLORS[i], marker=MARKERS[i], ms=3, lw=1.1, elinewidth=0.6, capsize=0, label=name)
    ax.set_title(title, loc="left", color=INK)
    ax.set_ylim(0, 0.16)
    ax.set_ylabel("Realised risk")
axes[1].set_xticks(range(1, 6))
axes[1].set_xlabel("Severity")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "col_s1")

# ------------------------------------------------------------ Fig. 3: H2
H2 = RES["hypotheses"]["H2"]["per_condition"]
fig, axes = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True)
ax = axes[0]
ax.axhline(0.03, color=INK, lw=0.8, ls="--")
for i, (c, name) in enumerate(CORR):
    pts = [H2[f"{c}_{s}"] for s in range(1, 6)]
    x = np.arange(1, 6) + (i - 2) * 0.07
    ax.errorbar(x, [p["d_plus"] for p in pts], yerr=[[p["d_plus"] - p["lo"] for p in pts], [p["hi"] - p["d_plus"] for p in pts]],
                color=COLORS[i], marker=MARKERS[i], ms=3, lw=1.1, elinewidth=0.6, label=name)
ax.set_ylabel(r"$D^{+}$ (95% CI)")
ax.set_title("(a) Largest increase over all thresholds", loc="left")
ax = axes[1]
ax.axhline(0, color=INK, lw=0.8)
for i, (c, name) in enumerate(CORR):
    ax.plot(range(1, 6), [H2[f"{c}_{s}"]["diff_at_tau"] for s in range(1, 6)], color=COLORS[i],
            marker=MARKERS[i], ms=3, lw=1.1)
ax.set_ylabel(r"$P(\tilde S>\hat\tau)-P(S>\hat\tau)$")
ax.set_title(r"(b) Change at the calibrated threshold $\hat\tau$", loc="left")
ax.set_xticks(range(1, 6))
ax.set_xlabel("Severity")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "col_h2")

# ------------------------------------------------------------ Fig. 4: forest
fig, axes = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True)
ys = np.arange(len(OTHER))[::-1]
style = {"violated": (COLORS[1], "s"), "preserved": (COLORS[0], "o"), "inconclusive": (COLORS[3], "D")}
for ax, (key, title) in zip(axes, LEVEL):
    band(ax, horizontal=False)
    for y, (c, _) in zip(ys, OTHER):
        m = LV[c][key]
        col, mk = style[m["decision"]]
        ax.plot([m["lo"], m["hi"]], [y, y], color=col, lw=1.3, solid_capstyle="round")
        ax.plot(m["point"], y, marker=mk, color=col, ms=4, mec="white", mew=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 1.0)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(ys)
    ax.set_yticklabels([n for _, n in OTHER])
    ax.set_title(title, loc="left")
axes[1].set_xlabel("Realised risk (log scale), 95% CI")
fig.legend(handles=[Line2D([], [], color=COLORS[0], marker="o", lw=1.2, ms=4, label="preserved"),
                    Line2D([], [], color=COLORS[3], marker="D", lw=1.2, ms=4, label="inconclusive"),
                    Line2D([], [], color=COLORS[1], marker="s", lw=1.2, ms=4, label="violated")],
           loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.05))
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "col_forest")

# ------------------------------------------------------------ Fig. 5: VLM tails (exploratory)
fig, ax = plt.subplots(figsize=(W, 2.35))
tau = VLM["tau_full_cal"]
t = np.linspace(-3, 1.5, 400)
for i, (name, v) in enumerate(VLM["vlms"].items()):
    s = np.array(v["hallucinated_scores"])
    ax.plot(t, [(s > x).mean() for x in t], color=COLORS[[0, 2, 1][i]], lw=1.3, label=f"{name} (n = {len(s)})")
ax.axvline(tau, color=INK, lw=0.8, ls="--")
ax.axhline(TARGET, color=MUTED, lw=0.7, ls=":")
ax.text(tau + 0.07, 0.30, r"$\hat\tau$ (calibrated" + "\non LLaVA)", fontsize=6.6)
ax.set_xlabel("OWLv2 logit score of hallucinated claims")
ax.set_ylabel(r"$P(S>t)$")
ax.set_xlim(-3, 1.5)
ax.set_ylim(0, 0.6)
ax.legend(loc="upper right")
save(fig, "col_vlm")

# ------------------------------------------------------------ Fig. 6: utility
fig, ax = plt.subplots(figsize=(W, 2.45))
ax.axvspan(TARGET, TARGET + MARGIN, color="#efefeb", zorder=0)
ax.axvline(TARGET, color=INK, lw=0.8, ls="--")
for i, (c, name) in enumerate(CORR):
    ax.plot([LV[f"{c}_{s}"]["claim_risk"]["point"] for s in range(1, 6)],
            [LV[f"{c}_{s}"]["supported_kept"]["point"] for s in range(1, 6)],
            color=COLORS[i], marker=MARKERS[i], ms=3, lw=1.1, label=name)
ax.plot(LV["clean"]["claim_risk"]["point"], LV["clean"]["supported_kept"]["point"], marker="*", ms=8,
        color=INK, ls="none", label="Clean")
ax.set_xlabel("Claim-level risk")
ax.set_ylabel("Share of supported claims kept")
ax.legend(loc="lower right", ncol=2)
save(fig, "col_utility")
print("column figures written")
