"""Build every figure and table of the manuscript from committed result files.

    python paper/make_figures.py

Inputs : reports/phase4/phase4_results.json, reports/phase4/exploratory_vlm.json
Outputs: paper/figures/*.pdf, paper/tables/*.tex, paper/numbers.tex (macros used in the text)
No number in the manuscript is typed by hand: the text uses the macros in numbers.tex.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "reports/phase4/phase4_results.json").read_text())
VLM = json.loads((ROOT / "reports/phase4/exploratory_vlm.json").read_text())
FIG, TAB = ROOT / "paper/figures", ROOT / "paper/tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
LV = RES["levels"]["0.1"]
TARGET, MARGIN = 0.10, 0.03

# Validated categorical palette (dataviz reference palette, slots 1-5); markers are the secondary encoding.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e1"
CORR = [("gaussian_blur", "Gaussian blur"), ("jpeg", "JPEG"), ("pixelate", "Pixelate"),
        ("gaussian_noise", "Gaussian noise"), ("contrast", "Contrast")]
OTHER = [("clean", "Clean (reference)"),
         ("swap_owlvit_v1", "Detector: OWL-ViT v1"), ("swap_owlvit_v1+recalibrated", "OWL-ViT v1, recalibrated"),
         ("swap_grounding_dino", "Detector: Grounding DINO"), ("swap_grounding_dino+recalibrated", "Grounding DINO, recalibrated"),
         ("vlm_qwen", "VLM: Qwen2.5-VL-3B"), ("vlm_internvl", "VLM: InternVL3-2B"),
         ("domain_amber", "Domain: AMBER")]
LEVEL = [("claim_risk", "Claim-level filter"), ("image_risk", "Image-level filter")]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def target_band(ax, horizontal=True):
    if horizontal:
        ax.axhspan(TARGET, TARGET + MARGIN, color="#f2f2ef", zorder=0)
        ax.axhline(TARGET, color=INK, lw=0.9, ls="--", zorder=1)
    else:
        ax.axvspan(TARGET, TARGET + MARGIN, color="#f2f2ef", zorder=0)
        ax.axvline(TARGET, color=INK, lw=0.9, ls="--", zorder=1)


# ---------------------------------------------------------------- Figure 1: S1 severity curves
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for ax, (key, title) in zip(axes, LEVEL):
    target_band(ax)
    clean = LV["clean"][key]["point"]
    ax.axhline(clean, color=MUTED, lw=0.8, ls=":")
    for i, (c, name) in enumerate(CORR):
        pts = [LV[f"{c}_{s}"][key] for s in range(1, 6)]
        x = np.arange(1, 6) + (i - 2) * 0.07
        y = [p["point"] for p in pts]
        ax.errorbar(x, y, yerr=[[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]],
                    color=COLORS[i], marker=MARKERS[i], ms=4, lw=1.4, elinewidth=0.8, capsize=0, label=name)
    ax.set_title(title, color=INK, fontsize=9)
    ax.set_xticks(range(1, 6))
    ax.set_xlim(0.55, 5.4)
    ax.set_xlabel("Severity")
    ax.set_ylim(0, 0.16)
axes[0].set_ylabel("Realised risk (95% CI)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(FIG / "fig_s1_severity.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: forest plot, non-S1 shifts
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
ys = np.arange(len(OTHER))[::-1]
for ax, (key, title) in zip(axes, LEVEL):
    target_band(ax, horizontal=False)
    for y, (c, _) in zip(ys, OTHER):
        m = LV[c][key]
        col = {"violated": COLORS[1], "preserved": COLORS[0], "inconclusive": COLORS[3]}[m["decision"]]
        mk = {"violated": "s", "preserved": "o", "inconclusive": "D"}[m["decision"]]
        ax.plot([m["lo"], m["hi"]], [y, y], color=col, lw=1.6, solid_capstyle="round")
        ax.plot(m["point"], y, marker=mk, color=col, ms=5, mec="white", mew=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 1.0)
    ax.set_xticks([0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(title, color=INK, fontsize=9)
    ax.set_xlabel("Realised risk (log scale, 95% CI)")
axes[0].set_yticks(ys)
axes[0].set_yticklabels([n for _, n in OTHER])
from matplotlib.lines import Line2D  # noqa: E402
axes[1].legend(handles=[Line2D([], [], color=COLORS[0], marker="o", label="preserved"),
                        Line2D([], [], color=COLORS[3], marker="D", label="inconclusive"),
                        Line2D([], [], color=COLORS[1], marker="s", label="violated")],
               loc="upper center", bbox_to_anchor=(-0.05, -0.2), ncol=3, fontsize=7.5)
fig.tight_layout()
fig.savefig(FIG / "fig_forest.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3: H2 - D+ vs change at tau
H2 = RES["hypotheses"]["H2"]["per_condition"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axes[0]
ax.axhline(0.03, color=INK, lw=0.9, ls="--")
ax.text(5.45, 0.03, "0.03", fontsize=7, color=INK, va="center")
for i, (c, name) in enumerate(CORR):
    pts = [H2[f"{c}_{s}"] for s in range(1, 6)]
    x = np.arange(1, 6) + (i - 2) * 0.07
    ax.errorbar(x, [p["d_plus"] for p in pts],
                yerr=[[p["d_plus"] - p["lo"] for p in pts], [p["hi"] - p["d_plus"] for p in pts]],
                color=COLORS[i], marker=MARKERS[i], ms=4, lw=1.4, elinewidth=0.8, label=name)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Severity")
ax.set_ylabel(r"$D^{+}$ (95% CI)")
ax.set_title("Dominance gap over all thresholds", fontsize=9, color=INK)
ax = axes[1]
ax.axhline(0, color=INK, lw=0.9)
for i, (c, name) in enumerate(CORR):
    ax.plot(range(1, 6), [H2[f"{c}_{s}"]["diff_at_tau"] for s in range(1, 6)], color=COLORS[i],
            marker=MARKERS[i], ms=4, lw=1.4, label=name)
ax.set_xticks(range(1, 6))
ax.set_xlabel("Severity")
ax.set_ylabel(r"$P(\tilde S>\tau)-P(S>\tau)$")
ax.set_title(r"Change at the calibrated $\tau$", fontsize=9, color=INK)
ax.legend(fontsize=6.5, loc="lower left")
axes[0].set_xlim(0.7, 5.8)
fig.tight_layout()
fig.savefig(FIG / "fig_h2.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 4: utility (supported kept) vs claim risk
fig, ax = plt.subplots(figsize=(3.4, 2.7))
ax.axvspan(TARGET, TARGET + MARGIN, color="#f2f2ef", zorder=0)
ax.axvline(TARGET, color=INK, lw=0.9, ls="--")
for i, (c, name) in enumerate(CORR):
    ax.plot([LV[f"{c}_{s}"]["claim_risk"]["point"] for s in range(1, 6)],
            [LV[f"{c}_{s}"]["supported_kept"]["point"] for s in range(1, 6)],
            color=COLORS[i], marker=MARKERS[i], ms=4, lw=1.2, label=name)
ax.plot(LV["clean"]["claim_risk"]["point"], LV["clean"]["supported_kept"]["point"], marker="*", ms=10,
        color=INK, ls="none", label="clean")
ax.set_xlabel("Claim-level risk")
ax.set_ylabel("Share of supported claims kept")
ax.legend(fontsize=6.5, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig_utility.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 5 (exploratory): VLM score tails
fig, ax = plt.subplots(figsize=(3.4, 2.6))
tau = VLM["tau_full_cal"]
for i, (name, v) in enumerate(VLM["vlms"].items()):
    s = np.array(v["hallucinated_scores"])
    t = np.linspace(-3, 1.5, 400)
    ax.plot(t, [(s > x).mean() for x in t], color=COLORS[[0, 2, 1][i]], lw=1.6,
            label=f"{name} (n={len(s)})")
ax.axvline(tau, color=INK, lw=0.9, ls="--")
ax.axhline(TARGET, color=MUTED, lw=0.8, ls=":")
ax.text(tau + 0.08, 0.28, "$\\tau$ (calibrated\non LLaVA)", fontsize=7, ha="left")
ax.set_xlabel("OWLv2 logit score of hallucinated claims")
ax.set_ylabel(r"$P(S > t)$")
ax.set_xlim(-3, 1.5)
ax.set_ylim(0, 0.6)
ax.legend(fontsize=6.5, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "fig_vlm_tails.pdf")
plt.close(fig)


# ---------------------------------------------------------------- tables
def f3(x):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def ci(m):
    return f"{m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def dec(m):
    return {"violated": r"\textbf{violated}", "preserved": "preserved", "inconclusive": r"\emph{inconcl.}"}[m["decision"]]


rows = []
for c, name in OTHER + [("s5_object_removed", "Object removed (S5, exploratory)")]:
    m = LV[c]
    rows.append(f"{name} & {ci(m['claim_risk'])} & {dec(m['claim_risk'])} & {ci(m['image_risk'])} & "
                f"{dec(m['image_risk'])} & {f3(m['supported_kept']['point'])} & {f3(m['abstention']['point'])} \\\\")
(TAB / "tab_main.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

rows = []
for c, name in CORR:
    for s in range(1, 6):
        m = LV[f"{c}_{s}"]
        rows.append(f"{name if s == 1 else ''} & {s} & {ci(m['claim_risk'])} & {dec(m['claim_risk'])} & "
                    f"{ci(m['image_risk'])} & {dec(m['image_risk'])} & {f3(m['supported_kept']['point'])} & "
                    f"{f3(m['abstention']['point'])} \\\\")
    rows.append(r"\addlinespace")
(TAB / "tab_s1.tex").write_text("\n".join(rows[:-1]) + "\n\\bottomrule\n")

HN = {"H1": "Degradation (S1) is safe", "H3": "A detector swap breaks the claim-level target",
      "H4": "Recalibration repairs detector swaps", "H5": "LLaVA-calibrated thresholds transfer to other VLMs",
      "H6": "Transfer to AMBER"}
hy = RES["hypotheses"]
rows = []
for h in ["H1", "H3", "H4", "H5", "H6"]:
    a, b = hy["claim_risk"][h], hy["image_risk"][h]
    rows.append(f"{h} & {HN[h]} & {'supported' if a['supported'] else 'not supported'} ({a['p_holm']:.3f}) & "
                f"{'supported' if b['supported'] else 'not supported'} ({b['p_holm']:.3f}) \\\\")
n_h2 = sum(v["supported"] for v in H2.values())
rows.append(f"H2 & $\\tilde S \\preceq_{{st}} S$ under S1 (upper CI of $D^+\\le 0.03$) & "
            f"\\multicolumn{{2}}{{l}}{{not supported ({n_h2} of {len(H2)} conditions meet the bound)}} \\\\")
(TAB / "tab_hyp.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

P = RES["pgm"]
rows = []
for split, sname in (("train_cal_a", "Cal-A"), ("test", "Test")):
    for mod, mname in (("unary_J0", "Unary ($J=0$)"), ("ising_tree_J", "Chain Ising (learned $J$)")):
        m = P[split][mod]
        cf = m["confusion"]
        rows.append(f"{sname} & {mname} & {m['accuracy']:.3f} & {m['precision']:.3f} & {m['recall']:.3f} & "
                    f"{m['f1']:.3f} & {m['roc_auc']:.3f} & {cf['tp']} & {cf['fp']} & {cf['tn']} & {cf['fn']} \\\\")
(TAB / "tab_pgm.tex").write_text("\n".join(rows) + "\n\\bottomrule\n")

# ---------------------------------------------------------------- numbers used in the prose
C = RES["counts"]
ri = P["robust_intervals"]
macros = {
    "NCal": C["cal"], "NTest": C["test"], "NQwen": C["qwen"], "NIntern": C["internvl"], "NAmber": C["amber"],
    "NSfive": C["s5"], "NAmberUnknown": RES["amber_counts"]["unknown"], "NAmberClaims": RES["amber_counts"]["claims"],
    "NTestHall": LV["clean"]["n_hallucinated"], "NTestImg": LV["clean"]["n_images"],
    "CleanClaim": f"{LV['clean']['claim_risk']['point']:.3f}", "CleanImage": f"{LV['clean']['image_risk']['point']:.3f}",
    "CleanKept": f"{LV['clean']['supported_kept']['point']:.3f}",
    "GdinoClaim": ci(LV["swap_grounding_dino"]["claim_risk"]), "GdinoImage": ci(LV["swap_grounding_dino"]["image_risk"]),
    "GdinoRecClaim": ci(LV["swap_grounding_dino+recalibrated"]["claim_risk"]),
    "GdinoRecImage": ci(LV["swap_grounding_dino+recalibrated"]["image_risk"]),
    "OwlvOneClaim": ci(LV["swap_owlvit_v1"]["claim_risk"]),
    "QwenClaim": ci(LV["vlm_qwen"]["claim_risk"]), "InternClaim": ci(LV["vlm_internvl"]["claim_risk"]),
    "QwenImage": ci(LV["vlm_qwen"]["image_risk"]), "InternImage": ci(LV["vlm_internvl"]["image_risk"]),
    "InternKeptHall": f"{LV['vlm_internvl']['kept_hallucinated_share']['point']:.3f}",
    "CleanKeptHall": f"{LV['clean']['kept_hallucinated_share']['point']:.3f}",
    "AmberClaim": ci(LV["domain_amber"]["claim_risk"]), "AmberImage": ci(LV["domain_amber"]["image_risk"]),
    "SfiveClaim": ci(LV["s5_object_removed"]["claim_risk"]), "SfiveImage": ci(LV["s5_object_removed"]["image_risk"]),
    "NHtwo": n_h2, "Jlearned": f"{P['J_learned']:.2f}", "LLgain": f"{P['loglik_gain_vs_J0']:.2f}",
    "CorrCal": f"{P['within_image_corr']['cal']:.3f}", "CorrTest": f"{P['within_image_corr']['test']:.3f}",
    "AucUnary": f"{P['test']['unary_J0']['roc_auc']:.3f}", "AucIsing": f"{P['test']['ising_tree_J']['roc_auc']:.3f}",
    "RobEps": f"{ri['eps']:.3f}", "RobB": f"{ri['B']:.3f}", "RobImgs": ri["images"],
    "WidthJ": f"{ri['J_learned']['mean_width_robust']:.3f}", "WidthJzero": f"{ri['J0']['mean_width_robust']:.3f}",
    "TanhJ": f"{math.tanh(P['J_learned']):.3f}",
    "ChairLlava": f"{RES['chair']['llava']['chair_i']:.3f}", "ChairQwen": f"{RES['chair']['qwen']['chair_i']:.3f}",
    "ChairIntern": f"{RES['chair']['internvl']['chair_i']:.3f}", "TruncQwen": RES["truncated_captions"]["qwen"],
}
for name, (lbl, key) in {"Llava": ("LLaVA-1.5-7B (test)", None), "Qwen": ("Qwen2.5-VL-3B", None),
                         "Intern": ("InternVL3-2B", None)}.items():
    v = VLM["vlms"][lbl]
    macros[f"HallRate{name}"] = f"{v['hallucinated_rate']:.3f}"
    macros[f"HallAbove{name}"] = f"{v['hallucinated_above_tau']:.3f}"
    macros[f"HallQninety{name}"] = f"{v['hallucinated_q90']:.2f}"
macros["TauFull"] = f"{VLM['tau_full_cal']:.3f}"
# extremes over S1 (for the Discussion) and the P2 slack in the measured regime
s1 = [f"{c}_{s}" for c, _ in CORR for s in range(1, 6)]
kmin = min(s1, key=lambda c: LV[c]["supported_kept"]["point"])
amax = max(s1, key=lambda c: LV[c]["abstention"]["point"])
macros["MinKeptSone"] = f"{LV[kmin]['supported_kept']['point']:.2f}"
macros["MinKeptSoneCond"] = kmin.replace("_", " ")
macros["MaxAbstSone"] = f"{LV[amax]['abstention']['point']:.2f}"
macros["MaxAbstSoneCond"] = amax.replace("_", " ")
macros["NClaimPreservedSone"] = sum(LV[c]["claim_risk"]["decision"] == "preserved" for c in s1)
macros["NImagePreservedSone"] = sum(LV[c]["image_risk"]["decision"] == "preserved" for c in s1)
macros["PtwoSlack"] = f"{math.tanh(P['J_learned']) * (ri['B'] - ri['eps']):.2f}"
macros["ShareBgtEps"] = f"{ri['share_B_gt_eps']:.2f}"
macros["GdinoKept"] = f"{LV['swap_grounding_dino']['supported_kept']['point']:.3f}"
macros["GdinoRecKept"] = f"{LV['swap_grounding_dino+recalibrated']['supported_kept']['point']:.3f}"
macros["InternKept"] = f"{LV['vlm_internvl']['supported_kept']['point']:.3f}"
# appendix table of S1 parameters, read from the frozen code
import sys as _sys
_sys.path.insert(0, str(ROOT))
from src.phase4.shifts import S1_PARAMS  # noqa: E402
_meaning = {"gaussian_blur": "Gaussian blur, $\\sigma$ (pixels)", "jpeg": "JPEG quality",
            "pixelate": "Pixelate, resize factor", "gaussian_noise": "Gaussian noise, s.d.\\ on $[0,1]$",
            "contrast": "Contrast factor $c$"}
(TAB / "tab_s1params.tex").write_text("\n".join(
    f"{_meaning[k]} & " + " & ".join(f"{v:g}" for v in vals) + " \\\\" for k, vals in S1_PARAMS.items()) + "\n\\bottomrule\n")
(ROOT / "paper/numbers.tex").write_text(
    "% generated by paper/make_figures.py - do not edit\n"
    + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
print("figures, tables and numbers.tex written")
