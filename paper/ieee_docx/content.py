"""Text of the IEEE Word manuscript, as structured blocks.

Every number is read from reports/phase4/*.json or paper/numbers.tex (written by
paper/make_figures.py); none is typed by hand.

Inline markup used in strings:
  $...$          inline math (LaTeX, converted to Word equations)
  [[k1,k2]]      citation keys -> IEEE numbers in order of first use
  ~~text~~       italic
  {fig:x} {tab:x} {eq:x} {sec:x}   cross-references resolved by build.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = json.loads((ROOT / "reports/phase4/phase4_results.json").read_text())
LV = RES["levels"]["0.1"]
N = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "paper/numbers.tex").read_text()))
HY = RES["hypotheses"]
PGM = RES["pgm"]


def r(c, m="claim_risk"):
    return f"{LV[c][m]['point']:.3f}"


def ci(c, m="claim_risk"):
    x = LV[c][m]
    return f"{x['point']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"


def lv(level, c, m="claim_risk"):
    return f"{RES['levels'][level][c][m]['point']:.3f}"


def dec(c, m="claim_risk"):
    return LV[c][m]["decision"]


H2 = HY["H2"]["per_condition"]

TITLE = ("When Do Conformal Hallucination Guarantees Survive Distribution Shift? "
         "A Pre-Registered Audit of Filters for Vision-Language Models")

AUTHORS = [
    ("S. Raja", "Department of Mathematics", "nellairajaa@yahoo.com", "ORCID: 0000-0003-1281-9948"),
    ("A. Antony Selva Jasfer", "Department of Artificial Intelligence and Data Science",
     "nc.ai.u4aid24068@nc.students.amrita.edu", None),
    ("Akash Narayanan C", "Department of Artificial Intelligence and Data Science",
     "nc.ai.u4aid24003@nc.students.amrita.edu", None),
    ("U. Priyan", "Department of Artificial Intelligence and Data Science",
     "nc.ai.u4aid24056@nc.students.amrita.edu", None),
    ("Nisanth S", "Department of Artificial Intelligence and Data Science",
     "nc.ai.u4aid24049@nc.students.amrita.edu", None),
]
ORG = ("Amrita School of Computing", "Amrita Vishwa Vidyapeetham", "Nagercoil 629001, India")

# IEEE asks for no symbols or math in the abstract.
ABSTRACT = (
    "Vision-language models (VLMs) often describe objects that are not in the image. Conformal filters address this "
    "by keeping an object claim only when an open-vocabulary detector finds enough visual evidence for it, and they "
    "guarantee that at most a chosen fraction of hallucinated claims survives. The guarantee assumes that test data "
    "are exchangeable with the calibration data, which deployed systems routinely violate. We pre-registered and ran "
    "an audit of two such filters, one acting on individual claims and one on whole images, calibrated on COCO "
    "captions from LLaVA-1.5-7B scored by OWLv2, under 25 image degradations, two detector swaps, two VLM swaps and "
    "a move to the AMBER benchmark. All 25 degradations kept the claim-level guarantee, although a simple "
    f"stochastic-dominance condition that suffices for this held in only {N['NHtwo']} of them. Replacing the detector "
    f"with Grounding DINO broke both filters, raising the claim-level risk to {r('swap_grounding_dino')} against a "
    "target of 0.10; recomputing the threshold on the new detector's scores for the same labelled calibration claims "
    "restored the guarantee without new annotation. Replacing the VLM broke the claim-level guarantee "
    f"({r('vlm_internvl')} for InternVL3-2B) but not the image-level one: the newer model hallucinates less often, but "
    "the objects it does hallucinate are more often ones the detector partly sees. A root-dominance bound for Ising "
    "trees under a shared perturbation budget explains why linking a caption's claims in a graphical model does not "
    "make these filters more robust at the measured coupling strength."
)
KEYWORDS = ("conformal prediction, distribution shift, object hallucination, vision-language models, "
            "stochastic dominance, belief propagation, Ising model, pre-registration")

B = []  # blocks


def h1(t, label=None): B.append(("h1", t, label))
def h2(t, label=None): B.append(("h2", t, label))
def p(t): B.append(("p", t))
def eq(tex, label=None): B.append(("eq", tex, label))
def bullets(items): B.append(("bullets", items))
def fig(path, caption, label): B.append(("fig", path, caption, label))
def table(caption, label, header, rows, widths, wide=False, font=7, align=None):
    B.append(("table", caption, label, header, rows, widths, wide, font, align))


# ======================================================================== I
h1("Introduction", "sec:intro")
p("Vision-language models (VLMs) write fluent descriptions of images, but they often mention objects that "
  "are not there. This failure, called object hallucination, has been measured for years with CHAIR, which "
  "compares the objects named in a caption with the objects annotated in the image [[rohrbach2018chair]], "
  "and with benchmarks such as POPE and AMBER [[li2023pope,wang2023amber]]. A practical defence is to check "
  "every mentioned object against an independent source of visual evidence, usually an open-vocabulary "
  "object detector, and to remove the mentions that the evidence does not support.")
p("Two recent methods make this check statistically principled. CEBC sets the detector threshold at the "
  "$(1-\\delta)$ quantile of the detector scores of hallucinated mentions in a calibration set, so that "
  "\"only a small fraction (about $\\delta$) of hallucinated mentions\" pass the test [[cebc]]. ConfLVLM "
  "filters the claims in a VLM's answer so that, with probability at least $1-\\alpha$, the retained answer "
  "contains no more than a tolerated amount of error [[conflvlm]]. When the threshold is chosen by split "
  "conformal prediction, both kinds of filter come with finite-sample guarantees that hold without any model "
  "of the VLM or of the detector [[vovk2005algorithmic,angelopoulos2023gentle]].")
p("These guarantees rest on one assumption: the calibration data and the data the filter sees later must be "
  "exchangeable. Deployed systems break this assumption all the time. Images arrive blurred, compressed or "
  "noisy; the detector that supplies the evidence is upgraded or replaced; the filter is put in front of a "
  "newer VLM; the images come from another source. The authors of ConfLVLM name this as their main "
  "limitation, writing that their framework \"assumes exchangeability of the calibration and test datasets\" "
  "and \"may not always hold in applications where distribution shifts frequently occur\" [[conflvlm]]. CEBC "
  "reports reduced robustness \"under domain shift\" and recalibrates its threshold for each dataset [[cebc]].")
p("Robust conformal methods exist, but each needs the shift to be described in advance: as an $\\ell_2$ "
  "perturbation of known size at the test point [[gendler2022adversarially]], as an $f$-divergence ball whose "
  "radius must be at least the true divergence [[cauchois2024robust]], through a known or estimated "
  "likelihood ratio between covariate distributions [[tibshirani2019covariate]], or through fixed weights "
  "whose coverage loss stays small only when the data are nearly exchangeable in a total-variation sense "
  "[[barber2023beyond]]. Swapping the detector or the caption generator is not naturally described in any of "
  "these ways. For text-only language models, recent work shows that conformal factuality can fall below its "
  "target when the calibration claims come from a different generator [[chen2026conformalrag]] or when the "
  "model configuration changes, and that recalibrating with a small labelled sample from deployment largely "
  "closes the gap [[zhu2026configuration]]. Whether the same happens to detector-evidence filters for VLMs, "
  "and for which kinds of shift, has not been tested.")
p("This paper asks one empirical question and one theoretical question. Which realistic shifts leave the "
  "guarantee of a calibrated hallucination filter intact without recalibration, which ones break it, and why? "
  "And what is the cheapest repair when it breaks? We study a claim-level filter in the style of CEBC and an "
  "image-level filter in the style of ConfLVLM, both calibrated on clean COCO captions from LLaVA-1.5-7B and "
  "scored by OWLv2, and we expose them to four families of shift ({fig:pipeline}). Our contributions are:")
bullets([
    "A pre-registered shift audit. Hypotheses, decision rules, shift parameters and analysis code were frozen "
    "in a public repository before any test caption or score was computed. To our knowledge this is the first "
    "systematic test of detector-evidence conformal filters for VLMs under image degradation, detector swap, "
    "VLM swap and domain change.",
    "A checkable sufficient condition, and evidence that it is far from necessary. Adapting known one-sided "
    "arguments for conformal prediction under shift and contamination [[clarkson2024split,doula2026conformal]] "
    "to the scores of hallucinated claims, a shift that makes those scores stochastically no larger keeps the "
    f"clean guarantee (Proposition 1). All 25 degradations kept the claim-level guarantee, yet the condition held "
    f"in only {N['NHtwo']}; what matters is the change at the calibrated threshold.",
    f"A detector swap breaks the guarantee and a label-free recalibration repairs it. Replacing OWLv2 with "
    f"Grounding DINO raised the claim-level risk from {N['CleanClaim']} to {N['GdinoClaim']} against a target of "
    f"0.10. Recomputing the threshold on the new detector's scores for the same calibration claims brought it "
    f"back to {N['GdinoRecClaim']}.",
    f"A VLM swap breaks the claim-level guarantee but not the image-level one. Thresholds calibrated on "
    f"LLaVA-1.5 were violated on InternVL3-2B ({N['InternClaim']}), which is the VLM counterpart of cross-generator "
    "failures reported for language models [[chen2026conformalrag]]. An exploratory analysis suggests why: the "
    "newer VLM hallucinates less often, but its hallucinations are more often objects that the detector partly sees.",
    "A graphical-model explanation of why coupling claims does not help. For an attractive Ising tree over a "
    "caption's claims we prove a root-dominance bound (Proposition 2). It combines the structure-free "
    "sensitivity bound of Chan and Darwiche [[chan2002numbers]] with the per-edge message contraction of Ihler "
    "et al. [[ihler2005loopy]]; the new part is the analysis under one perturbation budget shared by all claims, "
    "which gives an exact interval when the budget is no larger than the target's own radius.",
])
p("{sec:related} reviews related work. {sec:setting} defines the filters and their guarantees, and "
  "{sec:theory} gives the two propositions. {sec:design} describes the pre-registered design, {sec:results} "
  "reports the results and {sec:discussion} discusses what they mean in practice and where they stop. "
  "The appendix contains the full proof of Proposition 2 and the results for every degradation.")
fig("col_pipeline.png", "Setting of the audit. A VLM describes an image; each object it mentions is a claim. "
    "A detector scores every claim, and the claim is kept only if its score exceeds a threshold calibrated "
    "once on clean data. Orange labels mark where each family of shift enters the pipeline.", "fig:pipeline")

# ======================================================================== II
h1("Related Work", "sec:related")
h2("Object hallucination and how it is measured")
p("CHAIR counts the COCO objects [[lin2014microsoft]] that a caption mentions but the image annotations do not "
  "contain [[rohrbach2018chair]]; it is the labelling rule we use. POPE turns hallucination into yes/no questions "
  "about objects [[li2023pope]], and AMBER provides an LLM-free benchmark with annotated objects that are present "
  "and annotated \"hallucinatory target objects\" that models are likely to imagine [[wang2023amber]]. Detection "
  "methods use the VLM's own signals, such as the uncertainty of its answers to semantically equivalent perturbed "
  "prompts [[zhang2024vluncertainty]], the drift of cross-modal attention between layers combined with "
  "masking-based verification [[wen2026cadmp]], or a Bayesian combination of internal attention and logit "
  "signals with caption statistics [[zohrabi2026haloprobe]]. Huang et al. detect visual hallucinations by measuring, with Dempster-Shafer theory, the degree of conflict "
  "between positive and negative evidence built from the model's features [[huang2025evidential]]. These methods "
  "produce scores. Our question is what happens to a guarantee attached to such a score when conditions change.")
h2("Conformal guarantees for generated content")
p("Split conformal prediction turns any score into a decision rule with a finite-sample guarantee under "
  "exchangeability [[vovk2005algorithmic,angelopoulos2023gentle]], and conformal risk control extends the idea "
  "from coverage to expected losses [[angelopoulos2022conformal]]. For language models, Mohri and Hashimoto "
  "remove low-confidence sub-claims until the output is correct with probability at least $1-\\alpha$ "
  "[[mohri2024language]], and Cherian et al. add group-conditional guarantees and lower the guaranteed level "
  "adaptively to keep outputs useful [[cherian2024large]]. For VLMs, ConfLVLM applies conformal filtering to "
  "claims about images [[conflvlm]]; CEBC calibrates a detector-evidence threshold on hallucinated object "
  "mentions and combines it with candidate selection and constrained rewriting [[cebc]]; and BCEA lets a VLM "
  "answer, abstain, or re-examine the image under a compute budget, with a high-probability bound on the "
  "hallucination rate among asserted claims [[xu2026bcea]]. All of these guarantees are stated for "
  "exchangeable calibration and test data (in BCEA's theorem, i.i.d. data).")
p("Under shift, Chen et al. find that conformal factuality for retrieval-augmented LLMs \"is not robust to "
  "distribution shifts\", for example when GPT-4 calibration claims are used to filter claims from smaller "
  "open-source LLMs, and that adding distractors to the calibration set raises factuality at a high cost in "
  "informativeness [[chen2026conformalrag]]. Zhu et al. show that changing the prompt template, decoding "
  "temperature or quantization \"consistently erodes\" coverage, and that recalibrating with 50 labelled "
  "deployment examples, triggered by a label-free shift check, largely closes the gap [[zhu2026configuration]]. "
  "CoFact reweights calibration samples by online density-ratio estimation, with an exact guarantee when the "
  "true ratio is known and an asymptotic, time-averaged one when it is estimated [[hu2026cofact]]. For VLMs, BCEA "
  "shows that reusing a threshold after its own evidence-acquisition step raises scores and breaks the guarantee, "
  "and recalibrates [[xu2026bcea]]; IntroConformal assumes calibration and test data from the same distribution "
  "and states that model updates \"require recalibration\" [[atabuzzaman2026introconformal]]. We step back and ask "
  "which realistic shifts make recalibration necessary for detector-evidence VLM filters, and which do not.")
h2("Conformal prediction under distribution shift")
p("Under covariate shift with the conditional distribution of the label unchanged and a known likelihood ratio, "
  "weighting the calibration scores restores validity; an estimated ratio is supported empirically "
  "[[tibshirani2019covariate]]. Without exchangeability, the coverage loss of (weighted) split conformal "
  "prediction is bounded by a weighted sum of total-variation distances between the data and the data with the "
  "test point swapped with each calibration point [[barber2023beyond]]. That bound also applies to standard "
  "unweighted split conformal prediction, which is one reason why mild shifts can cost little coverage. Robust "
  "approaches guarantee coverage for every test distribution in an $f$-divergence ball [[cauchois2024robust]] or "
  "under bounded $\\ell_2$ perturbations of the input [[gendler2022adversarially]]. These methods need the form "
  "of the shift in advance, together with its size, an upper bound on it, or the likelihood ratio. Closest to our "
  "Proposition 1, one-sided comparisons of score distributions describe when transported or contaminated "
  "calibration stays valid: Doula bounds miscoverage by the one-sided gap between transported and target score "
  "distributions, notes that the opposite deviation makes the calibration conservative, and tests this on image "
  "classifiers under ImageNet-C-style corruptions [[doula2026conformal]]; Clarkson et al. obtain over- or "
  "under-coverage from stochastic-dominance assumptions on contaminated calibration scores "
  "[[clarkson2024split]]; and Lee observes that coverage collapses when test scores shift to the right "
  "[[lee2026diagnosing]]. Our shifts are discrete changes of a pipeline component, or realistic corruptions "
  "[[hendrycks2019benchmarking]]. We measure them instead of bounding them, apply the one-sided condition to the "
  "scores of hallucinated claims, and test both the condition and the guarantee directly.")
h2("Sensitivity of probabilistic graphical models")
p("How far a posterior can move when a parameter moves is a classical question for Bayesian networks. Chan and "
  "Darwiche show that for a binary variable, changing one parameter changes the log-odds of any query by at most "
  "the parameter's own log-odds change, \"with no assumptions whatsoever about the structure\" of the network, "
  "and remark that specific topologies can be more robust [[chan2002numbers]]. They also bound the change in any "
  "conditional odds by a distance between distributions [[chan2005distance]] and extend sensitivity analysis to "
  "Markov networks [[chan2005markov]]. For belief propagation, Ihler et al. prove that message errors contract at "
  "each update with slope $\\tanh(\\log d(\\psi))$ for an edge potential of dynamic range $d(\\psi)$, which for an "
  "Ising edge is $\\tanh J$, with a separate error cap per message [[ihler2005loopy]]. Mooij and Kappen propagate "
  "box bounds on marginals [[mooij2008bounds]], and Hopkins and Li study belief propagation on ferromagnetic Ising "
  "trees when an adversary flips a fraction of the observed leaves [[hopkins2024adversarially]]. These bounds are "
  "either tight in the worst case regardless of where the query sits in the graph, or they budget the observations. "
  "Proposition 2 perturbs the potentials, couples the per-node radii through one shared budget, and uses the tree "
  "contraction to show that coupling a caption's claims cannot narrow their robust intervals at the coupling "
  "strengths we measure.")

# ======================================================================== III
h1("Setting and Filters", "sec:setting")
h2("Claims, labels and scores")
p("A VLM writes a caption for an image $x$. Following CHAIR [[rohrbach2018chair]], we extract the COCO object "
  "categories that the caption mentions; each mention is a claim. A claim is hallucinated ($Y=1$) if its category "
  "is not among the image's annotated objects and supported ($Y=0$) otherwise. An open-vocabulary detector scores "
  "every claim: we query it with \"a photo of a <category>\", take the highest box probability, and use its logit, "
  "so a larger score $S$ means more visual evidence that the object is present.")
h2("Claim-level filter")
p("Let $S_1,\\dots,S_m$ be the scores of the hallucinated claims in a calibration set. The filter keeps a claim "
  "if and only if its score exceeds the threshold")
eq("\\hat\\tau=S_{(k)},\\qquad k=\\lceil (m+1)(1-\\delta)\\rceil", "eq:tau")
p("where $S_{(k)}$ is the $k$-th smallest score ($\\hat\\tau=+\\infty$ if $k>m$). Its risk is the probability that "
  "a hallucinated test claim is kept,")
eq("R_c=P(S>\\hat\\tau\\mid Y=1).", "eq:rc")
p("This is the threshold component of CEBC, which sets $\\tau=\\operatorname{clip}(\\operatorname{Quantile}_{1-\\delta}(ℋ),"
  "\\tau_{\\min},\\tau_{\\max})$ over the detector scores $ℋ$ of hallucinated calibration mentions and argues "
  "that only \"about $\\delta$\" of them then pass [[cebc]]. We drop the clipping and use the finite-sample index $k$ "
  "in {eq:tau}, which turns \"about $\\delta$\" into the guarantee below; CEBC states no formal guarantee. We study "
  "only this filtering step, not CEBC's candidate selection and rewriting.")
h2("Image-level filter")
p("For image $j$, let $M_j$ be the largest score among its hallucinated claims ($M_j=-\\infty$ if it has none). The "
  "threshold $\\hat\\tau_I$ is the conformal $(1-\\alpha)$ upper quantile of $M_1,\\dots,M_n$ over calibration images, "
  "computed as in {eq:tau}, and the risk is the probability that a test image keeps at least one hallucinated claim:")
eq("R_I=P(M>\\hat\\tau_I).", "eq:ri")
p("This is the guarantee $P(L\\le\\lambda)\\ge1-\\alpha$ of ConfLVLM [[conflvlm]] with $L$ the indicator that a kept "
  "claim is hallucinated and tolerance $\\lambda=0$; as in ConfLVLM, an image with no kept claim has zero loss. Our "
  "implementation differs from theirs in using CHAIR labels and detector scores, where ConfLVLM decomposes answers "
  "into claims by prompting a language model and scores them with internal or external confidence such as token "
  "log-probabilities or CLIP [[conflvlm]].")
h2("The clean guarantee")
p("If a test claim is exchangeable with the calibration claims, the standard conformal argument gives $R_c\\le\\delta$; "
  "if a test image is exchangeable with the calibration images, $R_I\\le\\alpha$ "
  "[[vovk2005algorithmic,angelopoulos2023gentle]]. Both are unconditional false-acceptance probabilities. Neither "
  "controls the share of hallucinated claims among the claims that are kept, a point we return to in "
  "{sec:discussion}. We use $\\delta=\\alpha=0.10$ throughout, with 0.05 and 0.20 as secondary targets.")

# ======================================================================== IV
h1("Theory", "sec:theory")
h2("A sufficient condition for a guarantee to survive a shift")
p("Under a shift, a hallucinated test claim has score $\\tilde S$ instead of $S$ (at image level, $\\tilde M$ instead "
  "of $M$). The calibration data, and therefore $\\hat\\tau$, are unchanged.")
p("~~Proposition 1.~~ (a) If the shifted score of a hallucinated claim is stochastically no larger than its clean "
  "score,")
eq("P(\\tilde S>t)\\le P(S>t)\\quad \\text{for all } t\\in ℝ,", "eq:dom")
p("then the clean-calibrated filter keeps its guarantee: $P(\\tilde S>\\hat\\tau)\\le P(S>\\hat\\tau)\\le\\delta$. "
  "(b) The same holds at image level with $M$, $\\tilde M$ and $\\alpha$. (c) If the shift never raises the score of a "
  "hallucinated claim ($\\tilde S\\le S$ almost surely), the conditions of (a) and (b) hold.")
p("~~Proof.~~ Write $f(t)=P(S>t)$ and $g(t)=P(\\tilde S>t)$, so that {eq:dom} says $g\\le f$. The test point is "
  "independent of the calibration sample, and hence of $\\hat\\tau$, so conditioning on $\\hat\\tau$ gives")
eq("P(\\tilde S>\\hat\\tau)=E\\,g(\\hat\\tau)\\le E\\,f(\\hat\\tau)=P(S>\\hat\\tau)\\le\\delta,",
   "eq:p1")
p("where the last step is the clean guarantee. Part (b) is identical. For (c), $\\{\\tilde S>t\\}\\subseteq\\{S>t\\}$ for "
  "every $t$, and $\\tilde M=\\max_i\\tilde S_i\\le\\max_i S_i=M$. $\\square$")
p("Two remarks matter for the experiments. First, one-sided arguments of this kind are known for conformal "
  "prediction under shift and contamination [[doula2026conformal,clarkson2024split]]; what is useful here is that "
  "the condition concerns only hallucinated claims and can be checked on a few labelled shifted claims. Second, the "
  "proof uses {eq:dom} only at $t=\\hat\\tau$, so the condition is sufficient but far from necessary. "
  "{sec:res-s1} shows that this gap matters in practice. Nothing is assumed about supported claims: a shift that "
  "lowers supported scores more than hallucinated ones keeps the risk below $\\delta$ while raising the share of "
  "hallucinated claims among those kept.")
h2("Why coupling claims does not narrow robust intervals", "sec:p2")
p("One could hope to make the filter more robust by modelling the claims of a caption jointly, so that confident "
  "neighbours support or contradict an uncertain claim. We model the labels $h_i\\in\\{-1,+1\\}$ of an image's claims "
  "($+1$ for supported) as an attractive Ising tree,")
eq("p(h)\\propto\\exp\\Big(\\sum_i\\theta_ih_i+\\sum_{(i,j)}J_{ij}h_ih_j\\Big),\\qquad J_{ij}\\ge0,", "eq:ising")
p("where $\\theta_i$ is the evidence for claim $i$ alone and $J_{ij}$ couples neighbouring claims. Belief propagation "
  "is exact on a tree and gives $P(h_r=+1)=\\sigma(2\\eta_r)$, where the effective field $\\eta_r$ of a target claim $r$ "
  "is its own field plus the messages from its neighbours,")
eq("\\eta_r=\\theta_r+\\delta_r+\\sum_{c}\\nu_{c\\to r},", "eq:bp")
p("where $\\delta_r$ is a perturbation of the target's field, introduced below, and the message from a neighbour $c$ is "
  "$\\nu_{c\\to r}=\\tanh^{-1}(\\tanh J_{cr}\\tanh\\eta_{c\\to r})$, with $\\eta_{c\\to r}$ the field of $c$ in the "
  "subtree on its side of the edge.")
p("Suppose the fields are perturbed by $\\delta_i$ with $|\\delta_i|\\le\\varepsilon_i$ for each claim and "
  "$|\\delta_1|+\\dots+|\\delta_n|\\le B$ in total. For each claim $i\\ne r$ let")
eq("\\gamma_i=\\prod_{e\\in\\operatorname{path}(i\\to r)}\\tanh J_e,", "eq:gamma")
p("let $\\gamma_{\\max}<1$ be the largest $\\gamma_i$, and let $a=\\min(\\varepsilon_r,B)$.")
p("~~Proposition 2 (root dominance).~~ The largest effective field $\\eta_r^{\\operatorname{sup}}$ reachable under the "
  "budget satisfies")
eq("\\eta_r+a\\le\\eta_r^{\\operatorname{sup}}\\le\\eta_r+a+\\gamma_{\\max}\\max(0,B-\\varepsilon_r),", "eq:p2")
p("and symmetrically for the infimum. In particular, if $B\\le\\varepsilon_r$ the robust interval for $P(h_r=+1)$ is "
  "exactly")
eq("[\\sigma(2(\\eta_r-B)),\\ \\sigma(2(\\eta_r+B))],", "eq:p2int")
p("which does not depend on the graph. The proof is in Appendix A. In words: under a shared budget, spending "
  "perturbation on the target claim moves its field one-for-one, while spending it on a neighbour moves it by at most "
  "$\\tanh J<1$ per edge, so the worst case always puts as much budget as allowed on the target itself. The graph can "
  "only add a further channel of size at most $\\gamma_{\\max}(B-\\varepsilon_r)$. The \"gain $\\le1$\" ingredient is the "
  "structure-free result of Chan and Darwiche [[chan2002numbers]], and the path product $\\gamma_i$ is the Ising case of the "
  "contraction of Ihler et al. [[ihler2005loopy]]; our addition is the shared budget and the exact interval it yields.")

# ======================================================================== V
h1("Experimental Design", "sec:design")
h2("Pre-registration")
p("The study was pre-registered. Hypotheses, decision rules, shift parameters and the analysis code were committed "
  "to a public repository and frozen under the git tag prereg-v2 before any test caption or detector score was "
  "computed (https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection; "
  "archived at https://doi.org/10.5281/zenodo.23810032). The tests were then run once. Earlier exploratory pilots, "
  "on 200 COCO val2014 images from a development split kept apart from calibration and test and on 200 AMBER images, motivated the hypotheses; those images are "
  "excluded from every confirmatory analysis by code. Analyses that were not pre-registered are labelled "
  "exploratory.")
h2("Data")
p(f"We split COCO val2014 [[lin2014microsoft]] with a fixed seed into calibration sets Cal-A (300 images) and Cal-B "
  f"(400 images) and a locked test set of 500 images. Calibration uses all 700 calibration images, which give "
  f"{N['NCal']} claims. The {N['NTestImg']} test images that produced at least one claim give {N['NTest']} claims, "
  f"{N['NTestHall']} of them hallucinated. For the domain shift we use the 804 generative-task images of AMBER "
  f"[[wang2023amber]] that were not in the pilot. Claims there are labelled with the two lists of AMBER's annotation "
  f"file: objects present in the image (supported) and annotated hallucinatory target objects (hallucinated). Of "
  f"{N['NAmberClaims']} AMBER claims, {N['NAmberUnknown']} are in neither list and are excluded. This differs from "
  "AMBER's own CHAIR, which counts every vocabulary object not marked present as hallucinated; we keep only claims "
  "whose label was annotated, because AMBER's authors note that the hallucinatory list is incomplete and that noun "
  "extraction can mislabel words such as \"orange\" [[wang2023amber]].")
h2("Source pipeline and implementation")
p("Captions come from LLaVA-1.5-7B [[liu2024improved]] with greedy decoding, the prompt \"Please describe this image "
  "in detail.\" and at most 512 new tokens. Claims and labels come from our re-implementation of CHAIR, which uses the "
  "official synonym list and differs from the original only in its tokeniser and singulariser. Scores come from OWLv2 "
  "[[minderer2023scaling]] (checkpoint owlv2-base-patch16-ensemble). All GPU work ran on Kaggle (two NVIDIA T4 GPUs, "
  "PyTorch 2.11, Transformers 5.16), one process per GPU, in sessions that resumed from saved results. The analysis "
  "was re-run from the raw files on a separate machine and reproduced the reported results exactly.")
h2("Shifts")
p("Every shift is applied to the test data only, one at a time ({fig:pipeline}).")
bullets([
    "S1, degradation: four of the 15 ImageNet-C corruptions (Gaussian noise, contrast reduction, pixelation and "
    "JPEG compression) and Gaussian blur, which ImageNet-C places in its extra validation set "
    "[[hendrycks2019benchmarking]], each at five severities. The severity parameters are those of the ImageNet-C code "
    "({tab:s1params}), re-implemented with PIL and NumPy; Gaussian noise is seeded by image and severity, so every run "
    "produces the same images.",
    "S2, detector swap: OWLv2 replaced by OWL-ViT v1 [[minderer2022simple]] or by Grounding DINO (base) "
    "[[liu2024grounding]]. We also test recalibration: the threshold is recomputed on the new detector's scores for "
    "the same labelled calibration claims, which needs no new labels because a claim's label does not depend on the "
    "detector.",
    "S3, VLM swap: LLaVA-1.5-7B replaced by Qwen2.5-VL-3B-Instruct [[bai2025qwen25]] or InternVL3-2B "
    "[[zhu2025internvl3]], with the same prompt, greedy decoding and at most 512 new tokens.",
    "S4, domain: COCO val2014 replaced by AMBER. AMBER's images come from the MS-COCO 2014 test set and from "
    "Unsplash [[wang2023amber]], so this is a partial domain shift: new images and annotations, partly from the same "
    "source.",
    "S5, object removal (exploratory): for one supported claim per test image, all COCO boxes of that category, "
    "enlarged by 10%, are filled with the image's mean colour, which makes the claim hallucinated by construction.",
])
table("ImageNet-C severity parameters used for S1", "tab:s1params",
      ["Corruption", "1", "2", "3", "4", "5"],
      [["Gaussian blur, sigma (px)", "1", "2", "3", "4", "6"],
       ["JPEG quality", "25", "18", "15", "10", "7"],
       ["Pixelate, resize factor", "0.6", "0.5", "0.4", "0.3", "0.25"],
       ["Gaussian noise, s.d. on [0, 1]", "0.08", "0.12", "0.18", "0.26", "0.38"],
       ["Contrast factor", "0.4", "0.3", "0.2", "0.1", "0.05"]],
      [1.55, 0.38, 0.38, 0.38, 0.38, 0.38])
h2("Hypotheses")
p("Six hypotheses were frozen before testing. H1: every S1 condition preserves both targets (reported together with "
  "the share of supported claims kept and the abstention rate, since a filter that keeps nothing is trivially safe). "
  "H2: under S1, the scores of hallucinated claims are stochastically no larger than on clean images, which is the "
  "condition of Proposition 1. H3: at least one S2 swap violates the claim-level target. H4: recalibration repairs "
  "every S2 swap. H5: thresholds calibrated on LLaVA transfer to each S3 VLM. H6: they transfer to AMBER.")
h2("Estimation and decision rules")
p("Thresholds are averaged over 200 calibration resamples, each using 80% of the calibration images. Uncertainty is "
  "quantified by an image-level cluster bootstrap with 2,000 resamples [[efron1993introduction]], because claims from "
  "one caption are not independent. A condition is violated if the 95% confidence interval of its risk lies entirely "
  "above the target, preserved if the upper bound is at most the target plus 0.03, and inconclusive otherwise. H2 is "
  "assessed per condition by")
eq("D^{+}=\\max_t\\,(P(\\tilde S>t)-P(S>t))", "eq:dplus")
p("on paired hallucinated test claims, and it holds for a condition if the upper bound of its 95% interval is at most "
  "0.03. A hypothesis is supported if all its conditions meet the rule. For H1 and H3 to H6 we also report a bootstrap "
  "p-value, the share of resamples on the wrong side of the margin taken over the hypothesis's conditions, "
  "Holm-adjusted within each filter [[holm1979simple]].")
h2("Graphical-model analysis (descriptive)")
p("We fit a logistic model of each claim's label on its OWLv2 score on Cal-A, and a chain Ising model over each "
  "caption's claims in caption order with one shared coupling $J\\in[0,2]$, chosen by exact likelihood on Cal-A. "
  "Inference is by exact belief propagation. For robust intervals, $\\varepsilon$ and $B$ are set to the 0.9 quantiles "
  "of the changes in claim fields caused by the S1 degradations on Cal-A. These results describe the data and test no "
  "hypothesis.")

# ======================================================================== VI
h1("Results", "sec:results")
p("{tab:hyp} summarises the pre-registered hypotheses. {fig:forest} and {tab:main} give the shifts other than "
  "degradation, and {fig:s1} the 25 degradations.")
hyp_rows = []
HN = {"H1": "Degradation (S1) is safe", "H3": "A detector swap breaks the claim-level target",
      "H4": "Recalibration repairs detector swaps", "H5": "LLaVA thresholds transfer to other VLMs",
      "H6": "Transfer to AMBER"}
for h in ["H1", "H3", "H4", "H5", "H6"]:
    a, b = HY["claim_risk"][h], HY["image_risk"][h]
    hyp_rows.append([h, HN[h], f"{'Yes' if a['supported'] else 'No'} ({a['p_holm']:.3f})",
                     f"{'Yes' if b['supported'] else 'No'} ({b['p_holm']:.3f})"])
nh2 = sum(v["supported"] for v in H2.values())
hyp_rows.insert(1, ["H2", "Hallucinated scores do not rise under S1", f"No ({nh2} of 25 hold)", "n/a"])
table("Pre-registered hypotheses (target 0.10). Supported? (Holm-adjusted p-value)", "tab:hyp",
      ["", "Hypothesis", "Claim level", "Image level"], hyp_rows, [0.3, 1.65, 0.75, 0.75])
h2("Clean reference")
p(f"On the clean test set the claim-level filter kept {N['CleanClaim']} of hallucinated claims and the image-level "
  f"filter let {N['CleanImage']} of images keep at least one hallucination, both within the target of 0.10. The "
  f"claim-level filter kept {N['CleanKept']} of supported claims. The clean claim-level risk sits below 0.10 partly "
  "because of sampling: its 95% interval, "
  f"[{LV['clean']['claim_risk']['lo']:.3f}, {LV['clean']['claim_risk']['hi']:.3f}], contains the target.")
h2("Image degradation is safe even where the sufficient condition fails", "sec:res-s1")
p("All 25 degradations preserved the claim-level target ({fig:s1}). Risk generally fell as severity rose, because "
  "degradation lowers the detector's evidence for every object, present or not. At image level, three mild "
  f"conditions were inconclusive: Gaussian noise at severity 1 ({ci('gaussian_noise_1', 'image_risk')}) and contrast "
  f"at severities 1 and 2 ({ci('contrast_1', 'image_risk')} and {ci('contrast_2', 'image_risk')}). Their upper bounds "
  "lie just above the 0.13 margin, so H1 is supported for the claim-level filter only. Safety came at a cost in "
  f"utility ({{fig:utility}}): under the strongest Gaussian noise the claim-level filter kept "
  f"only {N['MinKeptSone']} of supported claims and the image-level filter abstained on {N['MaxAbstSone']} of images.")
fig("col_s1.png", "Realised risk under image degradation (S1) with image-level cluster-bootstrap 95% intervals. "
    "Dashed line: target 0.10; shaded band: preservation margin (0.10 to 0.13); dotted line: clean risk.", "fig:s1")
p(f"The condition of Proposition 1 held in only {N['NHtwo']} of the 25 conditions ({{fig:h2}}a): "
  "mild JPEG compression, noise and contrast changes raised some hallucinated scores somewhere in the distribution. "
  "What matters for validity is the change at the calibrated threshold, which was small and mostly negative "
  "({fig:h2}b). Where it was positive, as for Gaussian blur at severity 1 "
  f"(+{H2['gaussian_blur_1']['diff_at_tau']:.3f}), the clean filter's slack below the target absorbed it.")
fig("col_h2.png", "Hypothesis H2. (a) One-sided Kolmogorov-Smirnov gap between shifted and clean scores of "
    "hallucinated test claims (95% interval; H2 needs the upper bound at most 0.03, dashed). (b) The same "
    "difference evaluated only at the calibrated threshold, which is all that Proposition 1 needs.", "fig:h2")
fig("col_utility.png", "Utility under degradation: share of supported claims kept against claim-level risk; "
    "points along each line are severities 1 to 5.", "fig:utility")
h2("Changing the evidence source breaks the guarantee; label-free recalibration repairs it")
p(f"Replacing OWLv2 with Grounding DINO broke both filters, with claim-level risk {N['GdinoClaim']} and image-level "
  f"risk {N['GdinoImage']} ({{fig:forest}}). Grounding DINO gives high scores to absent "
  "objects (its authors note that it \"will produce false positive results in some cases\" [[liu2024grounding]]), "
  f"and its scores are on a different scale from OWLv2's: almost every claim cleared the old threshold, so "
  f"{N['GdinoKept']} of supported claims were kept. OWL-ViT v1 gives hallucinated claims lower scores and made the "
  f"filters conservative ({N['OwlvOneClaim']}). Recomputing the threshold on Grounding DINO's scores for the same "
  f"labelled calibration claims restored both targets ({N['GdinoRecClaim']} and {N['GdinoRecImage']}) with no new "
  "annotation.")
fig("col_forest.png", "Realised risk (log scale, 95% interval) under detector swaps, with and without "
    "recalibration, VLM swaps and the AMBER domain shift. Marker shape and colour give the pre-registered decision.",
    "fig:forest")
h2("Changing the VLM breaks the claim-level guarantee but not the image-level one")
p(f"Thresholds calibrated on LLaVA did not transfer at claim level: InternVL3-2B was violated ({N['InternClaim']}) and "
  f"Qwen2.5-VL-3B was inconclusive ({N['QwenClaim']}). The image-level filter was preserved for both "
  f"({N['InternImage']} and {N['QwenImage']}). The ordering held at the secondary targets: at 0.05, InternVL3-2B was "
  f"again violated ({lv('0.05', 'vlm_internvl')}) while Qwen2.5-VL-3B was preserved ({lv('0.05', 'vlm_qwen')}).")
p(f"~~Exploratory.~~ The two newer VLMs hallucinate less often: {N['HallRateIntern']} and {N['HallRateQwen']} of their "
  f"claims, against {N['HallRateLlava']} for LLaVA (instance-level CHAIR {N['ChairIntern']}, {N['ChairQwen']} and "
  f"{N['ChairLlava']}). The hallucinations they do make have a heavier upper tail of detector scores; the 90th "
  f"percentile is {N['HallQninetyIntern']} and {N['HallQninetyQwen']} logits against {N['HallQninetyLlava']} "
  f"({{fig:vlm}}). In other words, which objects get hallucinated is part of what the threshold "
  "is calibrated to, and that changes with the VLM. For a user the consequence is mild: among the claims kept from "
  f"InternVL3-2B captions, the share that was hallucinated was {N['InternKeptHall']}, against {N['CleanKeptHall']} for "
  f"LLaVA. {N['TruncQwen']} of the 500 Qwen2.5-VL captions reached the 512-token limit.")
fig("col_vlm.png", "Exploratory. Survival function of OWLv2 scores of hallucinated claims on the test images for each "
    "VLM, with the threshold calibrated on LLaVA (dashed). The newer VLMs' hallucinations cross the threshold more "
    "often.", "fig:vlm")
h2("Domain shift and object removal")
p(f"On AMBER both filters were preserved and conservative ({N['AmberClaim']} at claim level and {N['AmberImage']} at "
  f"image level), although the image-level filter abstained on {r('domain_amber', 'abstention')} of images. In the "
  f"exploratory object-removal test, the clean-calibrated claim-level filter still kept the removed object's claim "
  f"with probability {N['SfiveClaim']}, and {N['SfiveImage']} of images kept at least one hallucinated claim. Filling an object's boxes with the "
  "mean colour does not always remove the detector's evidence for it; a plausible but untested explanation is that "
  "the object's outline and its context outside the boxes remain visible.")
h2("Graphical-model view")
p(f"Claim labels within a caption were almost uncorrelated ({N['CorrCal']} on calibration data, {N['CorrTest']} on "
  f"test data). The learned chain coupling was $J={N['Jlearned']}$, so a neighbour can move a claim's field by at most "
  f"$\\tanh J={N['TanhJ']}$ per unit of its own change. The coupling improved the log-likelihood over $J=0$ by "
  f"{N['LLgain']} nats on Cal-A and left test ROC-AUC essentially unchanged ({N['AucIsing']} against {N['AucUnary']}; "
  f"{{tab:pgm}}). Robust intervals over {N['RobImgs']} test images "
  f"($\\varepsilon={N['RobEps']}$, $B={N['RobB']}$, so every claim was in the regime $B>\\varepsilon_r$) were wider with "
  f"the learned coupling (mean width {N['WidthJ']}) than without it ({N['WidthJzero']}), as Proposition 2 predicts: "
  "coupling gives the perturbation an extra channel and never removes the target's own.")
main_rows = []
OTH = [("clean", "Clean (reference)"), ("swap_owlvit_v1", "Detector: OWL-ViT v1"),
       ("swap_owlvit_v1+recalibrated", "OWL-ViT v1, recalibrated"),
       ("swap_grounding_dino", "Detector: Grounding DINO"),
       ("swap_grounding_dino+recalibrated", "Grounding DINO, recalibrated"),
       ("vlm_qwen", "VLM: Qwen2.5-VL-3B"), ("vlm_internvl", "VLM: InternVL3-2B"),
       ("domain_amber", "Domain: AMBER"), ("s5_object_removed", "Object removal (S5, exploratory)")]
DEC = {"violated": "violated", "preserved": "preserved", "inconclusive": "inconcl."}
for c, name in OTH:
    m = LV[c]
    kept = m["supported_kept"]["point"]
    main_rows.append([name, ci(c), DEC[dec(c)], ci(c, "image_risk"), DEC[dec(c, "image_risk")],
                      "n/a" if kept != kept else f"{kept:.3f}", f"{m['abstention']['point']:.3f}"])
table("Shifts other than degradation (target 0.10): risk [95% interval], pre-registered decision, share of supported "
      "claims kept by the claim-level filter, and abstention rate of the image-level filter", "tab:main",
      ["Condition", "Claim-level risk", "Decision", "Image-level risk", "Decision", "Kept", "Abstained"],
      main_rows, [1.85, 1.3, 0.7, 1.3, 0.7, 0.55, 0.65], wide=True, font=8)
pgm_rows = []
for split, sname in (("train_cal_a", "Cal-A"), ("test", "Test")):
    for mod, mname in (("unary_J0", "Unary"), ("ising_tree_J", "Ising")):
        m = PGM[split][mod]
        cf = m["confusion"]
        pgm_rows.append([sname, mname, f"{m['accuracy']:.3f}", f"{m['precision']:.3f}", f"{m['recall']:.3f}",
                         f"{m['f1']:.3f}", f"{m['roc_auc']:.3f}", str(cf['tp']), str(cf['fp']), str(cf['tn']), str(cf['fn'])])
table("Claim classification (supported if posterior above 0.5): unary model (J = 0) against chain Ising model",
      "tab:pgm", ["Data", "Model", "Acc.", "Prec.", "Rec.", "F1", "AUC", "TP", "FP", "TN", "FN"], pgm_rows,
      [0.4, 0.4, 0.33, 0.33, 0.33, 0.31, 0.33, 0.27, 0.24, 0.3, 0.27], font=6.5)

# ======================================================================== VII
h1("Discussion", "sec:discussion")
h2("A practical rule")
p(f"For detector-evidence filters, the shifts we tested fall into two groups. Changes to the images, namely the 25 "
  f"degradations and the move to AMBER, left the clean-calibrated guarantee in place: the claim-level target was "
  f"preserved in {N['NClaimPreservedSone']} of 25 degradations and the image-level target in "
  f"{N['NImagePreservedSone']}, with the other three inconclusive and close to the margin. Changes to the pipeline, "
  "meaning the detector that supplies the evidence and, at claim level, the VLM that writes the captions, broke it. "
  "A threshold is a property of the detector's score scale and of the population of hallucinations it was calibrated "
  "on, so it must be recalibrated whenever either changes, and it need not be when only the images degrade. "
  "Recalibration after a detector swap is cheap, because the labels of the calibration claims do not depend on the "
  "detector: rescoring the same claims with the new detector is enough.")
h2("Why degradation was benign")
p(f"Degradation lowers detector evidence for every object and so moves hallucinated claims away from the threshold. "
  f"Proposition 1 formalises the safe direction, but its condition, dominance at every threshold, held in only "
  f"{N['NHtwo']} of 25 conditions while the guarantee held in all of them. Two facts explain the gap. The proof only "
  "needs dominance at the calibrated threshold, and the change there was small or negative throughout. And standard "
  "split conformal prediction degrades gracefully when the test distribution is close to the calibration "
  "distribution [[barber2023beyond]], so the small increases we did see were absorbed by the clean filter's slack. "
  "For practitioners, this suggests checking the change at the threshold on a few labelled shifted claims rather than "
  "testing full dominance. The other side of safety is utility: under the strongest noise the claim-level filter kept "
  f"only {N['MinKeptSone']} of supported claims and the image-level filter abstained on {N['MaxAbstSone']} of images, "
  "so a preserved guarantee does not by itself make a filter useful.")
h2("Per-hallucination and per-output guarantees")
p(f"The VLM swap separates two quantities that are easy to conflate. The claim-level risk is the probability that a "
  f"hallucinated claim is kept; on InternVL3-2B it rose to {N['InternClaim']}. Yet among the claims the filter kept from "
  f"InternVL3-2B captions, the share that was hallucinated was {N['InternKeptHall']}, slightly lower than for LLaVA "
  f"({N['CleanKeptHall']}), and the image-level guarantee held. Which guarantee matters depends on the application. A "
  "system that must suppress a set fraction of every hallucination needs recalibration on the new VLM's claims, which "
  "requires new labels. A system that cares about how much of its output is wrong may be served by the image-level "
  "filter, which transferred. Similar cross-generator failures have been reported for language models "
  "[[chen2026conformalrag]]; our results suggest that the image-level formulation can be more robust to them.")
h2("Graph structure does not buy robustness here")
p(f"Coupling a caption's claims in an Ising chain barely changed accuracy and ROC-AUC, and it made robust intervals "
  f"wider. Proposition 2 explains why: under a shared budget a perturbation is always at least as effective on the "
  f"target claim as on a neighbour, whose influence is damped by $\\tanh J$ per edge. With the coupling we measured, "
  f"$J={N['Jlearned']}$, the graph can only add an extra channel for perturbation. Coupled models could help only if "
  f"claims within a caption were strongly correlated, and they were not (correlation {N['CorrTest']}).")
h2("Limitations")
p(f"The audit uses one source pipeline (LLaVA-1.5-7B and OWLv2) and COCO images, plus AMBER, whose images partly come "
  "from the same collection; other source VLMs, detectors and domains may behave differently. Labels come from CHAIR's "
  "synonym matching on COCO, and AMBER claims were restricted to annotated objects, so the AMBER hallucinations are "
  f"those its annotators thought likely to be imagined. The swapped VLMs are small (2 to 3 billion parameters), and "
  f"{N['TruncQwen']} of the 500 Qwen2.5-VL captions reached the token limit. The object-removal test masks boxes with "
  "the mean colour, which leaves context and object outline partly visible. Finally, the mechanism we propose for the "
  "VLM result is exploratory and was not pre-registered.")

# ======================================================================== VIII
h1("Conclusion")
p("Conformal filters for VLM hallucinations carry guarantees only under exchangeability. In a pre-registered audit "
  "we found that this assumption can be relaxed for image degradation and for a partial domain shift, where the clean "
  "guarantee survived, but not for changes to the pipeline. Swapping the detector broke both filters until the "
  "threshold was recalibrated on the same labelled claims, and swapping the VLM broke the claim-level guarantee while "
  "the image-level one held. A one-sided condition explains the safe cases but is far from necessary, and a "
  "root-dominance bound shows why coupling claims in a graphical model does not make the filters more robust. "
  "Recalibrating when the evidence source or the generator changes, and otherwise checking the score change at the "
  "calibrated threshold, is a cheap way to keep these guarantees valid in deployment.")

BACK = []  # unnumbered back matter (Heading 5 style)
BACK.append(("Data and Code Availability",
             "All code, the pre-registration, the frozen analysis (tag prereg-v2) and the scripts that produce every "
             "figure, table and number in this paper are at "
             "https://github.com/antonyjasfer/Budget-Coupled-Robust-Belief-Propagation-for-Visual-Hallucination-Detection "
             "and are archived on Zenodo (https://doi.org/10.5281/zenodo.23810032), together with the raw captions and "
             "detector scores (no images). Code is under the MIT licence and data under CC BY 4.0."))
BACK.append(("Acknowledgment",
             "The authors thank the Department of Mathematics and the Department of Artificial Intelligence and Data "
             "Science, Amrita School of Computing, Amrita Vishwa Vidyapeetham, Nagercoil, and Kaggle for GPU time. "
             "In line with IEEE policy on AI-generated content, the authors disclose that Claude (Anthropic) was used to "
             "write and test analysis code, check results against the raw files, locate quotations in cited papers and "
             "draft and edit text; the authors reviewed all such content and take full responsibility for this paper."))

# ======================================================================== appendix
APPX = []
APPX.append(("h1n", "Appendix A: Proof of Proposition 2"))
APPX.append(("p", "Root the tree at $r$. For a child $c$ of $r$, belief propagation gives the message "
             "$\\nu_{c\\to r}=\\tanh^{-1}(\\tanh J_{cr}\\tanh\\eta_{c\\to r})$, where $\\eta_{c\\to r}$ is the field of "
             "$c$ in the subtree below it, and $\\eta_r=\\theta_r+\\delta_r+\\sum_c\\nu_{c\\to r}$."))
APPX.append(("p", "~~Step 1 (one edge).~~ For $h(\\eta)=\\tanh^{-1}(\\tanh J\\tanh\\eta)$ with $J\\ge0$,"))
APPX.append(("eq", "h'(\\eta)=\\tanh J\\,\\frac{1-\\tanh^2\\eta}{1-\\tanh^2J\\tanh^2\\eta}\\in[0,\\tanh J],", "eq:a1"))
APPX.append(("p", "because $0\\le\\tanh^2J\\le1$ puts the fraction in $[0,1]$. This is the Ising case of the contraction "
             "in Theorem 8 of Ihler et al. [[ihler2005loopy]]."))
APPX.append(("p", "~~Step 2 (paths).~~ The field $\\delta_i$ of a node $i\\ne r$ enters $\\eta_r$ only through the messages "
             "along the unique path from $i$ to $r$. By the chain rule, $\\partial\\eta_r/\\partial\\delta_i$ is the product of "
             "the derivatives of Step 1 along that path, so $0\\le\\partial\\eta_r/\\partial\\delta_i\\le\\gamma_i$, while "
             "$\\partial\\eta_r/\\partial\\delta_r=1$. In particular $\\eta_r$ is non-decreasing in every $\\delta_i$."))
APPX.append(("p", "~~Step 3 (budget).~~ By Step 2 the supremum over the uncertainty set is attained with $\\delta\\ge0$. "
             "Writing $\\eta_r(\\delta)=\\eta_r(0,\\delta_{-r})+\\delta_r$ and applying the mean value theorem on the segment "
             "from $0$ to $(0,\\delta_{-r})$,"))
APPX.append(("eq", "\\eta_r(\\delta)-\\eta_r\\le\\delta_r+\\gamma_{\\max}(B-\\delta_r),", "eq:a2"))
APPX.append(("p", "because the perturbations of the other claims sum to at most $B-\\delta_r$. The right side equals "
             "$\\gamma_{\\max}B+(1-\\gamma_{\\max})\\delta_r$ and increases in $\\delta_r\\le a=\\min(\\varepsilon_r,B)$, which gives the upper bound in "
             "{eq:p2}; the lower bound is attained by putting $\\delta_r=a$ and every other perturbation to zero. If "
             "$B\\le\\varepsilon_r$ the two bounds coincide, the supremum is $\\eta_r+B$, and by symmetry the infimum is "
             "$\\eta_r-B$. Since $\\eta\\mapsto\\sigma(2\\eta)$ is increasing, the interval is {eq:p2int}. $\\square$"))
APPX.append(("h1n", "Appendix B: Results for Every Degradation"))
s1_rows = []
CORR = [("gaussian_blur", "Gaussian blur"), ("jpeg", "JPEG"), ("pixelate", "Pixelate"),
        ("gaussian_noise", "Gaussian noise"), ("contrast", "Contrast")]
for c, name in CORR:
    for s in range(1, 6):
        k = f"{c}_{s}"
        m = LV[k]
        s1_rows.append([name if s == 1 else "", str(s), ci(k), DEC[dec(k)], ci(k, "image_risk"), DEC[dec(k, "image_risk")],
                        f"{m['supported_kept']['point']:.3f}", f"{m['abstention']['point']:.3f}",
                        f"{H2[k]['d_plus']:.3f}", f"{H2[k]['diff_at_tau']:+.3f}"])
APPX.append(("table", "All S1 conditions (target 0.10): risk [95% interval], decision, supported claims kept, "
             "abstention, and the H2 statistics (D+ and the change at the threshold)", "tab:s1",
             ["Corruption", "Sev.", "Claim-level risk", "Decision", "Image-level risk", "Decision", "Kept", "Abst.",
              "D+", "At thr."], s1_rows, [1.1, 0.35, 1.2, 0.65, 1.2, 0.65, 0.48, 0.48, 0.45, 0.5], True, 7, None))

