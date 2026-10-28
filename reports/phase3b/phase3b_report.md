# Phase 3b — Shifts that can raise the evidence for absent objects

**Status: exploratory.** Calibration always on the Phase 1 COCO pilot (LLaVA-1.5 captions, OWLv2, clean images); 500 random image splits (50% calibration). Targets: claim-level δ = 0.1, image-level α = 0.1; robust rows target +β = 0.05. Brackets: 2.5–97.5% range across splits.

## Diagnostics

| Condition | Claims | Images | Hallucination rate | AUROC (evidence) | Median logit, hallucinated | Median logit, supported |
|---|---|---|---|---|---|---|
| coco_llava_owlv2 | 660 | 200 | 0.271 | 0.884 | -2.26 | -0.38 |
| coco_llava_owlvit_v1 | 660 | 200 | 0.271 | 0.857 | -2.81 | -0.92 |
| coco_llava_grounding_dino | 660 | 200 | 0.271 | 0.840 | -0.76 | 0.17 |
| coco_qwen_owlv2 | 574 | 200 | 0.197 | 0.884 | -2.39 | -0.24 |
| amber_llava_owlv2 | 357 | 200 | 0.095 | 0.929 | -3.21 | 0.35 |

- coco_llava_owlvit_v1: share of hallucinated claims scored HIGHER than by OWLv2: 0.30

- coco_llava_grounding_dino: share of hallucinated claims scored HIGHER than by OWLv2: 0.94

## Realised risk

Claim-level (CEBC-style): share of hallucinated claims that pass. Image-level (ConfLVLM-style, λ = 0): share of images whose kept claims include a hallucination.

| Level | Condition | Target | Risk (mean [2.5%, 97.5%]) | Supported claims kept | Exceeds target? |
|---|---|---|---|---|---|
| claim | COCO, same pipeline (reference) | 0.10 | 0.096 [0.020, 0.207] | 0.613 | no |
| image | COCO, same pipeline (reference) | 0.10 | 0.099 [0.030, 0.180] | 0.672 | no |
| claim | Detector swap → OWL-ViT v1 | 0.10 | 0.048 [0.011, 0.087] | 0.458 | no |
| image | Detector swap → OWL-ViT v1 | 0.10 | 0.053 [0.020, 0.090] | 0.509 | no |
| claim |   + one-sided measured radius | 0.15 | 0.000 [0.000, 0.000] | 0.124 | no |
| image |   + one-sided measured radius | 0.15 | 0.002 [0.000, 0.020] | 0.180 | no |
| claim | Detector swap → Grounding DINO | 0.10 | 0.528 [0.360, 0.813] | 0.928 | YES |
| image | Detector swap → Grounding DINO | 0.10 | 0.410 [0.290, 0.525] | 0.970 | YES |
| claim |   + one-sided measured radius | 0.15 | 0.000 [0.000, 0.000] | 0.002 | no |
| image |   + one-sided measured radius | 0.15 | 0.000 [0.000, 0.000] | 0.012 | no |
| claim | VLM swap → Qwen2.5-VL-3B | 0.10 | 0.100 [0.017, 0.221] | 0.633 | no |
| image | VLM swap → Qwen2.5-VL-3B | 0.10 | 0.073 [0.020, 0.130] | 0.688 | no |
| claim | Domain shift → AMBER images | 0.10 | 0.040 [0.000, 0.088] | 0.655 | no |
| image | Domain shift → AMBER images | 0.10 | 0.014 [0.005, 0.025] | 0.693 | no |

## Counts

```
{
 "coco_claims_with_swap": 660,
 "qwen_claims": 574,
 "qwen_chair": {
  "captions": 200,
  "chair_i": 0.08056133056133057,
  "chair_s": 0.425
 },
 "amber": {
  "captions": 200,
  "claims": 426,
  "supported": 323,
  "hallucinated": 34,
  "unknown": 69
 }
}
```

## Reading guide (mechanical; conclusions are the author's)

- A shift *breaks* a clean-calibrated filter if its mean risk exceeds the target.
- Detector swaps: the robust rows show whether a radius measured on calibration images restores the target, and at what cost in supported claims kept.
- AMBER labels: claims whose object is in neither AMBER's present list nor its verified-absent list are excluded (see Counts).
