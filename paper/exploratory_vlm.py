"""EXPLORATORY (not pre-registered): why does the LLaVA-calibrated claim-level
threshold fail on Qwen2.5-VL-3B and InternVL3-2B?

Reads the raw Phase 4 shard files (unzipped phase4_results.zip plus synonyms.txt
and amber/annotations.json, as created by `scripts/phase4.py setup`) and writes
reports/phase4/exploratory_vlm.json with, per VLM, the hallucinated claims' clean
OWLv2 logit scores and summary statistics against the full-calibration tau.

    python paper/exploratory_vlm.py --raw /path/to/phase4_out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.phase4 import analysis as an  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "phase4" / "exploratory_vlm.json")
    a = ap.parse_args()
    d = an.load(a.raw)
    tau = float(an.tau_claim(d["cal"], 0.10))
    res = {"label": "EXPLORATORY - not a pre-registered analysis", "tau_full_cal": tau, "vlms": {}}
    for name, key in (("LLaVA-1.5-7B (test)", "test"), ("Qwen2.5-VL-3B", "qwen"), ("InternVL3-2B", "internvl")):
        h = np.array([c.score["clean"] for c in d[key] if c.hallucinated])
        s = np.array([c.score["clean"] for c in d[key] if not c.hallucinated])
        res["vlms"][name] = {
            "claims": int(len(h) + len(s)), "hallucinated": int(len(h)),
            "hallucinated_rate": float(len(h) / (len(h) + len(s))),
            "hallucinated_above_tau": float((h > tau).mean()),
            "supported_above_tau": float((s > tau).mean()),
            "hallucinated_median": float(np.median(h)),
            "hallucinated_q90": float(np.quantile(h, 0.9)),
            "hallucinated_scores": sorted(map(float, h)),
        }
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
