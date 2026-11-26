"""Exploratory 'orange' mini-audit (NOT pre-registered; separate from the frozen 600-claim audit).

Samples CHAIR-hallucinated 'orange' claims from the COCO test split of Qwen3-VL-4B,
Qwen2.5-VL-3B and InternVL3-2B (34 / 33 / 33, seed 2029), skipping any claim already in the
frozen audit sample, and writes a blind workbook in which the annotator decides how the
caption uses the word: fruit / color_adjective / unclear.

    python scripts/phase5_orange_audit.py --out OUT --dst DIR
Writes DIR/phase5_orange_audit.xlsx and DIR/phase5_orange_audit_KEY_do_not_share.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import phase4 as P4  # noqa: E402
import phase5 as P5  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.cell.rich_text import CellRichText, TextBlock  # noqa: E402
from openpyxl.cell.text import InlineFont  # noqa: E402
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402
from src.phase1 import splits as sp  # noqa: E402

PLAN = (("qwen3_4b", 34), ("qwen25", 33), ("internvl3", 33))
OPTIONS = ["fruit", "color_adjective", "unclear"]
SEED = 2029
F = "Arial"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args()
    out, dst = a.out, a.dst
    dst.mkdir(parents=True, exist_ok=True)
    prep, vocab = P5._prep(out), P4._vocab(out)
    gt = {int(k): set(v) for k, v in prep["gt"].items()}
    test = set(prep["test"])
    frozen = set()
    key_path = out / "audit" / "audit_key.csv"
    if key_path.exists():
        frozen = {(r["generator"], int(r["image_id"]), r["category"]) for r in csv.DictReader(open(key_path))}

    rng = np.random.default_rng(SEED)
    rows = []
    for g, n in PLAN:
        pool = []
        for r in P4.read_all(out, P5.cap_file(g, "coco")):
            iid = int(r["image_id"])
            if iid not in test or "orange" in gt.get(iid, set()) or (g, iid, "orange") in frozen:
                continue
            cl = next((c for c in vocab.extract_claims(r["caption"]).claims if c.node_word == "orange"), None)
            if cl is not None:
                pool.append({"generator": g, "image_id": iid, "caption": r["caption"], "spans": cl.char_spans})
        if len(pool) < n:
            raise ValueError(f"{g}: need {n}, have {len(pool)}")
        rows += [pool[int(i)] | {"pool_size": len(pool)} for i in rng.choice(len(pool), n, replace=False)]
    rows = [rows[int(i)] for i in rng.permutation(len(rows))]
    for k, r in enumerate(rows):
        r["oid"] = f"O{k:03d}"

    plain, hl = InlineFont(rFont=F, sz=10), InlineFont(rFont=F, sz=10, b=True, color="C00000", u="single")
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    text = [
        ("Exploratory mini-audit: how is the word 'orange' used?", True),
        ("This is separate from the 600-claim audit. Label every row of the 'Orange' sheet on your own.", False),
        ("", False),
        ("Read the caption in column D. Every highlighted 'orange' (red, underlined) is the word being judged.", False),
        ("Column E (required): what does the caption mean by it?", False),
        ("     fruit            the orange fruit (e.g. 'a bowl of oranges', 'an orange on the table')", False),
        ("     color_adjective  the colour (e.g. 'an orange safety vest', 'orange and white cones', 'orange accents')", False),
        ("     unclear          you cannot tell from the caption, even with the image", False),
        ("If the same caption uses 'orange' both ways, choose fruit and say so in column F.", False),
        ("Use the image link (column B) only when the caption alone does not settle it.", False),
        ("", False),
        ("Fill in only the yellow columns E and F. Do not sort, delete or reorder rows.", True),
    ]
    for i, (t, b) in enumerate(text, 1):
        ins.cell(i, 1, t).font = Font(name=F, size=14 if i == 1 else 10, bold=b)
    ins.column_dimensions["A"].width = 110

    ws = wb.create_sheet("Orange")
    hdr = ["item_id", "image", "image_file", "caption ('orange' in red)", "sense", "notes"]
    fill, inp = PatternFill("solid", start_color="1F3864"), PatternFill("solid", start_color="FFF2CC")
    thin = Border(bottom=Side(style="thin", color="D9D9D9"))
    for j, h in enumerate(hdr, 1):
        c = ws.cell(1, j, h)
        c.font, c.fill = Font(name=F, size=10, bold=True, color="FFFFFF"), fill
    for i, r in enumerate(rows, 2):
        cap, parts, pos = r["caption"], [], 0
        for s, e in sorted(r["spans"]):
            if s > pos:
                parts.append(TextBlock(plain, cap[pos:s]))
            parts.append(TextBlock(hl, cap[s:e]))
            pos = e
        if pos < len(cap):
            parts.append(TextBlock(plain, cap[pos:]))
        fname = sp.coco_val2014_filename(r["image_id"])
        ws.cell(i, 1, r["oid"])
        c = ws.cell(i, 2, "open image")
        c.hyperlink = f"http://images.cocodataset.org/val2014/{fname}"
        c.font = Font(name=F, size=10, color="0563C1", underline="single")
        ws.cell(i, 3, fname)
        ws.cell(i, 4).value = CellRichText(*parts)
        for j in (1, 3):
            ws.cell(i, j).font = Font(name=F, size=10)
        for j in (5, 6):
            ws.cell(i, j).fill, ws.cell(i, j).font = inp, Font(name=F, size=10)
        for j in range(1, 7):
            ws.cell(i, j).alignment = Alignment(vertical="top", wrap_text=j in (4, 6))
            ws.cell(i, j).border = thin
    dv = DataValidation(type="list", formula1='"' + ",".join(OPTIONS) + '"', allow_blank=True,
                        showErrorMessage=True, errorTitle="sense", error="Choose a value from the list.")
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(rows) + 1}")
    for col, w in zip("ABCDEF", (9, 12, 30, 95, 17, 30)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "B2"
    wb.active = 1
    wb.save(dst / "phase5_orange_audit.xlsx")

    with open(dst / "phase5_orange_audit_KEY_do_not_share.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "generator", "image_id", "pool_size", "seed"])
        for r in rows:
            w.writerow([r["oid"], r["generator"], r["image_id"], r["pool_size"], SEED])
    print(len(rows), "items;", {g: next(r["pool_size"] for r in rows if r["generator"] == g) for g, _ in PLAN})


if __name__ == "__main__":
    main()
