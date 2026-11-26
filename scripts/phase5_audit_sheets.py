"""Build the two blind annotator workbooks (A, B) from the frozen prereg-v3 audit sample.

    python scripts/phase5.py audit --out OUT --gens llava,qwen25,internvl3
    python scripts/phase5_audit_sheets.py --out OUT --dst DIR

The workbooks show only claim_id, image link, COCO category and the caption with the judged
mention highlighted; generator, score and CHAIR label stay in OUT/audit/audit_key.csv.
Primary label: supported / hallucinated / unclear (mapped to the audit module's
present / absent / ambiguous for kappa); optional error_type tag.
"""
import argparse, csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
_ap = argparse.ArgumentParser(); _ap.add_argument("--out", type=Path, required=True); _ap.add_argument("--dst", type=Path, required=True)
_a = _ap.parse_args()
import phase4 as P4
from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation

OUT, DST = _a.out, _a.dst; DST.mkdir(parents=True, exist_ok=True)
tasks = list(csv.DictReader(open(OUT / "audit/annotator_tasks.csv", encoding="utf-8")))
vocab = P4._vocab(OUT)
LABELS = ["supported", "hallucinated", "unclear"]
ERRS = ["color_or_homonym", "synonym_miss", "ambiguous", "none"]
F = "Arial"; plain = InlineFont(rFont=F, sz=10); hl = InlineFont(rFont=F, sz=10, b=True, color="C00000", u="single")
head_fill = PatternFill("solid", start_color="1F3864"); in_fill = PatternFill("solid", start_color="FFF2CC")
thin = Border(bottom=Side(style="thin", color="D9D9D9"))

def rich(caption, category):
    cl = next((c for c in vocab.extract_claims(caption).claims if c.node_word == category), None)
    if cl is None or not cl.char_spans:
        raise ValueError(f"no span for {category!r}")
    parts, pos = [], 0
    for a, b in sorted(cl.char_spans):
        if a > pos: parts.append(TextBlock(plain, caption[pos:a]))
        parts.append(TextBlock(hl, caption[a:b])); pos = b
    if pos < len(caption): parts.append(TextBlock(plain, caption[pos:]))
    return CellRichText(*parts), len(cl.char_spans)

rows = [(t, *rich(t["caption"], t["object"])) for t in tasks]

def build(who):
    wb = Workbook(); ins = wb.active; ins.title = "Instructions"
    lines = [
        ("Phase 5 human label audit: annotator " + who, True),
        ("Label every row of the 'Claims' sheet on your own. Do not discuss rows with the other annotator until both sheets are returned.", False),
        ("", False),
        ("What to do for each row", True),
        ("1. Click 'open image' (column B) to see the photo.", False),
        ("2. Read the caption (column E). The words in red, underlined, are the mention being judged; column D gives its COCO object category.", False),
        ("3. Column F (required): is that object really in the image?", False),
        ("     supported     the object is visible in the image", False),
        ("     hallucinated  the object is not in the image", False),
        ("     unclear       you cannot decide (too small, occluded, cut off, or the image will not load)", False),
        ("4. Column G (optional): why the automatic matching may have gone wrong.", False),
        ("     color_or_homonym  the word is used in another sense, e.g. 'orange' as a colour", False),
        ("     synonym_miss      the word names something else than the category, e.g. 'container' that is not a bowl", False),
        ("     ambiguous         the wording itself is unclear", False),
        ("     none              no wording problem", False),
        ("5. Column H: free notes (optional).", False),
        ("", False),
        ("Fill in only the yellow columns F, G and H. Do not sort, delete or reorder rows; claim_id must stay with its row.", True),
        ("Judge the highlighted mention only, as the caption uses it. Each row stands alone.", False),
        ("", False),
        ("Example row (not part of the audit)", True),
    ]
    for i, (t, b) in enumerate(lines, 1):
        c = ins.cell(i, 1, t); c.font = Font(name=F, size=14 if i == 1 else 10, bold=b)
    ex_hdr = ["claim_id", "image", "image_file", "COCO category", "caption", "label", "error_type", "notes"]
    ex = ["EXAMPLE", "open image", "COCO_val2014_000000000000.jpg", "orange",
          "A man in an orange safety vest stands next to traffic cones.", "hallucinated", "color_or_homonym",
          "orange is the vest's colour; no fruit"]
    r0 = len(lines) + 1
    for j, (h, v) in enumerate(zip(ex_hdr, ex), 1):
        ins.cell(r0, j, h).font = Font(name=F, size=10, bold=True)
        ins.cell(r0 + 1, j, v).font = Font(name=F, size=10, italic=True, color="808080")
    ins.column_dimensions["A"].width = 120
    for col in "BCDEFGH": ins.column_dimensions[col].width = 18

    ws = wb.create_sheet("Claims")
    hdr = ["claim_id", "image", "image_file", "COCO category", "caption (judged words in red)",
           "label", "error_type", "notes"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(1, j, h); c.font = Font(name=F, size=10, bold=True, color="FFFFFF"); c.fill = head_fill
        c.alignment = Alignment(vertical="center", wrap_text=True)
    for i, (t, rt, _) in enumerate(rows, 2):
        url = t["image_url"]
        ws.cell(i, 1, t["audit_id"])
        c = ws.cell(i, 2, "open image"); c.hyperlink = url; c.font = Font(name=F, size=10, color="0563C1", underline="single")
        ws.cell(i, 3, url.rsplit("/", 1)[1]); ws.cell(i, 4, t["object"]); ws.cell(i, 5).value = rt
        for j in (1, 3, 4): ws.cell(i, j).font = Font(name=F, size=10)
        for j in (6, 7, 8): ws.cell(i, j).fill = in_fill; ws.cell(i, j).font = Font(name=F, size=10)
        for j in range(1, 9):
            ws.cell(i, j).alignment = Alignment(vertical="top", wrap_text=(j == 5 or j == 8)); ws.cell(i, j).border = thin
    n = len(rows) + 1
    for col, opts, title in (("F", LABELS, "label"), ("G", ERRS, "error_type")):
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle=title, error="Choose a value from the list.")
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n}")
    for col, w in zip("ABCDEFGH", (10, 12, 30, 15, 90, 14, 18, 30)): ws.column_dimensions[col].width = w
    ws.freeze_panes = "B2"
    wb.active = 1
    p = DST / f"phase5_audit_annotator_{who}.xlsx"; wb.save(p); return p

for who in ("A", "B"):
    print(build(who))
print("rows", len(rows), "| rows with >1 highlighted span", sum(k > 1 for *_, k in rows))
