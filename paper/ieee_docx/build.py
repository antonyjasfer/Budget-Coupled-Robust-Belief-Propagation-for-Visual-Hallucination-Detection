"""Build the IEEE Word manuscript from content.py into the IEEE conference template.

    python paper/ieee_docx/build.py --template <transitional tmpl.docx> --out BCRBP_IEEE.docx

The template must be the transitional-OOXML version of the IEEE conference template
(the official file is strict OOXML; convert it once with LibreOffice). All styles,
numbering (headings, "Fig. n.", "TABLE N.", "[n]"), page size, margins and column
layout come from the template; this script writes only word/document.xml, adds the
figure images and blanks the copyright placeholder in the footer.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from html import escape
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import content as C  # noqa: E402

FIG = HERE.parent / "figures"
BIB = HERE.parent / "refs.bib"
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
MNS = 'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
PGMAR = '<w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:left="893" w:right="893" w:gutter="0" w:header="0" w:top="1080" w:footer="720" w:bottom="1440"/>'
TAIL = '<w:titlePg/><w:textDirection w:val="lrTb"/><w:docGrid w:type="default" w:linePitch="360" w:charSpace="8192"/>'


def sect(cols=1, space=360, first=False):
    foot = ('<w:footerReference w:type="even" r:id="rId2"/><w:footerReference w:type="default" r:id="rId3"/>'
            '<w:footerReference w:type="first" r:id="rId4"/>') if first else ""
    typ = "" if first else '<w:type w:val="continuous"/>'
    c = f'<w:cols w:num="{cols}" w:space="{space}" w:equalWidth="true" w:sep="false"/>' if cols > 1 else ""
    return f"<w:sectPr>{foot}{typ}{PGMAR}{c}<w:formProt w:val=\"false\"/>{TAIL}</w:sectPr>"


# ------------------------------------------------------------------ references
def parse_bib():
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*)\}\s*$", BIB.read_text(), re.M):
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        f = {}
        for fm in re.finditer(r"(\w+)=\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}", body):
            f[fm.group(1).lower()] = fm.group(2)
        f["_type"] = typ
        out[key] = f
    return out


def detex(s):
    s = s.replace("{\\`e}", "è").replace("{\\'a}", "á").replace("\\&", "&").replace("--", "–")
    return s.replace("{", "").replace("}", "")


def ieee_authors(a):
    names = [x.strip() for x in a.split(" and ")]
    others = names[-1] == "others"
    if others:
        names = names[:-1]
    fmt = []
    for n in names:
        n = detex(n)
        if "," not in n:
            fmt.append(n)
            continue
        last, first = [t.strip() for t in n.split(",", 1)]
        ini = " ".join(p[0] + "." if not p.endswith(".") else p for p in re.split(r"\s+", first) if p)
        ini = re.sub(r"(\w)\w*-(\w)\w*\.", r"\1.-\2.", ini) if "-" in first else ini
        fmt.append(f"{ini} {last}")
    if others or len(fmt) > 6:
        return fmt[0] + " et al."
    if len(fmt) == 1:
        return fmt[0]
    if len(fmt) == 2:
        return f"{fmt[0]} and {fmt[1]}"
    return ", ".join(fmt[:-1]) + ", and " + fmt[-1]


VENUE = {  # IEEE-style abbreviations of the venues in refs.bib
    "Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)":
        "in Proc. 64th Annu. Meeting Assoc. Comput. Linguistics (ACL), vol. 1",
    "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing":
        "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)",
    "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing":
        "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)",
    "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing":
        "in Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)",
    "International Conference on Learning Representations": "in Proc. Int. Conf. Learn. Represent. (ICLR)",
    "European Conference on Computer Vision (ECCV)": "in Proc. Eur. Conf. Comput. Vis. (ECCV)",
    "Computer Vision -- ECCV 2014": "in Proc. Eur. Conf. Comput. Vis. (ECCV)",
    "Computer Vision -- ECCV 2022": "in Proc. Eur. Conf. Comput. Vis. (ECCV)",
    "Computer Vision -- ECCV 2024": "in Proc. Eur. Conf. Comput. Vis. (ECCV)",
    "Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)":
        "in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)",
    "Advances in Neural Information Processing Systems": "in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)",
    "Proceedings of the 43rd International Conference on Machine Learning (PMLR 306)":
        "in Proc. 43rd Int. Conf. Mach. Learn. (ICML), PMLR, vol. 306",
    "Proceedings of the 41st International Conference on Machine Learning":
        "in Proc. 41st Int. Conf. Mach. Learn. (ICML), PMLR",
}
JOUR = {"Journal of Artificial Intelligence Research": "J. Artif. Intell. Res.",
        "Journal of Machine Learning Research": "J. Mach. Learn. Res.",
        "Scandinavian Journal of Statistics": "Scand. J. Statist.",
        "International Journal of Approximate Reasoning": "Int. J. Approx. Reason.",
        "Foundations and Trends in Machine Learning": "Found. Trends Mach. Learn.",
        "The Annals of Statistics": "Ann. Statist.",
        "Journal of the American Statistical Association": "J. Amer. Statist. Assoc.",
        "ACM Journal of Data Science": "ACM J. Data Sci."}


def ref_runs(key, f):
    """Return a list of (text, italic) for one reference, in IEEE style."""
    au = ieee_authors(f["author"])
    title = detex(f["title"])
    y = f.get("year", "")
    t = f["_type"]
    pg = detex(f.get("pages", ""))
    R = [(au + ", ", False)]
    if t == "book":
        R += [(title, True), (f", {f.get('address', '')}: {detex(f['publisher'])}, {y}", False)]
    else:
        R.append((f"\u201c{title},\u201d ", False))
        if t == "article":
            R.append((JOUR.get(f["journal"], f["journal"]), True))
            s = ""
            if "volume" in f:
                s += f", vol. {f['volume']}"
            if "number" in f:
                s += f", no. {f['number']}"
            s += (f", pp. {pg}" if "–" in pg else f", Art. no. {pg}") if pg else ""
            R.append((s + f", {y}", False))
        elif t == "inproceedings":
            v = VENUE.get(f["booktitle"], "in " + f["booktitle"])
            if "volume" in f and "vol." not in v:
                v += (", LNCS" if f.get("series") == "LNCS" else "") + f", vol. {f['volume']}"
            R.append((v + f", {y}" + (f", pp. {pg}" if pg else ""), False))
        else:
            hp = f.get("howpublished", "")
            if key == "chan2005markov":
                R.append(("in Proc. 19th Int. Joint Conf. Artif. Intell. (IJCAI), " + y, False))
            else:
                arx = re.match(r"(arXiv:[0-9.]+(?:v\d+)?)", hp)
                R.append(((arx.group(1) if arx else detex(hp)) + f", {y}", False))
    if "doi" in f:
        R.append((f", doi: {f['doi']}", False))
    R.append((".", False))
    return R


ORDER = {"m:dPr": ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"],
         "m:rPr": ["lit", "nor", "scr", "sty", "brk", "aln"]}


def fix_omml(x):
    """Put the children of m:dPr and m:rPr in schema order (pandoc does not)."""
    for tag, order in ORDER.items():
        def sort(m):
            kids = re.findall(r"<m:(\w+)[^>]*?/>", m.group(1))
            items = re.findall(r"<m:\w+[^>]*?/>", m.group(1))
            if "".join(items) != re.sub(r"\s+(?=<)", "", m.group(1)).strip():
                return m.group(0)
            pairs = sorted(zip(kids, items), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
            return f"<{tag}>" + "".join(i for _, i in pairs) + f"</{tag}>"
        x = re.sub(rf"<{tag}>(.*?)</{tag}>", sort, x, flags=re.S)
    # m:nor (plain text, from \\text) excludes m:sty in the schema
    x = re.sub(r"(<m:nor ?/>)<m:sty [^>]*/>", r"\1", x)
    # function names (min, max, sup, clip...) and lone operators in scripts as upright text runs;
    # Word renders both forms the same, LibreOffice only renders this one correctly
    x = re.sub(r'<m:rPr><m:sty m:val="p" ?/></m:rPr><m:t>([A-Za-z]{2,})</m:t>', r"<m:rPr><m:nor/></m:rPr><m:t>\1</m:t>", x)
    x = re.sub(r'(<m:(?:sub|sup)><m:r>)<m:rPr><m:sty m:val="p" ?/></m:rPr>(<m:t>[+\u2212]</m:t></m:r></m:(?:sub|sup)>)',
               r"\1<m:rPr><m:nor/></m:rPr>\2", x)
    # size the math like the 10 pt body text
    x = re.sub(r"(<m:r>(?:<m:rPr>.*?</m:rPr>)?)", r'\1<w:rPr><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>', x)
    return x


def smart_quotes(text):
    """Curly quotes and apostrophes outside $...$ math."""
    out, inmath, opened = [], False, False
    for ch in text:
        if ch == "$":
            inmath = not inmath
        elif not inmath and ch == '"':
            ch = "\u201d" if opened else "\u201c"
            opened = not opened
        elif not inmath and ch == "'":
            ch = "\u2019"
        out.append(ch)
    return "".join(out)


# ------------------------------------------------------------------ inline markup
TOK = re.compile(r"(\$[^$]+\$|~~.+?~~|\[\[[^\]]+\]\]|\{(?:fig|tab|eq|sec):[\w-]+\})")


class Doc:
    def __init__(self):
        self.cite = {}
        self.math = {}
        self.labels = {}
        self.media = []
        self.pid = 100
        self.eqn = 0

    # --- pass 1: numbering
    def number(self, blocks):
        h1 = h2 = fig = tab = eq = 0
        for b in blocks:
            k = b[0]
            if k == "h1":
                h1 += 1
                h2 = 0
                if b[2]:
                    self.labels[b[2]] = f"Section {ROMAN[h1]}"
            elif k == "h2":
                h2 += 1
                if b[2]:
                    self.labels[b[2]] = f"Section {ROMAN[h1]}-{chr(64 + h2)}"
            elif k == "fig":
                fig += 1
                self.labels[b[3]] = f"Fig. {fig}"
            elif k == "table":
                tab += 1
                self.labels[b[2]] = f"Table {ROMAN[tab]}"
            elif k == "eq":
                eq += 1
                if b[2]:
                    self.labels[b[2]] = f"({eq})"
            for txt in self.texts(b):
                for m in re.finditer(r"\[\[([^\]]+)\]\]", txt):
                    for c in m.group(1).split(","):
                        self.cite.setdefault(c.strip(), len(self.cite) + 1)
                for m in re.finditer(r"\$([^$]+)\$", txt):
                    self.math.setdefault(m.group(1), None)
            if k == "eq":
                self.math.setdefault(b[1], None)

    @staticmethod
    def texts(b):
        k = b[0]
        if k in ("p", "h1", "h2", "h1n"):
            return [b[1]]
        if k == "bullets":
            return list(b[1])
        if k == "fig":
            return [b[2]]
        if k == "table":
            return [b[1]] + [c for r in b[4] for c in r]
        return []

    # --- math via pandoc
    def convert_math(self):
        snips = list(self.math)
        md = "\n\n".join(f"MATHMARK{i} ${s}$" for i, s in enumerate(snips))
        with tempfile.TemporaryDirectory() as td:
            Path(td, "m.md").write_text(md)
            subprocess.run(["pandoc", "m.md", "-o", "m.docx"], cwd=td, check=True)
            x = zipfile.ZipFile(Path(td, "m.docx")).read("word/document.xml").decode()
        paras = re.findall(r"<w:p>.*?</w:p>|<w:p .*?</w:p>", x, re.S)
        found = {}
        for p in paras:
            mm = re.search(r"MATHMARK(\d+)", p)
            if not mm:
                continue
            om = re.search(r"<m:oMath>.*</m:oMath>", p, re.S)
            if not om:
                raise SystemExit(f"pandoc failed on math: {snips[int(mm.group(1))]}")
            found[int(mm.group(1))] = om.group(0)
        for i, s in enumerate(snips):
            self.math[s] = fix_omml(found[i])

    # --- runs
    @staticmethod
    def typo(t):
        return smart_quotes(t)

    def run(self, t, italic=False, sz=None, extra=""):
        rp = ("<w:i/><w:iCs/>" if italic else "") + extra + (f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>' if sz else "")
        return f'<w:r><w:rPr>{rp}</w:rPr><w:t xml:space="preserve">{escape(t, quote=False)}</w:t></w:r>'

    def cites(self, keys):
        nums = sorted(self.cite[k.strip()] for k in keys.split(","))
        parts, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"[{nums[i]}]–[{nums[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in nums[i:j + 1]))
            i = j + 1
        return ", ".join(parts)

    def inline(self, text, sz=None, italic=False):
        out = []
        text = smart_quotes(text)
        for piece in TOK.split(text):
            if not piece:
                continue
            if piece.startswith("$"):
                out.append(self.math[piece[1:-1]])
            elif piece.startswith("~~"):
                out.append(self.run(self.typo(piece[2:-2]), True, sz))
            elif piece.startswith("[["):
                out.append(self.run(self.cites(piece[2:-2]), italic, sz))
            elif piece.startswith("{") and ":" in piece:
                out.append(self.run(self.labels[piece[1:-1]], italic, sz))
            else:
                t = self.typo(piece)
                t = re.sub(r"(https?://\S+)", lambda m: m.group(1).replace("/", "/​"), t)
                out.append(self.run(t, italic, sz))
        return "".join(out)

    # --- blocks
    @staticmethod
    def para(style, body, ppr=""):
        return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr}</w:pPr>{body}</w:p>'

    def image(self, name, width_in=3.45):
        w, h = Image.open(FIG / name).size
        cx = int(width_in * 914400)
        cy = int(cx * h / w)
        rid = f"rIdImg{len(self.media) + 1}"
        self.media.append((rid, name))
        self.pid += 1
        return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                f'<wp:docPr id="{self.pid}" name="Picture {self.pid}"/><wp:cNvGraphicFramePr>'
                '<a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
                '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
                f'<pic:nvPicPr><pic:cNvPr id="{self.pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
                f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
                f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
                '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
                '</wp:inline></w:drawing></w:r>')

    def table(self, caption, header, rows, widths, font, wide):
        sz = int(round(font * 2))
        tw = [int(round(x * 1440)) for x in widths]
        # IEEE table rules: horizontal lines only, above the header, below the header and below the last row
        def rule(side, w):
            return f'<w:{side} w:val="single" w:sz="{w}" w:space="0" w:color="000000"/>'
        mar = ('<w:tblCellMar><w:top w:w="15" w:type="dxa"/><w:left w:w="43" w:type="dxa"/>'
               '<w:bottom w:w="15" w:type="dxa"/><w:right w:w="43" w:type="dxa"/></w:tblCellMar>')
        x = [self.para("tablehead", self.inline(caption), '<w:keepNext/>')]
        x.append(f'<w:tbl><w:tblPr><w:tblW w:w="{sum(tw)}" w:type="dxa"/><w:jc w:val="center"/>'
                 f'<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/>'
                 f'<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
                 f'<w:tblLayout w:type="fixed"/>{mar}<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>'
                 + "".join(f'<w:gridCol w:w="{w}"/>' for w in tw) + "</w:tblGrid>")

        def row(cells, head, last=False):
            r = '<w:tr><w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if head else '') + '</w:trPr>'
            bord = ((rule("top", 8) if head else '<w:top w:val="nil"/>') + '<w:left w:val="nil"/>'
                    + (rule("bottom", 4) if head else rule("bottom", 8) if last else '<w:bottom w:val="nil"/>')
                    + '<w:right w:val="nil"/>')
            for i, (c, w) in enumerate(zip(cells, tw)):
                jc = "left" if i == 0 and not head else "center"
                style = "tablecolhead" if head else "tablecopy"
                body = self.inline(c, sz=sz)
                r += (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/><w:tcBorders>{bord}</w:tcBorders>'
                      f'<w:vAlign w:val="center"/></w:tcPr>'
                      + self.para(style, body, ('<w:keepNext/>' if head or len(rows) < 12 else '') + f'<w:spacing w:before="0" w:after="0"/><w:jc w:val="{jc}"/>'
                                  f'<w:rPr><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>') + "</w:tc>")
            return r + "</w:tr>"
        x.append(row(header, True))
        for i, rr in enumerate(rows):
            x.append(row(rr, False, last=(i == len(rows) - 1)))
        x.append("</w:tbl>")
        return x

    def render(self, blocks):
        out = []
        prev = None
        for b in blocks:
            k = b[0]
            if k == "h1":
                out.append(self.para("Heading1", self.inline(b[1])))
            elif k == "h1n":
                out.append(self.para("Heading5", self.inline(b[1]), '<w:keepNext/><w:jc w:val="center"/>'))
            elif k == "h2":
                out.append(self.para("Heading2", self.inline(b[1])))
            elif k == "p":
                # a sentence that continues after a display equation is not indented
                cont = prev == "eq" and (b[1][:1].islower() or b[1].startswith("$"))
                out.append(self.para("BodyText", self.inline(b[1]), '<w:ind w:firstLine="0"/>' if cont else ""))
            elif k == "bullets":
                for it in b[1]:
                    out.append(self.para("bulletlist", self.inline(it),
                                         '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
                                         '<w:spacing w:after="60"/><w:ind w:hanging="288" w:left="576"/><w:jc w:val="both"/>'))
            elif k == "eq":
                self.eqn += 1
                eqn = self.eqn
                num = ('<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
                       f'</w:rPr><w:tab/><w:t>({eqn})</w:t></w:r>')
                out.append(self.para("equation", '<w:r><w:tab/></w:r>' + self.math[b[1]] + num,
                                     '<w:spacing w:before="120" w:after="120" w:line="240" w:lineRule="auto"/>'))
            elif k == "fig":
                out.append(self.para("Normal", self.image(b[1]),
                                     '<w:keepNext/><w:spacing w:before="120" w:after="0"/><w:jc w:val="center"/>'))
                out.append(self.para("figurecaption", self.inline(b[2]),
                                     '<w:keepLines/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="3"/></w:numPr><w:ind w:hanging="0" w:left="0"/>'))
            elif k == "table":
                _, cap, label, header, rows, widths, wide, font, align = b
                tb = self.table(cap, header, rows, widths, font, wide)
                if wide:
                    # end the two-column section at the previous paragraph, put the table in a
                    # one-column section, then return to two columns
                    keep = []
                    while 'w:val="Heading' in out[-1]:
                        keep.insert(0, out.pop())
                    last = out.pop()
                    out.append(last.replace("</w:pPr>", sect(2) + "</w:pPr>", 1))
                    out += keep + tb
                    out.append(self.para("Normal", "", '<w:spacing w:before="0" w:after="120" w:line="120" w:lineRule="exact"/>'
                                         + sect(1)))
                else:
                    out += tb
                    out.append(self.para("Normal", "", '<w:spacing w:before="0" w:after="60" w:line="120" w:lineRule="exact"/>'))
            prev = k
        return out


def author_par(name, dept, email, extra, last=False):
    rp = '<w:sz w:val="18"/><w:szCs w:val="18"/>'
    lines = [(name, False, rp.replace("18", "22")), (dept, True, rp), (C.ORG[0], True, rp), (C.ORG[1], True, rp),
             (C.ORG[2], False, rp), (email, False, rp)] + ([(extra, False, rp)] if extra else [])
    body = ""
    for i, (t, it, r) in enumerate(lines):
        br = "<w:br/>" if i else ""
        body += f'<w:r><w:rPr>{"<w:i/>" if it else ""}{r}</w:rPr>{br}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'
    if not last:
        body += '<w:r><w:br w:type="column"/></w:r>'
    return body


def build(template: Path, out: Path):
    d = Doc()
    d.number(C.B + C.APPX)
    for t in (C.ABSTRACT, C.KEYWORDS) + tuple(x[1] for x in C.BACK):
        assert "$" not in t and "[[" not in t
    d.convert_math()
    bib = parse_bib()
    missing = [k for k in d.cite if k not in bib]
    assert not missing, missing

    X = []
    X.append(d.para("papertitle", d.run(C.TITLE), sect(1, first=True)))
    A = C.AUTHORS
    row1 = "".join(author_par(*a, last=(i == 2)) for i, a in enumerate(A[:3]))
    X.append(d.para("Author", row1, '<w:spacing w:before="120" w:after="120"/>' + sect(3, 216)))
    row2 = "".join(author_par(*a, last=(i == 1)) for i, a in enumerate(A[3:]))
    X.append(d.para("Author", row2, '<w:spacing w:before="120" w:after="240"/>' + sect(2, 216)))
    X.append(d.para("Abstract", d.run("Abstract", True) + d.run("—" + d.typo(C.ABSTRACT))))
    X.append(d.para("Keywords", d.run("Keywords—" + C.KEYWORDS)))
    X += d.render(C.B)
    # back matter: data availability, appendices, references (IEEE order)
    title, txt = C.BACK[0]
    X.append(d.para("Heading5", d.run(title), '<w:keepNext/><w:jc w:val="center"/>'))
    X.append(d.para("BodyText", d.inline(txt)))
    X += d.render(C.APPX)
    for title, txt in C.BACK[1:]:
        X.append(d.para("Heading5", d.run(title), '<w:keepNext/><w:jc w:val="center"/>'))
        X.append(d.para("BodyText", d.inline(txt)))
    X.append(d.para("Heading5", d.run("References"), '<w:keepNext/><w:jc w:val="center"/>'))
    for key, n in sorted(d.cite.items(), key=lambda kv: kv[1]):
        body = "".join(d.run(t, it) for t, it in ref_runs(key, bib[key]))
        X.append(d.para("references", body, '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="5"/></w:numPr>'
                        '<w:ind w:hanging="354" w:left="354"/>'))
    X[-1] = X[-1].replace("</w:pPr>", sect(2) + "</w:pPr>", 1)
    unused = sorted(set(bib) - set(d.cite))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(template) as z:
            z.extractall(td)
        docx = (td / "word/document.xml").read_text()
        head = docx[:docx.index("<w:body>")]
        if "xmlns:m=" not in head:
            head = head.replace("<w:document ", f"<w:document {MNS} ", 1)
        body = "".join(X) + sect(1)
        (td / "word/document.xml").write_text(head + "<w:body>" + body + "</w:body></w:document>")
        nb = td / "word/numbering.xml"
        num = re.sub(r"(<w:pStyle [^>]*/>)(<w:numFmt [^>]*/>)", r"\2\1", nb.read_text())

        def space_suffix(m):  # "VIII. Conclusion", "TABLE V. Caption": a space, not a tab, after the number
            return re.sub(r'(<w:lvl w:ilvl="[01]"[^>]*>(?:(?!</w:lvl>).)*?)(<w:lvlText )', r'\1<w:suff w:val="space"/>\2', m.group(0), flags=re.S)
        num = re.sub(r'<w:abstractNum w:abstractNumId="(?:1|6)".*?</w:abstractNum>', space_suffix, num, flags=re.S)
        nb.write_text(num)
        for f in ("footer1.xml", "footer2.xml", "footer3.xml"):
            p = td / "word" / f
            p.write_text(p.read_text().replace("XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE", ""))
        rels = (td / "word/_rels/document.xml.rels").read_text()
        rels = re.sub(r'<Relationship Id="rId[56]"[^>]*/>', "", rels)
        (td / "word/media").mkdir(exist_ok=True)
        add = ""
        for rid, name in d.media:
            shutil.copy(FIG / name, td / "word/media" / name)
            add += (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                    f'relationships/image" Target="media/{name}"/>')
        (td / "word/_rels/document.xml.rels").write_text(rels.replace("</Relationships>", add + "</Relationships>"))
        core = td / "docProps/core.xml"
        if core.exists():
            c = core.read_text()
            c = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(C.TITLE)}</dc:title>", c)
            c = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator>S. Raja et al.</dc:creator>", c)
            core.write_text(c)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            ct = td / "[Content_Types].xml"
            z.write(ct, "[Content_Types].xml")
            for p in sorted(td.rglob("*")):
                if p.is_file() and p != ct:
                    z.write(p, p.relative_to(td).as_posix())
    print(f"wrote {out}: {len(d.cite)} references, {len(d.media)} figures; unused bib keys: {unused}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.template, a.out)
