# IEEE Word manuscript

`content.py` holds the text; every number is read from `reports/phase4/*.json` and
`paper/numbers.tex`. `build.py` writes it into the IEEE conference Word template
(converted once from strict to transitional OOXML with LibreOffice), keeping the
template's styles, numbering, page size, margins and column layout.

    python paper/make_figures.py          # numbers.tex
    python paper/make_ieee_figures.py     # figures/col_*.png
    python paper/ieee_docx/build.py \
        --template paper/ieee_docx/template/ieee_conference_template_transitional.docx \
        --out BCRBP_IEEE_final.docx

Needs pandoc (math is converted to Word equations) and Pillow.
For submission, export the PDF from Word so that Times New Roman and Cambria Math are embedded.
