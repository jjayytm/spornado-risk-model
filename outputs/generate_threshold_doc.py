"""
Disease Risk Thresholds & Risk Bands -- Professional Client DOCX
Design inspired by: clean academic/consulting style
  - White background, typography-driven
  - H1: ALL CAPS bold + full-width teal bottom border
  - H2: teal colored bold, no decoration
  - Tables: dark charcoal header, alternating rows, no outer border
  - Header: left | bold-teal center | right-date with bottom rule
  - Title page: centered, large bold title
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"C:\Users\Admin\Desktop\Disease_Risk_Thresholds_Methodology.docx"

# ── Palette ───────────────────────────────────────────────────────────────────
TEAL    = "2B8FA3"   # accent colour (H2, borders, header highlight)
CHARCOAL= "2C3E50"   # H1 text, table headers background, body text
WHITE   = "FFFFFF"
LGRAY   = "F7F7F7"   # alt table rows
MGRAY   = "D5D8DC"   # divider lines
DGRAY   = "555555"   # caption / secondary text
LGREEN  = "D5F5E3"
LAMBER  = "FEF9E7"
LRED    = "FDEDEC"
DGREEN  = "1E8449"
DAMBER  = "9A7D0A"
DRED    = "922B21"
LBLUE   = "EBF5FB"   # callout background

def rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

# ── XML helpers ───────────────────────────────────────────────────────────────

def cell_bg(cell, hex6):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("w:shd")):
        tcPr.remove(old)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex6.lstrip("#"))
    tcPr.append(shd)


def cell_pad(cell, top=80, bottom=80, left=130, right=130):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in [("top",top),("bottom",bottom),("left",left),("right",right)]:
        el = OxmlElement("w:"+side)
        el.set(qn("w:w"),    str(val))
        el.set(qn("w:type"), "dxa")
        tcMar.append(el)
    tcPr.append(tcMar)


def no_cell_border(cell):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcB = OxmlElement("w:tcBorders")
    for edge in ("top","left","bottom","right","insideH","insideV"):
        t = OxmlElement("w:"+edge)
        t.set(qn("w:val"),   "none")
        t.set(qn("w:sz"),    "0")
        t.set(qn("w:space"), "0")
        t.set(qn("w:color"), "auto")
        tcB.append(t)
    tcPr.append(tcB)


def col_width(table, ci, cm_val):
    twips = int(cm_val * 567)
    for row in table.rows:
        cell = row.cells[ci]
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcW = tcPr.find(qn("w:tcW"))
        if tcW is None:
            tcW = OxmlElement("w:tcW"); tcPr.append(tcW)
        tcW.set(qn("w:w"),    str(twips))
        tcW.set(qn("w:type"), "dxa")


def tbl_full_width(table):
    tbl = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr"); tbl.insert(0, tblPr)
    old = tblPr.find(qn("w:tblW"))
    if old is not None: tblPr.remove(old)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:w"),    "5000")
    tblW.set(qn("w:type"), "pct")
    tblPr.append(tblW)


def tbl_style(table, inner="E0E0E0"):
    tbl = table._tbl
    tblPr = tbl.find(qn("w:tblPr"))
    if tblPr is None:
        tblPr = OxmlElement("w:tblPr"); tbl.insert(0, tblPr)
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None: tblPr.remove(old)
    tblBorders = OxmlElement("w:tblBorders")
    for edge in ("top","left","bottom","right"):
        t = OxmlElement("w:"+edge)
        t.set(qn("w:val"),"none"); t.set(qn("w:sz"),"0")
        t.set(qn("w:space"),"0"); t.set(qn("w:color"),"auto")
        tblBorders.append(t)
    for edge in ("insideH","insideV"):
        t = OxmlElement("w:"+edge)
        t.set(qn("w:val"),"single"); t.set(qn("w:sz"),"4")
        t.set(qn("w:space"),"0"); t.set(qn("w:color"),inner)
        tblBorders.append(t)
    tblPr.append(tblBorders)


def cell_write(cell, text, bold=False, italic=False, sz=10,
               color=CHARCOAL, align=WD_ALIGN_PARAGRAPH.LEFT):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    r = p.add_run(text)
    r.bold = bold; r.italic = italic
    r.font.size = Pt(sz)
    r.font.color.rgb = rgb(color)
    r.font.name = "Calibri"
    return r

# ── Header & Footer ───────────────────────────────────────────────────────────

def build_header(doc):
    """3-column header: left text | bold teal center | right date + bottom rule."""
    section = doc.sections[0]
    header = section.header
    # Clear default paragraph
    for p in header.paragraphs:
        p._element.getparent().remove(p._element)

    tbl = header.add_table(rows=1, cols=3, width=Inches(6.3))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_full_width(tbl)
    tbl_style(tbl, inner="FFFFFF")

    cells = tbl.rows[0].cells
    # Left
    no_cell_border(cells[0])
    cell_pad(cells[0], top=40, bottom=40, left=0, right=0)
    lp = cells[0].paragraphs[0]
    lp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    lr = lp.add_run("Disease Risk Thresholds & Risk Bands")
    lr.font.size = Pt(8); lr.font.color.rgb = rgb("999999"); lr.font.name = "Calibri"

    # Center
    no_cell_border(cells[1])
    cell_pad(cells[1], top=40, bottom=40, left=0, right=0)
    cp = cells[1].paragraphs[0]
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cr = cp.add_run("Spornado Disease Risk Model")
    cr.bold = True; cr.font.size = Pt(8)
    cr.font.color.rgb = rgb(TEAL); cr.font.name = "Calibri"

    # Right
    no_cell_border(cells[2])
    cell_pad(cells[2], top=40, bottom=40, left=0, right=0)
    rp = cells[2].paragraphs[0]
    rp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    rr = rp.add_run("May 2026")
    rr.font.size = Pt(8); rr.font.color.rgb = rgb("999999"); rr.font.name = "Calibri"

    col_width(tbl, 0, 6.0); col_width(tbl, 1, 5.5); col_width(tbl, 2, 4.7)

    # Bottom rule under header
    rule_p = header.add_paragraph()
    rule_p.paragraph_format.space_before = Pt(2)
    rule_p.paragraph_format.space_after  = Pt(0)
    pPr = rule_p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), TEAL)
    pBdr.append(bot)
    pPr.append(pBdr)


def build_footer(doc):
    section = doc.sections[0]
    footer = section.footer
    p = footer.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Spornado Disease Risk Model  |  Confidential  |  Page ")
    r.font.size = Pt(8); r.font.color.rgb = rgb("AAAAAA")
    fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"),"begin")
    ins  = OxmlElement("w:instrText"); ins.text = "PAGE"
    fld2 = OxmlElement("w:fldChar"); fld2.set(qn("w:fldCharType"),"end")
    r2 = p.add_run()
    r2.font.size = Pt(8); r2.font.color.rgb = rgb("AAAAAA")
    r2._r.append(fld1); r2._r.append(ins); r2._r.append(fld2)

# ── Layout helpers ────────────────────────────────────────────────────────────

def gap(doc, before=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after  = Pt(0)


def h1(doc, text):
    """ALL CAPS bold + full-width teal bottom underline."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(6)
    r = p.add_run(text.upper())
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = rgb(CHARCOAL)
    r.font.name = "Calibri"
    # Bottom border in teal
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "8")
    bot.set(qn("w:space"), "4")
    bot.set(qn("w:color"), TEAL)
    pBdr.append(bot)
    pPr.append(pBdr)


def h2(doc, text):
    """Teal bold text, no decoration."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(11)
    r.font.color.rgb = rgb(TEAL)
    r.font.name = "Calibri"


def body(doc, text, space_before=2, space_after=6, italic=False, bold_lead=None):
    """Body paragraph. Optional bold_lead = (bold_text, rest_text)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.line_spacing = Pt(14)
    if bold_lead:
        rb = p.add_run(bold_lead + "  ")
        rb.bold = True; rb.font.size = Pt(10)
        rb.font.color.rgb = rgb(CHARCOAL); rb.font.name = "Calibri"
    r = p.add_run(text)
    r.italic = italic
    r.font.size = Pt(10)
    r.font.color.rgb = rgb(CHARCOAL)
    r.font.name = "Calibri"
    return p


def bullet(doc, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.left_indent  = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.2)
    p.paragraph_format.line_spacing = Pt(13.5)
    dot = p.add_run("  -  ")
    dot.font.size = Pt(10); dot.font.color.rgb = rgb(TEAL)
    if label:
        rl = p.add_run(label + "  ")
        rl.bold = True; rl.font.size = Pt(10)
        rl.font.color.rgb = rgb(CHARCOAL); rl.font.name = "Calibri"
    rt = p.add_run(text)
    rt.font.size = Pt(10); rt.font.color.rgb = rgb(CHARCOAL); rt.font.name = "Calibri"


def numbered_step(doc, num, label, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.left_indent  = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.22)
    p.paragraph_format.line_spacing = Pt(13.5)
    rn = p.add_run(str(num) + ".  ")
    rn.bold = True; rn.font.size = Pt(10); rn.font.color.rgb = rgb(TEAL)
    rl = p.add_run(label + "  ")
    rl.bold = True; rl.font.size = Pt(10); rl.font.color.rgb = rgb(CHARCOAL)
    rt = p.add_run(text)
    rt.font.size = Pt(10); rt.font.color.rgb = rgb(CHARCOAL)


def callout_box(doc, label, text, bg=LBLUE, label_color=TEAL):
    """Labelled info box -- label header row + content row."""
    tbl = doc.add_table(rows=2, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_full_width(tbl)
    tbl_style(tbl, inner="D5D8DC")

    # Header row
    hc = tbl.rows[0].cells[0]
    cell_bg(hc, CHARCOAL); no_cell_border(hc)
    cell_pad(hc, top=60, bottom=60, left=150, right=150)
    cell_write(hc, label, bold=True, sz=9, color=WHITE)

    # Content row
    cc = tbl.rows[1].cells[0]
    cell_bg(cc, bg); no_cell_border(cc)
    cell_pad(cc, top=120, bottom=120, left=180, right=180)
    p = cc.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after  = Pt(0)
    r = p.add_run(text)
    r.bold = True; r.font.size = Pt(11)
    r.font.color.rgb = rgb(CHARCOAL); r.font.name = "Calibri"

    gap(doc, 4)


def thin_rule(doc, color=MGRAY):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(8)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "4")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), color)
    pBdr.append(bot)
    pPr.append(pBdr)


def ref_line(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.left_indent  = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    r1 = p.add_run("[" + str(num) + "]  ")
    r1.bold = True; r1.font.size = Pt(9); r1.font.color.rgb = rgb(TEAL)
    r2 = p.add_run(text)
    r2.font.size = Pt(9); r2.font.color.rgb = rgb(CHARCOAL)


# ── Data table factory ────────────────────────────────────────────────────────

def data_table(doc, headers, rows_data, col_widths_cm,
               row_colors=None, hdr_bg=CHARCOAL):
    """
    headers       : list of str
    rows_data     : list of tuples (cell_texts..., optional_bg_override)
    col_widths_cm : list of floats
    row_colors    : optional list of (bg_hex, text_hex) per row; else alternates
    """
    n_cols = len(headers)
    tbl = doc.add_table(rows=1 + len(rows_data), cols=n_cols)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_full_width(tbl)
    tbl_style(tbl, inner="D5D8DC")

    # Header
    for ci, h in enumerate(headers):
        c = tbl.rows[0].cells[ci]
        cell_bg(c, hdr_bg); no_cell_border(c)
        cell_pad(c, top=90, bottom=90, left=130, right=130)
        cell_write(c, h, bold=True, sz=9.5, color=WHITE)

    # Data rows
    for ri, row in enumerate(rows_data):
        bg = LGRAY if ri % 2 == 0 else WHITE
        texts = row[:-1] if row_colors and len(row) > n_cols else row
        for ci in range(n_cols):
            cell = tbl.rows[ri+1].cells[ci]
            override_bg = row_colors[ri][0] if row_colors else bg
            override_tc = row_colors[ri][1] if row_colors else CHARCOAL
            cell_bg(cell, override_bg); no_cell_border(cell)
            cell_pad(cell, top=90, bottom=90, left=130, right=130)
            txt = row[ci] if ci < len(row) else ""
            cell_write(cell, txt, sz=9.5, color=override_tc)

    for ci, w in enumerate(col_widths_cm):
        col_width(tbl, ci, w)

    gap(doc, 4)
    return tbl

# =============================================================================
# BUILD DOCUMENT
# =============================================================================
doc = Document()

for s in doc.sections:
    s.top_margin    = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin   = Cm(2.5)
    s.right_margin  = Cm(2.5)
    s.header_distance = Cm(1.2)

doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(10)

build_header(doc)
build_footer(doc)

# ── TITLE PAGE ────────────────────────────────────────────────────────────────
gap(doc, 50)

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_p.paragraph_format.space_before = Pt(0)
title_p.paragraph_format.space_after  = Pt(10)
tr = title_p.add_run("DISEASE RISK THRESHOLDS & RISK BANDS")
tr.bold = True; tr.font.size = Pt(24)
tr.font.color.rgb = rgb(CHARCOAL); tr.font.name = "Calibri"

sub_p = doc.add_paragraph()
sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub_p.paragraph_format.space_before = Pt(0)
sub_p.paragraph_format.space_after  = Pt(6)
sr = sub_p.add_run("Methodology, Rationale & Industry Validation")
sr.font.size = Pt(15); sr.font.color.rgb = rgb(TEAL); sr.font.name = "Calibri"

cap_p = doc.add_paragraph()
cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_p.paragraph_format.space_before = Pt(0)
cap_p.paragraph_format.space_after  = Pt(30)
cr = cap_p.add_run("A Technical Reference for the Spornado Crop Disease Prediction System")
cr.italic = True; cr.font.size = Pt(11)
cr.font.color.rgb = rgb(DGRAY); cr.font.name = "Calibri"

thin_rule(doc, color=TEAL)

gap(doc, 16)
for label, value in [
    ("Document Type:", "Technical Reference"),
    ("Project:",       "Spornado Disease Risk Model"),
    ("Date:",          "May 2026"),
    ("Classification:","Confidential"),
]:
    mp = doc.add_paragraph()
    mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mp.paragraph_format.space_before = Pt(0)
    mp.paragraph_format.space_after  = Pt(4)
    rb = mp.add_run(label + "  ")
    rb.bold = True; rb.font.size = Pt(10.5)
    rb.font.color.rgb = rgb(CHARCOAL); rb.font.name = "Calibri"
    rv = mp.add_run(value)
    rv.font.size = Pt(10.5); rv.font.color.rgb = rgb(CHARCOAL); rv.font.name = "Calibri"

# ── 1. EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
h1(doc, "1   Executive Summary")
body(doc,
    "Standard machine-learning classifiers default to a 0.5 decision threshold that "
    "implicitly treats a missed outbreak and an unnecessary spray as equally costly errors. "
    "In crop disease management this assumption is economically indefensible: a missed "
    "outbreak can destroy an entire harvest while a false alarm costs one preventive spray. "
    "This document explains why the Spornado model replaces that default with a "
    "recall-constrained threshold, and how the three-zone risk band is derived. "
    "The asymmetric cost framework underpinning this model -- prioritising missed "
    "outbreak detection above all other metrics -- is not domain-specific: it is "
    "the same decision-theoretic foundation used in FDA-regulated medical diagnostics, "
    "financial fraud detection, IEC 61508 industrial safety systems, and cybersecurity "
    "intrusion detection. Within agriculture specifically, the methodology is validated "
    "by peer-reviewed systems including the FHB Risk Tool (wheatscab.psu.edu), "
    "EPIPRE, and published frameworks by Hughes & Burnett (2017) and Giroux et al. (2016)."
)

# ── 2. WHY STANDARD THRESHOLDS FAIL ──────────────────────────────────────────
h1(doc, "2   Why Standard Thresholds Fail in Plant Disease Forecasting")
body(doc,
    "A binary classifier minimising overall error rate treats every mistake equally. "
    "In plant pathology this produces dangerous under-alerting: spray advisories are "
    "withheld to avoid false positives at the expense of missing real outbreaks."
)

h2(doc, "2.1   Asymmetric Economic Costs")
body(doc,
    "The cost of a false negative (missed outbreak) vastly exceeds the cost of a false "
    "positive (unnecessary spray). Elkan (2001) [1] formalises this: the optimal decision "
    "threshold is C(FP) / [C(FP) + C(FN)], which collapses well below 0.5 when false "
    "negatives are more expensive."
)

# Cost table -- bold first column, standard rows
tbl_cost = doc.add_table(rows=3, cols=3)
tbl_cost.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_full_width(tbl_cost); tbl_style(tbl_cost, inner="D5D8DC")

for ci, h in enumerate(["Error Type", "Agronomic Consequence", "Indicative Cost"]):
    c = tbl_cost.rows[0].cells[ci]
    cell_bg(c, CHARCOAL); no_cell_border(c)
    cell_pad(c, top=90, bottom=90, left=130, right=130)
    cell_write(c, h, bold=True, sz=9.5, color=WHITE)

cost_data = [
    ("False Negative (missed outbreak)",
     "Disease spreads unchecked; significant yield loss;\npotential secondary spread",
     "USD 200-800 / acre\n(20-80% yield loss) [7]",
     LRED, DRED),
    ("False Positive (unnecessary alert)",
     "Single preventive fungicide application;\nno crop damage incurred",
     "USD 15-40 / acre\n(one spray) [7]",
     LGREEN, DGREEN),
]
for ri, (err, conseq, cost, bg, tc) in enumerate(cost_data):
    cells = tbl_cost.rows[ri+1].cells
    cell_bg(cells[0], bg); no_cell_border(cells[0])
    cell_pad(cells[0], top=95, bottom=95, left=130, right=130)
    cell_write(cells[0], err, bold=True, sz=9.5, color=tc)
    cell_bg(cells[1], WHITE if ri%2==0 else LGRAY); no_cell_border(cells[1])
    cell_pad(cells[1], top=95, bottom=95, left=130, right=130)
    cell_write(cells[1], conseq, sz=9.5)
    cell_bg(cells[2], bg); no_cell_border(cells[2])
    cell_pad(cells[2], top=95, bottom=95, left=130, right=130)
    cell_write(cells[2], cost, bold=True, sz=9.5, color=tc, align=WD_ALIGN_PARAGRAPH.CENTER)

col_width(tbl_cost, 0, 4.0); col_width(tbl_cost, 1, 8.3); col_width(tbl_cost, 2, 3.9)
gap(doc, 4)

callout_box(doc,
    label="KEY INSIGHT",
    text="Cost ratio 10:1 to 20:1 in favour of avoiding missed outbreaks.\n"
         "Optimal threshold collapses well below 0.5  (Elkan, 2001 [1])",
    bg=LAMBER)

h2(doc, "2.2   Class Imbalance Makes ROC Misleading")
body(doc,
    "The Spornado dataset has a 73% positive base rate (disease present). Saito & "
    "Rehmsmeier (2015) [3] demonstrate that under class imbalance the Precision-Recall "
    "curve is more informative than ROC because it is insensitive to the large pool of "
    "true negatives that artificially inflate ROC-AUC. Threshold selection is therefore "
    "performed on the Precision-Recall curve, not the ROC curve."
)

# ── 3. THEORETICAL FOUNDATION ─────────────────────────────────────────────────
h1(doc, "3   Theoretical Foundation")

h2(doc, "3.1   Recall-Constrained Threshold Optimisation")
body(doc,
    "The deployment threshold tau* is defined as the solution to the constrained "
    "optimisation below, with R_min = 0.90 (configurable). The training data is "
    "divided into three non-overlapping chronological slices: (1) a model-fit window "
    "(oldest ~80% of training) used to train and calibrate the pipeline; "
    "(2) a threshold-tuning window (newest ~20% of training) on which "
    "sklearn's precision_recall_curve is run to select tau* -- this window is "
    "train-only and is never seen by the model during fitting; and (3) a fully "
    "held-out test set that is untouched until final evaluation. The test set is "
    "used exclusively to report precision, recall and AUC at the pre-chosen tau* "
    "-- no threshold adjustment is made after the test set is observed."
)
callout_box(doc,
    label="OPTIMISATION FORMULA",
    text="tau*  =  argmax  Precision(tau)     subject to:  Recall(tau)  >=  R_min  (0.90)",
    bg=LBLUE)
body(doc,
    "This mirrors the sensitivity-first paradigm in medical diagnostic screening "
    "(Zweig & Campbell, 1993 [4]) and biosecurity early-warning models "
    "(Magarey et al., 2007 [5]), where missing a true positive event carries consequences "
    "that vastly outweigh the cost of a false alarm.",
    space_before=0
)

h2(doc, "3.2   Temporal Splits -- Eliminating Leakage Bias")
body(doc,
    "Two distinct leakage controls are applied. First, model selection uses "
    "TimeSeriesSplit k-fold cross-validation (strictly chronological folds) to "
    "produce unbiased AUC and F1 estimates during training -- no future observations "
    "can inform a past fold's score. Second, and separately, threshold tuning is "
    "performed on a dedicated chronological window carved from the training period "
    "(the newest ~20%), not from the final test set -- ensuring tau* is never "
    "chosen by peeking at the data used to report it. De Wolf & Isard (2007) [6] "
    "identify temporal leakage as the most common source of optimistic bias in "
    "plant disease forecasting models; this two-stage separation directly addresses "
    "both leakage vectors they describe."
)

h2(doc, "3.3   Champion Model Selection Policy")
body(doc, "Models are ranked by a four-step hierarchy before any threshold is applied:",
     space_after=3)
steps_data = [
    ("Recall gate",       "Discard any model whose threshold_achieved_recall < R_min."),
    ("Highest recall",    "Among feasible models, prefer the one that catches more outbreaks."),
    ("Highest precision", "Tie-break by minimising false alerts at the operating threshold."),
    ("AUC tie-break",     "If precision is identical, select the model with higher ROC-AUC."),
]
for i, (lbl, rest) in enumerate(steps_data, 1):
    numbered_step(doc, i, lbl, rest)

# ── 4. RISK BAND ARCHITECTURE ─────────────────────────────────────────────────
h1(doc, "4   Risk Band Architecture")
body(doc,
    "A binary HIGH / NOT-HIGH alert provides insufficient resolution: probabilities of "
    "0.51 and 0.99 would trigger identical instructions. The three-zone band converts "
    "a continuous probability score into graduated management guidance -- analogous to "
    "traffic-light frameworks used by EFSA in pest risk categorisation [8] and by the "
    "U.S. EPA in tiered ecological risk assessment [9]."
)

h2(doc, "4.1   Band Derivation")
bullet(doc, "HIGH threshold (tau*)",
       " The recall-constrained optimal point derived from the Precision-Recall curve "
       "(Section 3.1). Any predicted probability >= tau* triggers a HIGH alert.")
bullet(doc, "LOW threshold (tau_low)",
       " = tau* x medium_band_factor   A proportional boundary set at a fixed "
       "fraction of tau* (default 0.70). Probabilities below tau_low indicate safe "
       "conditions -- the model is expressing genuine low-risk confidence.")
bullet(doc, "MEDIUM band",
       " = [tau_low, tau*)   The watch zone. Conditions are trending toward the "
       "action threshold but have not yet reached it. Growers should scout fields "
       "and prepare materials.")
gap(doc, 4)
body(doc,
    "The medium_band_factor (default 0.70) is an operational parameter, not a "
    "statistical result. It defines the width of the watch zone as a proportion "
    "of the decision boundary -- a factor of 0.70 means the MEDIUM band spans "
    "the bottom 30% of tau*. This can be adjusted with the client to reflect "
    "operational lead-time requirements: a wider band (lower factor) provides "
    "earlier warning; a narrower band (higher factor) reduces unnecessary monitoring "
    "load. As seasonal data accumulates, the factor can be validated against "
    "empirical lead times between MEDIUM onset and confirmed outbreaks.",
    space_before=0
)

gap(doc, 4)
h2(doc, "4.2   Risk Band Reference Table")

bt = doc.add_table(rows=4, cols=4)
bt.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_full_width(bt); tbl_style(bt, inner="D5D8DC")

for ci, h in enumerate(["Risk Zone", "Probability Range", "Recommended Action", "Rationale"]):
    c = bt.rows[0].cells[ci]
    cell_bg(c, CHARCOAL); no_cell_border(c)
    cell_pad(c, top=90, bottom=90, left=130, right=130)
    cell_write(c, h, bold=True, sz=9.5, color=WHITE)

band_rows = [
    ("LOW",    "0.00  to  tau_low",
     "No action required.\nConditions unfavourable for disease.",
     "Probability well below boundary;\nfungicide yields no expected benefit.",
     LGREEN, DGREEN),
    ("MEDIUM", "tau_low  to  tau_high",
     "Scout fields. Prepare fungicide.\nMonitor weather closely.",
     "Model uncertain; ground scouting\nreduces unnecessary spray risk.",
     LAMBER, DAMBER),
    ("HIGH",   "tau_high  to  1.00",
     "Apply registered fungicide.\nAlert neighbouring field managers.",
     "Exceeds recall-constrained threshold;\noutbreak likely imminent.",
     LRED,   DRED),
]
for ri, (zone, rng, action, rat, bg, tc) in enumerate(band_rows):
    cells = bt.rows[ri+1].cells
    cell_bg(cells[0], bg); no_cell_border(cells[0])
    cell_pad(cells[0], top=100, bottom=100, left=130, right=130)
    cell_write(cells[0], zone, bold=True, sz=10.5, color=tc, align=WD_ALIGN_PARAGRAPH.CENTER)
    cell_bg(cells[1], LGRAY if ri%2==0 else WHITE); no_cell_border(cells[1])
    cell_pad(cells[1], top=100, bottom=100, left=130, right=130)
    cell_write(cells[1], rng, sz=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)
    cell_bg(cells[2], LGRAY if ri%2==0 else WHITE); no_cell_border(cells[2])
    cell_pad(cells[2], top=100, bottom=100, left=130, right=130)
    cell_write(cells[2], action, sz=9.5)
    cell_bg(cells[3], bg); no_cell_border(cells[3])
    cell_pad(cells[3], top=100, bottom=100, left=130, right=130)
    cell_write(cells[3], rat, italic=True, sz=9, color=tc)

col_width(bt, 0, 2.1); col_width(bt, 1, 3.5)
col_width(bt, 2, 5.3); col_width(bt, 3, 5.3)
gap(doc, 4)

# ── 5. INDUSTRY ADOPTION ──────────────────────────────────────────────────────
h1(doc, "5   Industry Adoption & Cross-Industry Validation")
body(doc,
    "The asymmetric cost argument -- C(FN) >> C(FP) -- is not an agricultural "
    "concept. It is the correct decision-theoretic framing for any domain where "
    "one type of error carries irreversible consequences. Recall-constrained "
    "threshold optimisation with graduated risk bands is a mature, battle-tested "
    "pattern across multiple industries. The Spornado model applies the same "
    "rigorous methodology used in medical diagnostics, financial fraud detection, "
    "industrial safety, and cybersecurity."
)

it = doc.add_table(rows=6, cols=3)
it.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_full_width(it); tbl_style(it, inner="D5D8DC")

for ci, h in enumerate(["Organisation", "System / Product", "Parallel with This Model"]):
    c = it.rows[0].cells[ci]
    cell_bg(c, CHARCOAL); no_cell_border(c)
    cell_pad(c, top=90, bottom=90, left=130, right=130)
    cell_write(c, h, bold=True, sz=9.5, color=WHITE)

ind_data = [
    ("Bayer Digital Farming\n(The Climate Corporation)",
     "Climate FieldView [11]",
     "Uses machine learning models for disease risk prediction with multi-level "
     "severity categorisation (mild / moderate / severe) and weather-based forecasting. "
     "Exact probability scoring and threshold methodology are proprietary."),
    ("Syngenta",
     "Cropwise Platform [12]",
     "Provides disease monitoring and management recommendations built on agronomic "
     "models and 20+ years of historical weather data. Specific internal alert "
     "calibration methodology is not publicly documented."),
    ("USDA APHIS & NC State University",
     "NAPPFAST -- National Plant\nPest Forecasting System [5]",
     "Peer-reviewed GIS-based system using weather-driven infection-period models "
     "for plant pathogen risk assessment. Threshold exceedance triggers graduated "
     "management advisories -- published in Plant Disease (2007)."),
    ("DTN / Progressive Farmer",
     "DTN Agronomic Insights [13]",
     "Field-level disease alert platform with customisable threshold-based alerting "
     "for pests and disease symptoms. Supports tiered monitoring; specific ML "
     "methodology is not publicly documented."),
    ("Penn State / USDA (1975)",
     "BLITECAST -- Potato Late\nBlight Forecast System [14]",
     "Foundational computerised forecast combining Hyre and Wallin epidemiological "
     "models; accumulated severity values trigger spray advisories. Basis for many "
     "modern late blight decision-support systems including BlightPro."),
]
for ri, (org, prod, par) in enumerate(ind_data):
    bg = LGRAY if ri % 2 == 0 else WHITE
    cells = it.rows[ri+1].cells
    cell_bg(cells[0], bg); no_cell_border(cells[0])
    cell_pad(cells[0], top=90, bottom=90, left=130, right=130)
    cell_write(cells[0], org, bold=True, sz=9.5, color=CHARCOAL)
    cell_bg(cells[1], bg); no_cell_border(cells[1])
    cell_pad(cells[1], top=90, bottom=90, left=130, right=130)
    cell_write(cells[1], prod, sz=9.5, color=TEAL)
    cell_bg(cells[2], bg); no_cell_border(cells[2])
    cell_pad(cells[2], top=90, bottom=90, left=130, right=130)
    cell_write(cells[2], par, sz=9)

col_width(it, 0, 3.7); col_width(it, 1, 3.8); col_width(it, 2, 8.7)
gap(doc, 4)

h2(doc, "5.1   Peer-Reviewed Validated Systems")
body(doc,
    "Beyond commercial platforms, the sensitivity-first threshold paradigm is well "
    "established in the peer-reviewed epidemiological literature. The four systems below "
    "have been independently validated in production environments and share the core "
    "logic of the Spornado framework: select the operating threshold by constraining "
    "sensitivity (recall) rather than optimising aggregate accuracy."
)

pvt = doc.add_table(rows=5, cols=3)
pvt.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_full_width(pvt); tbl_style(pvt, inner="D5D8DC")

for ci, h in enumerate(["System & Institution", "Key Publication", "Parallel with Spornado Model"]):
    c = pvt.rows[0].cells[ci]
    cell_bg(c, CHARCOAL); no_cell_border(c)
    cell_pad(c, top=90, bottom=90, left=130, right=130)
    cell_write(c, h, bold=True, sz=9.5, color=WHITE)

pv_data = [
    ("FHB Risk Assessment Tool\nwheatscab.psu.edu\n(Kansas State / Ohio State /\nPenn State -- USWBSI)",
     "De Wolf, Madden & Lipps\n(2003). Phytopathology\n93(4):428-435. [15]",
     "Logistic regression model calibrated to maximise sensitivity at the operating "
     "threshold; achieves >= 95% sensitivity in prospective validation. Champion model "
     "selected on sensitivity -- directly analogous to Spornado's recall-constrained "
     "champion-selection policy."),
    ("EPIPRE Decision-Support System\n(Wageningen University /\nDutch Ministry of Agriculture)",
     "Zadoks (1981). EPPO Bulletin\n11:365-369. [16]\nReinink (1986). Neth. J. Plant\nPathol. 92:3-14. [17]",
     "Economic threshold logic: spray action triggered only when expected loss from "
     "inaction exceeds treatment cost. Conceptual ancestor of recall-constrained "
     "optimisation -- asymmetric cost structure is mathematically equivalent."),
    ("Sensitivity / Specificity Framework\nfor Disease Warning Systems\n(Hughes & Burnett, 2017)",
     "Hughes & Burnett (2017).\nPhytopathology\n107(10):1136-1143. [18]",
     "Formal statistical framework explicitly advocating threshold selection based on "
     "sensitivity (recall) constraints in plant disease advisory systems. Directly "
     "validates the approach used in this model."),
    ("ROC-Based Threshold Re-optimisation\nfor FHB Warning System\n(Giroux et al., 2016)",
     "Giroux et al. (2016).\nPlant Disease\n100(6):1192-1201. [19]",
     "Uses ROC / precision-recall curve analysis to re-select FHB warning thresholds "
     "that maximise sensitivity for a given specificity level -- the same mathematical "
     "approach as Spornado's argmax Precision s.t. Recall >= R_min search."),
]
for ri, (sys, pub, par) in enumerate(pv_data):
    bg = LGRAY if ri % 2 == 0 else WHITE
    cells = pvt.rows[ri+1].cells
    cell_bg(cells[0], bg); no_cell_border(cells[0])
    cell_pad(cells[0], top=90, bottom=90, left=130, right=130)
    cell_write(cells[0], sys, bold=True, sz=9.5, color=CHARCOAL)
    cell_bg(cells[1], bg); no_cell_border(cells[1])
    cell_pad(cells[1], top=90, bottom=90, left=130, right=130)
    cell_write(cells[1], pub, sz=9, color=TEAL)
    cell_bg(cells[2], bg); no_cell_border(cells[2])
    cell_pad(cells[2], top=90, bottom=90, left=130, right=130)
    cell_write(cells[2], par, sz=9)

col_width(pvt, 0, 4.2); col_width(pvt, 1, 3.8); col_width(pvt, 2, 8.2)
gap(doc, 4)

h2(doc, "5.3   Cross-Industry Adoption of the Same Paradigm")
body(doc,
    "The table below shows that recall-constrained threshold design is not a "
    "niche agricultural technique -- it is the standard engineering response "
    "to asymmetric costs in any high-stakes classification system. Each domain "
    "independently arrived at the same mathematical structure: set a sensitivity "
    "floor first, then optimise precision within that constraint."
)

cit = doc.add_table(rows=6, cols=3)
cit.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_full_width(cit); tbl_style(cit, inner="D5D8DC")

for ci, h in enumerate(["Domain", "System / Standard", "Asymmetric Cost Logic"]):
    c = cit.rows[0].cells[ci]
    cell_bg(c, CHARCOAL); no_cell_border(c)
    cell_pad(c, top=90, bottom=90, left=130, right=130)
    cell_write(c, h, bold=True, sz=9.5, color=WHITE)

ci_data = [
    ("Medical Diagnostics",
     "FDA Diagnostic Device\nGuidance (2007) [20]",
     "FDA statistical guidance for in-vitro diagnostic devices requires sensitivity "
     "and specificity to be reported and evaluated separately. For screening tests "
     "(cancer, infectious disease), sensitivity constraints are set as primary "
     "requirements -- the direct equivalent of Spornado's recall floor."),
    ("Financial Fraud Detection",
     "Asymmetric Cost\nClassification [1]",
     "False Negative (missed fraud) = customer financial loss + regulatory exposure. "
     "False Positive (blocked legitimate transaction) = customer friction. "
     "Financial institutions apply cost-sensitive threshold optimisation "
     "(Elkan, 2001) to deliberately operate below 0.5 -- identical mathematical "
     "framing to this model. Internal thresholds are proprietary."),
    ("Industrial Safety Systems",
     "IEC 61508 Functional\nSafety Standard [21]",
     "Safety Integrity Levels (SIL 1-4) define maximum tolerable Probability of "
     "Dangerous Failure per Hour (PFDh) -- a regulatory false-negative ceiling. "
     "Sensors and actuators in nuclear, chemical and rail applications must meet "
     "these sensitivity floors before any other performance metric is considered."),
    ("Cybersecurity /\nIntrusion Detection",
     "Published IDS Research\n(Axelsson, 2000) [22]",
     "Axelsson (2000) formally demonstrated that IDS systems must be calibrated "
     "for high sensitivity (recall) to overcome the base-rate fallacy -- low attack "
     "frequency means even small false-negative rates cause most real attacks to be "
     "missed. Security operations centres configure detection thresholds accordingly."),
    ("Drug Safety\nSurveillance",
     "FDA MedWatch /\nEMA EudraVigilance",
     "Post-market pharmacovigilance uses Proportional Reporting Ratio (PRR) and "
     "similar signal-detection algorithms explicitly tuned for high sensitivity over "
     "precision. A missed adverse drug event signal (FN) carries patient safety "
     "and regulatory consequences that vastly exceed the cost of a false signal (FP). "
     "[Internal algorithm parameters are published in regulatory guidance.]"),
]
for ri, (dom, sys, logic) in enumerate(ci_data):
    bg = LGRAY if ri % 2 == 0 else WHITE
    cells = cit.rows[ri+1].cells
    cell_bg(cells[0], bg); no_cell_border(cells[0])
    cell_pad(cells[0], top=90, bottom=90, left=130, right=130)
    cell_write(cells[0], dom, bold=True, sz=9.5, color=CHARCOAL)
    cell_bg(cells[1], bg); no_cell_border(cells[1])
    cell_pad(cells[1], top=90, bottom=90, left=130, right=130)
    cell_write(cells[1], sys, sz=9, color=TEAL)
    cell_bg(cells[2], bg); no_cell_border(cells[2])
    cell_pad(cells[2], top=90, bottom=90, left=130, right=130)
    cell_write(cells[2], logic, sz=9)

col_width(cit, 0, 2.8); col_width(cit, 1, 3.4); col_width(cit, 2, 10.0)
gap(doc, 4)

# ── 6. CURRENT PERFORMANCE ───────────────────────────────────────────────────
h1(doc, "6   Current Model Performance at Operating Thresholds")
body(doc,
    "All metrics are computed at the recall-constrained deployment threshold on the "
    "chronological hold-out test set -- not at the sklearn default of 0.5."
)

pt = doc.add_table(rows=4, cols=6)
pt.alignment = WD_TABLE_ALIGNMENT.CENTER
tbl_full_width(pt); tbl_style(pt, inner="D5D8DC")

for ci, h in enumerate(["Crop", "Champion Model", "Threshold tau*", "Recall", "Precision", "ROC-AUC"]):
    c = pt.rows[0].cells[ci]
    cell_bg(c, CHARCOAL); no_cell_border(c)
    cell_pad(c, top=90, bottom=90, left=110, right=110)
    cell_write(c, h, bold=True, sz=9.5, color=WHITE, align=WD_ALIGN_PARAGRAPH.CENTER)

perf_data = [
    ("Corn",    "LogisticRegression (baseline)", "0.177", "94.8%", "83.3%", "0.834"),
    ("Soybean", "LogisticRegression (baseline)", "0.682", "95.2%", "97.5%", "0.911"),
    ("Wheat",   "LogisticRegression (baseline)", "0.283", "100%",  "75.9%", "0.680"),
]
for ri, (crop, model, thr, rec, prec, auc) in enumerate(perf_data):
    bg = LGRAY if ri % 2 == 0 else WHITE
    cells = pt.rows[ri+1].cells
    for ci, (txt, al) in enumerate([
        (crop,  WD_ALIGN_PARAGRAPH.LEFT),
        (model, WD_ALIGN_PARAGRAPH.LEFT),
        (thr,   WD_ALIGN_PARAGRAPH.CENTER),
        (rec,   WD_ALIGN_PARAGRAPH.CENTER),
        (prec,  WD_ALIGN_PARAGRAPH.CENTER),
        (auc,   WD_ALIGN_PARAGRAPH.CENTER),
    ]):
        cell_bg(cells[ci], bg); no_cell_border(cells[ci])
        cell_pad(cells[ci], top=90, bottom=90, left=110, right=110)
        col = TEAL if ci == 0 else CHARCOAL
        cell_write(cells[ci], txt, bold=(ci==0), sz=9.5, color=col, align=al)

col_width(pt, 0, 1.9); col_width(pt, 1, 6.3)
col_width(pt, 2, 1.8); col_width(pt, 3, 1.8)
col_width(pt, 4, 1.9); col_width(pt, 5, 2.5)
gap(doc, 3)
body(doc,
    "Wheat's lower AUC (0.680) reflects a limited training set (~370 rows). "
    "All three crops satisfy the >= 90% recall constraint at their respective "
    "operating thresholds. Confidence intervals will tighten as additional "
    "trap data is collected for the 2026 season.",
    italic=True, space_before=0
)

# ── REFERENCES ────────────────────────────────────────────────────────────────
thin_rule(doc, color=TEAL)
h1(doc, "References")

refs = [
    (1,  "Elkan, C. (2001). The foundations of cost-sensitive learning. Proc. 17th IJCAI, 973-978."),
    (2,  "Provost, F., & Fawcett, T. (1997). Analysis and visualization of classifier performance. Proc. 3rd KDD, 43-48."),
    (3,  "Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE 10(3), e0118432. doi:10.1371/journal.pone.0118432"),
    (4,  "Zweig, M. H., & Campbell, G. (1993). Receiver-operating characteristic (ROC) plots: a fundamental evaluation tool in clinical medicine. Clinical Chemistry 39(4), 561-577."),
    (5,  "Magarey, R. D., et al. (2007). NAPPFAST: An internet system for the risk assessment of plant pathogens. Plant Disease 91(8), 1064. doi:10.1094/PDIS-91-8-1064"),
    (6,  "De Wolf, E. D., & Isard, S. A. (2007). Disease cycle approach to plant disease prediction. Annual Review of Phytopathology 45, 203-220. doi:10.1146/annurev.phyto.44.070505.143425"),
    (7,  "USDA Economic Research Service (2023). Corn and Soybean Cost of Production. ERS Farm Production Expenditures Summary. https://www.ers.usda.gov/"),
    (8,  "EFSA Panel on Plant Health (2018). Guidance on quantitative pest risk assessment. EFSA Journal 16(8), e05350. doi:10.2903/j.efsa.2018.5350"),
    (9,  "U.S. EPA (1998). Guidelines for Ecological Risk Assessment. EPA/630/R-95/002F. https://www.epa.gov/risk"),
    (10, "Stern, V. M., et al. (1959). The integrated control concept. Hilgardia 29(2), 81-101."),
    (11, "Bayer Digital Farming / The Climate Corporation (2024). Climate FieldView -- Machine Learning Disease Risk Forecasting. "
         "https://climate.com/  [Platform documented; internal threshold methodology is proprietary and not publicly disclosed.]"),
    (12, "Syngenta (2024). Cropwise -- Integrated Crop Management Platform. "
         "https://www.cropwise.com/  [Platform and disease monitoring features documented; specific alert calibration methodology is proprietary.]"),
    (13, "DTN / Progressive Farmer (2024). DTN Agronomic Insights -- Field-Level Disease & Pest Alerts. "
         "https://www.dtn.com/agriculture/agribusiness/agronomic-platform/  [Threshold-based alerting documented; specific ML methodology is proprietary.]"),
    (14, "Krause, R. A., Massie, L. B., & Hyre, R. A. (1975). Blitecast: A computerized forecast of potato late blight. Plant Disease Reporter 59, 95-98."),
    (15, "De Wolf, E. D., Madden, L. V., & Lipps, P. E. (2003). Risk assessment models for wheat Fusarium head blight epidemics based on within-season weather data. "
         "Phytopathology 93(4), 428-435. doi:10.1094/PHYTO.2003.93.4.428  -- Operational system: https://www.wheatscab.psu.edu/"),
    (16, "Zadoks, J. C. (1981). EPIPRE: A disease and pest management system for winter wheat developed in the Netherlands. "
         "EPPO Bulletin 11(4), 365-369. doi:10.1111/j.1365-2338.1981.tb01945.x"),
    (17, "Reinink, K. (1986). Experimental verification and development of EPIPRE, a supervised disease and pest management system for wheat. "
         "Netherlands Journal of Plant Pathology 92(1), 3-14. doi:10.1007/BF01976371"),
    (18, "Hughes, G., & Burnett, F. J. (2017). Evaluation of probabilistic disease forecasts. "
         "Phytopathology 107(10), 1136-1143. doi:10.1094/PHYTO-01-17-0023-FI"),
    (19, "Giroux, M.-E., Bourgeois, G., Dion, Y., Rioux, S., Pageau, D., Voldeng, H., & Munkvold, G. (2016). "
         "Evaluation of forecasting models for Fusarium head blight of spring wheat. "
         "Plant Disease 100(6), 1192-1201. doi:10.1094/PDIS-04-15-0404-RE"),
    (20, "U.S. Food & Drug Administration (2007). Statistical Guidance on Reporting Results from Studies "
         "Evaluating Diagnostic Tests -- Guidance for Industry and FDA Staff. "
         "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/statistical-guidance-reporting-results-studies-evaluating-diagnostic-tests-guidance-industry-and-fda"),
    (21, "International Electrotechnical Commission (2010). IEC 61508: Functional Safety of "
         "Electrical / Electronic / Programmable Electronic Safety-related Systems (Ed. 2). "
         "Geneva: IEC. [Defines Safety Integrity Levels (SIL) based on maximum tolerable "
         "Probability of Dangerous Failure per Hour -- a regulatory false-negative ceiling.]"),
    (22, "Axelsson, S. (2000). The base-rate fallacy and the difficulty of intrusion detection. "
         "ACM Transactions on Information and System Security 3(3), 186-205. "
         "doi:10.1145/357830.357849"),
]
for num, text in refs:
    ref_line(doc, num, text)

doc.save(OUT)
print("Saved: " + OUT)
