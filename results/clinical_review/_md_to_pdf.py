# -*- coding: utf-8 -*-
"""Minimal, self-contained Markdown -> PDF renderer for the clinical-review dossier.
Handles: #/##/### headings, paragraphs, **bold**/*italic*, > blockquotes (boxed),
pipe tables (with header), '---' rules, '- ' bullets. Uses Windows Arial TTF so all
French accents and symbols (>= , x, << >> , em-dash) render correctly."""
import re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                HRFlowable, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

MD = r"results/clinical_review/dossier_evaluation_clinique.md"
PDF = r"results/clinical_review/dossier_evaluation_clinique.pdf"

# ---- fonts (Windows Arial) ----
pdfmetrics.registerFont(TTFont("Body", r"C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Body-Bold", r"C:/Windows/Fonts/arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Body-Italic", r"C:/Windows/Fonts/ariali.ttf"))
pdfmetrics.registerFont(TTFont("Body-BoldItalic", r"C:/Windows/Fonts/arialbi.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                              italic="Body-Italic", boldItalic="Body-BoldItalic")

NAVY = colors.HexColor("#1F3B6E")
DARK = colors.HexColor("#13242B")
GREY = colors.HexColor("#5B6B6E")
LIGHT = colors.HexColor("#EEF2F7")
HEADBG = colors.HexColor("#1F3B6E")

ss = getSampleStyleSheet()
def mk(name, **kw):
    base = dict(fontName="Body", textColor=DARK, leading=14, fontSize=10)
    base.update(kw); return ParagraphStyle(name, parent=ss["Normal"], **base)

S = {
 "h1": mk("h1", fontName="Body-Bold", fontSize=18, leading=22, textColor=NAVY,
          spaceBefore=4, spaceAfter=6, alignment=TA_CENTER),
 "h2": mk("h2", fontName="Body-Bold", fontSize=13, leading=16, textColor=NAVY,
          spaceBefore=12, spaceAfter=5),
 "h3": mk("h3", fontName="Body-Bold", fontSize=11, leading=14, textColor=DARK,
          spaceBefore=8, spaceAfter=3),
 "p":  mk("p", spaceAfter=5),
 "li": mk("li", leftIndent=12, spaceAfter=3),
 "quote": mk("quote", fontName="Body-Italic", textColor=colors.HexColor("#243B55"),
             leftIndent=2, rightIndent=2, leading=14),
 "cell": mk("cell", fontSize=9, leading=12),
 "cellb": mk("cellb", fontName="Body-Bold", fontSize=9, leading=12, textColor=colors.white),
 "sub": mk("sub", fontSize=10, leading=13, textColor=GREY, alignment=TA_CENTER, spaceAfter=2),
}

def inline(t):
    t = t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", t)
    return t

def split_table_row(line):
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]

lines = open(MD, encoding="utf-8").read().splitlines()
flow = []
i = 0
title_done = False
while i < len(lines):
    ln = lines[i]
    s = ln.strip()

    # headings
    if s.startswith("# "):
        flow.append(Paragraph(inline(s[2:]), S["h1"])); i+=1; continue
    if s.startswith("## "):
        flow.append(Paragraph(inline(s[3:]), S["h2"])); i+=1; continue
    if s.startswith("### "):
        flow.append(Paragraph(inline(s[4:]), S["h3"])); i+=1; continue

    # horizontal rule
    if s == "---":
        flow.append(Spacer(1,4))
        flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#C3CEDA")))
        flow.append(Spacer(1,4)); i+=1; continue

    # subtitle italics right after title (center)
    if not title_done and s.startswith("*") and s.endswith("*") and "|" not in s:
        flow.append(Paragraph(inline(s.strip("*")), S["sub"])); i+=1
        # keep consuming consecutive italic subtitle lines
        while i < len(lines) and lines[i].strip().startswith("*") and lines[i].strip().endswith("*"):
            flow.append(Paragraph(inline(lines[i].strip().strip("*")), S["sub"])); i+=1
        title_done = True
        continue

    # blockquote block
    if s.startswith(">"):
        buf=[]
        while i < len(lines) and lines[i].strip().startswith(">"):
            buf.append(lines[i].strip()[1:].strip()); i+=1
        text = " ".join(x for x in buf if x!="")
        qp = Paragraph(inline(text), S["quote"])
        box = Table([[qp]], colWidths=[165*mm])
        box.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,-1),LIGHT),
            ("LEFTPADDING",(0,0),(-1,-1),10),("RIGHTPADDING",(0,0),(-1,-1),10),
            ("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7),
            ("LINEBEFORE",(0,0),(0,-1),3,NAVY),
        ]))
        flow.append(box); flow.append(Spacer(1,5)); continue

    # table block
    if s.startswith("|") and i+1 < len(lines) and set(lines[i+1].strip()) <= set("|:- "):
        header = split_table_row(lines[i]); i+=2  # skip separator
        rows=[]
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append(split_table_row(lines[i])); i+=1
        ncols=len(header)
        avail=165*mm
        if ncols==4 and header[0] in ("#",""):
            widths=[avail*w for w in (0.05,0.55,0.13,0.27)]
        elif ncols==4:  # flag validation grid
            widths=[avail*w for w in (0.46,0.12,0.20,0.22)]
        elif ncols==2:
            widths=[avail*0.5, avail*0.5]
        else:
            widths=[avail/ncols]*ncols
        data=[[Paragraph(inline(c), S["cellb"]) for c in header]]
        for r in rows:
            r=(r+[""]*ncols)[:ncols]
            data.append([Paragraph(inline(c), S["cell"]) for c in r])
        tb=Table(data, colWidths=widths, repeatRows=1)
        st=[("BACKGROUND",(0,0),(-1,0),HEADBG),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#B8C4D2")),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
            ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F5F8FB")])]
        tb.setStyle(TableStyle(st))
        flow.append(tb); flow.append(Spacer(1,6)); continue

    # bullet / numbered list
    if re.match(r"^(-|\d+\.)\s+", s):
        m=re.match(r"^(-|\d+\.)\s+(.*)", s)
        bullet = "•" if m.group(1)=="-" else m.group(1)
        text=m.group(2); i+=1
        # merge indented continuation lines into this item
        while i<len(lines):
            raw=lines[i]; nx=raw.strip()
            if nx=="" or not raw.startswith(" ") or nx.startswith(("-","#",">","|")) \
               or re.match(r"^\d+\.\s", nx):
                break
            text+=" "+nx; i+=1
        flow.append(Paragraph(f"{bullet}&nbsp;&nbsp;{inline(text)}", S["li"])); continue

    # blank
    if s=="":
        i+=1; continue

    # paragraph (gather until blank/structural)
    buf=[s]; i+=1
    while i<len(lines):
        nx=lines[i].strip()
        if nx=="" or nx.startswith(("#",">","|","---","- ")) or re.match(r"^\d+\.\s",nx):
            break
        buf.append(nx); i+=1
    flow.append(Paragraph(inline(" ".join(buf)), S["p"]))

doc = SimpleDocTemplate(PDF, pagesize=A4,
                        leftMargin=22*mm, rightMargin=22*mm,
                        topMargin=18*mm, bottomMargin=16*mm,
                        title="Dossier d'evaluation clinique")

def footer(canvas, d):
    canvas.saveState()
    canvas.setFont("Body", 8); canvas.setFillColor(GREY)
    canvas.drawString(22*mm, 10*mm, "Prototype de recherche - non destine a un usage clinique en l'etat")
    canvas.drawRightString(A4[0]-22*mm, 10*mm, "Page %d" % d.page)
    canvas.restoreState()

doc.build(flow, onFirstPage=footer, onLaterPages=footer)
print("PDF written:", PDF)
