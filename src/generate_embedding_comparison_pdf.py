"""
Generates a PDF comparing two RAG embedding models (all-MiniLM-L6-v2 vs a
biomedical alternative, NeuML/pubmedbert-base-embeddings) on the same 276-
chunk knowledge base and the same test queries - presentation of an
already-run comparison, no new analysis happens here. Brief, bullet-point
format by request (not paragraph-heavy).
"""
import json
import sys

sys.path.insert(0, "src")
from rag_pearson_evaluation import QUERIES_AND_POSITIVE_HEADERS

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _table_style():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ])


def _bullets(items, styles, style_name="Body"):
    return ListFlowable(
        [ListItem(Paragraph(item, styles[style_name]), leftIndent=10) for item in items],
        bulletType="bullet", start="-",
    )


def generate_embedding_comparison_pdf(output_path, comparison_json_path="results/rag_embedding_comparison.json"):
    with open(comparison_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=3))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=2 * cm, bottomMargin=2 * cm)
    s = []

    s.append(Paragraph("RAG Embedding Model Comparison", styles["Title"]))
    s.append(Paragraph("all-MiniLM-L6-v2 (generalist) vs. NeuML/pubmedbert-base-embeddings (biomedical)", styles["Heading3"]))
    s.append(Spacer(1, 0.3 * cm))

    s.append(Paragraph("Setup", styles["Heading2"]))
    s.append(_bullets([
        "Same 276 chunks (identical chunking, 4 sources), embedded separately by each model",
        "10 queries: 6 direct + 4 paraphrased (no keyword overlap, to test synonym handling)",
        "results/rag_knowledge_base/ (general) vs. results/rag_knowledge_base_clinical/ (clinical)",
    ], styles))

    s.append(Paragraph("Full results (top-1 retrieval)", styles["Heading2"]))
    header_row = ["Query", "General: top section (score)", "Clinical: top section (score)", "Agree?"]
    rows = [header_row]
    for q, r in data.items():
        agree = "Yes" if r["general_header"] == r["clinical_header"] else "No"
        rows.append([
            Paragraph(q, styles["Small"]),
            Paragraph(f"{r['general_header'][:45]} ({r['general_score']:.3f})", styles["Small"]),
            Paragraph(f"{r['clinical_header'][:45]} ({r['clinical_score']:.3f})", styles["Small"]),
            agree,
        ])
    t = Table(rows, colWidths=[4.2 * cm, 4.9 * cm, 4.9 * cm, 1.5 * cm], repeatRows=1)
    t.setStyle(_table_style())
    s.append(t)

    s.append(PageBreak())
    s.append(Paragraph("Honest analysis", styles["Heading2"]))
    s.append(_bullets([
        "6/10: both models agree - clinical model scores higher confidence on the same answer (e.g. 0.428 vs. 0.743)",
        "4/10: models disagree",
        "<b>2 clear clinical-model failures</b>, both on paraphrase queries: \"ECG preprocessing...\" -> general correctly found the real source (0.671), clinical found an unrelated section (0.724, HIGHER confidence, WRONG)",
        "Same pattern on \"irregular heartbeat...\" (AFib): general correct (0.470), clinical wrong (0.563)",
        "2/10: both defensible, different granularity - not a real disagreement",
        "<b>Key concern</b>: clinical model's wrong answers scored HIGHER confidence than when correct - harder to catch downstream",
    ], styles))

    s.append(Paragraph("Round 1 - Pearson correlation (easy negatives)", styles["Heading2"]))
    s.append(_bullets([
        "Methodology from NeuML's own model card: correlate similarity score against known relevance",
        "10 queries x (1 correct chunk + 3 clearly-unrelated chunks) = 40 labeled pairs",
    ], styles))
    data3 = [
        ["Model", "Pearson r", "Mean sim (relevant)", "Mean sim (irrelevant)"],
        ["General", "0.601", "0.468", "0.246"],
        ["Clinical", "0.678", "0.512", "0.264"],
    ]
    t3 = Table(data3, colWidths=[4.5 * cm, 3 * cm, 4 * cm, 4 * cm])
    t3.setStyle(_table_style())
    s.append(t3)
    s.append(_bullets([
        "Clinical model wins here (both p &lt; 0.0001) - but easy negatives can't detect the failure mode found above",
    ], styles))

    s.append(PageBreak())
    s.append(Paragraph("Round 2 - harder Pearson test (25 queries, hard negatives)", styles["Heading2"]))
    s.append(_bullets([
        "Expanded to 25 queries (10 from round 1 + 15 new), covering all clean sections of the source document",
        "Added 2 <b>hard negatives</b> per query: another query's correct answer - plausible but wrong",
        "150 pairs total: 25 positive + 50 hard negative + 75 easy negative",
    ], styles))
    s.append(Spacer(1, 0.15 * cm))
    s.append(Paragraph("The 25 queries and their ground-truth (ok) section:", styles["Body"]))
    query_rows = [["#", "Query", "Ground-truth section"]]
    for i, (q, header) in enumerate(QUERIES_AND_POSITIVE_HEADERS, start=1):
        is_new = i > 10
        marker = " (new)" if is_new else ""
        query_rows.append([
            str(i),
            Paragraph(q + marker, styles["Small"]),
            Paragraph(header, styles["Small"]),
        ])
    tq = Table(query_rows, colWidths=[0.8 * cm, 10.2 * cm, 5.5 * cm], repeatRows=1)
    tq.setStyle(_table_style())
    s.append(tq)
    s.append(Spacer(1, 0.2 * cm))
    data4 = [
        ["Model", "Pearson r (all 150)", "Pearson r (positive vs. hard negatives, 75)"],
        ["General", "0.693", "0.796"],
        ["Clinical", "0.716", "0.773"],
    ]
    t4 = Table(data4, colWidths=[4.5 * cm, 4.5 * cm, 6.5 * cm])
    t4.setStyle(_table_style())
    s.append(t4)
    s.append(_bullets([
        "Full mix: clinical still wins (0.716 vs. 0.693) - coarse separation, consistent with round 1",
        "<b>Hard negatives isolated: general model wins (0.796 vs. 0.773)</b> - this is the task RAG retrieval actually performs",
        "Triangulates with the top-1 test's 2 clinical-model failures, via an independent, larger method",
    ], styles))

    s.append(Paragraph("Final conclusion", styles["Heading2"]))
    s.append(_bullets([
        "<b>Keep all-MiniLM-L6-v2 in production</b> - firm conclusion, not a close call",
        "Two independent tests agree: general model is more reliable at picking the correct section among plausible candidates",
        "Clinical model's advantage (coarse relevant/irrelevant separation) doesn't transfer to this harder, decision-relevant task",
        "Real cost already observed: 5x larger download, extended network instability during setup",
        "No case for switching",
    ], styles))

    doc.build(s)
    return str(output_path)


if __name__ == "__main__":
    print(generate_embedding_comparison_pdf("results/rag_embedding_comparison.pdf"))
