"""
PDF case report generation
=============================
Renders a single human-readable PDF from data already produced by the ECG
agent, MRI agent, and Sep Agent - a presentation layer only, no new data is
computed here. Intended for handing a case to a human reviewer who would
otherwise have to read raw JSON.

Dependency: reportlab a python library that serves for generating PDFs. Install with pip
(pip install reportlab) - already installed in this
project's environment.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

__all__ = ["generate_case_report_pdf"]


def _table_style():
    """Shared visual style for the findings/traceability tables below."""
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ]
    )


def _findings_table(findings, styles):
    """
    Build a reportlab Table summarizing a list of agent_common.Finding
    objects (feature, value, confidence, evidence type, rationale).

    Inputs:
        findings: list of Finding instances.
        styles: reportlab stylesheet (must include "Small" and "Identifier"
            paragraph styles - see generate_case_report_pdf).

    Output:
        reportlab Table flowable.

    Note: Paragraph is used for the feature, evidence type, value and rationale columns to allow line wrapping of long text.
    The confidence columns are rendered as plain text since they are typically short and numeric.
    """
    data = [["Feature", "Value", "Confidence", "Evidence type", "Rationale"]]
    for finding in findings:
        data.append(
            [
                Paragraph(finding.feature, styles["Identifier"]),
                Paragraph(str(finding.value), styles["Identifier"]),
                f"{finding.confidence:.2f}",
                Paragraph(finding.evidence_type, styles["Identifier"]),
                Paragraph(finding.rationale, styles["Small"]),
            ]
        )
    table = Table(data, colWidths=[3.4 * cm, 2 * cm, 1.8 * cm, 2.4 * cm, 6.9 * cm], repeatRows=1)
    #repeatRows=1 ensures the header row is repeated on each page if the table spans multiple pages.
    table.setStyle(_table_style())
    return table


def _traceability_table(traceability, styles):
    """
    Build a reportlab Table for a SepAgentDecision's traceability entries
    (statement, agent_id, feature).

    Inputs:
        traceability: list of TraceabilityEntry instances.
        styles: reportlab stylesheet (must include "Small" and "Identifier"
            paragraph styles - see generate_case_report_pdf).

    Output:
        reportlab Table flowable.
    """
    data = [["Statement", "Agent", "Feature"]]
    for entry in traceability:
        data.append(
            [
                Paragraph(entry.statement, styles["Small"]),
                entry.agent_id,
                Paragraph(entry.feature, styles["Identifier"]),
            ]
        )
    table = Table(data, colWidths=[9 * cm, 2 * cm, 5.5 * cm], repeatRows=1)
    table.setStyle(_table_style())
    return table


def generate_case_report_pdf(
    output_path,
    sep_decision,
    contributions,
    figure_paths=None,
    synthetic_pairing_note=None,
    patient_context_lines=None,
):
    """
    Render a single human-readable PDF combining a Sep Agent decision and
    the detailed findings of the modality agents that fed into it. Pure
    presentation of already-computed data - no new analysis happens here.

    Inputs:
        output_path: path (str or Path) to write the PDF to.
        sep_decision: sep_agent.SepAgentDecision instance.
        contributions: list of agent_common.AgentContribution instances that
            fed into sep_decision (rendered as one detailed section each).
        figure_paths: optional list of PNG file paths to embed at the end
            (e.g. MRI ED/ES figures, the ECG pipeline plot).
        synthetic_pairing_note: optional string; if given, rendered as a
            prominent disclaimer near the top (e.g. to flag that a case
            combines records from datasets with no real shared patients).
        patient_context_lines: optional dict mapping agent_id -> a
            human-readable patient-context line (agent_common.
            build_patient_context_line's output, e.g. "Patient: 56-year-old
            female, 63kg."). Rendered ONLY here, in the PDF, under each
            contributing agent's own findings section - NOT fed into any
            LLM prompt (deliberately: the Sep Agent already only combines
            findings the modality agents already interpreted, it never
            needs the raw demographics itself - see the modality agents'
            own patient_info parameter for where that context actually
            informs analysis). None (the default) omits it entirely -
            existing callers unaffected. Needs no same-patient/different-
            patient logic: each agent already gets its own page/section
            here, so one line per contribution is already unambiguous,
            even when - as with today's PTB-XL+ACDC test data - the two
            sources are different real people (see synthetic_pairing_note
            above for that disclaimer).

    Output:
        output_path as a string, for convenience.
    """
    styles = getSampleStyleSheet() #default styles include Title, Heading1, Heading2, Heading3, Normal, etc.
    #.add adds personalized small text style for fine print, e.g. traceability rationale
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10)) 
    # wordWrap="CJK" lets long, space-free snake_case identifiers (e.g.
    # "myocardium_volume_ed_ml") break at any character to fit a narrow
    # table column, instead of overflowing (Paragraph normally only breaks
    # at whitespace). Left as plain text (no character insertion), unlike
    # using zero-width spaces, which rendered as visible
    # tofu boxes since the default font has no glyph for that character.
    styles.add(ParagraphStyle(name="Identifier", parent=styles["Small"], wordWrap="CJK"))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    story = [] #elements to render in the PDF, in order. Each element is a "flowable" (Paragraph, Table, Image, Spacer, etc.)

    story.append(Paragraph("Multi-Agent Case Report", styles["Title"]))
    story.append(Paragraph(f"Case: {sep_decision.record_id}", styles["Heading3"]))
    story.append(
        Paragraph(f"Contributing agents: {', '.join(sep_decision.contributing_agents)}", styles["Normal"])
    )
    story.append(Spacer(1, 0.3 * cm))

    if synthetic_pairing_note:
        disclaimer_style = ParagraphStyle(
            name="Disclaimer", parent=styles["Normal"], textColor=colors.red, fontSize=9
        )
        story.append(Paragraph(f"Note: {synthetic_pairing_note}", disclaimer_style))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Integrated Summary (Sep Agent)", styles["Heading2"]))
    story.append(Paragraph(sep_decision.summary, styles["Normal"]))
    story.append(Spacer(1, 0.2 * cm))

    flags_text = ", ".join(sep_decision.flags_for_review) if sep_decision.flags_for_review else "none"
    story.append(Paragraph(f"Flags for review: {flags_text}", styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))

    # Deterministic cross-modality convergence from the knowledge graph (present
    # only when the Sep Agent ran with use_kg=True). LLM-independent and auditable.
    convergent = (sep_decision.trace or {}).get("kg_convergent_diagnoses")
    if convergent:
        story.append(Paragraph("Cross-Modality Convergent Evidence (Knowledge Graph)", styles["Heading2"]))
        for entry in convergent.values():
            story.append(Paragraph(
                f"<b>{entry['label']}</b> - independently supported by "
                f"{', '.join(entry['agents'])} ({', '.join(entry['supporting_findings'])})",
                styles["Normal"],
            ))
        story.append(Paragraph(
            "Convergence across independent modalities is stronger evidence than any single "
            "modality. This block is computed deterministically from the clinical knowledge graph, "
            "not generated by the language model.",
            styles["Small"],
        ))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("Traceability", styles["Heading2"]))
    story.append(_traceability_table(sep_decision.traceability, styles))
    story.append(Spacer(1, 0.3 * cm))
    story.append(
        Paragraph(
            "Note: traceability confirms each statement cites a real finding from a contributing agent - "
            'it does not verify that the summary\'s characterization of that finding (e.g. "low" vs "high") '
            "is clinically correct.",
            styles["Small"],
        )
    )

    for contribution in contributions:
        story.append(PageBreak())
        story.append(Paragraph(f"{contribution.agent_id.upper()} Agent Findings", styles["Heading2"]))
        story.append(Paragraph(f"Record: {contribution.record_id}", styles["Normal"]))
        if patient_context_lines and patient_context_lines.get(contribution.agent_id):
            story.append(Paragraph(patient_context_lines[contribution.agent_id], styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))
        agent_flags = ", ".join(contribution.flags_for_review) if contribution.flags_for_review else "none"
        story.append(Paragraph(f"Flags: {agent_flags}", styles["Normal"]))
        story.append(Spacer(1, 0.2 * cm))
        if contribution.findings:
            story.append(_findings_table(contribution.findings, styles))
        else:
            story.append(Paragraph("(no findings - insufficient signal for this modality)", styles["Normal"]))

    if figure_paths:
        story.append(PageBreak())
        story.append(Paragraph("Figures", styles["Heading2"]))
        for path in figure_paths:
            story.append(Paragraph(str(path), styles["Small"]))
            story.append(Spacer(1, 0.2 * cm))
            story.append(Image(str(path), width=14 * cm, height=7 * cm, kind="proportional")) #kind="proportional" preserves the image's aspect ratio, scaling to fit within the given width and height.
            story.append(Spacer(1, 0.5 * cm))

    doc.build(story)
    return str(output_path)
