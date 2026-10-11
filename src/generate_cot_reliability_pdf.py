"""
Generates a PDF documenting the CoT reliability test: the original 5-case
test (one patient per ACDC pathology group, paired with a fixed real ECG
record, use_rag=True throughout), the 2 real clinical-characterization
errors it found (which traceability cannot catch by design - a cited
feature can be real while its direction is stated backwards), the
deterministic directional-consistency fix added to sep_agent.py in
response, and a before/after re-run on the SAME 5 cases to check whether
the fix actually resolves them. Presentation only - no new analysis
happens here, brief bullet-point format (established convention).
"""
import json

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


def generate_cot_reliability_pdf(
    output_path,
    before_json_path="results/cot_reliability_test.json",
    after_json_path="results/cot_reliability_test_after_fix.json",
):
    with open(before_json_path, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(after_json_path, "r", encoding="utf-8") as f:
        after = json.load(f)
    after_by_id = {r["patient_id"]: r for r in after}

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=8.5, leading=11))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=13, spaceAfter=3))
    styles.add(ParagraphStyle(name="Quote", parent=styles["Normal"], fontSize=8.5, leading=11, leftIndent=10, textColor=colors.HexColor("#555555")))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=2 * cm, bottomMargin=2 * cm)
    s = []

    s.append(Paragraph("Chain-of-Thought Reliability Test &amp; Directional-Consistency Fix", styles["Title"]))
    s.append(Paragraph("sep_agent.py - 5 held-out patients, one per ACDC pathology group, use_rag=True throughout", styles["Heading3"]))
    s.append(Spacer(1, 0.3 * cm))

    s.append(Paragraph("1. Setup", styles["Heading2"]))
    s.append(_bullets([
        "5 test patients, one per ACDC pathology group (DCM, HCM, MINF, NOR, RV), all unseen by YOLO/U-Net/classifier training",
        "Same fixed real ECG record (00001_hr) paired with each, to isolate the MRI-side variation",
        "Each case: generate_sep_agent_decision run twice on the identical inputs - use_cot=False, then use_cot=True (both use_rag=True)",
        "Goal: does CoT change the output? Does it parse reliably? And - the actual question that matters - is the resulting text clinically consistent?",
    ], styles))

    s.append(Paragraph("2. Original test - structural results", styles["Heading2"]))
    n_differs = sum(r["summary_differs"] for r in before)
    n_first_no_cot = sum(r["attempts_no_cot"] == 1 for r in before)
    n_first_cot = sum(r["attempts_cot"] == 1 for r in before)
    n_reasoning = sum(bool(r["reasoning"]) for r in before)
    s.append(_bullets([
        f"Summary changed by CoT: {n_differs}/5",
        f"First-attempt parse success (no CoT): {n_first_no_cot}/5",
        f"First-attempt parse success (CoT): {n_first_cot}/5",
        f"Reasoning field present when required: {n_reasoning}/5",
        "Mechanically, CoT is fully reliable - never crashes, never fails to parse, always changes the output",
    ], styles))

    s.append(Paragraph("3. But reading the text: 2/5 real characterization errors", styles["Heading2"]))
    s.append(_bullets([
        "Traceability guarantees a cited (agent_id, feature) pair is real - it does NOT guarantee the model's qualitative word for that value (\"low\"/\"high\"/\"reduced\"/\"elevated\") is the correct direction",
        "This is the same failure class already documented in sep_agent_demo.ipynb (a 295.5 mL LV volume once called \"low\" instead of elevated/dilated) - still present here, even with RAG's correctness_rule active",
    ], styles))

    s.append(Paragraph("patient102 (true NOR, LV EF 70.9%, flag = high_lv_ejection_fraction_suspected)", styles["Heading3"]))
    p102 = next(r for r in before if r["patient_id"] == "patient102")
    s.append(Paragraph(f"CoT reasoning: “...a decreased ejection fraction (70.9%)...”", styles["Quote"]))
    s.append(Paragraph(f"CoT summary: “{p102['summary_cot']}”", styles["Quote"]))
    s.append(_bullets([
        "70.9% is elevated - it is exactly what triggered the high-EF flag, not a low-EF one",
        "The no-CoT summary for this same case avoided the error entirely by staying purely numeric (no qualitative adjective)",
    ], styles))

    s.append(Paragraph("patient109 (true RV, RV EF 33.7%, flag = low_rv_ejection_fraction_suspected)", styles["Heading3"]))
    p109 = next(r for r in before if r["patient_id"] == "patient109")
    s.append(Paragraph(f"CoT summary: “{p109['summary_cot']}”", styles["Quote"]))
    s.append(_bullets([
        "Reasoning text itself was accurate (“...the right ventricle has a low ejection fraction (33.7%)”)",
        "But the final summary understates it (“mildly reduced”) then, one sentence later, flatly contradicts itself: “no signs of reduced ejection fraction”",
        "Error introduced only during the reasoning-to-summary compression step, not in the reasoning itself",
    ], styles))

    s.append(Paragraph("Root cause common to both", styles["Heading3"]))
    s.append(_bullets([
        "The correct direction was already known - it's the flag itself (high_lv_.../low_rv_..._suspected), computed deterministically in detect_flags_mri",
        "The model didn't need to judge the number's direction at all, and got it wrong anyway when it tried to",
    ], styles))

    s.append(PageBreak())
    s.append(Paragraph("4. The fix", styles["Heading2"]))
    s.append(_bullets([
        "Same pattern already used once in agent_common.py (MedGemma's image over-citation was fixed by force-correcting evidence_ref in code, not by prompting harder alone)",
        "New: _check_directional_consistency(text, flags_for_review) in sep_agent.py - a sentence-level keyword heuristic (not NLI/a formal proof)",
        "For each flag with a known direction (the 2 EF flags, tachycardia, bradycardia), scans reasoning + summary text for a sentence mentioning the relevant measurement (and ventricle, with 1-sentence lookback for anaphoric follow-ups like “...no signs of reduced ejection fraction”) alongside a word implying the OPPOSITE direction",
        "A violation is treated exactly like a fabricated citation: it raises SepAgentParseError, which re-enters the existing retry loop (no new retry infrastructure needed)",
        "Also added a fixed, unconditional prompt rule (not gated behind use_rag/use_cot): flags already indicate the correct direction, describe it consistently rather than independently judging the number",
        "Limitation, stated plainly: this is a keyword heuristic scoped only to the flags explicitly listed in the code - it can miss phrasing it doesn't recognize, and is not a general correctness prover",
    ], styles))

    s.append(Paragraph("5. Re-run on the same 5 cases, after the fix", styles["Heading2"]))
    n_differs_a = sum(r["summary_differs"] for r in after)
    n_ok_no_cot = sum(r.get("no_cot_ok", True) for r in after)
    n_ok_cot = sum(r.get("cot_ok", True) for r in after)
    n_retry_no_cot = sum(r.get("no_cot_ok", True) and r["attempts_no_cot"] and r["attempts_no_cot"] > 1 for r in after)
    n_retry_cot = sum(r.get("cot_ok", True) and r["attempts_cot"] and r["attempts_cot"] > 1 for r in after)
    n_clean_no_cot = sum(r.get("no_cot_ok", True) and not r["remaining_violations_no_cot"] for r in after)
    n_clean_cot = sum(r.get("cot_ok", True) and not r["remaining_violations_cot"] for r in after)
    n_exhausted = sum((not r.get("no_cot_ok", True)) or (not r.get("cot_ok", True)) for r in after)
    s.append(_bullets([
        f"Summary still changed by CoT: {n_differs_a}/5 (fix does not suppress CoT's actual effect on output)",
        f"Succeeded within the retry budget: {n_ok_no_cot}/5 (no CoT), {n_ok_cot}/5 (CoT)",
        f"Of those, a retry was triggered and then succeeded: {n_retry_no_cot}/{n_ok_no_cot} (no CoT), {n_retry_cot}/{n_ok_cot} (CoT)",
        f"Final accepted output free of directional violations: {n_clean_no_cot}/{n_ok_no_cot} (no CoT), {n_clean_cot}/{n_ok_cot} (CoT)",
        f"Cases that exhausted the retry budget without the model ever self-correcting: {n_exhausted}/5 - reported honestly below, not hidden",
    ], styles))
    if n_exhausted:
        exhausted_rows = [["Patient", "Which call failed", "Final rejection reason"]]
        for r in after:
            for tag, ok_key, err_key in (("no_cot", "no_cot_ok", "error_no_cot"), ("cot", "cot_ok", "error_cot")):
                if not r.get(ok_key, True):
                    exhausted_rows.append([
                        r["patient_id"], tag,
                        Paragraph(str(r.get(err_key) or ""), styles["Small"]),
                    ])
        te = Table(exhausted_rows, colWidths=[2.5 * cm, 2.5 * cm, 13 * cm], repeatRows=1)
        te.setStyle(_table_style())
        s.append(te)

    s.append(Paragraph("Before / after, the two known failure cases specifically", styles["Heading2"]))
    rows = [["Patient", "Flag", "Before-fix CoT summary said...", "After-fix CoT summary says...", "Retry triggered?"]]
    for pid in ("patient102", "patient109"):
        b = next(r for r in before if r["patient_id"] == pid)
        a = after_by_id.get(pid, {})
        after_summary = a.get("summary_cot") if a.get("cot_ok", True) else f"(FAILED after retries: {a.get('error_cot')})"
        after_summary = after_summary or "(not yet re-run)"
        attempts = a.get("attempts_cot")
        rows.append([
            pid,
            Paragraph(", ".join(b["mri_flags"]), styles["Small"]),
            Paragraph(b["summary_cot"], styles["Small"]),
            Paragraph(after_summary, styles["Small"]),
            "Yes" if (attempts and attempts > 1) else "No",
        ])
    t = Table(rows, colWidths=[2 * cm, 3.2 * cm, 5.4 * cm, 5.4 * cm, 1.8 * cm], repeatRows=1)
    t.setStyle(_table_style())
    s.append(t)

    s.append(Spacer(1, 0.3 * cm))
    s.append(Paragraph("6. Conclusion", styles["Heading2"]))
    conclusion_bullets = [
        "Structural CoT reliability was already solid (5/5) and remains so after the fix",
        f"Directional-consistency violations in the final ACCEPTED output: {n_clean_cot}/5 clean with CoT, {n_clean_no_cot}/5 clean without CoT",
        "Both previously-broken cases (patient102, patient109) are covered by the new check - see table above for whether each specific case was corrected via retry",
        "This closes the loop opened by the sep_agent_demo.ipynb finding: traceability alone (citation is real) is necessary but not sufficient - a deterministic direction check was needed on top of it, same as evidence_ref needed a deterministic force-correction in agent_common.py",
        "Remaining honest limitation: the check only covers the flags explicitly listed in _DIRECTIONAL_FLAG_CHECKS (2 EF flags + tachycardia/bradycardia) - a similar characterization error on a flag not in that list (e.g. a pathology_<CLASS>_pattern_suspected mischaracterization) would not be caught",
    ]
    s.append(_bullets(conclusion_bullets, styles))

    s.append(PageBreak())
    s.append(Paragraph("7. Additional finding from this same re-run - not yet fixed", styles["Heading2"]))
    s.append(_bullets([
        "patient104's CoT summary: “...The RV ejection fraction is slightly low at 65.6%.”",
        "65.6% is not low - no low_rv_ejection_fraction_suspected flag fired for this patient (65.6% doesn't cross the 45% threshold), so the new directional check had no flag to check this claim against",
        "More importantly: this claim has NO traceability entry at all - none of the 8 entries in this response cite it",
        "This slips past every existing layer at once: traceability validation only checks that entries WHICH EXIST cite a real pair, not that every claim in the summary has one; the correctness_rule requiring a grounding citation is a prompt instruction only, not code-enforced; and the new directional check only fires for flags actually present",
        "Different bug category from the two fixed above (those were wrong-direction claims about an ALREADY-FLAGGED measurement) - this is an ungrounded, uncited claim about a measurement that never crossed any threshold",
        "Reported here honestly as a further, real gap found via the same test run, not folded into “fixed” - a proper fix would need a code-level check that every quantitative sentence in the summary has a matching traceability entry, which is future work, not implemented in this pass",
    ], styles))

    doc.build(s)
    return str(output_path)


if __name__ == "__main__":
    print(generate_cot_reliability_pdf("results/cot_reliability_test.pdf"))
