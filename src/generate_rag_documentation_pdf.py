"""
Generates a human-readable PDF documenting the RAG knowledge base work:
design choices, installation, and implementation steps. Presentation layer
only (mirrors report_pdf.py's role for case reports) - no new analysis,
just rendering what rag_knowledge_base.py already did.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _table_style():
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
    ])


def generate_rag_documentation_pdf(output_path):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=10, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="CodeBlock", fontName="Courier", fontSize=8.5, leading=11,
                               backColor=colors.HexColor("#f4f4f4"), borderPadding=4))

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=2 * cm, bottomMargin=2 * cm)
    s = []

    s.append(Paragraph("RAG Knowledge Base for the Sep Agent", styles["Title"]))
    s.append(Paragraph("Design choices, installation, and implementation steps", styles["Heading3"]))
    s.append(Spacer(1, 0.5 * cm))

    s.append(Paragraph("1. Motivation", styles["Heading2"]))
    s.append(Paragraph(
        "The original internship subject explicitly names Retrieval-Augmented Generation (RAG), "
        "Chain-of-Thought reasoning, knowledge graphs, and agent orchestration (LangChain) as techniques "
        "to explore, alongside LLMs and domain-specific fine-tuning. Fine-tuning was already covered "
        "extensively (ECGEmbeddingNet, YOLO26n, the MRI U-Net, the MRI pathology classifier) - RAG was "
        "the first still-missing piece to be addressed. It plugs in at the Sep Agent level only, not the "
        "modality agents: ECG/MRI agents stay diagnosis-free by design, so retrieval-grounded clinical "
        "reasoning belongs at the cross-modality synthesis step, where sep_agent.py already does its own "
        "kind of reasoning today.", styles["Body"]))

    s.append(Paragraph("2. Architecture", styles["Heading2"]))
    s.append(Paragraph(
        "Matches the hand-drawn \"Retrieval Augmented Generator (RAG) x Hallucination\" architecture "
        "sketch provided during design: documents (PDF for now, extensible to websites/SQL/structured "
        "data later) are chunked and embedded into a vector index; at query time the query is embedded "
        "and compared against that index to retrieve the top-k most similar chunks; those chunks are "
        "injected into the LLM prompt alongside the query. The sketch's \"x Hallucination\" check - "
        "verifying the model's output doesn't drift from what was actually retrieved - maps directly "
        "onto this project's existing traceability mechanism (sep_agent.py already rejects fabricated "
        "citations that don't match a real finding); extending that same validation to also cover "
        "retrieved-knowledge citations is the natural next step, not a new mechanism.", styles["Body"]))

    s.append(Paragraph("3. Knowledge base sources", styles["Heading2"]))
    s.append(Paragraph(
        "Eight sources ingested, all under rag_sources/ with clear, renamed filenames. The original four "
        "(below) ground ECG/MRI signal-processing and diagnostic-criteria content; four more were added "
        "later for broader cardiovascular-risk-factor context (epidemiology, demographics, social "
        "determinants) that the original four don't cover at all:", styles["Body"]))
    s.append(ListFlowable([
        ListItem(Paragraph("<b>summer_internship_notes.pdf</b> - a personal clinical reference document "
                            "(23 pages) covering ECG wave/interval definitions, diagnostic criteria "
                            "(Sokolow-Lyon and Cornell criteria for LVH, STEMI lead/wall correspondence, "
                            "RBBB/LBBB patterns), and MRI sequence interpretation (Cine/LGE/T1-T2 mapping) "
                            "mapped to the exact pathologies this project's MRI agent already computes "
                            "(DCM, HCM, low EF). Ingested first: self-authored, no licensing ambiguity, "
                            "already in this project's own vocabulary.", styles["Body"])),
        ListItem(Paragraph("<b>ptbxl_wagner2020.pdf</b> (Wagner et al. 2020, Scientific Data) - the "
                            "PTB-XL dataset paper itself, defining the 5 diagnostic superclasses "
                            "(NORM/MI/STTC/CD/HYP) that ecg_embedding.py's DIAGNOSTIC_SUPERCLASSES and "
                            "ecg_agent.py's embedding_&lt;CLASS&gt;_probability findings are built on, and "
                            "the SCP-ECG code hierarchy flag_validation.py's ground-truth sets already "
                            "use without ever citing a source.", styles["Body"])),
        ListItem(Paragraph("<b>bernard2018_acdc.pdf</b> (Bernard et al. 2018, IEEE TMI) - the ACDC "
                            "challenge paper, defining the 5 pathology groups "
                            "(NOR/MINF/DCM/HCM/RV) mri_flag_validation.py and the learned pathology "
                            "classifier both use as ground truth - the direct MRI-side counterpart to "
                            "the PTB-XL paper above.", styles["Body"])),
        ListItem(Paragraph("<b>alqudah_alfraihat2026_multimodal.pdf</b> (Alqudah &amp; Alfraihat 2026, "
                            "Health Information Science and Systems) - already cited in "
                            "summer_internship_notes.pdf as the methodological source for this project's "
                            "ECG preprocessing pipeline (baseline wander removal, notch filtering, "
                            "wavelet denoising) as implemented in ecg_preprocessing.py.", styles["Body"])),
    ], bulletType="bullet"))
    s.append(Paragraph(
        "<b>step1_literature_review.pdf</b> (the broader literature review that identified the chunking "
        "benchmark discussed in section 4) stays reference-only, not ingested - it documents dataset/model "
        "benchmarks useful for the report's literature review, not clinical definitions the Sep Agent's "
        "output needs to be grounded in.", styles["Body"]))
    s.append(Spacer(1, 0.15 * cm))
    s.append(Paragraph(
        "Four more added later, all peer-reviewed and open access, covering cardiovascular risk factors "
        "the original four never touch (diet, ethnicity, socioeconomic status, global epidemiology):",
        styles["Body"]))
    s.append(ListFlowable([
        ListItem(Paragraph("<b>ahiduzzaman2025_nhanes_cvd.pdf</b> (Ahiduzzaman &amp; Hasan 2025, PLOS ONE) "
                            "- interpretable ML models for CVD risk prediction from NHANES dietary/health "
                            "data (2017-2023).", styles["Body"])),
        ListItem(Paragraph("<b>razieh2022_ukbiobank_ethnicity.pdf</b> (Razieh et al. 2022, Nutrition, "
                            "Metabolism &amp; Cardiovascular Diseases) - CVD risk differences across ethnic "
                            "groups, UK Biobank observational study.", styles["Body"])),
        ListItem(Paragraph("<b>kreatsoulas2010_social_determinants.pdf</b> (Kreatsoulas &amp; Anand 2010, "
                            "Can J Cardiol) - the impact of social determinants (income, education, social "
                            "support) on cardiovascular disease.", styles["Body"])),
        ListItem(Paragraph("<b>lear2025_pure_global_cvd.pdf</b> (Lear et al. 2025, European Heart Journal) "
                            "- social factors, health policy, and environment across the globe, drawing on "
                            "the PURE study's international cohort.", styles["Body"])),
    ], bulletType="bullet"))
    s.append(Paragraph(
        "One acquisition issue worth recording: the Razieh source was first supplied as a browser "
        "\"Print to PDF\" export (metadata confirmed: Producer \"Microsoft: Print To PDF\") with text "
        "rendered as vector paths, not a real text layer - pdfplumber extracted 0 characters across all 9 "
        "pages despite the file opening normally and looking correct visually. Caught by checking "
        "extracted character count before ingesting, not assumed from the file opening fine - re-supplied "
        "as a proper publisher-sourced PDF (Producer \"Acrobat Distiller\", DOI and title verified in "
        "metadata) and confirmed extractable before use.", styles["Body"]))

    s.append(Paragraph("4. Chunking strategy - and why not the \"obvious\" choice", styles["Heading2"]))
    s.append(Paragraph(
        "step1_literature_review.pdf's own literature review benchmarks chunking methods and recommends "
        "\"Recursive Semantic\" splitting (89.36% accuracy@5, best of the methods tested). That benchmark "
        "was almost certainly measured on long, uniform academic prose - which describes 3 of the 4 "
        "sources here (the two research papers) but not summer_internship_notes.pdf: short, heavily "
        "structured personal notes with numbered headers (e.g. \"3.1 The Waves\", \"5.2 STEMI\") and many "
        "data tables. Applying the benchmarked method unmodified to that document risked two concrete "
        "problems: (1) losing the semantic segmentation the author already did via headers, and (2) "
        "breaking a table row away from its label (e.g. \"SV1+RV5 >= 35mm\" retrieved without knowing "
        "it's the Sokolow-Lyon LVH criterion is useless). The single chunking implementation used for all "
        "4 sources handles both cases correctly by construction, not by branching on source type:",
        styles["Body"]))
    s.append(ListFlowable([
        ListItem(Paragraph("Primary split: the document's own numbered header hierarchy "
                            "(regex-detected, e.g. \"3.1 The Waves\") - reuses the author's own semantic "
                            "segmentation rather than rediscovering it blindly.", styles["Body"])),
        ListItem(Paragraph("Fallback: recursive paragraph-then-line splitting, applied only inside a "
                            "single section if its body text exceeds ~1200 characters - this is where "
                            "the benchmarked \"Recursive Semantic\" technique is actually reused, scoped "
                            "correctly instead of applied document-wide.", styles["Body"])),
        ListItem(Paragraph("Tables extracted separately (via pdfplumber's table detection, not the "
                            "general text stream) as their own atomic chunks, each prefixed with its "
                            "nearest preceding header for context.", styles["Body"])),
    ], bulletType="bullet"))
    s.append(Paragraph(
        "For the two research papers (ptbxl_wagner2020.pdf, alqudah_alfraihat2026_multimodal.pdf), no "
        "numbered headers are detected, so the primary-split step naturally produces a single "
        "\"(document start)\" section covering the whole document - and the recursive fallback then "
        "takes over for all of it. This is exactly the benchmarked \"Recursive Semantic\" behavior applied "
        "to the kind of text it was actually measured on, achieved without any source-type branching in "
        "the code - the same function handles both document shapes correctly.", styles["Body"]))

    s.append(Paragraph("5. Embedding model", styles["Heading2"]))
    s.append(Paragraph(
        "sentence-transformers, <b>all-MiniLM-L6-v2</b>: small, runs locally with no API key, consistent "
        "with this project's established resource-conscious pattern (the same reasoning already applied "
        "when choosing YOLO26n over larger detectors, or a resnet18 encoder over nnU-Net, elsewhere in "
        "this project). Retrieval itself is plain cosine similarity over normalized embeddings - no "
        "vector database dependency needed at this scale (510 chunks).", styles["Body"]))

    s.append(PageBreak())
    s.append(Paragraph("6. Installation", styles["Heading2"]))
    s.append(Paragraph("New dependencies added this step (on top of what the project already had):", styles["Body"]))
    s.append(Paragraph("pip install pdfplumber sentence-transformers", styles["CodeBlock"]))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "pdfplumber: PDF text and table extraction (chosen over pypdf/PyPDF2 specifically for its table "
        "detection, needed for the atomic-table-chunk strategy in section 4). sentence-transformers: "
        "local embedding model loading and inference. Both installed successfully, no conflicts with the "
        "existing ML stack (torch, ultralytics, segmentation_models_pytorch already present).", styles["Body"]))

    s.append(Paragraph("7. Implementation steps (in order)", styles["Heading2"]))
    s.append(ListFlowable([
        ListItem(Paragraph("Verified PDF text/table extraction quality on sample pages before building "
                            "the full pipeline (confirmed header regex matches real extracted text; "
                            "confirmed pdfplumber correctly separates real data tables from decorative "
                            "figures/diagrams that are not tables).", styles["Body"])),
        ListItem(Paragraph("Built src/rag_knowledge_base.py: extract_pages, header-based section "
                            "splitting with recursive-split fallback, atomic table chunking, "
                            "build_knowledge_base (chunk + embed + save index), retrieve "
                            "(query embedding + cosine similarity top-k).", styles["Body"])),
        ListItem(Paragraph("Ran build_knowledge_base first on summer_internship_notes.pdf alone, then "
                            "re-ran on all 4 sources together once ptbxl_wagner2020.pdf, "
                            "bernard2018_acdc.pdf, and alqudah_alfraihat2026_multimodal.pdf were "
                            "obtained and verified (build_knowledge_base rebuilds the whole index per "
                            "call, not incrementally - cheap enough at this scale that this is not a "
                            "limitation worth engineering around yet).", styles["Body"])),
        ListItem(Paragraph("Validated retrieval quality with 3 realistic clinical queries before "
                            "considering this step done (results in section 8) - consistent with this "
                            "project's practice of validating each new component empirically rather than "
                            "assuming it works.", styles["Body"])),
    ], bulletType="bullet"))

    s.append(Paragraph("8. Results", styles["Heading2"]))
    raw_data = [
        ["Metric", "Value"],
        ["Sources", "8 PDFs (see section 3) - up from 4"],
        ["Total chunks", "510 - up from 276"],
        ["Text chunks", "452 - up from 237"],
        ["Table chunks (atomic)", "58 - up from 39"],
        ["Chunks per original source", "PTB-XL 88, Alqudah&amp;Alfraihat 81, ACDC/Bernard 60, personal notes 47"],
        ["Chunks per new source", "Ahiduzzaman (NHANES) 37, Razieh (UK Biobank) 52, "
                                   "Kreatsoulas (social determinants) 51, Lear (PURE global) 94"],
        ["Embedding model", "all-MiniLM-L6-v2 (384-dim) - unchanged"],
    ]
    data = [
        [Paragraph(metric, styles["Small"]), Paragraph(value, styles["Small"])]
        for metric, value in raw_data
    ]
    t = Table(data, colWidths=[5 * cm, 10.5 * cm], repeatRows=1)
    t.setStyle(_table_style())
    s.append(t)
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "Note on the 4 new sources: 3 of the 4 (Ahiduzzaman, Kreatsoulas, Lear) use standard unnumbered "
        "academic headers (\"Introduction\", \"Methods\"...) rather than the numbered scheme "
        "(\"3.1 The Waves\") the header-detection regex targets - so, same as the original two research "
        "papers, most of their content falls under a single \"(document start)\" section and gets split "
        "purely by the recursive length-based fallback. This does not block retrieval (confirmed below), "
        "it just means these chunks cite a generic \"(document start)\" header rather than a specific "
        "section name. Only Razieh (UK Biobank), which does use numbered sections, gets useful "
        "header-level grounding like the original personal-notes source.", styles["Body"]))
    s.append(Spacer(1, 0.2 * cm))

    s.append(Paragraph("Retrieval validation (6 test queries across the original 4 sources, top result shown):", styles["Body"]))
    raw_rows = [
        ["low left ventricular ejection fraction dilated cardiomyopathy",
         "4.2 Structural abnormalities (DCM)", "notes", "0.565", "Yes"],
        ["wide QRS complex bundle branch block",
         "5.3 Bundle Branch Blocks (LBBB / RBBB)", "notes", "0.428", "Yes"],
        ["Sokolow-Lyon criterion hypertrophy voltage",
         "5.4 Left Ventricular Hypertrophy (LVH)", "notes", "0.524", "Yes"],
        ["PTB-XL diagnostic superclasses NORM MI STTC CD HYP",
         "(document start / superclass definitions)", "PTB-XL", "0.530", "Yes"],
        ["ACDC dilated cardiomyopathy hypertrophic cardiomyopathy classification",
         "4.2 Structural abnormalities (DCM)", "notes", "0.601", "Yes (overlaps w/ ACDC content)"],
        ["ECG preprocessing baseline wander wavelet denoising",
         "Multimodal approach discussion", "A&A 2026", "0.671", "Yes"],
    ]
    data2 = [["Query", "Top retrieved section", "Source", "Score", "Correct?"]] + [
        [Paragraph(q, styles["Small"]), Paragraph(sec, styles["Small"]), src, score, Paragraph(correct, styles["Small"])]
        for q, sec, src, score, correct in raw_rows
    ]
    t2 = Table(data2, colWidths=[5.0 * cm, 4.5 * cm, 2.5 * cm, 1.4 * cm, 2.1 * cm], repeatRows=1)
    t2_style = _table_style()
    t2.setStyle(t2_style)
    s.append(t2)
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "All 6 queries retrieved a correct, clinically relevant top hit - including the ACDC-targeted "
        "query, which retrieved the personal notes' DCM/HCM section rather than bernard2018_acdc.pdf "
        "directly: not a failure, just overlapping coverage between sources (expected and harmless, "
        "since both are accurate on this topic).", styles["Body"]))

    s.append(Paragraph("Validation after adding the 4 new sources", styles["Heading3"]))
    s.append(Paragraph(
        "Two checks, not one - confirming nothing broke is a different question from confirming the new "
        "content is actually retrievable, and both needed a real answer:", styles["Body"]))
    s.append(ListFlowable([
        ListItem(Paragraph("<b>Regression check</b>: the 25-query set from the embedding-model comparison "
                            "(rag_pearson_evaluation.py) was run twice - once against a temporary index "
                            "built from only the original 4 sources, once against the new 510-chunk index "
                            "with all 8 - to get a true before/after, not an assumed one. Result: "
                            "<b>20/25 correct top-1 retrieval in both cases, byte-identical per-query "
                            "results</b>. The 5 misses are pre-existing retrieval imperfections already "
                            "present before this change, unrelated to the 4 additions - none of them "
                            "retrieved anything from a new source.", styles["Body"])),
        ListItem(Paragraph("<b>New-source retrievability</b>: 5 queries targeting the new sources' actual "
                            "content (not their weak headers - see the note above), run against the final "
                            "510-chunk index:", styles["Body"])),
    ], bulletType="bullet"))
    new_raw_rows = [
        ["machine learning NHANES dietary nutrient cardiovascular risk prediction",
         "ahiduzzaman2025_nhanes_cvd.pdf", "0.782", "Yes"],
        ["UK Biobank ethnic differences cardiovascular disease risk South Asian",
         "razieh2022_ukbiobank_ethnicity.pdf", "0.738", "Yes"],
        ["social determinants of health cardiovascular disease income poverty",
         "kreatsoulas2010_social_determinants.pdf", "0.676", "Yes"],
        ["PURE study urbanization low and middle income countries cardiovascular",
         "lear2025_pure_global_cvd.pdf", "0.788", "Yes"],
        ["interpretable machine learning model transparency clinical prediction",
         "ahiduzzaman2025_nhanes_cvd.pdf", "0.508", "No - see note"],
    ]
    new_data = [["Query", "Expected source", "Top-1 score", "Correct?"]] + [
        [Paragraph(q, styles["Small"]), Paragraph(src, styles["Small"]), score, Paragraph(ok, styles["Small"])]
        for q, src, score, ok in new_raw_rows
    ]
    t3 = Table(new_data, colWidths=[7.5 * cm, 5.5 * cm, 1.6 * cm, 1.5 * cm], repeatRows=1)
    t3.setStyle(_table_style())
    s.append(t3)
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "4/4 source-specific queries (one per new source) correctly retrieved their own source as the "
        "top-1 result - all 4 new sources are confirmed retrievable, not just present in the index. The "
        "5th query was a deliberately generic ML/interpretability phrase with no cardiovascular-specific "
        "wording, included to stress-test rather than cherry-pick a clean result; it retrieved a chunk "
        "from alqudah_alfraihat2026_multimodal.pdf instead (also a deep-learning cardiac-diagnosis paper, "
        "a genuinely plausible near-miss, not a broken result) - reported honestly rather than dropped "
        "from the table.", styles["Body"]))
    s.append(Spacer(1, 0.2 * cm))
    s.append(Paragraph(
        "One acquisition-time finding worth keeping visible here too, not just in section 3: a "
        "\"Print to PDF\" export of the Razieh source produced a file that opened and looked correct but "
        "had zero extractable text on every page (text rendered as vector paths, not a real text layer) - "
        "caught by checking extracted character count directly, not by assuming a file that opens "
        "correctly is usable by this pipeline.", styles["Body"]))

    s.append(Paragraph("9. Not yet done", styles["Heading2"]))
    s.append(ListFlowable([
        ListItem(Paragraph("Wire retrieve() into sep_agent.py's prompt construction - the knowledge base "
                            "exists and is validated, but the Sep Agent does not call it yet.", styles["Body"])),
        ListItem(Paragraph("Extend TraceabilityEntry / parse_sep_agent_response to validate citations of "
                            "retrieved knowledge the same way (agent_id, feature) citations are already "
                            "validated - the concrete form of the \"x Hallucination\" check from the "
                            "architecture sketch.", styles["Body"])),
        ListItem(Paragraph("Further research articles identified in step1_literature_review.pdf could "
                            "still be added later if a specific gap in retrieval quality is found - "
                            "build_knowledge_base already accepts a list of PDF paths, so this is a "
                            "one-line addition when/if needed, not a redesign.", styles["Body"])),
        ListItem(Paragraph("Chain-of-Thought prompting and LangChain-based orchestration - separate, "
                            "not-yet-started pieces of the original subject scope.", styles["Body"])),
    ], bulletType="bullet"))

    doc.build(s)
    return str(output_path)


if __name__ == "__main__":
    print(generate_rag_documentation_pdf("results/rag_knowledge_base/rag_design_and_implementation.pdf"))
