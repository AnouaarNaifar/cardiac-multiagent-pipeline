"""
RAG knowledge base - extraction, header-aware chunking, embedding, retrieval
================================================================================
First piece of the RAG layer discussed for the Sep Agent (see the "Retrieval
Augmented Generator (RAG) x Hallucination" architecture sketch this mirrors):
sources (PDF for now, "Extend database" later per that sketch) -> chunking ->
embedding -> a small vector index -> top-k retrieval at query time.

Chunking strategy - deliberately NOT the plain "Recursive Semantic" method
benchmarked in STEP 1 - ref.pdf (89.36% accuracy@5), because that benchmark
was almost certainly measured on long, uniform academic prose, not on a
document like this one (short, heavily structured personal notes with
numbered headers and many tables). Instead:
  1. Split by the document's OWN header hierarchy (numbered sections like
     "3.1 The Waves", "5.2 STEMI") - the author already did the semantic
     segmentation via headers, so reuse it rather than re-discovering it
     blindly paragraph-by-paragraph.
  2. Recursive paragraph/line splitting is used only as a FALLBACK, inside a
     single section, if that section's body text is too long for one chunk.
  3. Tables are extracted as their own ATOMIC chunks (via pdfplumber's table
     detection, not the general text stream), each prefixed with its
     nearest preceding header - so a chunk like the Sokolow-Lyon LVH
     criterion table is never separated from the context that makes it
     interpretable in isolation.

Embedding: sentence-transformers, "all-MiniLM-L6-v2" - small, local, no API
key, consistent with this project's resource-conscious, local-first
convention (same reasoning as choosing YOLO26n/resnet18 over heavier
alternatives elsewhere in this project).
"""

import json
import os
import re

import numpy as np
import pdfplumber

_HEADER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+([A-Z][^\n]{0,100})$")
_MAX_CHUNK_CHARS = 1200
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Cache the SentenceTransformer instance(s), keyed by model name, so repeated
# retrieve()/build_knowledge_base() calls reuse an already-loaded model instead
# of reloading weights from disk every time (a real cost when retrieving in a
# loop, e.g. a use_rag on/off A/B or an evaluation run). Keyed by name so a
# comparison run with a different embedding model still gets its own instance.
_MODEL_CACHE = {}


def _get_model(embedding_model_name):
    """Return a cached SentenceTransformer for embedding_model_name, loading it once."""
    if embedding_model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE[embedding_model_name] = SentenceTransformer(embedding_model_name)
    return _MODEL_CACHE[embedding_model_name]


def extract_pages(pdf_path):
    """Return a list of {"page_num", "text", "tables"} dicts, one per page."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            pages.append({"page_num": i + 1, "text": text, "tables": tables})
    return pages


def _split_into_sections(pages):
    """
    Walk every page's text line by line, splitting on numbered-header lines
    into (header_path, body_text, page_num) sections. Lines belonging to a
    table (already captured separately via extract_tables) are left in the
    text stream too - pdfplumber does not dedupe - so table content ends up
    both in a section's prose AND as its own atomic table chunk; this is a
    deliberate, harmless redundancy (a table's prose neighbors, e.g. row
    values folded into paragraph text, still help retrieval even without
    header formatting).
    """
    sections = []
    current_header = "(document start)"
    current_lines = []
    current_page = pages[0]["page_num"] if pages else 1

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"header": current_header, "text": body, "page_num": current_page})

    for page in pages:
        for line in page["text"].split("\n"):
            match = _HEADER_RE.match(line.strip())
            if match:
                flush()
                current_header = line.strip()
                current_lines = []
                current_page = page["page_num"]
            else:
                current_lines.append(line)
    flush()
    return sections


def _recursive_split(text, max_chars):
    """Fallback splitter for an over-long section: paragraphs -> lines."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split("\n\n")
    if len(paragraphs) > 1:
        chunks, buf = [], ""
        for para in paragraphs:
            if len(buf) + len(para) + 2 <= max_chars:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                if buf:
                    chunks.append(buf)
                buf = para
        if buf:
            chunks.append(buf)
        return [c for chunk in chunks for c in _recursive_split(chunk, max_chars)]
    lines = text.split("\n")
    chunks, buf = [], ""
    for line in lines:
        if len(buf) + len(line) + 1 <= max_chars:
            buf = f"{buf}\n{line}" if buf else line
        else:
            if buf:
                chunks.append(buf)
            buf = line
    if buf:
        chunks.append(buf)
    return chunks


def _format_table(table):
    if not table or not table[0]:
        return ""
    header, rows = table[0], table[1:]
    lines = [" | ".join(str(c) if c is not None else "" for c in header)]
    for row in rows:
        lines.append(" | ".join(str(c) if c is not None else "" for c in row))
    return "\n".join(lines)


def chunk_document(pdf_path, source_name=None, max_chunk_chars=_MAX_CHUNK_CHARS):
    """
    Output: list of {"text", "header", "page_num", "source", "chunk_type"}
    dicts - "chunk_type" is "text" or "table".
    """
    source_name = source_name or os.path.basename(pdf_path)
    pages = extract_pages(pdf_path)
    sections = _split_into_sections(pages)

    chunks = []
    for section in sections:
        for piece in _recursive_split(section["text"], max_chunk_chars):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append({
                "text": f"Section: {section['header']}\n{piece}",
                "header": section["header"],
                "page_num": section["page_num"],
                "source": source_name,
                "chunk_type": "text",
            })

    # Tables: atomic chunks, prefixed with the header active on their page.
    header_by_page = {}
    for section in sections:
        header_by_page.setdefault(section["page_num"], section["header"])
    last_header = "(document start)"
    for page in pages:
        if page["page_num"] in header_by_page:
            last_header = header_by_page[page["page_num"]]
        for table in page["tables"]:
            formatted = _format_table(table)
            if not formatted.strip():
                continue
            chunks.append({
                "text": f"Section: {last_header}\nTable:\n{formatted}",
                "header": last_header,
                "page_num": page["page_num"],
                "source": source_name,
                "chunk_type": "table",
            })

    return chunks


_YEAR_RE = re.compile(r"\b(?:19|20)\d\d\b")
# Bibliography-specific signal for reference sections that the year+comma rule
# misses: journal citations like "2009;361:2241", "2021;42:3599". Near-absent
# from clinical-criteria prose, so multiple occurrences are a safe reference-list
# signal.
# NOTE: an author-initial heuristic ("Surname AB") was tried and REMOVED - it
# fired on medical abbreviations (LVEF, HFrEF, NYHA, NT-proBNP) and Roman-numeral
# classes ("Class I/II/III"), wrongly dropping real criteria (the NYHA class
# definitions). It was only ever a band-aid for GARBLED bibliography from a
# complex-layout PDF (mangled ESC extraction), which is no longer ingested -
# clean sources' bibliographies are caught by the doi/"et al"/year+comma rules.
_CITATION_RE = re.compile(r"\b(?:19|20)\d\d\s?;\s?\d")

# A short chunk is only dropped if it ALSO lacks any clinical/quantitative signal.
# Terse-but-valuable clinical definitions ("EF: normal ~55%, <35% = severe heart
# failure", "LGE extent %", "wall motion score") are exactly the criteria content
# the KB exists for, so a digit, a "%", or a clinical keyword rescues a short chunk
# from the length filter. Verified against real dropped content: without this,
# cardiac_ai_internship.pdf's EF/EDV/LGE criteria lines were being discarded.
_SIGNAL_RE = re.compile(
    r"\d|%|\b(ejection fraction|EF|EDV|ESV|hypertroph|infarct|ischemi|stenos|regurgit|"
    r"fibrillat|tachycard|bradycard|bundle branch|systol|diastol|scar|LGE|wall motion|"
    r"ventric|atri|criteri|threshold|severe|moderate|myocard|perfusion|voltage)\b",
    re.I,
)


def _is_low_value_chunk(chunk):
    """
    Ingestion quality gate: return (is_low_value, reason) for one chunk, to keep
    bibliography/contact/fragment boilerplate out of the index. A KB content
    analysis found ~28% of chunks were low-value (110 reference-list chunks alone,
    plus emails, tiny fragments), which diluted retrieval - an A/B test
    (results/rag_ab_test/) showed the Sep Agent then retrieved dataset-doc
    boilerplate instead of clinical content.

    Deliberately conservative to avoid dropping real clinical text:
    - TABLE chunks are never dropped - they are the atomic criteria content
      (e.g. a Sokolow-Lyon LVH table) and are number/symbol-heavy by nature, so
      no length/non-alpha rule applies to them.
    - The non-alpha bar is set low (0.55, not 0.75) so accented French clinical
      notes and PDF-extraction-damaged text (stray U+FFFD chars) still survive.
    - The year-based reference-list rule requires corroboration (many commas or
      an "et al") so ordinary prose that merely mentions a few years is kept.
    """
    if chunk.get("chunk_type") == "table":
        return False, None

    text = chunk["text"]
    low = text.lower()
    # Strip the leading "Section: <header>\n" prefix for the length measure.
    body = text.split("\n", 1)[1].strip() if "\n" in text else text.strip()

    # Length is measured on the body (excludes the "Section: <header>" prefix),
    # but the clinical signal is checked on the FULL text: numbered list items
    # (e.g. "2. End-diastolic volume (EDV): Enlarged = cardiomyopathy") carry
    # their content in the header line, so checking body alone would wrongly drop
    # them.
    if len(body) < 150 and not _SIGNAL_RE.search(text):
        return True, "short(<150)"
    if "@" in text or "e-mail" in low or "email" in low:
        return True, "email/contact"
    years = len(_YEAR_RE.findall(text))
    if (
        "doi" in low
        or low.count("et al") >= 2
        or (years >= 4 and (low.count("et al") >= 1 or text.count(",") >= 6))
        or len(_CITATION_RE.findall(text)) >= 2          # multiple journal citations
    ):
        return True, "reference-list"
    alpha = sum(ch.isalpha() or ch.isspace() for ch in text)
    if len(text) and alpha / len(text) < 0.55:
        return True, "garbled(nonalpha)"
    return False, None


def build_knowledge_base(pdf_paths, output_dir="results/rag_knowledge_base", embedding_model_name=_EMBEDDING_MODEL_NAME):
    """Chunk + embed one or more PDFs into a single saved index.

    A per-chunk ingestion quality gate (_is_low_value_chunk) drops bibliography/
    contact/fragment boilerplate before embedding; the dropped chunks (with their
    reason) are written to dropped_chunks.json in output_dir for auditing, so the
    filter's effect is always inspectable rather than silent.

    embedding_model_name: any sentence-transformers-compatible model name,
        defaults to this module's standard all-MiniLM-L6-v2 - overridable
        for comparison runs (e.g. a clinical/biomedical embedding model)
        without touching the production index at the default output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_chunks = []
    for pdf_path in pdf_paths:
        raw_chunks.extend(chunk_document(pdf_path))

    all_chunks, dropped = [], []
    for chunk in raw_chunks:
        low_value, reason = _is_low_value_chunk(chunk)
        if low_value:
            dropped.append({**chunk, "_drop_reason": reason})
        else:
            all_chunks.append(chunk)

    from collections import Counter
    drop_summary = dict(Counter(d["_drop_reason"] for d in dropped))
    with open(os.path.join(output_dir, "dropped_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(dropped, f, indent=2, ensure_ascii=False)

    model = _get_model(embedding_model_name)
    embeddings = model.encode([c["text"] for c in all_chunks], show_progress_bar=True, normalize_embeddings=True)

    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    return {
        "n_chunks": len(all_chunks),
        "n_dropped": len(dropped),
        "drop_summary": drop_summary,
        "output_dir": output_dir,
        "embedding_model_name": embedding_model_name,
    }


def retrieve(query, top_k=3, kb_dir="results/rag_knowledge_base", embedding_model_name=_EMBEDDING_MODEL_NAME):
    """Return the top_k most similar chunks to query, each with a similarity score."""
    embeddings = np.load(os.path.join(kb_dir, "embeddings.npy"))
    with open(os.path.join(kb_dir, "chunks.json"), "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = _get_model(embedding_model_name)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    scores = embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:top_k]
    return [{"score": float(scores[i]), **chunks[i]} for i in top_indices]


if __name__ == "__main__":
    import glob

    # Ingest EVERY PDF currently in rag_sources/ rather than a hardcoded list,
    # so the index always reflects the actual source set (the previous hardcoded
    # list went stale when summer_internship_notes.pdf was replaced by
    # cardiac_ai_internship.pdf and step1_literature_review.pdf was added).
    pdf_paths = sorted(glob.glob("rag_sources/*.pdf"))
    print(f"Ingesting {len(pdf_paths)} source PDFs from rag_sources/:")
    for p in pdf_paths:
        print("  -", os.path.basename(p))
    result = build_knowledge_base(pdf_paths)
    print(result)
