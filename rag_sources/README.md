# RAG Knowledge Base — Sources

This document describes the source corpus behind the Sep Agent's opt-in
retrieval-augmented generation (RAG) layer: what is in the production index, what
is deliberately kept out, and why.

## How the index is built (one-line recap)
Only the **top-level PDFs** in `rag_sources/` are ingested (the ingestion glob is
non-recursive, so anything in a subfolder is automatically excluded without being
deleted). Each PDF is chunked with a header-aware splitter, passed through a
quality gate that drops boilerplate (bibliographies, footers, garbled text), and
embedded with a local sentence-embedding model. The result is saved as the
production index under `results/rag_knowledge_base/`.

**Production index: 3 sources / 83 chunks** (2 chunks dropped by the quality gate).

---

## 1. Sources IN the production index (diagnostic index)

These are curated for **clean, retrieval-oriented clinical criteria** — the genre
that actually helps the aggregator ground its synthesis.

| Source file | Chunks | What it is | What it contributes |
|---|---:|---|---|
| `cardiac_ai_internship.pdf` | 59 | The project's own consolidated clinical reference notes (single-column, clean text). | General diagnostic criteria and normal ranges across ECG / MRI / clinical findings; the backbone of the index. |
| `ESC_2021_Heart_Failure_RAG_Source.pdf` | 13 | A cleanly reformatted, single-column extract of ESC 2021 Heart-Failure guideline content. | Ejection-fraction categories (HFrEF ≤40 %, HFmrEF 41–49 %, HFpEF ≥50 %), NYHA functional class, advanced-HF criteria. |
| `AHA_ACCF_HRS_2009_ECG_Chamber_Hypertrophy_RAG_Source.pdf` | 11 | A cleanly reformatted extract of the AHA/ACCF/HRS 2009 recommendations on ECG chamber-hypertrophy interpretation. | Voltage-based chamber-hypertrophy (LVH) criteria that back the Sokolow–Lyon flag. |

**Why these two guideline files are re-authored extracts, not the raw PDFs.**
A raw guideline PDF (complex two-column layout) extracts with spaces stripped
(~26 % of "words" merged), which degrades retrieval and is unreadable to the model.
The two guideline sources above were therefore reformatted as clean, single-column
text (<1 % merged words) specifically for reliable retrieval.

---

## 2. Sources kept but EXCLUDED from the index (subfolders)

These are preserved for provenance / future use but are **out of the ingestion
glob** on purpose. The controlled A/B study
(`results/rag_ab_test/rag_utility_ab_synthesis.md`) showed why: dataset-description
and epidemiology papers are dense lists of annotation codes and population
statistics that dominate similarity retrieval and crowd out real clinical
criteria, so they must not sit in the *diagnostic* index.

| Subfolder | Files | Nature | Why excluded from the diagnostic index |
|---|---|---|---|
| `dataset_docs/` | PTB-XL (Wagner 2020), ACDC (Bernard 2018), Alqudah 2026 multimodal | Dataset-description papers | Cited for the datasets themselves; their annotation-code lists pollute diagnostic retrieval. |
| `epidemiology/` | PURE global CVD (Lear 2025), NHANES CVD (Ahiduzzaman 2025), UK Biobank ethnicity (Razieh 2022), social determinants (Kreatsoulas 2010) | Population CVD epidemiology | Not diagnostic criteria; reserved for a possible future, separately scoped risk-context index. |
| `guidelines_raw/` | Raw `2021_ESC_Guidelines_for_HF.pdf` | Full raw guideline PDF | Two-column layout extracts garbled (~26 % merged words); replaced by the clean ESC extract above. |
| `New/` | Recent cardiology papers (aficamten trials, aortic stenosis, HF device therapy, etc.) | Recent literature, staged | Newly added, not yet curated/ingested; left in a subfolder so it is out of the index until reviewed. |

---

## 3. Provenance and usage note

The guideline extracts (`ESC_2021_*`, `AHA_ACCF_HRS_2009_*`) are short, reformatted
excerpts of published clinical-guideline content, prepared **for local research
retrieval only**. Guideline text is copyrighted by its issuing societies; these
extracts are used on-device, are not redistributed, and are not a substitute for
the official guideline documents. The dataset and epidemiology papers retain their
original authorship as cited in the report bibliography.

---

## 4. Design conclusion (from the A/B study)

Curation of knowledge-base **content** — not the retrieval mechanism or the
embedding model — is the dominant factor in whether RAG helps: with the clean,
targeted diagnostic index above, the aggregator cites real criteria (e.g. the
HFrEF ≤40 % threshold); with an unfiltered index dominated by dataset/epidemiology
text, it cites boilerplate or nothing. Full evidence:
`results/rag_ab_test/rag_utility_ab_synthesis.md`. Adding one or two more clean,
authoritative guideline extracts to `rag_sources/` (top level) would strengthen
the index; a rebuild picks them up automatically.
