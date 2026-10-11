# RAG downstream-utility: A/B synthesis (Sep Agent, `use_rag` on vs off)

**Status: open item resolved into a concrete, evidence-backed conclusion.** This
document records whether the opt-in RAG layer actually improves the Sep Agent's
final synthesis (not just retrieval quality), which the RAG section of the report
package (`07_rag_layer/`) had flagged as never measured. It supersedes that
caveat.

## Question

The RAG layer was validated at the *retrieval* level (embedding-model Pearson
evaluation) but never at the *downstream synthesis* level: does
`generate_sep_agent_decision(..., use_rag=True)` produce a better integrated
impression than `use_rag=False`? "Better" here means: does the model actually
retrieve relevant clinical knowledge, and does it *use/cite* it in the synthesis?

## Method

- **Isolation of the RAG effect.** Same three `AgentContribution` inputs
  (the 3-modality case: PTB-XL ECG `00001_hr` + ACDC `patient001` + UCI Cleveland
  row 0), same **fixed seed (42)** for both the `use_rag=False` (A) and
  `use_rag=True` (B) runs, so the *only* difference between A and B is the
  retrieved-knowledge block injected into the Sep Agent prompt. Any difference in
  output is attributable to that block, not to sampling noise.
- **Metrics.** (1) retrieval relevance of the top-k chunks; (2) whether B's
  traceability cites `agent_id="knowledge"` at all; (3) *what* it cites (real
  clinical criteria vs. dataset-documentation boilerplate); (4) whether the prose
  summary is grounded in retrieved knowledge.
- **Caveat throughout: n = 1** synthetic case (PTB-XL/ACDC/UCI share no real
  patients), seeded. Illustrative and internally consistent, not a statistical
  claim.

## Two production changes made along the way

Both are real improvements applied to `src/` (not experiment-only workarounds):

1. **`build_retrieval_query` (sep_agent.py) — clinical phrasing.** The query was
   a raw concatenation of flag names ("low lv ejection fraction suspected ..."),
   i.e. internal jargon. Replaced with a curated `_FLAG_QUERY_PHRASES` map that
   renders flags as natural clinical language ("reduced left ventricular ejection
   fraction, systolic dysfunction; ..."). Unmapped flags fall back to the old
   underscore→space rendering. Default behaviour unchanged (`use_rag=False`
   remains the default).

2. **`_is_low_value_chunk` + ingestion gate (rag_knowledge_base.py) — quality
   filter.** A per-chunk gate drops bibliography/contact/fragment/garbled
   boilerplate before embedding, dumping the dropped chunks to
   `dropped_chunks.json` for audit. Table chunks are never dropped (atomic
   criteria content); short chunks carrying a clinical signal (a digit, `%`, or a
   clinical keyword) are rescued from the length filter. On the 8-source set:
   **405 chunks kept / 118 dropped** (102 reference-list, 9 email/contact,
   5 garbled, 2 empty fragments). *A real bug was caught here:* the length filter
   initially deleted terse clinical criteria ("EF: normal ~55%, <35% = severe
   heart failure", LGE/EDV/wall-motion lines) — fixed by the clinical-signal
   rescue, verified no clinical content is lost.

## Results — four A/B configurations

| # | Query | Knowledge base | Retrieval scores | B cites knowledge? | Nature of the citation |
|---|---|---|---|---|---|
| 1 | flag jargon | full, unfiltered (510 chunks, 8 src) | 0.39–0.43 | No | — (includes an email/boilerplate chunk) |
| 2 | clinical | full, unfiltered (510) | 0.51–0.54 | No | — (4/5 = PTB-XL annotation-code lists) |
| 3 | clinical | filtered (405, 8 src) | 0.51–0.54 | **Yes** | **Superficial** — cites a PTB-XL SCP code list; summary contains no knowledge-derived content |
| 4 | clinical | **cardiac_ai_internship.pdf only (59)** | 0.47–0.51 | **Yes** | **Meaningful** — cites real clinical criteria (HFrEF definition, STEMI ECG criteria, wall-motion abnormalities) |

### Key diagnosis (why 1–3 failed downstream)

The Sep Agent builds **one combined query over all flags at once** (EF + thallium
+ BP + ST). The unfiltered/full KB is dominated by **dataset-description papers**
whose text is dense **lists of annotation codes** (PTB-XL Wagner: "ISCAL =
ischemic in anterolateral leads", "INJAL = subendocardial injury", ...). Those
lists share heavy vocabulary with the combined query, so similarity retrieval
surfaces **code lists rather than clinical criteria**. The ingestion filter does
*not* remove them (they are legitimate tables, not junk-by-heuristic), so run 3
still retrieved them — and B's single knowledge citation was therefore to a code
list: a **decorative citation, not clinical grounding**. Note that a *single*-flag
query ("reduced LV EF criteria") *did* retrieve the real EF-threshold criterion
cleanly on the filtered KB — it is specifically the **combined multi-flag query
that re-dilutes** retrieval.

### The isolation test (run 4) — KB content is the dominant lever

Removing every specialist/dataset-description paper and keeping only the author's
own clinical notes (`cardiac_ai_internship.pdf`, 59 chunks) **flipped the outcome**:

- Retrieval returned real clinical criteria (HFrEF vs HFpEF, STEMI "ST elevation
  ≥1 mm in ≥2 contiguous leads", waves/intervals normal ranges, Cine MRI), 4/5
  chunks clinical.
- B cited knowledge with **meaningful** clinical content for the first time
  (HFrEF definition, wall-motion abnormalities, dyssynchrony, STEMI criteria) —
  not SCP code boilerplate.

This confirms the hypothesis that the downstream RAG failure was driven mainly by
**KB content** (dataset-doc/epidemiology papers polluting retrieval), not by the
retrieval mechanism or the query alone.

## Honest limitations of the run-4 improvement

- **Citation ≠ deep integration.** The model now *cites* real criteria in
  traceability, but the prose summary is still findings-centric — it reports
  "LV EF low (23.7%)" without explicitly writing "23.7% meets HFrEF criteria,
  well below the 40% threshold". Downstream integration (G2) is **improved but
  still shallow** in the narrative.
- **Imperfect targeting + one broken chunk.** It cited STEMI criteria for a
  patient whose thallium showed a *fixed* defect (not an acute STEMI), and chunk
  [0] was a mangled STEMI table ("Table: None | | direct") — a PDF-extraction
  artifact an empty-table filter should catch.
- **n = 1, seeded.** Single synthetic case.

## Conclusions and next steps

1. **KB content curation is the highest-impact lever — confirmed empirically.**
   Dataset-description/epidemiology papers should not sit in the *diagnostic*
   retrieval index; the diagnostic index should hold clinical-criteria content.
   (Epidemiology, if kept, belongs in a separate index scoped to a future
   risk-context use-case, never mixed in.)
2. **Down-scope the PTB-XL / dataset annotation tables** by source/section — they
   are legitimate tables (so the heuristic filter keeps them) but are the wrong
   genre for diagnostic grounding and dominate combined-query retrieval.
3. **Per-modality retrieval** (each agent retrieves from its own domain KB with a
   single-modality query and forwards the sources; the Sep Agent stops issuing one
   diluted combined query) directly removes the combined-query dilution shown in
   runs 1–3. This is the proposed architectural change to validate next.
4. **Force downstream integration** (e.g. require the summary to cite a retrieved
   criterion when one is relevant, mirroring the existing directional-consistency
   checks) to convert "cites in traceability" into "grounded in the prose".

## Applied to the production index

Conclusion 1 (content curation) was **applied**, not just recommended:

- `rag_sources/` was reorganized so the non-diagnostic sources are preserved but
  excluded from the diagnostic index (the `__main__` glob is non-recursive, so a
  PDF in a subfolder is automatically out of the index without being deleted):
  - `rag_sources/cardiac_ai_internship.pdf` — the only diagnostic source (for now)
  - `rag_sources/dataset_docs/` — Wagner (PTB-XL), Bernard (ACDC), Alqudah
  - `rag_sources/epidemiology/` — lear, razieh, kreatsoulas, ahiduzzaman
    (reserved for a future, separate risk-context index)
- The production diagnostic index (`results/rag_knowledge_base/`, used by the Sep
  Agent when `use_rag=True`) was rebuilt to **59 chunks** from `cardiac_ai` alone
  — i.e. the content that behaved well in run 4 is now the default. Previous
  indexes are kept as backups (`chunks.prefilter.json`, `chunks.8source.json`);
  no git.
- **Remaining weakness:** this index is clean but thin (one informal-notes
  source). Adding 1–2 authoritative guideline sources (ESC EF thresholds,
  Sokolow-Lyon LVH criteria) into `rag_sources/` would strengthen it; rebuild is
  automatic via the glob.

## ESC 2021 HF Guidelines experiment — a source-extraction limitation

Adding an authoritative guideline (`2021_ESC_Guidelines_for_HF.pdf`) to strengthen
the thin diagnostic index produced an important negative-but-instructive result.

- **Filter hardening (applied).** Guideline bibliographies (hundreds of entries)
  slipped through the year+comma reference rule (mangled years, no "et al",
  merged author names). `_is_low_value_chunk` gained two bibliography-specific
  signals — dense author-initial groups (`_AUTHOR_RE`) and journal-citation
  patterns (`_CITATION_RE`). Reference-list drops rose from 148 to 461; verified
  0 false positives on `cardiac_ai` clinical chunks.
- **Retrieval (G1) improved:** the top-5 for the case query became 5/5 ESC
  clinical content (LVEF<25% transplant criteria, echocardiography, acute
  pulmonary oedema table), no bibliography.
- **But downstream still failed:** B cited no knowledge and the summary did not
  improve. **Root cause = PDF text-extraction quality, not filtering.** The ESC
  PDF (complex two-column layout) extracts with spaces stripped: **26.1% of ESC
  "words" are >15 characters (merged), vs 0.0% for cardiac_ai** — e.g.
  "severeTR,withastablepsychosocialbackground... LVEF<25%and". This garbled text
  (a) degrades embedding retrieval, (b) makes the hardened author-heuristic
  over-fire (merged words / dense abbreviations like LVEF/NYHA/EAST-AFNET4 look
  like author initials → 159 clinically-flavoured chunks over-dropped, some real
  recommendations), and (c) is unreadable to the LLM, so retrieved criteria are
  ignored.
- **Conclusion:** a naive `pdfplumber` ingestion of a complex guideline is
  insufficient — the blocker is extraction, not the RAG mechanism. Fix options:
  a layout-aware extractor (PyMuPDF / pymupdf4llm), a space-repair step, or a
  clean text source. Until then, the working diagnostic index is the clean
  `cardiac_ai` notes; the ESC PDF should be set aside (moved out of the glob) so
  it does not degrade the index with garbled text.

## Clean guideline sources — RAG utility demonstrated

After the raw ESC PDF failed (garbled extraction), two **cleanly formatted,
retrieval-oriented** guideline extracts were authored and added:
`ESC_2021_Heart_Failure_RAG_Source.pdf` and
`AHA_ACCF_HRS_2009_ECG_Chamber_Hypertrophy_RAG_Source.pdf` (single-column, real
text: 0.9% and 0.2% merged words vs 26% for the raw ESC). Index = cardiac_ai +
these two = 83 chunks.

- **A filter false positive was found and fixed first:** the hardened
  `_AUTHOR_RE` author-initial heuristic fired on medical abbreviations (LVEF,
  HFrEF, NYHA, NT-proBNP) and Roman-numeral classes ("Class I/II/III"), wrongly
  dropping the NYHA class definitions as "reference-list". `_AUTHOR_RE` was
  **removed** (it was only ever a band-aid for garbled-PDF bibliographies, no
  longer ingested); `_CITATION_RE` (year;volume) and the doi/"et al"/year+comma
  rules remain and are false-positive-free on clean text.
- **Retrieval (G1):** top-5 led by real criteria — "HFrEF: LVEF ≤40%. HFmrEF
  41–49%. HFpEF ≥50%", advanced-HF criteria.
- **Downstream (G2) — the payoff:** B cited knowledge with **specific, correct
  guideline thresholds**: HFrEF ≤40% / HFmrEF 41–49% / HFpEF ≥50%, NYHA class
  III/IV, "severe cardiac dysfunction LVEF ≤30%", "6-min walk <300 m, peak VO₂
  <12 mL/kg/min". The summary also improved ("LVEF is 23.7%, which suggests
  reduced systolic function"), with traceability grounded on the ESC HFrEF
  criterion.
- **Conclusion — RAG is useful when the KB holds clean, targeted clinical
  content.** The combined levers (ingestion filter + clinical query + source
  curation + clean extraction) move the Sep Agent from "cites boilerplate /
  nothing" to "cites HFrEF ≤40%, NYHA III/IV". Remaining: prose still doesn't
  explicitly write "23.7% = HFrEF" (deep integration → per-modality retrieval +
  forced citation); one mangled STEMI table chunk in cardiac_ai; n=1, seeded.

## G2 deep-integration experiment (integrate_criteria directive) — tried, not adopted

The remaining gap after clean-guideline curation is stylistic depth: the Sep
Agent *cites* the criterion (HFrEF ≤40%) in traceability but does not weave it
into the prose ("LVEF 23.7%, which suggests reduced systolic function" rather
than "...below the 40% HFrEF threshold"). An opt-in prompt directive
(`integrate_criteria`, in `build_sep_agent_prompt` / `generate_sep_agent_decision`,
default False) was added to instruct the model to state the value together with
the threshold/classification it crosses. A/B on the 3-modality case, same clean
guideline index, same seed, directive off vs on:

| | Directive OFF (B1) | Directive ON (B2) |
|---|---|---|
| Prose weaves the threshold | No | No |
| Knowledge citations | 7 | **0** |
| Summary | "LVEF 23.7%, which suggests reduced systolic function" | "LVEF of 23.7%" (flatter) |

**Negative result:** the directive did not achieve prose integration and
**degraded** grounding — over-constraining MedGemma 4B lost all knowledge
citations (7 → 0). Conclusion: the deep-integration gap is bounded by the small
model's capability, not by prompt engineering. `integrate_criteria` is kept as a
documented, default-off experiment knob (same convention as the temperature/seed
experiment); the realistic levers for G2 are a stronger model (cloud, not run
here) or per-modality retrieval — the latter addresses retrieval, not prose
weaving. n=1, seeded. Result saved to `g2_ab_result.json`.

## Stronger-model retest (Groq Llama-3.3-70B) — G2 is unblocked by model capability

The G2 negative result above concluded the deep-integration gap is bounded by
MedGemma 4B's *capability*, not by the prompt. That claim was tested directly by
swapping only the generator model for a much stronger free one — **Groq's
`llama-3.3-70b-versatile`** (OpenAI-compatible REST API, `generate_report_groq`
in `agent_common.py`) — while holding everything else fixed: same 3-modality
contributions, same production clean-guideline index (cardiac_ai + ESC + AHA),
same `seed=42`, same `integrate_criteria` directive. So the *only* variable vs the
MedGemma runs is the model. (Trade-off: Groq models are general, not medically
fine-tuned, and the prompt is sent to the cloud — acceptable here because the
3-modality case is synthetic, not a real patient.)

| Run | use_rag | integrate | Knowledge cites | Prose weaves a real threshold |
|---|---|---|---|---|
| A_norag | False | — | 0 | No¹ |
| B_rag | True | False | 1 (`low LV ejection fraction`) | No (cites, doesn't weave) |
| **C_rag_integrate** | **True** | **True** | **1** (`below the 40% HFrEF threshold`) | **Yes** |

¹ A_norag's summary said "…consistent with the flag for low LV ejection fraction
suspected" — restates the flag name, no KB, no numeric criterion (a regex artifact,
not real integration).

**Key result (C):** *"…reduced left ventricular ejection fraction (LVEF) of 23.7%,
below the 40% HFrEF threshold, indicating heart failure with reduced ejection
fraction."* This is exactly the value↔threshold weaving MedGemma 4B could not
produce — and, unlike MedGemma, the same `integrate_criteria=True` directive here
did **not** collapse grounding (1 citation retained, vs MedGemma's 7 → 0).

### Second strong model (gpt-oss-120b) — rules out a Llama-specific fluke

To check the G2-unblock is a general property of strong models and not an
idiosyncrasy of one Llama checkpoint, the same three runs were repeated with a
model from a **different family**, `openai/gpt-oss-120b` (also free via Groq).
Two operational notes: it is a **reasoning model** (hidden reasoning tokens spend
the completion budget, so the JSON truncated at the default 2048 — fixed with
`max_tokens`/`reasoning_effort="low"`, added to `generate_report_groq`; defaults
unchanged so the Llama/MedGemma runs stay byte-identical), and it has a lower
tokens-per-minute free cap (429 backoff added in the runner).

| Run | use_rag | integrate | Knowledge cites | Prose weaves a real threshold |
|---|---|---|---|---|
| A_norag | False | — | 0 | No (no KB → no criterion to weave) |
| **B_rag** | **True** | **False** | **1** (`HFrEF (LV EF ≤40% per ESC guideline)`) | **Yes — spontaneously, no directive** |
| C_rag_integrate | True | True | 1 (`23.7 % below the ≤40 % HFrEF threshold`) | Yes |

**Key refinement:** gpt-oss-120b weaves the criterion into the prose already at
`integrate_criteria=False` — *"LVEF of 23.7 % consistent with HFrEF (LV EF ≤40%
per ESC guideline)"* — i.e. a strong enough model needs no integration directive
at all; the RAG-injected criterion alone suffices, and the directive becomes
redundant rather than helpful. A_norag (no RAG) weaves nothing, confirming the
criterion comes from the KB and the model only supplies the integration.

**Capability ladder (the headline for the report):**

| Model | Weaves at directive OFF | Weaves at directive ON |
|---|---|---|
| MedGemma 4B (local, default) | No (cites 7×, no weave) | No (grounding collapses 7→0) |
| Llama-3.3-70B (Groq) | No (cites only) | Yes |
| gpt-oss-120b (Groq) | **Yes (spontaneous)** | Yes |

Prose integration **emerges with model scale**: absent at 4B, directive-triggered
at 70B, spontaneous at 120B. Two independent strong models (different families)
both unblock G2 → not a Llama artifact. Saved to `gptoss_ab_result.json`.

**Conclusions:**
- **G2 was a model-capability ceiling, confirmed twice.** The identical
  prompt/directive that degraded MedGemma achieves genuine prose integration on a
  70B model, and a 120B model integrates the criterion even without the directive.
- **The two design levers are separable:** RAG (retrieval + clean KB) surfaces the
  right criterion regardless of model; deep prose integration additionally needs a
  model strong enough to reason over it — the 4B has the former but not the latter.
- **RAG stays useful under the stronger model** (B/C still cite real KB criteria),
  so RAG and model capability are complementary, not substitutes.
- Caveats: n=1, seeded; Groq is general-purpose (loses MedGemma's medical
  fine-tuning); cloud call (fine for synthetic data only). This validates the
  *hypothesis*, it is not a production switch — the project's default remains the
  free, local, medically-tuned MedGemma 4B. Result saved to `groq_ab_result.json`.

## Master comparison — models × options (G2 chapter close)

One consolidated view of every run, over the same synthetic 3-modality case, same
clean guideline index (cardiac_ai + ESC + AHA), same `seed=42`. The two levers are
**RAG** (`use_rag`, inject retrieved clinical criteria) and the **integration
directive** (`integrate_criteria`, instruct the model to state value + threshold
together). "Knowledge citations" = grounded traceability entries citing the KB;
"Weaves real criterion in prose" = the summary states the value against the
criterion it crosses (e.g. "23.7% below the ≤40% HFrEF threshold"), not just that
it is "low".

| Model | use_rag | integrate | Knowledge citations | Weaves real criterion in prose |
|---|:---:|:---:|:---:|:---:|
| **MedGemma 4B** (local, default) | False | — | 0 (no KB) | No |
| **MedGemma 4B** | True | False | **7** | No |
| **MedGemma 4B** | True | True | **0** (collapsed) | No |
| **Llama-3.3-70B** (Groq) | False | — | 0 | No¹ |
| **Llama-3.3-70B** | True | False | 1 | No |
| **Llama-3.3-70B** | True | True | 1 | **Yes** |
| **gpt-oss-120b** (Groq) | False | — | 0 | No |
| **gpt-oss-120b** | True | False | 1 | **Yes** |
| **gpt-oss-120b** | True | True | 1 | **Yes** |

¹ Llama's no-RAG summary tripped the weave-regex on "consistent with the flag for
low LV EF suspected" — a restatement of the flag name, not a real numeric
criterion (no KB was injected) — so it is scored **No** here after manual review.

**How to read it (three findings in one grid):**
- **RAG usefulness** (compare `use_rag` False→True at `integrate=False`): every
  model gains real KB citations only with RAG on; without a KB there is no
  criterion to cite or weave, at any model size.
- **Grounding vs. over-constraining on the small model** (MedGemma rows): the
  4B grounds best of all (7 citations) but never weaves, and forcing it to weave
  (`integrate=True`) collapses grounding to 0 — it cannot do both at once.
- **Integration emerges with scale** (weave column, top to bottom): No at 4B (any
  option) → Yes at 70B only with the directive → Yes at 120B even without it. Two
  strong models from different families both cross the threshold; the 120B makes
  the directive redundant.

**Chapter verdict:** RAG (retrieval + clean KB) and model capability are
**complementary, separable levers**. RAG decides whether the right criterion is
*available*; model capability decides whether it is *woven into the reasoning*.
The project default (MedGemma 4B) has strong grounding but a hard integration
ceiling; that ceiling is a model property, confirmed by two independent strong
models. G2 is closed. Cloud backends stay opt-in / synthetic-data-only; the
production default remains local MedGemma 4B.

## Reproduce

- Groq stronger-model A/B (RAG utility + G2): `experiments/groq_ab.py` →
  `groq_ab_result.json` (Llama-3.3-70B); `experiments/gptoss_ab.py` →
  `gptoss_ab_result.json` (gpt-oss-120b, with 429 backoff + reasoning-model token
  budget). Both need `GROQ_API_KEY`; see `groq_backend_setup.md`.
- Filtered full-KB A/B: `experiments/rag_ab_test.py` → `rag_ab_result.json`.
- Cardiac-only isolation A/B: `experiments/rag_ab_cardiaconly.py` →
  `rag_ab_result_cardiaconly.json` (also rebuilds
  `results/rag_knowledge_base_cardiaconly/`).
- KB content analysis: `experiments/kb_analysis.py`.
- Both A/B scripts hold `AgentContribution` inputs fixed and set seed=42 so A vs B
  differ only by the injected knowledge block.
