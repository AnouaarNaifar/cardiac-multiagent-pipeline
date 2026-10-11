# Cardiac Multi-Agent Clinical Pipeline

Internship project (*stage 25-26*) - a multi-agent decision-**support** system for cardiac signal/image
analysis, on the public **PTB-XL** (ECG), **ACDC** (cardiac MRI) and **UCI Cleveland** (tabular) datasets.

Each modality is handled by its own agent that produces **structured findings** (never a final diagnosis);
a **Sep Agent** aggregates them into one traceable clinical impression. Every statement links back to a real,
computed finding.

> ⚠️ **Research / demo on public data - not a medical device and not a diagnosis.** Cross-modality cases are
> *synthetic pairings* (the three datasets share no real patients), used for architecture validation.

---

## Quickstart

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Local LLM (default backend, free, runs on your machine)
#    install Ollama, then:
ollama pull medgemma:4b

# 3. Launch the app
streamlit run Home.py        # → http://localhost:8501
```

Optional: a free **Groq** cloud backend (stronger text model) is available in the UI if you set
`GROQ_API_KEY` - the prompt then leaves the machine (fine for this public data).

## Using the app

Log in with a **role** (demo passwords, a UX *mode* - not real security):

| Role | Password | Sees |
|---|---|---|
| Patient | `patient2026` | Overview, ECG, Patient-data, MRI, Discuss (plain-language, no diagnosis) |
| Doctor | `doctor2026` | + Case aggregation |
| Researcher | `researcher2026` | + Validation, Models, prompt lab, backend selection |

Typical flow: run a modality page (ECG / MRI / Patient-data) → each adds a contribution to the **current
case** → **Case aggregation** fuses them (opt-in RAG / CoT / knowledge-graph) and exports a **PDF report**.
**Discuss** is a conversational assistant grounded in the case (and it *sees* the MRI image).

### Ready-made test cases
`results/synthetic_test_cases.md` lists **exact patients** to select on each page and the **field-by-field
Form B values**, all verified by running the agents - the fastest way to see cross-modality convergence.

## What's built

- **ECG agent** - single-lead pipeline (Pan-Tompkins, wavelet denoise) + multi-lead voltage/territorial flags
  (Sokolow-Lyon LVH, inferior-Q MI) + a learned 12-lead CNN embedding.
- **MRI agent** - ACDC preprocessing + ejection-fraction flags; a full **raw-image pipeline** (YOLO + U-Net +
  pathology classifier) so an MRI with no expert mask still works.
- **Patient-data agent** - 13-field UCI form → validated deterministic flags.
- **Sep Agent** - cross-modality aggregation with traceability, an optional **RAG** knowledge base and a
  clinical **knowledge graph** (cross-modality convergent diagnoses).
- **Streamlit UI** - upload + guardrails, prompt lab, live flag-validation, model documentation, role-based
  access, PDF export.

## Scope note (detection / diagnosis / prognosis)

- **Detection** — solid (validated deterministic flags per modality).
- **Diagnosis** — decision *support* only, by design (structured findings + convergence, never a verdict).
- **Prognosis**  — **out of scope**, and documented as such: PTB-XL/ACDC/UCI are cross-sectional (no
  follow-up/outcome data), so no prognostic model can be trained or validated here. See the project notes
  and local results write-ups for the detailed rationale.

## Layout

```
src/         Python modules (agents, pipelines, Sep Agent, RAG, KG)
Home.py      Streamlit entry (role router)  ·  views/  UI pages  ·  ui_*.py  UI glue
experiments/ reproducible experiment scripts
notebooks/   exploratory + validation notebooks
results/     local generated outputs; only synthetic_test_cases.md is versioned
kaggle/ ACDC/ RTLP/   local reference datasets, ignored by Git
uci_heart_disease/ rag_sources/   small local data and reference sources
CLAUDE.md    full architecture / design rationale (developer reference)
```

Large datasets, model weights, experiment outputs and archived runs are intentionally excluded from Git.
Download or generate them locally before running workflows that require them.

For the full design rationale, per-module detail and validation history, see **`CLAUDE.md`**.
