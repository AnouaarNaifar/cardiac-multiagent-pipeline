# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research/prototyping project (internship, "stage 25-26") for ECG/MRI signal processing and an LLM-based
multi-agent clinical pipeline, on the PTB-XL and ACDC datasets. There is no build system, package manifest,
or test suite — work happens through the Python modules and the Jupyter notebooks directly.

Each modality agent (ECG built, MRI built, structured patient data built) is one of several parallel agents
in a larger multi-agent framework: a separate aggregating component (the "Sep Agent", `src/sep_agent.py`)
combines each agent's output into a final integrated clinical impression plus a traceability summary. Because
of this, modality agents do **not** produce a final diagnostic report — they produce structured findings (see
`src/agent_common.py`, `src/ecg_agent.py`, `src/mri_agent.py`, `src/patient_data_agent.py` below) meant to be
consumed by that aggregator. All three modalities have been validated aggregating together through the Sep
Agent end-to-end (`src/sep_agent_3modality_test.py`, real LLM calls) - the shared `AgentContribution` schema
supports N modalities with no change to `sep_agent.py`, confirming the modular design.

**Current status** (detail per piece in Architecture below): all three modality agents (ECG through its
learned-embedding integration, MRI, patient-data) and the Sep Agent built and validated; each agent's
deterministic flags validated against ground truth (some threshold recalibrations, one new MRI flag). The MRI
agent gained a full raw-MRI pipeline (YOLO + U-Net + learned pathology classifier, replacing ACDC's expert
mask). **A full ECG-IMAGE branch was added** (realistic input = a photo/scan of a 12-lead ECG): a VLM image
agent + a GenECG-trained image classifier (real 5-superclass probabilities, macro F1 0.643 / MI 0.619 on
held-out fold 10, ~2× the VLM-only baseline) + an image **digitization** brick (`src/ecg_digitization.py`,
recovers a signal from the rhythm strip → measured heart rate/rhythm, HR MAE ≈ 1 bpm on GenECG), routed by
`src/ecg_router.py`, wired into the Sep Agent + KG convergence - see the "ECG IMAGE branch" architecture
section. A report-ready visual walkthrough of both branches lives in `results/ecg_pipeline_doc/`
(`ecg_pipeline_complete.html` + `ecg_digitization_report.md`). The ECG embedding also gained recurrent (GRU)
variants + early stopping + a duration-augmentation study, with the CNN retained as the production model. The
Sep Agent gained an opt-in RAG layer (`src/rag_knowledge_base.py`, production index 3 clean-guideline
sources / 83 chunks), opt-in Chain-of-Thought with deterministic directional/citation-consistency checks, an
opt-in clinical **knowledge-graph** layer (`src/knowledge_graph.py`, `use_kg`) that surfaces cross-modality
convergent diagnoses - A/B-validated positive on both the local 4B and a strong cloud model - and
an optional free Groq cloud backend used to prove the deep-integration gap ("G2") is a model-capability
ceiling, not a prompt problem (full write-up: `results/rag_ab_test/rag_utility_ab_synthesis.md`; production
default stays local MedGemma 4B). A standalone module scores ECG text against PTB-XL references (documented
metric limits), now with a biomedical clinical-cosine metric and a **per-agent doubling** that evaluates the
ECG SIGNAL and ECG IMAGE branches on the same 20 records (`src/agent_report_evaluation.py`). Multi-lead ECG support was **started** in two additive modules (`src/ecg_multilead.py`,
`src/ecg_territorial.py`) that read the raw 12-lead signal directly (bypassing the single-lead pipeline) and
add two validated deterministic flags to `ecg_agent.py`: **Sokolow-Lyon LVH** (HYP, F1 0.55, the project's
best ECG flag) and **territorial inferior-Q** (old inferior MI, F1 0.33, recall-favoring - the only
deterministic MI signal). Several other multi-lead features were tried and kept descriptive-only or dropped
after honest validation (axis, ST deviation, RVH; and BBB / acute ST elevation, which need proper QRS
delineation or an acute dataset PTB-XL lacks - see `results/ecg_multilead_validation/` and
`results/ecg_territorial_validation/`). **Framing note:** "passer au multi-lead" is a misnomer — the project
is already multi-lead where it matters, *heterogeneously and by design*, because each feature needs a different
preprocessing: voltage/amplitude criteria (Sokolow-Lyon, territorial-Q) use the **raw 12-lead signal in mV**
(the pipeline's z-score normalization + wavelet denoising would make a mV threshold meaningless), the learned
embedding uses its own **per-lead normalization** (matched to training), and rate/rhythm stay single-lead
(beat timing is shared across leads). So the real remaining gap is NOT "add the leads" (they're there) but a
**validated multi-lead QRS delineator** (onset/offset of P-QRS-T) — the one true lock, prerequisite for BBB and
a clean ST measurement. **Not yet started**: that validated QRS delineator; unifying beat detection into one
shared multi-lead pass (vs each additive module running its own Pan-Tompkins on lead II); per-beat chunking for
the embedding model; LangChain orchestration (no concrete need yet); and a stronger vision model for the MRI
agent's image grounding.

## Project structure

```
ECG/
├── src/                    Python modules (see Architecture below)
├── notebooks/              Jupyter notebooks
├── kaggle/                 PTB-XL dataset (reference data, not source)
├── ACDC/                   ACDC MRI dataset (reference data, not source)
├── uci_heart_disease/      UCI Heart Disease / Cleveland CSV + metadata (patient-data agent's source)
├── physionet/              small PhysioNet cross-check samples
├── RTLP/                   RTLP repo clone (PTB-XL English report translations)
├── rag_sources/            PDF sources ingested into the RAG knowledge base (reference data)
├── experiments/            reproducible one-off A/B scripts (RAG-utility, G2/model, KG), referenced by results/*.md
├── results/                generated output (figures, saved contributions, evaluation runs)
└── CLAUDE.md
```

Notebooks resolve the project root at runtime via a small `find_project_root()` helper (walks upward from
the notebook's cwd looking for `CLAUDE.md`), rather than assuming Jupyter's working directory - this is what
lets `notebooks/*.ipynb` import from `../src` and reference `../kaggle`, `../results`, etc. regardless of
where Jupyter was launched from. When running modules directly (not through a notebook), add `src/` to
`sys.path` (or run from within `src/`) - e.g. `python -c "import sys; sys.path.insert(0, 'src'); ..."`.

This project is **not** a git repository - there is no version-control safety net for destructive edits;
back up before large rewrites.

## Environment

A pinned **`requirements.txt`** now exists at the project root (`pip install -r requirements.txt` — versions
validated in the internship environment; the raw-MRI pipeline adds `ultralytics` + `segmentation-models-pytorch`,
see the torch/CUDA note in the file). No `pyproject.toml`. The per-module breakdown below still documents which
dependency each file needs (dependencies are also declared inline in each file's docstring/imports):

```
pip install numpy scipy pywavelets matplotlib wfdb pandas
pip install requests      # needed for the default local Ollama LLM backends
pip install anthropic     # only needed for the optional, non-default Claude API backend
pip install torch         # needed for ecg_embedding.py AND ecg_agent.py (learned embedding, now wired in)
pip install nibabel Pillow  # needed for mri_preprocessing.py
pip install reportlab      # needed for report_pdf.py
pip install langdetect sacrebleu rouge-score bert-score  # needed for report_evaluation.py
pip install sentence-transformers pdfplumber  # needed for rag_knowledge_base.py (the Sep Agent's RAG layer)
pip install ucimlrepo       # needed once to fetch UCI Heart Disease for patient_data_agent.py (then read locally)
pip install streamlit       # needed for the local UI (streamlit run Home.py, see below)
```

`requests` (already listed above for Ollama) also covers the optional Groq cloud backend
(`agent_common.generate_report_groq`, needs `GROQ_API_KEY`) - no extra dependency.

A free, local Ollama model (`ollama pull medgemma:4b`, MedGemma 4B) generates the actual clinical
findings/rationale text for every agent.

No API key is required or used by default; the Anthropic backend exists in `agent_common.py` only as an
optional alternative, unused unless explicitly wired in.

**Known resource constraint**: on a machine with ~16GB RAM, Ollama does not keep multiple models resident at
once (confirmed via `/api/ps` — loading one evicts another), and each cold load costs roughly 10-15s.

Running/testing code:
- Scripts are plain Python modules under `src/` (`python src/ecg_preprocessing.py` won't do anything on its
  own — they're meant to be imported). Exercise them via the notebooks or a throwaway `python -c` snippet
  with `src/` on `sys.path`.
- There is no test framework configured; `notebooks/test_verification.ipynb` is a manual data-integrity
  check, not an automated test suite.

## Architecture

The project is organized as a two-stage pipeline: signal processing → clinical features → structured
LLM-generated findings, aggregated by the Sep Agent, with a standalone evaluation/validation layer on top.

### `src/ecg_preprocessing.py` — signal pipeline
`ecg_preprocessing_pipeline(raw_signal, fs, powerline_freq=50.0)` runs, in order, and returns every
intermediate array/value in a dict (`raw`, `baseline_removed`, `powerline_removed`, `wavelet_denoised`,
`normalized`, `r_peaks`, `rr_intervals`, `qrs_durations_ms`, `rhythm_irregular`, `rhythm_cv`):
1. `remove_baseline_wander` — Butterworth high-pass (0.5 Hz) to remove drift.
2. `remove_powerline_interference` — IIR notch at 50/60 Hz.
3. `wavelet_denoise` — db6 wavelet, 5 levels, universal soft-threshold (MAD-based sigma estimate).
4. `normalize_signal` — z-score normalization.
5. `pan_tompkins_rpeaks` — Pan-Tompkins R-peak detection (band-pass → derivative → square → moving-window
   integrate → adaptive threshold), followed by `compute_rr_intervals`.
6. `compute_qrs_durations` — per-beat QRS duration estimate via an amplitude-threshold heuristic (not a
   validated clinical onset/offset detector — documented as such in its docstring).
7. `detect_rhythm_irregularity(rr_intervals, cv_threshold=0.15)` — flags rhythm irregularity from the RR
   interval coefficient of variation.

`plot_pipeline(results, fs, t_start, t_end, output_dir, file_name)` renders the 5-stage figure and saves a
PNG to `results/<...>/<file_name>.png` (defaults to `./results` next to the module if `output_dir` is
omitted). Downstream code (notebook, `ecg_agent.py`) consumes the dict returned by the pipeline — keep the
key names stable since multiple call sites depend on them structurally.

**Known limitation, partially addressed**: THIS pipeline still analyzes only a single lead - every step
(baseline removal, notch filter, wavelet denoise, normalize, R-peak detection, QRS duration, rhythm CV)
operates on one 1D array end to end (`raw_signal` never carries a lead dimension). Two parts of the project
now use all 12 leads via a separately-passed raw `multilead_signal`, both bypassing this pipeline: the learned
embedding model (`ecg_embedding.py`), and the new deterministic multi-lead feature module (`ecg_multilead.py`,
Sokolow-Lyon LVH voltage index - see its section below). So the rate/QRS/rhythm features here remain
single-lead, but multi-lead analysis is no longer entirely unstarted. **Why these parts bypass the pipeline
rather than extend it (deliberate, not a shortcut):** the right preprocessing differs per feature, so a single
uniform 5-step pipeline is the WRONG shape for them. Voltage/amplitude criteria (Sokolow-Lyon, territorial-Q)
must be measured on the **raw signal in real mV** — the pipeline's z-score normalization rescales into
standard-deviation units and wavelet denoising attenuates amplitudes, either of which makes a mV threshold
("≥ 3.5 mV") meaningless; the embedding re-normalizes **per-lead** to match its training. So "making this
*pipeline itself* multi-lead" does NOT mean running the same steps on 12 leads — it means a path-separated core
(a filtered/delineated branch for timing/morphology + preserved raw mV for voltage, both aligned on one shared
beat detection). The concrete missing piece is a **validated multi-lead QRS delineator** (prerequisite for BBB
and a clean ST measurement), plus unifying beat detection into one shared pass; "adding the leads" is already
done. Still future work.

### `src/agent_common.py` — shared building blocks across modality agents
Holds everything that is identical across every modality agent (ECG, MRI, and patient-data - all three
built), so a schema or prompt fix only has to be made once. Design principle: every agent calls the SAME
underlying model (MedGemma 4B via Ollama) — what differs between agents is the prompt and the
modality-specific preprocessing before it, not the model itself.
- `Finding` (dataclass) — `feature`, `value`, `rationale`, `confidence`, `evidence_ref`, and
  `evidence_type` (defaults to `"numeric_feature"`; set to `"image_region"` for vision-grounded findings —
  the default keeps text-only agents like ECG unaffected).
- `AgentContribution` (dataclass) — `agent_id`, `record_id`, `findings` (list of `Finding`),
  `flags_for_review`, `trace`. `agent_id` (e.g. `"ecg"`, `"mri"`) is what distinguishes one agent's output
  from another's, not the class name.
- `FindingsParseError` — raised by `parse_findings_response` on any validation failure.
- `build_generic_findings_prompt(agent_role, modality_label, summary, evidence_keys, patient_info=None, image_labels=None)` —
  the shared structured-findings prompt scaffold (no diagnosis, one JSON shape, evidence_ref validation
  rule). `agent_role` must include its own article (e.g. `"an ECG analysis agent"`) since "a" vs "an" can't
  be inferred generically. `image_labels` (ordered list of image identifiers matching the `images` sent to
  the LLM) adds a section telling the model which attached image is which - `None` by default, so text-only
  agents get no such section. Each agent's own `build_findings_prompt` is a thin wrapper around this.
- `parse_findings_response(raw_text, valid_evidence_keys, image_evidence_keys=None)` — validates the LLM's
  JSON: required fields per finding, `evidence_ref` must be in `valid_evidence_keys`, `confidence` must be in
  `[0.0, 1.0]`. Strips a leading ```` ```json ```` markdown fence if present (MedGemma wraps its JSON in
  one). `evidence_type` on each `Finding` is derived here, not trusted from the LLM: if a finding's own
  `feature` name exactly matches a known numeric key, `evidence_ref`/`evidence_type` are forced to that key
  regardless of what the model cited - needed because MedGemma was empirically found to over-cite an
  attached image as evidence even for findings restating a specific number already given in text (see
  `mri_agent.py` below).
- `build_patient_context_line(age=None, sex=None, height=None, weight=None)` — builds a short human-readable
  demographic line (e.g. `"Patient: 56-year-old female, 63kg."`) from PTB-XL's raw columns (accepts PTB-XL's
  `sex` int encoding directly; uses `numbers.Real` so numpy int64/float64 from a pandas column aren't
  silently dropped) or ACDC's height/weight. Omits missing fields gracefully rather than guessing; says
  `"age/sex not available"` only when BOTH are absent (the ACDC case). Used to prepend context to the ECG/MRI
  agents' own prompts, and rendered in the case-report PDF (see `report_pdf.py`) — deliberately NOT fed into
  the Sep Agent's prompt (it only combines already-interpreted findings, never needs raw demographics).
- LLM backends: `generate_report_medgemma(prompt, model="medgemma:4b", host=..., images=None, temperature=None, seed=None, num_predict=2048)`
  — **default backend**; `images` is an optional list of base64-encoded JPEG/PNG/WebP strings for multimodal
  input. The `host` default reads the **`OLLAMA_HOST` env var** (falling back to `http://localhost:11434`) so
  the app can point at a containerized/remote Ollama — used by the Docker setup (Option A points it at the host
  Ollama via `host.docker.internal`; see `DEPLOY.md`); local usage is unchanged. Always sends `options={"num_ctx": 8192, "num_predict": 2048}` to Ollama — **necessary**: without an
  explicit `num_ctx`, Ollama falls back to a small default (~2-4k tokens) regardless of medgemma:4b's real
  131k capacity, and long CoT+RAG prompts got cut off mid-JSON ("Unterminated string"); this fix dropped
  truncation-caused retries from 9/13 to ~2/9 in a 25-generation sample (see `sep_agent.py`'s CoT section).
  `temperature`/`seed` are optional overrides, default `None` (Ollama's own defaults) — an experiment knob
  (`src/temperature_seed_reliability_test.py` found lowering temperature made the qualitative-characterization
  error rate WORSE, not better, so it was NOT adopted). Note: this num_ctx change affects ALL agents,
  including `report_evaluation.py` — the published n=300 evaluation numbers predate it, so a re-run would use
  different generation settings.
  `generate_report_anthropic(prompt, api_key=None, model="claude-sonnet-5")` — optional, non-default, needs
  `ANTHROPIC_API_KEY`.
  `generate_report_groq(prompt, api_key=None, model="llama-3.3-70b-versatile", temperature=None, seed=None, max_tokens=2048, reasoning_effort=None, host=...)`
  — optional, non-default, **free** cloud backend (Groq's OpenAI-compatible API, `requests` only, no new
  dependency); reads `GROQ_API_KEY` from env. Added to test whether model capability (not prompt) is the
  ceiling on deep criterion integration — the "G2" experiment (see `sep_agent.py`'s `integrate_criteria` and
  `results/rag_ab_test/`). `max_tokens`/`reasoning_effort` were added for reasoning models like
  `openai/gpt-oss-120b`, whose hidden reasoning tokens spend the completion budget and truncated the JSON at
  the default 2048 (`reasoning_effort="low"` + a larger `max_tokens` fixes it); the defaults leave the
  Llama/MedGemma runs byte-identical. Cloud backend: the prompt leaves the machine, so use with
  synthetic/non-sensitive inputs only — the project's default remains local MedGemma.
  `call_ollama_chat_with_tools(messages, tools, model="qwen3:4b", think=True)` —
  generic Ollama `/api/chat` wrapper with tool schemas (MedGemma itself doesn't support Ollama's native tool
  calling - confirmed empirically, HTTP 400 "does not support tools"). Kept as a reusable, modality-agnostic
  building block; not currently called by any agent (the ECG agent's earlier tool-calling orchestration demo
  that used this, `run_ecg_agent_loop`, was removed - see `ecg_agent.py` below for why).
- `save_contribution(contribution, output_dir=None, file_name="contribution")` — serializes an
  `AgentContribution` to JSON for later auditing.

**Known inconsistency, intentionally not yet fixed:** evidence keys aren't namespaced by modality (e.g.
`heart_rate_mean_bpm`, not `ecg.heart_rate_mean_bpm`) even though fusion will eventually need that. Renaming
ECG's existing keys now would be a regression risk for no immediate benefit; new agents should adopt
namespaced keys (`mri.ejection_fraction`, `mri.image.*`) from the start, and reconciling ECG's naming is
deferred to when the Sep Agent's actual fusion logic is designed.

### `src/ecg_agent.py` — the ECG agent (structured findings, not a final report)
Pipeline: ECG pipeline output → numeric feature dict → deterministic decision → structured-findings prompt →
LLM → parsed/validated `AgentContribution` (from `agent_common.py`). Only genuinely ECG-specific logic lives
here; schema/prompt/parsing/LLM-backend code lives in `agent_common.py`.

Built incrementally in phases: **B** (numeric feature extraction) → **C** (deterministic decision
thresholds) → **D** (Qwen3 tool-calling orchestration demo, `run_ecg_agent_loop` - since removed, see below)
→ **E** (learned-embedding integration, current).

**Feature extraction and deterministic decision logic:**
- `compute_feature_dict(results, fs, multilead_signal=None, embedding_model=None)` — single source of truth
  for numeric features (duration, QRS count, heart rate mean/std, RR mean/std, QRS duration mean/std, rhythm
  irregularity flag/CV). Its keys are the only valid `evidence_ref` values an LLM finding may cite. Adding
  `multilead_signal`/`embedding_model` (both optional, default `None`) also adds the embedding-based features
  below - omitted, this behaves exactly as in phases B/C.
- `compute_embedding_features(multilead_signal, embedding_model)` — **Phase E: the trained `ecg_embedding.py`
  model, wired in.** Runs `ECGEmbeddingNet` on the FULL 12-lead signal (normalized identically to training)
  and returns `embedding_<CLASS>_probability` for each of `DIAGNOSTIC_SUPERCLASSES`. This is the only
  genuinely diagnostic-adjacent signal in the ECG agent - everything else is rate/duration statistics. Still
  just another probabilistic `Finding`, not an assertion - preserves the "findings only, no diagnosis"
  contract. Only computed if both `multilead_signal` and `embedding_model` are explicitly passed. Adds
  `torch` as a hard import-time dependency of this module now, not just of `ecg_embedding.py` itself.
- `build_clinical_summary(results, fs, patient_info=None, multilead_signal=None, embedding_model=None)` —
  human-readable text version of the same features, used as prompt context. When embedding features are
  present, spells out each class's full name (`_DIAGNOSTIC_SUPERCLASS_NAMES`, e.g. `"HYP (hypertrophy)"`) -
  **found necessary empirically**: given just the bare abbreviation "HYP", MedGemma glossed it in a finding's
  rationale as "hypokinesis" (a real but WRONG clinical concept - PTB-XL's HYP means hypertrophy). Spelling
  out the name up front eliminated this in a direct before/after re-test on the same record.
- `DecisionPlan` / `detect_flags(feature_dict, tachycardia_bpm=120.0, bradycardia_bpm=60.0, wide_qrs_ms=120.0, embedding_probability_threshold=0.5)`
  — fixed-threshold flags (`tachycardia_suspected`, `bradycardia_suspected`, `wide_qrs_suspected`,
  `irregular_rhythm_suspected`, and `embedding_<CLASS>_pattern_suspected` when a class probability exceeds
  the threshold - only checked for classes actually present in `feature_dict`) computed directly from
  features, so they're raised reliably even if the LLM misses them. Also decides `skip_llm_call` (fewer than
  2 QRS complexes detected → no LLM call is made at all). `generate_ecg_contribution` always merges these
  flags into the final `flags_for_review` alongside whatever the LLM proposes.
  **`tachycardia_bpm` was raised from its original 100.0 to 120.0** after validating against PTB-XL's
  `scp_codes` (`src/flag_validation.py` - see that module's section below for the full sweep and reasoning):
  at 100.0, precision was 0.015 with recall 0.893 (full dataset); 120.0 cuts false positives by ~72% while
  keeping recall at 0.786 - a deliberately recall-favoring choice over the F1-maximizing 165.0 (which drops
  recall to 0.286), since this flag feeds a human-review alert, not a final classification. `bradycardia_bpm`
  was NOT changed: `SBRAD` never occurs with nonzero likelihood anywhere in PTB-XL (confirmed directly), so
  precision/recall are undefined at any threshold for that flag.
- `build_findings_prompt(clinical_summary, feature_dict, patient_info=None)` — thin ECG-specific wrapper
  around `agent_common.build_generic_findings_prompt` (role: `"an ECG analysis agent"`, modality label
  `"ECG"`); new embedding keys flow through automatically since evidence_keys is just `feature_dict.keys()`.

**Orchestration:**
- `generate_ecg_contribution(results, fs, llm_call=None, record_id=None, patient_info=None, model_name=None, max_retries=1, multilead_signal=None, embedding_model=None)`
  — runs `detect_flags` first (short-circuits to an empty-findings `AgentContribution` if skipped),
  otherwise builds the prompt, calls the LLM, parses/retries on failure, and merges deterministic +
  LLM-proposed flags. `llm_call` defaults to `generate_report_medgemma` (free, local, no API key). This is
  the only path used anywhere in the project to produce an ECG `AgentContribution`.
- **Validated end-to-end** on real record `00001_lr` (a known NORM record): embedding model correctly
  predicted 94% NORM probability (all other classes ≤2%), `embedding_NORM_pattern_suspected` flag correctly
  raised, MedGemma correctly cited and grounded all 5 probability findings to their own `evidence_ref` with
  no hallucinated evidence, and the HYP-naming fix above was confirmed via direct before/after comparison.
- **Removed (Phase D)**: this module previously also had `run_ecg_agent_loop`, a Qwen3-orchestrated
  tool-calling demo (Qwen3 decided whether/when to call two read-only getters - `get_baseline_features`,
  `check_deterministic_flags` - over data `generate_ecg_contribution` recomputed independently right after
  anyway). On inspection it added no information to the final contribution: both tools only exposed values
  already computed elsewhere, nothing was derived or enriched, and nothing in the project ever called it
  (confirmed via a full-repo search - zero call sites outside its own definition). Removed rather than kept
  as unused infrastructure - unlike the embedding model (Phase E), which computed real, validated signal even
  before being wired in, this computed nothing new even when run.

### `src/ecg_embedding.py` — learned signal embedding (trained, wired into `ecg_agent.py`)
Complements the hand-crafted features above with a learned representation capturing morphology that R-peak
timing alone can't (P-wave/T-wave shape, ST-segment, etc.). Trained (see the training-run history below) and
**wired into `ecg_agent.py`** via `compute_embedding_features`/`predict_diagnostic_probabilities` - the
patient-disjoint `embedding_stratfold.best.pt` checkpoint is the one to use (see the leakage finding below).
- `DIAGNOSTIC_SUPERCLASSES` (`["CD", "HYP", "MI", "NORM", "STTC"]`) / `load_scp_superclass_map` /
  `scp_codes_to_multihot` — map each PTB-XL record's `scp_codes` to a multi-hot label over PTB-XL's standard
  5-class diagnostic superclass benchmark, via `scp_statements.csv`'s `diagnostic`/`diagnostic_class` columns.
- `PTBXLDataset(database_csv, scp_statements_csv, dataset_dir, resolution="lr")` — PyTorch `Dataset` reading
  PTB-XL records via `wfdb.rdsamp`, per-lead z-score normalized, paired with the multi-hot label. `"lr"` =
  100 Hz (`filename_lr`, faster), `"hr"` = 500 Hz (`filename_hr`). `.strat_fold_indices(train_folds=range(1,9),
  val_fold=9, test_fold=10)` splits record indices using PTB-XL's official patient-stratified `strat_fold`
  column (verified: no patient ever spans more than one fold) — the leakage-free split, used after the
  leakage finding below.
- `ECGEmbeddingNet(n_leads=12, embedding_dim=64, n_classes=5)` — small 1D-CNN ending in global average
  pooling, so it accepts either resolution (1000 or 5000 samples) unchanged. `.embed(x)` returns the
  pre-classifier-head embedding; `.forward(x)` returns classification logits (multi-label). **This is the
  FINAL production model** (checkpoint `embedding_stratfold.best.pt`).
- **Recurrent variants (per the supervisor's temporal-recurrence / variable-length note):** `ECGGRUNet`
  reads the signal as a temporal sequence (GRU, final hidden state = embedding), and `ECGGRU2LayerDropoutNet`
  a deeper dropout variant. Both keep the same `(batch, n_leads, n_samples)` input and 5-class output as the
  CNN. **Decision (see `results/ecg_embedding/rnn_threshold_validation.md` + `cnn_gru_preprocessing_comparison.md`):
  the CNN remains the final model** (best calibrated test F1 macro 0.703, best HYP 0.470); the **simple GRU is
  the retained recurrent baseline** (F1 0.684, does not replace it); the 2-layer/dropout, class-weighted, and
  baseline+notch(+wavelet) preprocessing variants were tried and NOT retained (per-lead z-score only). NB the
  belief "CNN = fixed input" does not apply here: the CNN's global-average-pool accepts variable length just
  like the GRU (both verified running at 1000 and 5000 samples).
- `embed_signal(sig_multilead, model, device=None)` — inference entry point; takes a `(12, n_samples)`
  normalized numpy array, returns a 1D embedding vector. `device=None` auto-detects (CUDA if available); pass
  `"cpu"` to force CPU (the param was previously silently overwritten — now honored).
- `predict_diagnostic_probabilities(sig_multilead, model, device=None)` — same input shape, returns
  per-class sigmoid probabilities (5 floats, `DIAGNOSTIC_SUPERCLASSES` order) via the classification head.
  This is what `ecg_agent.compute_embedding_features` calls. Works for the CNN or a GRU (model-agnostic).
- `train_embedding_model(dataset, model, epochs=5, batch_size=32, ..., val_dataset=None, checkpoint_path=None, epoch_log_path=None, resume_from=None, pos_weight=None, patience=None)`
  — full training loop (Adam, BCE loss), atomic checkpointing (`.latest.pt`/`.best.pt`), resumable, appends
  an epoch CSV log. `val_dataset` (e.g. from `PTBXLDataset.strat_fold_indices`) is used when `dataset` and
  the eventual held-out test set must both come from a patient-disjoint split. `pos_weight` weights the loss
  (class imbalance); **`patience` enables early stopping** (stop after N epochs without val-loss improvement;
  `None` = off, byte-identical to the original behavior). `save_model_checkpoint`/`load_model_checkpoint`
  persist/restore weights.
- **Duration-robustness (variable ECG length 10/5/2 s, per the supervisor's "not tied to PTB-XL's fixed 10 s"
  point):** experiment `src/ecg_duration_augmentation_experiment.py`. Finding: on truncated inputs the CNN
  degrades far less than the GRU out of the box (macro F1 drop 10→2 s: CNN −0.02 vs GRU −0.10 — GAP is
  length-invariant, the GRU's terminal hidden state is length-sensitive). **The real fix is training-time
  random-length-crop augmentation**, which helps BOTH: after it the CNN is perfectly duration-flat (−0.00) and
  the GRU's collapse is largely repaired (−0.01). Full numbers: `results/ecg_embedding/duration_augmentation/`.

**Training (full history + tables: `results/project_history_and_results.md`).** Three runs, each fixing the
last: quick subset (4k records) → full (~19.8k, but **leaked**: split was random over records not patients,
22.3% eval-patient overlap) → **stratfold** (`strat_fold` folds 1-8/9/10, patient-disjoint, 0 overlap — the
one to cite: accuracy 84.9-89.1%, checkpoint `embedding_stratfold.best.pt`). Key standing finding: **HYP lift
stays at the trivial baseline across all runs** despite 5x more data — consistent with the single-lead
limitation (LVH needs voltage criteria across multiple leads), not a data-volume problem. Requires
`pip install torch`; uses CUDA if available (validated RTX 4050, 6.4GB), CPU otherwise.

### `src/ecg_multilead.py` — multi-lead ECG features (B9, additive)
First step toward multi-lead ECG analysis, addressing the single-lead pipeline's documented diagnostic
ceiling (it structurally can't compute voltage-criteria hypertrophy - the standing HYP-stagnation
interpretation). **Additive and standalone** (mirrors the embedding / raw-MRI pattern): does NOT modify
`ecg_preprocessing.py`; consumes the SAME raw 12-lead `multilead_signal` already threaded into `ecg_agent`,
and measures amplitudes on the RAW signal in mV (not the z-scored/denoised pipeline output - a voltage
criterion is only meaningful in real mV).
- `compute_sokolow_lyon_mv(multilead_signal, fs, r_peaks=None)` / `compute_multilead_features(...)` — the
  **Sokolow-Lyon LVH index** = `S(V1) + max(R(V5), R(V6))`, per-beat amplitudes aggregated by median (R-peaks
  detected on lead II if not supplied). Returns `sokolow_lyon_index_mv` + components.
  `compute_multilead_features` also returns descriptive-only features (below): `cardiac_axis_deg` /
  left/right axis deviation, `max_st_elevation_mv` / `max_st_depression_mv`, `rs_ratio_v1`. `PTBXL_LEAD_ORDER`
  verified against a real header (I, II, III, AVR, AVL, AVF, V1..V6).
- **Only Sokolow-Lyon became a review flag; it is the one clear multi-lead win. Validated**
  (`src/ecg_multilead_validation.py`, mirrors `flag_validation.py`, no LLM) against the `LVH` scp_code
  specifically (Sokolow-Lyon is left-ventricular only, not the whole HYP superclass). n=9000, classic **3.5 mV**:
  P 0.676 / R 0.465 / **F1 0.551** - the **best deterministic ECG flag** in the project (next is `wide_qrs`
  0.440) and the only one reaching HYP. Textbook 3.5 mV is near the F1 plateau (no recalibration).
- **Other classes attempted, honest outcomes** (`src/ecg_multilead_validation2.py`, n=2000; full write-up in
  `results/ecg_multilead_validation/ecg_multilead_validation_report.md`): **ST deviation** (STTC) precision
  stuck at the base rate (crude ST, no J-point, T-waves unmeasured); **R/S in V1** (RVH) - RVH too rare in
  PTB-XL (0.25%) to validate; **bundle-branch block** (CD) - a homemade multi-lead QRS width could not reliably
  delineate the QRS, so it was **removed** (the single-lead `wide_qrs_suspected` flag already covers CD at
  F1 0.44). Axis/ST/R-S-V1 are kept as descriptive numeric context only (no flag) - real measurements, not
  reliable classifiers. The harder classes need proper waveform delineation, not amplitude criteria - future work.
- **QRS-delineator deep-dive for BBB (documented experiment, NOT shipped - `src/ecg_bbb_neurokit_experiment.py`,
  full notes `results/ecg_multilead_validation/neurokit2_delineation_notes.md`).** To check whether the BBB
  failure was our *measurement tool* or something deeper, the homemade width heuristic was replaced with
  **NeuroKit2**'s validated wavelet delineator (`ecg_delineate`, `method="dwt"`, on PTB-XL **500 Hz** for fine
  temporal resolution). Progression of best F1 at **natural** BBB prevalence (~3%): homemade ~0.04 → NeuroKit2
  single-lead-II 0.149 → NeuroKit2 **12-lead + robust p75 aggregation 0.190** - the best BBB attempt ever, ~5×
  the heuristic, but **still ≪ the single-lead `wide_qrs` 0.44**, so still not shippable. **Root cause =
  measurement quality, NOT data volume:** `dwt` systematically **over-reads** QRS width (a normal ~90 ms QRS
  reads ~178 ms, making the clinical 120 ms threshold meaningless), because it confuses the QRS-offset/T-wave
  onset boundary - a morphology problem no denoising fixes (P/T waves are wanted signal, not artifact). That
  over-read makes NORM (~196 ms) and BBB (~235 ms) distributions overlap heavily → precision collapses (8 true
  vs 62 false positives at the best threshold), while recall is fine (0.57). Proof it isn't a data problem: LVH
  is *equally rare* (~9.5%) yet reaches F1 0.55 - because voltage-in-mV is a *clean, specific* measurement.
  **Two findings worth keeping:** (1) the clinically "correct" cross-lead aggregation `min(onset)→max(offset)`
  (global QRS extent) is the **WORST** (F1 = base rate) - one lead with a bad onset inflates the whole beat, so
  the global extreme *amplifies* delineation error; robust aggregations (median, p75) that ignore the outlier
  lead recover real separation. (2) NeuroKit2 **downgrades pandas 3.x → 2.3.x** (it pins `pandas<3`), so it's
  kept optional/experimental, nothing imports it in production. A shippable BBB flag would need a
  better-calibrated delineator (or per-beat/per-lead QC) **+ intra-QRS morphology** (rSR' in V1, V6 notching) -
  a genuine signal-processing sub-project, which is the real content behind "validated QRS delineator = future
  work".
- **Wired into `ecg_agent.py`**: `compute_feature_dict` adds all these features when a `multilead_signal` is
  passed (no model needed, unlike the embedding); `detect_flags` raises `lvh_sokolow_lyon_suspected` only;
  `build_clinical_summary` reports the LVH index + axis + ST lines. No-op for single-lead-only callers. No new
  dependency (numpy + reuses `pan_tompkins_rpeaks`).

### `src/ecg_territorial.py` — territorial MI features (additive)
Second multi-lead module (same additive pattern as `ecg_multilead.py`), for the MI superclass, on the
anatomical logic that each cardiac wall is watched by specific leads with reciprocal leads opposite
(`WALLS`: inferior II/III/aVF, anterior V1-V4, lateral I/aVL/V5/V6). `compute_territorial_features` returns
per-wall ST elevation + reciprocal depression (`st_elev_<wall>_mv`, `st_recip_<wall>_mv`) and pathological-Q
scores (`qr_ratio_<wall>` = max Q/R in the wall, `qcount_<wall>` = count of leads with Q/R≥0.25). Reuses
`ecg_multilead`'s per-lead helpers; raw mV.
- **Validated** (`src/ecg_territorial_validation.py`, no LLM, n=4000; report `results/ecg_territorial_validation/`):
  - **Acute ST elevation (#1): NOT viable in PTB-XL** (F1 ~0) - a resting-ECG dataset has almost no acute
    STEMI (3-42 injury cases). The territorial+reciprocal logic is clinically correct but there's no data
    here to validate it. ST features kept descriptive-only.
  - **Chronic pathological Q (#2): real, moderate.** Territorial CONSENSUS (≥k contiguous leads) is the
    specificity lever (a lone septal q is physiological). Inferior wall at the clinical ≥2 leads: F1 0.328,
    **recall 0.915**, precision 0.200 = 2.2× the 9.2% base rate.
- **Wired into `ecg_agent.py`**: `compute_feature_dict` adds all territorial features when a `multilead_signal`
  is passed; `detect_flags` raises **`pathological_q_inferior_suspected`** (≥2 inferior leads, param
  `inferior_q_min_leads`) - a recall-favoring MI alert and the **only deterministic MI signal** in the agent
  (all other MI coverage is the embedding). Anterior/lateral Q + all ST stay descriptive-only (weaker/no
  data). `build_clinical_summary` reports the per-wall Q counts. No-op for single-lead-only callers.

### ECG IMAGE branch (a second, realistic ECG input: a photo/scan of a 12-lead ECG)
A user is far more likely to have an ECG **image** (printout/photo) than a raw WFDB signal. This branch handles
that input, dispatched by `src/ecg_router.py`, and returns the SAME `AgentContribution` shape as the signal
agent so the Sep Agent / UI don't care which branch produced it. **Architectural decision (endorsed by the
supervisor, validated empirically):** the image goes DIRECTLY through a VLM + a learned image classifier -
NOT through digitization back into the signal pipeline (that would re-couple to PTB-XL-signal-specific code).
Its findings are honestly QUALITATIVE (`evidence_type="image_region"`), complemented by a discriminative
classifier for calibrated probabilities. Full write-up + all comparative tests:
`results/ecg_image_pipeline/ecg_image_branch_report.md`. New dataset: **GenECG** (`edcci/GenECG` on HuggingFace)
= PTB-XL signals **rendered as images**, so every image inherits its PTB-XL 5-superclass label (used for
training + validation). Requires `pip install opencv-python pillow huggingface_hub torchvision json_repair`.

- **`src/adaptive_ecg_pipeline.py`** — adaptive image preprocessing before the VLM (OpenCV/PIL only, no heavy
  model): `process_ecg_image(path)` returns a base64 PNG + a quality report (before/after blur/contrast/
  brightness, corrections applied, `source_quality_level`). Corrections are PROPORTIONAL to the detected
  severity (unsharp mask, CLAHE, gamma, deskew, resize). Documented limitations (no perspective warp) at the
  bottom of the file - the former "no non-ECG detection" limitation is now addressed one level up by the
  opt-in CLIP gate (`ecg_image_gate.py`, below).
- **`src/ecg_image_gate.py`** — opt-in **"is this actually an ECG?" content gate** (CLIP zero-shot,
  `clip-ViT-B-32` via `sentence-transformers`, lazily imported + cached module-level; no new dependency - RAG
  already pulls `sentence-transformers`). `is_ecg_image(path, threshold=0.5)` encodes the image against a
  small set of positive ("a 12-lead ECG printout on grid paper") vs negative (person/landscape/text/object +
  two HARD negatives: "a chart/diagram that is not an ECG" and "an MRI/X-ray") captions, softmaxes, and
  returns `P(ECG)` = the positive mass. **Why CLIP, not a trained classifier:** "not an ECG" is open-set, so
  a supervised negative set is impractical; CLIP answers zero-shot. The prompt set was tuned on real images
  (two GenECG ECGs, an ACDC MRI slice, a block-diagram figure): **ECGs ~0.89, non-ECGs <0.1** - a clean,
  well-separated margin around 0.5 (the two hard negatives are what stop a matplotlib figure / an MRI slice
  from drifting to the ECG side). Wired into `ecg_image_agent.generate_image_ecg_contribution(ecg_gate=True,
  ecg_gate_threshold=0.5)`: runs FIRST on the RAW upload, and a non-ECG returns an empty-findings
  contribution with a **`not_an_ecg_image`** flag + `P(ECG)` in the trace, BEFORE any preprocessing / VLM /
  classifier work. **Fail-open** (opt-in, default off): if the gate model can't load (offline / dep missing)
  the check is skipped and the pipeline proceeds rather than blocking a genuine ECG - it's a convenience
  guard, not a hard boundary. Verified end-to-end: MRI/diagram blocked with NO LLM call, real ECG passes,
  gate-load failure proceeds. **Default policy (deliberate split):** the API defaults stay opt-in
  (`ecg_gate=False` in `generate_image_ecg_contribution` / `route_ecg_upload`) so programmatic + validation
  callers are byte-identical and unslowed; the **UI** ECG page "Upload image" toggle defaults it **ON** (a
  human may upload anything), uncheckable to force analysis. Report figures:
  `results/report/figure_image_ecg_gate.png` (utility - ECGs ~0.88 accepted vs non-ECGs <0.1 rejected) and
  `figure_image_gate_flow.png` (the opt-in/fail-open/accept/reject decision flow).
- **`src/ecg_image_agent.py`** — the ECG IMAGE agent (mirrors `ecg_agent.py`, reuses `agent_common`). Its only
  numeric "evidence" is the image-quality metrics; all clinical content comes from the VLM reading the image
  (per-lead `evidence_ref` keys `ecg_image_I..V6` + `ecg_image_overall`). `_repair_raw_response` (via
  `json_repair`) fixes the small vision models' malformed JSON. **LLM backend:** local Ollama, default the
  ECG-specialized fine-tune `hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M` with a `medgemma:4b`
  fallback (see `setup_medgemma_ecginstruct_ollama.sh`). **"Always respond" contract:** an unreadable image
  returns an empty-findings contribution, never raises. **Learned-classifier layer (opt-in, mirrors
  `ecg_agent`'s embedding wiring):** `compute_pathology_features(image_path, classifier_model)` →
  `pathology_<CLASS>_probability` (TRUE probabilities from the GenECG classifier, lazy-imported so the module
  stays torch-free otherwise), `detect_pathology_flags` → `pathology_<CLASS>_pattern_suspected`;
  `generate_image_ecg_contribution(..., classifier_model=None)` threads it through (probs in the summary +
  prompt + trace, flags merged; graceful degradation if it fails).
- **`src/ecg_router.py`** — single entry point `route_ecg_upload(file_path, ..., embedding_model=None, classifier_model=None)`:
  detects image vs WFDB vs CSV from the extension, runs the matching branch, and ALWAYS returns an
  `AgentContribution` (every failure caught → explanatory flag, never an unhandled exception). `embedding_model`
  flows to the signal branch, `classifier_model` to the image branch (symmetric); `ecg_gate`/
  `ecg_gate_threshold` also flow to the image branch (the CLIP content gate above).
- **GenECG image classifier (real probabilities + validation — the answer to "no values/thresholds").**
  `src/genecg_classifier_download.py` builds a class-stratified manifest of GenECG images (folds 1-9 train /
  fold 10 test) + PTB-XL labels (reuses `ecg_embedding.py`'s label logic); resumable (cached images skipped).
  `src/genecg_classifier_train.py` trains a ResNet18 (ImageNet-pretrained, torchvision) → 5 sigmoid
  probabilities, **with early stopping on a carved val split** (`MAX_EPOCHS`/`PATIENCE`, best-val weights
  restored — essential: it overfits by ~epoch 3); also holds the inference entry points
  `load_genecg_classifier` / `predict_pathology_probabilities` used by `ecg_image_agent`. **Validated on
  held-out fold 10: macro F1 0.643, MI F1 0.619** (`results/ecg_image_pipeline/genecg_classifier_validation.md`)
  - vs the 100% VLM zero-shot baseline **0.314 / MI 0.000** (`vlm_zeroshot_validation.md`): the classifier
  ~doubles reliability and closes the VLM's dangerous MI blind spot, approaching the signal branch's ~0.67.
- **Model comparison (why `qwen2.5vl:7b` is the recommended local VLM):** the ECG fine-tune produces the WORST
  JSON (trained for free-text description, not the schema) and hallucinates; base `medgemma:4b` is more
  reliable but shallow; `qwen2.5vl:7b` gives valid JSON + no hallucination; a frontier VLM reads best but is
  paid. Model choice fixes FORM (JSON), never the SUBSTANCE (no measured values) - only the classifier
  (probabilities) and the digitization (measured values, below) do.
- **Digitization (measured numbers from the image — `src/ecg_digitization.py`, wired in).** Recovers a signal
  from the image so the branch produces its OWN measured values (heart rate, rhythm), the answer to "a VLM
  never gives calibrated numbers". `digitize_rhythm_strip` digitizes the bottom **rhythm strip (lead II)**:
  ① grid-scale detection (autocorrelation of a reddish-grid column profile → px/mm), ② trace extraction
  (darkest ink pixel per column, grid excluded; **Fix A** `_clean_trace` despikes via a local-median-envelope
  reject + median filter), ③ calibration via the universal paper standard (25 mm/s, 10 mm/mV), ④ resample.
  `measure_rhythm_from_image` then counts R-peaks with a **combiner** (Pan-Tompkins first for tightest error,
  falling back to `robust_r_peaks` = **Fix B**, a MAD-threshold detector, for highest success) → HR + rhythm
  CV. One lead suffices because beat TIMING is shared across leads; the 5 superclasses are NOT recoverable
  this way (morphology/voltage need multiple leads). **Full 12-lead digitization → signal → embedding is
  explicitly NOT pursued** (deliberately dropped, not pending work): the learned **GenECG image classifier**
  already recovers the 5 superclasses straight from the pixels (macro F1 0.643, see below) with no signal
  reconstruction, so digitization's only remaining role is the *measured* values (HR/rhythm in real units),
  which the single-lead rhythm-strip path above already covers. Reconstructing all 12 traces to re-derive
  superclasses the classifier already gives would be redundant engineering risk for no gain. **Validated on
  GenECG** (`src/ecg_digitization_validation.py`, vs the true PTB-XL lead II):
  heart-rate **MAE ≈ 1 bpm** (median ~0.5-1.0), waveform correlation 0.77 after the two fixes (the fixes cut
  the pre-fix MAE ~10x; the failures were R-peak DETECTION on spurious spikes, not calibration - diagnosed by
  plotting detected peaks on the recovered trace). **Wired into `ecg_image_agent`** (opt-in `digitize=True`,
  mirrors the classifier): `compute_digitization_features` (lazy import) + `detect_digitization_flags`
  (`tachycardia_suspected`/`bradycardia_suspected`/`irregular_rhythm_suspected` - SAME names/thresholds as
  the signal branch, 120/60 bpm, CV 0.15); `digitized_*` keys flow into the summary/prompt/trace/flags,
  no-op when off; `ecg_router` passes `digitize` through. NOT a re-route into the signal AGENT (rhythm strip =
  1 lead only, and re-coupling was deliberately avoided) - it's the image branch computing its own rate/rhythm.
- **Sep Agent / KG integration:** an `ecg_image` contribution already participated in the Sep Agent core
  (summary/flag-union/traceability). Its `pathology_<CLASS>_pattern_suspected` flags are now ALSO nodes in
  `clinical_kg.json` (→ `cat_<CLASS>`, agent `ecg_image`) and in `sep_agent._FLAG_QUERY_PHRASES`, so the image
  branch participates in **cross-modality convergence** (e.g. image-MI + patient CAD flags → coronary artery
  disease) exactly like the signal branch.
- **Fixtures / tests:** `src/genecg_fixtures.py` (fetch a single GenECG image by ecg_id/variant, the single
  source of truth after fusing two older download scripts); `src/verify_image_pipeline.py` (3-layer smoke test),
  `src/sanity_check_image_ingestion.py` (does the VLM actually SEE the image?), `src/dry_run_test.py` (offline
  prompt+parser unit test), `src/debug_llm_raw_response.py` (single raw VLM call, locates the JSON failure).
  Notebook `src/ecg_image.ipynb` builds the GenECG↔PTB-XL mapping CSV.

### `src/mri_preprocessing.py` — MRI (ACDC) preprocessing
Mirrors `ecg_preprocessing.py`'s role for the MRI modality: turns a raw ACDC patient record into (a) a
representative, ROI-cropped 2D image per cardiac phase for MedGemma's vision encoder, and (b) mask-derived
numeric features.
- `load_patient_info(info_cfg_path)` — parses ACDC's `Info.cfg` (`ED`/`ES` frame numbers, `Group` pathology
  label, height/weight).
- `load_nifti_volume(path)` — loads a `.nii.gz` file via `nibabel`, returns `(volume, voxel_spacing_mm)`.
- `select_representative_slice(mask_volume)` — picks the slice with the most total foreground mask area
  per phase, rather than a fixed index — confirmed on real data that outer slices can have zero heart tissue.
- `compute_bounding_box(mask_2d, margin_fraction=0.15)` / `crop_to_bounding_box` / `union_bounding_boxes` —
  ROI cropping around the heart, display-only. Margin is proportional to the box's own size (heart size
  varies a lot by pathology, e.g. dilated DCM vs smaller HCM). ED and ES are cropped to the *union* of their
  two boxes, so both images share the same scale/framing for side-by-side comparison.
- `normalize_slice_to_uint8` / `encode_slice_to_base64_png` — MRI intensity → displayable image → base64,
  for `agent_common.generate_report_medgemma`'s `images` parameter. `save_slice_png` writes a PNG for visual
  sanity-checking.
- `compute_chamber_volumes_ml` / `compute_ejection_fraction_pct` — mask-derived numeric features, computed
  from the *full*, uncropped 3D mask. Validated against known pathology labels across **all 150 ACDC
  patients**: mean LV EF by group - DCM ~19.8%, MINF ~30.8%, RV ~56.5%, NOR ~62.4%, HCM ~69.8% - clinically
  coherent. Explicitly casts `voxel_spacing_mm` components to native `float` before multiplying (nibabel
  returns `numpy.float32`, which under NumPy's promotion rules stays `float32` when multiplied by a Python
  float, breaking JSON serialization until fixed).
- `preprocess_patient(patient_dir)` — orchestrates all of the above for one ACDC patient directory, returning
  a dict with `images`, `slice_indices`, `roi_bbox`, and `features` - mirrors
  `ecg_preprocessing_pipeline`'s output-dict pattern.
- Requires `pip install nibabel Pillow`.

### `src/mri_agent.py` — the MRI agent (structured findings, image-grounded)
Mirrors `ecg_agent.py`'s structure, consuming `preprocess_patient()`'s output. Uses the same shared MedGemma
backbone as ECG - MedGemma is natively multimodal, so only the prompt and `images=[...]` change.
- `build_mri_clinical_summary(preprocessing_result, patient_info=None)` — text summary of ED/ES volumes,
  ejection fractions.
- `MRIDecisionPlan` / `detect_flags_mri(feature_dict, low_lv_ef_pct=45.0, low_rv_ef_pct=45.0, high_lv_ef_pct=65.0)` —
  deterministic `low_lv_ejection_fraction_suspected` / `low_rv_ejection_fraction_suspected` /
  `high_lv_ejection_fraction_suspected` flags (the last one added after `mri_flag_validation.py`'s threshold
  sweep - see below); skips the LLM call if LV ejection fraction isn't computable.
- `build_mri_findings_prompt(summary, feature_dict, image_keys, patient_info=None)` — thin wrapper around
  `agent_common.build_generic_findings_prompt` (role `"an MRI analysis agent"`, evidence keys = numeric
  feature keys + image identifiers, `image_labels` set so the prompt tells the model which image is which).
- `generate_mri_contribution(preprocessing_result, ..., image_mode="ed_only")` — same shape as
  `generate_ecg_contribution`. `image_mode` is `"ed_only"` (default) or `"ed_es"` (both phases in one call).
- **Known LLM behavior, found and corrected**: MedGemma, when an image is attached, cited the image as
  `evidence_ref` for *every* finding - even ones restating a numeric feature already in the text summary -
  even after an explicit prompt rule against it. Fixed in `agent_common.parse_findings_response` instead: if
  a finding's `feature` name exactly matches a known numeric key, `evidence_ref` is forced to that key
  regardless of what the model cited. Validated on 2 patients (DCM, HCM), both `image_mode`s: valid JSON
  first attempt every time, correct flags, 100% correct evidence_ref attribution after the fix.
  `image_mode="ed_only"` kept as default despite `"ed_es"` also working (MedGemma's own evaluation is
  documented as primarily single-image, and 2 samples don't yet justify overriding that caution).
- **Deterministic flag validation done** (`src/mri_flag_validation.py`, mirrors `flag_validation.py`):
  validates `detect_flags_mri` against ACDC pathology group labels, all 150 patients, no LLM (reuses
  `results/mri_features_all_patients.csv`). Table + reasoning in
  `results/mri_flag_validation/mri_flag_validation_report.md` and `results/project_history_and_results.md`.
  **Much better calibrated than the ECG flags**: `low_lv_ejection_fraction_suspected` (45%) is nearly perfect
  (P 0.967 / F1 0.975 - cleanly separates DCM/MINF from the rest); `low_rv` (45%) moderate (F1 0.562, RV is
  naturally more variable). No recalibration needed for the original 2. **New flag
  `high_lv_ejection_fraction_suspected`** (65%, HCM-associated) added from a 62-75% sweep (F1 0.698), as an
  `elif` after the low-EF check (mutually exclusive); validated end-to-end on `patient021` (HCM, LV EF 67.6%).
- **Raw-MRI pipeline extension (YOLO + U-Net + learned pathology classifier), built and wired in.** Replaces
  ACDC's expert mask (a real patient's MRI has none). Full design/training/validation in
  `results/report_package/05_future_work_raw_mri_pipeline/raw_mri_pipeline_proposal.md` (summary in the archive
  doc); YOLO26n (mAP50 0.976) + U-Net (val Dice 0.906) + a 5-class pathology classifier (RV F1 0.80, its
  strongest, vs the weak `low_rv` flag). Wired via `compute_pathology_classifier_features(raw_ed_slice_uint8,
  yolo_model, unet_model, classifier_model, device="cpu")` (lazily imported so `mri_agent.py` stays importable
  without the heavy ML stack) + optional params on `build_mri_clinical_summary`/`detect_flags_mri`/
  `generate_mri_contribution` mirroring ECG's embedding-wiring conventions (spelled-out class names,
  `pathology_<CLASS>_pattern_suspected` flags incl. NOR, opt-in/no-op when omitted). Validated end-to-end on
  `patient021` (HCM): both `high_lv_ejection_fraction_suspected` and `pathology_HCM_pattern_suspected` (99.0%)
  raised, all 5 pathology findings self-cited, no hallucinated evidence.

### `src/patient_data_agent.py` — the structured patient-data agent (3rd modality)
Same contract as `ecg_agent.py`/`mri_agent.py` (structured findings only, no diagnosis; consumed by the Sep
Agent), on tabular clinical data instead of a signal or image. Full write-up: `results/patient_data_agent.md`.
- **Data source**: UCI Heart Disease, Cleveland subset, 303 patients (UCI repo id=45). Fetched once via
  `ucimlrepo`, saved locally to `uci_heart_disease/processed_cleveland.csv` (+ `variables_metadata.csv`) - the
  agent reads the local file, never a live network fetch (`ucimlrepo` doesn't cache).
- `compute_feature_dict` — 13-field feature dict (`age_years`, `sex`, `chest_pain_type`, `resting_bp_mmHg`,
  `cholesterol_mgdl`, `max_heart_rate_achieved_bpm`, `st_depression_oldpeak_mm`, thallium/vessel/slope fields,
  etc.); keys are un-namespaced (confirmed neither ECG nor MRI actually namespaces either - citation is always
  `(agent_id, feature)`, so a bare-name clash can't happen). Missing `ca`/`thal` (4/2 patients) → `None`,
  omitted gracefully, never guessed.
- `detect_flags_patient` — 7 deterministic flags validated against `heart_disease_present = (num > 0)` (see
  `patient_data_flag_validation.py` below). Two thresholds recalibrated by sweep (same discipline as ECG's
  tachycardia 100→120): `oldpeak` 1.0→0.8 mm, and max-HR `220-age@85%` → **Tanaka `208-0.7×age` @91%** (the
  F1-max, F1 0.669; the flag fires when `max_hr < pct×predicted`, so a higher pct fires more positives and 91%
  also beats 90% on recall — 0.705 vs 0.683 — no recall/F1 trade-off here, it dominates 90% on both; 92% adds
  marginal recall 0.712 but F1 drops to 0.651). `high_resting_bp`/`high_cholesterol` kept at clinically meaningful cutoffs
  (140 mmHg / 240 mg/dl) despite weak sweeps - both are simply weak CAD predictors in this dataset at any
  threshold, reported honestly rather than forced.
- `generate_patient_contribution` (`agent_id="patient"`) — same orchestration shape as the other agents.
- **Validated end-to-end**: real MedGemma call, patient `cleveland_0` - 13/13 findings, all `evidence_ref`
  valid, first-attempt parse, 3 flags correctly raised/merged. No changes made to `ecg_agent.py`,
  `mri_agent.py`, or `sep_agent.py`.

### `src/sep_agent.py` — the Sep Agent (cross-modality aggregation)
Combines multiple modality agents' `AgentContribution` objects into one integrated clinical impression with
full traceability. **Unlike** the modality agents, the Sep Agent IS allowed to synthesize a cross-modality
conclusion - that is its entire purpose; it never re-derives facts from raw signals/images, only combines
what's already reported. Uses the same shared MedGemma backbone, called text-only.
- `TraceabilityEntry` / `SepAgentDecision` (dataclasses) — `SepAgentDecision` holds `record_id`,
  `contributing_agents`, `summary` (LLM-synthesized 3-5 sentence impression), `flags_for_review`
  (deterministic union of every input contribution's flags - never LLM-generated), `traceability` (list of
  `TraceabilityEntry` linking each summary statement to a specific `(agent_id, feature)`), and `trace`.
- `build_sep_agent_prompt(contributions)` — lists each agent's findings grouped by agent, plus the exact
  valid `(agent_id, feature)` pairs the model may cite in traceability.
- `parse_sep_agent_response(raw_text, valid_pairs)` — validates the JSON and that every traceability entry
  cites a real `(agent_id, feature)` pair actually present in the inputs - rejects/retries on fabrication.
- `generate_sep_agent_decision(contributions, llm_call=None, record_id=None, model_name=None, max_retries=1, use_rag=False, top_k_knowledge=5, kb_dir="results/rag_knowledge_base", use_cot=False, integrate_criteria=False, use_kg=False)`
  — orchestrates the above; `contributions` is a list of `AgentContribution` (one or more modalities). The
  last five params are all opt-in (default off) so existing callers - notably `report_evaluation.py`'s ECG-only
  scoring - keep byte-identical behavior; enabling any by default would have silently changed the published
  n=300 numbers.
- **Opt-in RAG layer (`use_rag`)** — when True, `build_retrieval_query` turns this case's flags into a clinical
  query (via `_FLAG_QUERY_PHRASES`), `rag_knowledge_base.retrieve` fetches the top-k chunks, and they are
  injected as findings from a virtual `"knowledge"` agent (`feature="knowledge_<i>"`). Crucially this reuses
  the EXACT existing citation-validation path: a fabricated knowledge citation is rejected by the same
  `parse_sep_agent_response` logic that rejects a fabricated `(agent_id, feature)` pair - no new anti-
  hallucination code. Retrieved chunks are stored in `trace["retrieved_knowledge"]` for audit. See
  `src/rag_knowledge_base.py` below and `results/rag_ab_test/rag_utility_ab_synthesis.md` (RAG is genuinely
  useful only with a clean, curated, well-extracted KB).
- **Opt-in Chain-of-Thought (`use_cot`)** — adds a `reasoning` field the model must fill BEFORE `summary`
  (independent per-modality read → agreement/conflict → what retrieved knowledge confirms). Stored on
  `SepAgentDecision.reasoning`, deliberately kept OUT of the user-facing `summary`; `require_reasoning` in the
  parser rejects/retries a response missing it. Most valuable here specifically - the one place that actually
  combines cross-source evidence.
- **Deterministic directional/citation-consistency checks** (always on, not gated behind the opt-in flags) —
  `_check_directional_consistency` scans summary+reasoning clauses for a known direction-implying flag (the two
  EF flags, tachycardia, bradycardia - facts sourced from the clinical KG's `directional` blocks, see
  `knowledge_graph.py`; migrated from the former separate `directional_flags_config.json`) stated in the wrong
  direction, and `_check_ef_citation_coverage` checks an EF claim actually cites a backing entry; either
  failure triggers the same retry loop as a bad citation. This targets the documented "traceability guarantees
  the finding is real, not that its characterization is correct" gap (the `295.5 mL called low` bug): the
  correct direction already sits in `flags_for_review`, so a clause-level keyword heuristic (not NLI, precision
  over recall) catches the model contradicting it. Full history incl. two fixed false positives and one
  deliberately-unfixed negation-scope gap: module docstring + `results/cot_rag_narrative_accuracy_limitation.md`.
- **`integrate_criteria`** — opt-in experiment knob (default False): appends a directive telling the model to
  state a value together with the threshold/classification it crosses. TRIED AND NOT ADOPTED as a default on
  MedGemma 4B (dropped grounding 7→0 citations without achieving prose weaving) - but on stronger models
  (Groq Llama-70B / gpt-oss-120b) it does weave; this is the "G2" capability-ceiling experiment, full write-up
  in `results/rag_ab_test/rag_utility_ab_synthesis.md`.
- **Validated with real data**: a synthetic tachycardic ECG combined with a real DCM MRI patient - valid JSON
  first attempt, correct union of flags, correct synthesis, all 5 traceability entries pointed to real
  pairs. Fabricated-reference rejection and single-modality-only input both tested and working. Note: PTB-XL
  and ACDC share no real patients, so any ECG+MRI test case is a synthetic pairing for architecture
  validation, not a real joint clinical case.
- **Known narrative-accuracy limitation** (see `sep_agent_demo.ipynb` below): traceability guarantees a
  cited finding is real, NOT that the model's characterization of it is clinically correct - these are
  different guarantees, and only the former is currently enforced in code.

### `src/rag_knowledge_base.py` — RAG knowledge base (the Sep Agent's opt-in knowledge layer)
Builds and queries a small local vector index over curated cardiology PDFs; its single consumer is the Sep
Agent (`use_rag=True`). No new heavy service - just `sentence-transformers` (`all-MiniLM-L6-v2`) + a saved
`chunks.json`/`embeddings.npy` pair. **Current production index: 3 curated clean-guideline sources / 83
chunks** (`rag_sources/cardiac_ai_internship.pdf`, `ESC_2021_Heart_Failure_RAG_Source.pdf`,
`AHA_ACCF_HRS_2009_ECG_Chamber_Hypertrophy_RAG_Source.pdf`), saved under `results/rag_knowledge_base/`.
- `extract_pages(pdf_path)` — per-page `{page_num, text, tables}` via pdfplumber (tables extracted separately).
- `_split_into_sections` / `chunk_document(pdf_path, ...)` — header-aware chunking (splits on detected section
  headers, `_recursive_split` fallback for over-long sections; tables formatted via `_format_table`).
- `_is_low_value_chunk(chunk)` → `(is_low_value, reason)` — **ingestion quality filter** added this session:
  drops bibliography/citation lists, email/footer fragments, garbled (low alpha-ratio) text, and short
  non-clinical fragments (`<150` chars) - but **tables are exempt**, and a short chunk carrying a clinical
  signal (`_SIGNAL_RE`, e.g. an EF cutoff) is rescued. Dropped chunks are written to `dropped_chunks.json` for
  audit. (Note: an `_AUTHOR_RE` hardening was tried and REMOVED - it matched medical abbreviations LVEF/HFrEF/
  NYHA and Roman-numeral "Class I/II/III", wrongly dropping real NYHA-class definitions.)
- `build_knowledge_base(pdf_paths, output_dir=..., embedding_model_name=...)` — chunk+filter+embed one or more
  PDFs into a single index; returns `n_chunks`/`n_dropped`/`drop_summary`. `__main__` globs `rag_sources/*.pdf`
  (non-recursive - subfolders like `rag_sources/dataset_docs/`, `epidemiology/`, `guidelines_raw/` are
  deliberately OUT of the production glob after curation).
- `retrieve(query, top_k=3, kb_dir=..., ...)` — top-k cosine-similarity chunks with scores.
- `_get_model(embedding_model_name)` — module-level `_MODEL_CACHE` so the SentenceTransformer loads once, not
  per call (fixed a 66.77s → 0.03s reload cost when build + retrieve ran back-to-back).
- **Utility is content-dependent, documented in `results/rag_ab_test/rag_utility_ab_synthesis.md`**: with a
  dirty/jargon KB the Sep Agent ignored or only cited boilerplate; with clean, well-extracted guidelines it
  cites real criteria (HFrEF ≤40%, HFmrEF 41-49%, HFpEF ≥50%, NYHA III/IV). A complex 2-column PDF (raw ESC
  guideline) extracted poorly (~26% merged words) and was excluded - extraction quality, not RAG itself, was
  the blocker. Requires `pip install sentence-transformers pdfplumber`.

### `src/knowledge_graph.py` — clinical knowledge graph (the Sep Agent's opt-in structured-reasoning layer)
Second opt-in Sep Agent layer, complementing (not replacing) the RAG text layer: a `networkx` directed graph
over a human-editable JSON source (`src/clinical_kg.json`), giving structured reasoning RAG cannot. Finding
node ids = the agents' real flag names, so the graph connects directly to `AgentContribution` output; node
types `finding→criterion→diagnosis→severity`, relations `meets/indicates/supports`. **Not Neo4j/RDF** (external
service, overkill for a local prototype - noted as the production-scale evolution only).
- `load_kg` / `diagnoses_for_findings` / `convergent_diagnoses` / `reasoning_path`; plus the taxonomy queries
  `category_of` / `categories_for_findings` / `convergent_categories`. Lazy-imported by `sep_agent` only when
  `use_kg=True`, so `sep_agent` stays importable without `networkx` otherwise.
- **Pathology hierarchy (taxonomy).** `category` nodes for the PTB-XL diagnostic superclasses
  (CD/HYP/MI/STTC/NORM) + a clinical `heart failure` category, linked by an `is_a` relation
  (`old_inferior_mi is_a MI`, `lv_hypertrophy/hcm is_a HYP`, `hfref is_a heart failure`). The learned-embedding
  superclass flags (`embedding_<CLASS>_pattern_suspected`), the **ECG-image classifier flags
  (`pathology_<CLASS>_pattern_suspected`, agent `ecg_image`)**, and `pathology_DCM_pattern_suspected` point
  straight at their category node, so the KG rolls specific findings up to their superclass AND bridges the
  deterministic vs. embedding vs. image ECG pathways at the category level (e.g. the **MI** superclass is
  reached by the deterministic `pathological_q_inferior`, the learned `embedding_MI`, AND the image-branch
  `pathology_MI` flag - so an image-only case converges with the patient agent's CAD flags on coronary artery
  disease, just like a signal case). `convergent_categories` gives
  cross-modality convergence at the (more abstract) superclass level; both diagnosis- and category-level
  convergences are stored deterministically in `decision.trace` (`kg_convergent_diagnoses` /
  `kg_convergent_categories`) when `use_kg=True`. Note: `clinical_kg.json` is hand-authored (not extracted from
  the RAG PDFs) - a small, high-precision curated graph, the opposite of RAG's large auto-ingested text; the
  superclass taxonomy mirrors PTB-XL's own `scp_statements.csv` `diagnostic_class` hierarchy.
- **Consumer #3 (SHIPPED, the KG's real value): cross-modality convergence.** `sep_agent`'s new opt-in
  `use_kg` injects a "convergent evidence" block listing diagnoses supported by findings from ≥2 DIFFERENT
  modality agents (e.g. `lv_hypertrophy` ← ECG-Sokolow + MRI-HCM; `hypertensive_heart_disease` ← ECG+MRI+
  patient; `coronary_artery_disease` ← ECG-MI + the patient agent's 4 CAD/ischaemia flags, via the MI
  hierarchy) - something concatenating findings cannot make explicit. **A/B (KG off vs on) is robustly positive
  on the local MedGemma 4B** - across 5 seeds it names the convergent diagnosis 5/5 (clean case) / 4/5 (noisier
  case) with the KG vs 0/5 without (9/10 vs 0/10) - and gpt-oss-120b uses it even more richly ("convergent
  evidence across ECG, MRI, and BP" + clinical nuance). **Stronger than G2** (where the 4B could not use
  injected text criteria at all): the hard reasoning is done deterministically by the graph, so the small
  model only narrates a pre-computed fact. The convergent diagnoses are also stored **deterministically** in
  `decision.trace["kg_convergent_diagnoses"]` (LLM-independent, auditable, same spirit as `flags_for_review`).
- **Consumer #1 (SHIPPED as a source migration, not a logic change): the KG is the single source of the
  directional facts.** Each directional finding node carries a `directional` block (`ventricle_key`,
  `measure_terms`, `forbidden_words`, `anchor_description`); `sep_agent._load_directional_flags_config` now
  reads these (plain json, no networkx) instead of a separate `directional_flags_config.json`, which was
  **deleted**. The battle-tested check/anchor LOGIC (`_check_directional_consistency` with its ventricle-
  referent resolution) is UNCHANGED - only the data source moved, verified **config-byte-identical** before
  removal (zero regression; bradycardia coverage confirmed). A standalone KG-native check was tried first and
  found detection-equivalent to the keyword heuristic (no gain), so this migration keeps the proven logic and
  just unifies the source. Net: all clinical knowledge (relations + directions) now lives in one graph.
- Does NOT address G2 (prose integration, model-bound, already closed) - a different axis (structured
  reasoning / verification). Full design + A/B: `results/kg_integration/kg_design_and_results.md`. Requires
  `pip install networkx` (graph traversal only; the directional-facts read is plain json).

### `src/report_pdf.py` — PDF case report generation
Presentation-layer only (renders data already produced by the ECG/MRI/Sep agents into a single
human-readable PDF for a human reviewer) - no new analysis happens here.
- `_table_style()` — shared visual style (dark header row, alternating row backgrounds, grid lines) for the
  findings/traceability tables.
- `_findings_table(findings, styles)` / `_traceability_table(traceability, styles)` — render a list of
  `Finding`/`TraceabilityEntry` objects as a reportlab `Table`, with long snake_case identifiers (feature
  names, evidence types) wrapped via a custom `"Identifier"` paragraph style
  (`wordWrap="CJK"`) so they break inside narrow table columns instead of overflowing - a zero-width-space
  approach was tried first and rejected (rendered as visible tofu-box glyphs in the default font).
  `repeatRows=1` keeps the header row visible if a table spans multiple pages.
- `generate_case_report_pdf(output_path, sep_decision, contributions, figure_paths=None, synthetic_pairing_note=None, patient_context_lines=None)`
  — assembles the full report: title, an optional red disclaimer paragraph (`synthetic_pairing_note`, e.g.
  flagging a synthetic ECG+MRI pairing), the Sep Agent's integrated summary + flags, a **cross-modality
  convergent-evidence section** (rendered only when `sep_decision.trace["kg_convergent_diagnoses"]` is
  present, i.e. the Sep Agent ran with `use_kg=True` - the deterministic KG convergence, noted as computed by
  the graph not the LLM), the traceability table (with a fixed note that traceability only guarantees the
  finding is real, not that its characterization is correct), then one detailed findings section per
  contributing agent (each on its own page), then any supplied figures embedded at the end
  (`kind="proportional"` preserves aspect ratio).
  `patient_context_lines` (optional) maps `agent_id` → a human-readable demographic line (from
  `agent_common.build_patient_context_line`) rendered under that agent's own findings section — rendered ONLY
  in the PDF, deliberately NOT fed into any LLM prompt (the Sep Agent combines already-interpreted findings and
  never needs raw demographics).
- Used by `notebooks/sep_agent_demo.ipynb` to produce `results/sep_agent_demo/case_report.pdf`.
- Requires `pip install reportlab`.

### `src/report_evaluation.py` — ECG report evaluation (BLEU/ROUGE/BERTScore)
Standalone module (does not modify `ecg_agent.py`, `agent_common.py`, `sep_agent.py`, or `ecg_embedding.py`):
scores the ECG agent's generated narrative against real PTB-XL cardiologist reference reports. Scope is
ECG-only by construction: `generate_text_for_scoring`/`generate_brief_ecg_summary` call
`sep_agent.generate_sep_agent_decision` with ONLY the ECG contribution (no MRI), so the resulting summary
cannot reference MRI content, and `ecg_agent.py` itself never produces free text (only `Finding` objects) -
scoring raw JSON would be invalid, hence going through the Sep Agent for text at all.

**Reference pool construction** (`build_reference_subset(database_csv, rtlp_repo_dir, validated_only=True)`,
`rtlp_repo_dir` mandatory, cloned at `RTLP/`): built from **RTLP's English translations, not language
detection** - `langdetect` had a 13.3% false-positive rate on this dataset's short text, so it's used only as
a sanity cross-check now (`cross_check_native_vs_rtlp`), never for pool construction.
`is_administrative_placeholder_report` drops pure admin notes ("trace only requested."). **Final pool: 12040
records** (vs the old lang-detect 7702). Caveat: most references are now machine-translated from German
(added scoring noise). Detail in `results/project_history_and_results.md`.

**Scoring**: `compute_bleu` (sacrebleu, sentence BLEU) / `compute_rouge` (rouge-score, ROUGE-1/2/L F1,
stemmer on) / `compute_bertscore_batch` (bert-score, batched once per run for speed;
`distilbert-base-uncased` backbone rather than the package's heavier `roberta-large` default, matching this
project's 16GB-RAM constraint) / `compute_bertscore_contrastive_baseline` (scores each generated text against
a DIFFERENT record's reference, cyclic shift by 1 - a sanity check on whether real-pair BERTScore means
anything, since raw BERTScore is known to compress into a narrow high band even for unrelated text).

**Sampling**: `assign_primary_diagnostic_class`/`stratified_sample` sample ~equally across PTB-XL's 5
diagnostic superclasses instead of plain random sampling - a first 30-record random draw came out 53% NORM /
10% MI (vs. 46% / 21% in the full pool), and since NORM reports are shorter/more templated, that skew would
bias aggregate scores for reasons unrelated to quality.

**End-to-end**: `run_evaluation(...)` builds the reference subset, samples `n_samples` records, generates +
scores each pair (caching `AgentContribution`s under `contributions/`), saves `evaluation_results.csv`
(per-record + MEAN/STD, including per-record mismatched-pair scores and which record each was compared
against - not just the aggregate) and `evaluation_interpretation.md`. `brief_mode=True` switches text
generation to `generate_brief_ecg_summary` (its own dedicated prompt, does NOT touch `sep_agent.py` - see
finding below). `embedding_model=<ECGEmbeddingNet>` includes the learned diagnostic probabilities in each
contribution. Per-record failures are caught and logged, not fatal. `generate_evaluation_summary_pdf` renders
a one-page PDF (aggregate scores, best/worst 5 records with reference shown alongside generated text).

**Three findings** (full text + the run-1-to-7 result history table in `results/project_history_and_results.md`;
per-run interpretation in `results/evaluation_brief/evaluation_interpretation.md`, and the older runs'
in `archive/evaluation*/`):
1. **BERTScore doesn't discriminate here at all** - the contrastive gap (real minus mismatched pairs) stays
   within ±0.003 of zero across 5 runs (incl. n=300); ~0.70-0.82 is raw BERTScore's compressed baseline floor
   on short clinical text, not semantic alignment. With BLEU/ROUGE near-zero, **this setup cannot make a
   quantitative semantic-faithfulness claim** - a legitimate conclusion, not a failure.
2. **Category mismatch by design**: reference = diagnostic interpretation, generated text = quantitative
   description only (the prompt forbids diagnosis). Motivated two done follow-ups (`flag_validation.py`,
   embedding wired into `ecg_agent.py`). Ceiling remains for axis/morphology diagnoses the single-lead
   pipeline never computes (BBB, fascicular block, voltage-criteria hypertrophy) - same root as HYP stagnation.
3. **Brief number-free style gives a real BLEU/ROUGE win, but the BERTScore rise is a trap** (contrastive gap
   barely moved). **Current citable number: run 7, n=300** - BLEU 4.29±5.60, ROUGE-1 0.262, BERTScore F1
   0.821, gap +0.002 (NORM's outsized BLEU 11.14 is a real class effect).

**Clinical (biomedical) cosine similarity added** (`compute_clinical_cosine_similarity_batch`,
`load_clinical_embedding_model`): scores the generated-vs-reference pair as whole-report biomedical
embeddings (default `pritamdeka/S-PubMedBert-MS-MARCO`), a more meaningful semantic metric than BLEU/ROUGE on
this short clinical text. **Caveat found this session:** that default model ships only a torch `.bin`, which
the installed `transformers` refuses to load under torch < 2.6 (CVE-2025-32434) - use a safetensors biomedical
model (`NeuML/pubmedbert-base-embeddings`) instead until torch is upgraded.

### `src/agent_report_evaluation.py` — per-agent (signal vs image) report evaluation
Doubles the ECG report evaluation across the two ECG branches, on the SAME 20 held-out (PTB-XL fold 10)
records, against the SAME PTB-XL/RTLP cardiologist reference report (GenECG images ARE those records rendered
as images, so both branches share the reference). Reuses `report_evaluation.py`'s reference-pool builder,
stratified sampler, and scorers (BLEU / ROUGE-1 / clinical cosine). Phased to avoid Ollama model thrashing
(16GB RAM -> one resident model): Phase 1 signal (MedGemma) -> Phase 2 image contributions (qwen2.5vl:7b) ->
Phase 3 image Sep summaries (MedGemma) -> scoring; resumable via a per-record text cache (each phase is one
foreground chunk, since the harness reaps long background tasks). **Result** (`results/agent_evaluation/`):
signal BLEU 0.62 / ROUGE-1 0.044 / clin.cosine 0.560; image 1.41 / 0.131 / 0.572. **Important reading:** the
image branch scores HIGHER on BLEU/ROUGE, but that is partly generic "report-like" VLM prose overlapping the
reference vocabulary (e.g. it says "sinus rhythm, normal" on a record whose reference is atrial fibrillation) -
higher text-similarity is NOT higher clinical accuracy. The image branch's real accuracy is its classifier
(macro F1 0.64), not this text table. Notes: the signal branch is run single-lead here for tractability (the
multilead+embedding form emits ~35 findings, needs `num_predict=4096`, ~80s/record); clinical cosine uses
`NeuML/pubmedbert` (the torch<2.6 caveat above). Only ECG has a reference report, so MRI/patient "accuracy" is
their flag validation, not a text table.

### `src/mri_report_evaluation.py` — MRI report evaluation (structure ready, references pending)
MRI counterpart of `report_evaluation.py`: scores the MRI agent's generated narrative against reference
reports with the SAME three metrics (BLEU / ROUGE-1 / clinical cosine), **importing** `report_evaluation.py`'s
scorers unchanged (no re-implementation) and routing the MRI contribution through the Sep Agent (MRI-only, same
pattern as the ECG module) for scoreable text. Group-stratified sampler over ACDC's 5 pathology groups; caches
per-patient contributions. **Blocker, documented in the module: ACDC ships NO free-text cardiologist report**, so
there is currently no reference corpus — everything is wired EXCEPT `load_mri_reference_reports()`, the single
plug-in point (candidate sources listed in the module docstring). Until a reference source exists,
`run_mri_evaluation()` stops with an explicit message and returns `None` rather than fabricating references or
scores. No new dependency (reuses `report_evaluation.py`'s sacrebleu/rouge/sentence-transformers).

### RAG embedding-model comparison (general vs clinical) — re-run on the current index
`src/rag_pearson_evaluation.py` (the general `all-MiniLM-L6-v2` vs clinical `NeuML/pubmedbert` A/B) was written
against an OLDER, larger RAG index and no longer matches the current production index (3 clean guidelines / 83
chunks) - its query/negative chunk headers are absent, so it must be re-pointed at the current index (its
`run_pearson_evaluation` reads only `results/rag_knowledge_base/chunks.json`; the `kb_dir` arg is ignored, only
the model name matters). Re-run on the current index (14 clinical queries, + a per-query top-1 accuracy):
`results/rag_ab_test/rag_embedding_comparison.md` - **general r 0.770 / top-1 13/14 (93%) / separation 0.413**
vs **clinical r 0.770 / top-1 14/14 (100%) / separation 0.512**. So on the shipped index the two are TIED on
Pearson and the clinical model is marginally BETTER on top-1/separation - meaning **all-MiniLM's retention as
the production default is a PRAGMATIC choice (~5x smaller, faster, general-purpose, competitive), NOT because
it retrieves better** (the earlier docstring's "clinical fails top-1" finding was on the old index and no
longer holds). An honest defense point, not a superiority claim.

### `src/flag_validation.py` — deterministic flag validation against PTB-XL scp_codes
Standalone module: validates `ecg_agent.detect_flags`'s 4 fixed-threshold clinical flags against real
PTB-XL diagnostic ground truth via `scp_codes` - a classification agreement check (precision/recall/F1),
correctly scoped to what these flags actually claim, unlike `report_evaluation.py`'s free-text comparison.
**No LLM call anywhere** (`detect_flags` is pure feature-threshold logic) - runs on the full dataset in
~24 minutes, unlike the LLM-bottlenecked report evaluation.
- Ground-truth code sets **verified directly against `scp_statements.csv`**, not guessed: tachycardia =
  `STACH`/`SVTAC`/`PSVT`; bradycardia = `SBRAD`; irregular rhythm = `AFIB`/`SARRH`/`SVARR`/`AFLT`/`BIGU`/
  `TRIGU`; wide QRS = `CRBBB`/`CLBBB`/`ILBBB`/`IRBBB`/`WPW`/`IVCD` (AV blocks and fascicular blocks excluded -
  they affect PR interval/axis, not QRS width). Embedding-based flags are deliberately NOT covered here -
  already validated against a patient-disjoint held-out split; re-checking here would be a less rigorous,
  potentially circular duplicate.
- `compute_features_and_ground_truth(...)` / `sweep_threshold(features_df, flag, feature_col, candidate_thresholds, direction)`
  — caches raw per-record features once, then tries many candidate thresholds cheaply (no pipeline re-run
  per candidate). `bradycardia_suspected` is never swept - no threshold is evaluable with zero ground-truth
  positives. `wide_qrs_suspected` was never touched (already the best-performing flag).

**Full-dataset result (21799 records) — table + sweep detail in `results/flag_validation/flag_validation_report.md`
and `results/project_history_and_results.md`.** Only `wide_qrs_suspected` (P 0.376 / R 0.531 / F1 0.440)
performs well absolutely; the other three fire on ordinary physiological variation PTB-XL cardiologists didn't
code as diagnostic (a statistically elevated 10s heart rate ≠ a clinical "sinus tachycardia"). This matters
because these 4 flags feed real `flags_for_review` in every `AgentContribution`. Recalibration:
`tachycardia_suspected` **100→120 bpm** (F1 peaked at 165 but recall collapsed to 0.286 there; 120 keeps
recall 0.786 - recall-favoring since it's a human-review alert, not a classifier); `irregular_rhythm_suspected`
**kept at CV 0.15** (F1 barely moved 0.047→0.056 while recall dropped, not fixable by thresholds alone);
`bradycardia` unevaluable (`SBRAD` never occurs). `write_flag_validation_report` saves the table + sweep CSVs.

### `src/patient_data_flag_validation.py` — deterministic flag validation for the patient-data agent
Standalone, mirrors `flag_validation.py`'s approach on the 3rd modality: validates `detect_flags_patient`'s
7 flags against the UCI binary target `heart_disease_present = (num > 0)` - precision/recall/F1, no LLM, all
303 patients. Best flags: `abnormal_thallium_result_suspected` (F1 0.739), `multiple_vessels_involved_suspected`
(F1 0.702); drove the two recalibrations noted in the `patient_data_agent.py` section (oldpeak 1.0→0.8 mm;
max-HR `220-age@85%` → Tanaka `208-0.7×age`@91%). Full table: `results/patient_data_flag_validation_report.md`.
Also writes a per-record CSV (`build_per_record_results_df` → `results/patient_data_flag_validation_results.csv`,
303 rows, `<flag>_predicted`/`<flag>_ground_truth` per flag - ground truth is the single binary target repeated
per flag) in the same layout the ECG/MRI validations use, so the UI Validation page can recompute its
precision/recall/F1 table live for the patient modality too (added when the UI page revealed only patient
lacked this CSV).

### Streamlit UI (`Home.py` router + `views/`) — local presentation/orchestration layer
A local web UI over the pipeline, run with `streamlit run Home.py` (localhost, not deployed - the Ollama +
torch + dataset dependencies make cloud hosting impractical, and the data is clinical). Presentation layer
only, same discipline as `report_pdf.py`: it **calls the existing `src/` functions unchanged** and never
recomputes or duplicates any validated logic. **Phase P0 (MVP), current**: single-page ECG explorer - pick a
real PTB-XL record (`ecg_id`, lr/hr, single-lead choice), then it runs `ecg_preprocessing_pipeline` +
`compute_feature_dict`/`detect_flags` (deterministic features + flags, always shown), renders the 5-stage
pipeline figure inline (a local helper that *returns* a matplotlib fig for `st.pyplot`, unlike
`plot_pipeline` which saves + `plt.show`), and optionally the learned embedding (`@st.cache_resource`,
checkpoint `embedding_stratfold.best.pt`) and the LLM structured findings via `generate_ecg_contribution`.
Demographic context is built with `agent_common.build_patient_context_line`. **Important wiring detail**: the
canonical `multilead_signal` passed downstream is `wfdb.rdsamp`'s raw `(n_samples, 12)` output, NOT transposed
(the embedding transposes internally, see `ecg_agent.compute_embedding_features`). **Truncation fix (led to a
shared-backend change)**: with the multi-lead + embedding feature set (~32 numeric features), MedGemma emits
~32 findings, whose JSON exceeded the backend's hardcoded `num_predict=2048` and truncated mid-response
("Unterminated string") on essentially every record - the single-lead set (~10 findings) fit. So
`generate_report_medgemma` gained an optional `num_predict` param (default 2048 = byte-identical for existing
callers, same additive pattern as `temperature`/`seed`); the UI passes a custom `llm_call` with
`num_predict=4096`, which validated the full path end-to-end (record `ecg_id=1`: 35 findings incl. all 5
embedding findings, all `evidence_ref` valid, `embedding_NORM_pattern_suspected` raised). LLM and embedding
are opt-in toggles; the deterministic panel works with Ollama down / torch unused, and backend errors are
caught and surfaced (no crash). **Selectable LLM backend** (sidebar): MedGemma 4B (local, default, no key,
data stays on the machine) or GPT `openai/gpt-oss-120b` via the free Groq cloud (`generate_report_groq`, needs
`GROQ_API_KEY`; the UI passes `reasoning_effort="low"` + `max_tokens=4096` since gpt-oss is a reasoning model
whose hidden tokens otherwise truncate the JSON). Cloud choice shows a "prompt leaves the machine" caption
(fine for public PTB-XL data) and warns if the key is missing. Both backends validated end-to-end on
`ecg_id=1`. NOTE: Groq's catalog dropped the Llama chat models (`llama-3.3-70b-versatile` now 404s despite
being `generate_report_groq`'s documented default) - the current free general chat models are `openai/gpt-oss-120b`
/ `-20b` and `qwen/qwen3.6-27b`; a Llama backend option was intentionally not added for this reason.

**Phase P1 (multi-modality + aggregation), built.** The UI is **multipage** (see Phase P4 below for the current
role-aware `st.navigation` router — originally P1 relied on Streamlit's auto-discovery of a `pages/` folder,
since replaced). The working pages live in a **`views/`** folder: `0_Overview.py`, `1_ECG.py`,
`2_Patient_data.py`, `3_MRI.py`, `4_Case_aggregation.py`, `5_Validation.py`, `6_Models.py`, `8_Chat.py`. ECG is
one page among equals, not the home page. A small root module **`ui_common.py`** centralizes what every page
needs (it puts `src/` on `sys.path` on import, holds the dataset paths + `PTBXL_LEAD_ORDER`, the
selectable-backend dict via `get_llm_backends()`/`backend_selector()`, and `init_case_store()`); `Home.py` and
every page import it instead of duplicating that setup. Pages accumulate each modality's `AgentContribution` into a **cross-page case store**
on `st.session_state` (`contributions` / `patient_context_lines` / `figure_paths`, keyed by `agent_id`), which
`4_Case_aggregation.py` consumes: it runs `sep_agent.generate_sep_agent_decision` (with opt-in `use_rag`/
`use_cot`/`use_kg` toggles) over whatever contributions are present, shows the integrated impression +
deterministic flag union + traceability + KG convergent diagnoses (`trace["kg_convergent_diagnoses"]`), and
exports the PDF via `report_pdf.generate_case_report_pdf` (`st.download_button`, with the synthetic-pairing
disclaimer when >1 modality). `2_Patient_data.py` is **Form B** - a 13-field UCI form → `generate_patient_contribution`
(a form is the natural input since the modality IS tabular; `ca`/`thal` allowed missing → `None`). `3_MRI.py`
picks an ACDC patient → `preprocess_patient` → `generate_mri_contribution`; it uses **MedGemma only** (the MRI
agent attaches the image and MedGemma is the only multimodal backend - the Groq cloud backend is text-only and
cannot see the image). **Sex-encoding normalization (handled at the source, per the "correct at the source"
principle):** UCI encodes `1=male`/`0=female`, the OPPOSITE of PTB-XL's `1=female`; rather than any ad-hoc
per-page string building, each page normalizes sex to the canonical **label** ("male"/"female") and passes that
to `build_patient_context_line`, which already accepts the label form directly (so both datasets' raw encodings
converge on the same label, no conflict). Validated end-to-end (local MedGemma): ECG(10 findings) +
patient(13 findings, 3 flags) → Sep Agent (2 agents, 21 traceability entries) → PDF (8 KB); MRI on `patient001`
(DCM, LV EF 23.7%) → 8 findings + `low_lv_ejection_fraction_suspected`. Multipage boot check: HTTP 200, all
three pages discovered, no errors. Robustness note found here: MedGemma is **stochastically unreliable on the
big multi-lead ECG prompt** (~35 findings) - it intermittently returns JSON missing required keys / truncated;
the GPT (Groq) backend handles that prompt more reliably, and the single-lead ECG prompt is reliable on both.

**Phase P2 (ECG upload + guardrails), built.** The ECG page (`views/1_ECG.py`) now has a **source** toggle:
"PTB-XL dataset" (the original flow), **"Upload"** (signal), or **"Upload image"** (added later - see below).
Signal uploads go through a new additive module
**`src/ecg_ingest.py`** (UI-agnostic, no Streamlit) that normalizes any upload to the canonical
`(n_samples, 12)`-in-mV / explicit-`fs` / PTB-XL-lead-order form the pipeline already expects: `load_wfdb`
(the clean case - fs/units/lead-order come from the header, auto-reordered to PTB-XL) and `load_csv` (user
supplies fs + unit mV/µV + orientation + a "columns are in PTB-XL order" flag; auto-transposes when the longer
axis is leads). `validate(ingested) -> IngestReport` is the **guardrail**: `hard_errors` (block: bad `fs`,
empty/too-short/all-NaN signal), `warnings`, and an `enabled` map that the page uses to actually **disable**
(not merely warn about) invalid feature families. Key design point - the embedding and the multi-lead voltage
features share compute_feature_dict's single `multilead_signal` argument, so both are gated on the SAME
condition (**12 leads in PTB-XL order**); the unit does NOT gate them because the embedding re-normalizes
per-lead (unit-invariant) - a wrong unit only raises a warning that the voltage-amplitude criteria
(Sokolow-Lyon/territorial-Q) may be unreliable. So on upload: <12 leads or unconfirmed order → `multilead=None`
passed (single-lead rate/rhythm only, embedding auto-disabled with an on-page note); implausible amplitude
(µV/ADC mistaken for mV) → still runs incl. embedding, but a prominent voltage warning. Validated: real PTB-XL
WFDB → all enabled/0 warnings; ×1000 (µV) → voltage off, embedding kept; 2-lead → single-lead only; CSV both
orientations + µV→mV conversion. WFDB uploads are staged to a temp dir (both `.hea`+`.dat` needed) before
`load_wfdb`; optional age/sex demographic fields feed `build_patient_context_line`.

**ECG page "Upload image" source, built.** A third source on `views/1_ECG.py`, a self-contained branch (it
`st.stop()`s before any signal code) that goes DIRECTLY through the ECG-IMAGE branch: upload → adaptive
preprocessing + the GenECG **image classifier** (calibrated `pathology_<CLASS>_probability` + flags, shown
always, `@st.cache_resource`) + optional **VLM** structured findings. The VLM model is a sidebar selector of
LOCAL multimodal models (`qwen2.5vl:7b` default; the cloud Groq backend is excluded - text-only, can't see the
image). It saves an `ecg_image` contribution to the cross-page case store (→ Case aggregation + KG convergence).
Same UI conventions as the signal path (`*_ran` flag, cached result, deterministic panel always / LLM opt-in).
Verified: `ast.parse` OK, app boots HTTP 200, headless `AppTest` 0 exceptions on both sources.

**Phase P3 (MRI upload + pipeline-utility comparison, prompt lab, validation page), built.**
- **MRI upload / raw-mask pipeline** (`views/3_MRI.py` redesigned + new pure helper **`ui_mri.py`**): a
  **source** toggle - "ACDC patient" or **"Upload NIfTI"** (ED + ES volumes, both phases needed for EF, +
  optional expert masks). (Note: the "ACDC patient" selector lists ACDC's **`testing/` = held-out patient101-150**
  so the learned-model demos stay off training data; training patients are reachable via Upload.) **Upload
  auto-extraction (mirrors the ECG upload reading its header):** the **patient ID** is parsed from the uploaded
  filename (`patientXXX`), and an **optional `Info.cfg` uploader** auto-extracts the **pathology group** (+ ED/ES
  frames, height/weight via `load_patient_info`) - shown as a caption; when no `Info.cfg` is given the pathology
  classifier still predicts the group. From the raw volume ALONE, the trained YOLO+U-Net predicts a mask over the full ED
  and ES stacks (`ui_mri.predict_mask_volume`, reusing `mri_pipeline_validation._predict_mask_slice`), from
  which `ui_mri.features_from_masks` computes chamber volumes + EF in the exact keys `preprocess_patient`
  produces. **This resolves the EF short-circuit for uploads**: a real MRI has no expert mask, but the
  *predicted* mask yields `lv_ejection_fraction_pct`, so `generate_mri_contribution` no longer skips - no agent
  change needed (the user's own insight: "we already have a pipeline to make the mask"). When an expert mask is
  also present (always for ACDC, optional on upload) the page shows a **pipeline-utility check**: predicted vs
  expert EF + per-structure Dice. Validated on `patient001` (DCM): predicted LV EF 20.8% vs expert 23.7%, Dice
  RV/Myo/LV = 0.923/0.858/0.953; `features_from_masks(expert)` reproduces `preprocess_patient` byte-for-byte;
  full contribution on the predicted path = 14 findings + `low_lv_ejection_fraction_suspected` +
  `pathology_DCM_pattern_suspected`. Model loaders are `@st.cache_resource` (YOLO+U-Net, classifier; CUDA if
  available). Still MedGemma-only for the LLM findings (multimodal); the LLM step is an opt-in toggle.
  **Robustness added while debugging a striped-image report:** `ui_mri.to_canonical` reorients an uploaded
  volume so the short-axis stack is axis 2 (detected via the largest voxel spacing / through-plane, fallback
  fewest-slices) and rejects 4-D cines; a "Volume diagnostics" expander shows shape/spacing/detected-axis; and
  the slice-validity indicator under the image is **anatomical** (connected-component count + largest-piece
  share via `scipy.ndimage`, not just "is there tissue"). **The actual striped-image bug, though, was neither
  axes nor segmentation:** `ui_mri.representative_ed` was handing the raw **float64** slice to
  `encode_slice_to_base64_png` (which does `Image.fromarray(..., mode="L")`, needing **uint8**) - PIL
  reinterpreted each 8-byte float as 8 pixels → evenly-spaced vertical stripes. Volumes/EF were always correct
  (computed from the mask, and the U-Net got the properly-normalized slice via `predict_mask_volume`), so only
  the DISPLAY channel was wrong; the fix is one line (encode `raw_uint8`, not the float slice), verified by
  rendering patient001's slice to a real cardiac cross-section. Lesson: a `len(b64) > 0` check passes on a
  garbage PNG - visually inspect rendered images, don't just assert non-empty.
- **Prompt lab** — a shared `ui_common.prompt_lab(...)` expander on **every text agent** (ECG, Patient-data,
  Sep Agent; MRI is excluded because its prompt carries the image, which needs a multimodal run). Each shows
  the exact prompt (`build_findings_prompt` / `build_patient_findings_prompt` / `build_sep_agent_prompt`), an
  edit toggle runs the edited prompt through the SAME validating parser the agent uses
  (`parse_findings_response` / `parse_sep_agent_response`), so a fabricated `evidence_ref` or `(agent_id,
  feature)` citation is still rejected. Results are marked non-standard and **never saved to the case** - a
  transparency/limits testbench, not a way to bypass the guard (the guard lives in the parser, not the prompt).
- **Streamlit state fix (important pattern)**: a `st.button`/`st.form_submit_button` is only `True` on its own
  click, so the pages' original `if not run_btn: st.stop()` gating made ANY later interaction (a toggle, the
  prompt-lab's own button) rerun the script with the run button `False` and short-circuit to a near-blank page.
  Fixed on ECG/Patient/Sep by persisting a `*_ran` flag in `st.session_state` (gate on that, not the transient
  button) AND only (re)calling the LLM on an explicit run click while caching the contribution/decision in
  session_state - so clicking the lab button reuses the stored result instead of re-invoking the model. Also
  fixed a latent `backend_name` NameError on the ECG page left over from the backend-selector refactor.
- **Validation page** (`views/5_Validation.py`, read-only): recomputes precision/recall/F1 per flag on the fly
  from the saved `*_predicted`/`*_ground_truth` CSVs (`results/flag_validation/`, `results/mri_flag_validation/`),
  renders the threshold-sweep curves and the written validation reports. Verified the recomputed numbers match
  the documented ones (wide_qrs F1 0.440; MRI low_lv 0.975 / high_lv 0.698 / low_rv 0.562). Note: the patient
  modality had no per-record CSV (only ECG/MRI did), so `patient_data_flag_validation.py` gained
  `build_per_record_results_df` (writes `results/patient_data_flag_validation_results.csv`) AND an
  "Interpretation" prose section in its report, giving the patient tab the same live-table + reasoning depth as
  the others.
- **Smaller UI extras** (from usability feedback): the Case-aggregation page keeps a **configuration
  comparison** (each Sep run stored by a config tag `standard`/`rag`/`cot`/`kg` and shown side by side, so RAG
  vs CoT vs KG impressions compare at a glance); the exported **PDF name carries the config + timestamp**
  (`case_report_<config>_<YYYYMMDD_HHMMSS>.pdf`); the patient form **defaults every field to unknown → None**
  (fill only what you have; `detect_flags_patient` already guards None per field); the prompt-lab message now
  says whether the prompt was **edited vs unchanged** (was misleadingly "non-standard" for an unchanged run).

- **Models page** (`views/6_Models.py`, read-only): the full model inventory + each learned model's
  architecture and **training curves read live from the saved epoch logs** (`results/ecg_embedding/
  epoch_log_stratfold.csv`, `results/mri_yolo_training/final/results.csv`, `results/mri_unet_training/
  epoch_log.csv`, `results/mri_classifier_training/epoch_log.csv`) + final metric, plus the embedding's
  per-class eval figure and pointers to the Validation page (flag sweeps) and `evaluation_brief/` (text
  scores). Built to show/justify the modelling choices at a glance (e.g. for a supervisor review).
- **Reset case** button on the Overview page (`ui_common.reset_case`): clears the whole case store, Sep-Agent
  runs, cached per-page results, and the `*_ran` flags so every page returns to its initial state.

Full app boot check: HTTP 200, all pages load, no errors.

**Phase P4 (role-based router + conversational chat), built.** Two changes reshaped the app's entry:
- **`Home.py` is now a login gate + explicit router, not the landing page.** It shows a role selector
  (`patient` / `doctor` / `researcher`) with **fictitious demo passwords** (`ui_common.ROLES`, e.g.
  `patient2026` — explicitly a UX role **switch**, NOT real authentication; everything runs on public/synthetic
  data), then builds a **role-aware page list** and runs it via `st.navigation`/`st.Page` over the `views/`
  files. This **replaces Streamlit's `pages/` auto-discovery** (hence the folder rename `pages/` → `views/` —
  auto-discovery would otherwise show every page to every role). Role visibility: `patient` → Overview + ECG +
  Patient-data + MRI + **Case-aggregation** + Chat (no Validation/Models); `doctor` → the same set;
  `researcher` → everything (+ Validation, Models). **The Case-aggregation page is itself role-aware**
  (`4_Case_aggregation.py`): for a `patient` it renders a **simplified plain-language report** — a "Generate my
  report" button, the Sep summary + a downloadable PDF, WITHOUT the clinical-decision internals (RAG/CoT/KG
  toggles, convergent-diagnosis block, traceability table, prompt lab, backend selector); under the hood it
  runs the Sep Agent with **`use_cot=True`** (hidden reasoning to improve quality) but NOT KG/RAG (which would
  name diagnoses / inject criteria, breaching the patient "no diagnosis" contract). `doctor`/`researcher` get
  the full clinical aggregation (all toggles, traceability, KG convergence). The role badge + logout
  are pinned to the sidebar bottom via a small CSS flex-order trick (pages call `st.stop()`, so a footer added
  after `pg.run()` can't rely on normal ordering); **logout does a full `st.session_state.clear()`** so no
  case data leaks between roles. `ui_common` gained `ROLES`, `current_role`, `is_researcher`,
  `role_is`. Landing content moved to **`views/0_Overview.py`** (role-specific framing + current-case status +
  the save/reset buttons).
- **Conversational chat page (`views/8_Chat.py` + new pure helper `ui_chat.py`)** — a free-text assistant
  grounded in the current case's already-produced findings (`build_case_context` summarizes the Sep impression,
  flag union, and each contribution's findings) and, when present, the MRI slice image — so it uses the
  **local multimodal MedGemma** backend (it can actually SEE the MRI), not the text-only cloud backend. The
  system framing is **role-dependent** (`_ROLE_SYSTEM`): the `patient` role is instructed to never state a
  diagnosis/prognosis or give treatment advice and to redirect to a doctor (the same "support, not diagnosis"
  contract the rest of the project keeps); `doctor` gets differentials as decision support; `researcher` gets
  open discussion of model outputs/limits. Chat history lives in `st.session_state["chat_history"]` (cleared by
  `reset_case`). It only reads the case — it never writes an `AgentContribution` or a Sep decision, so it can't
  corrupt the auditable case store.

**Not yet built** (possible later UI work): a runnable prompt-lab for MRI (blocked on a multimodal run path);
a global backend selector; wiring a multimodal cloud backend (Claude) so MRI too could offer a backend choice.
Requires `streamlit` (see `requirements.txt`).

### Deployment (Docker) — `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `DEPLOY.md`, `DOCKERHUB_README.md`
Root-level Docker setup to build/share the app as a reproducible image (built & validated: 3.58 GB, container
serves `/_stcore/health` → 200). Presentation/packaging only, no analysis logic. **Option A** (the chosen run
mode): only the **app** is containerized; it reaches the **native host Ollama** (GPU) via
`OLLAMA_HOST=http://host.docker.internal:11434` — no NVIDIA Container Toolkit, LLM stays GPU-fast, the
container's torch runs CPU. Key build choices, each forced by a real constraint: **no `apt`** (uses
`opencv-python-headless`) because restricted networks block apt's HTTP/port-80 mirrors; **CPU-only torch** from
PyTorch's CPU index so the image is ~3.5 GB not ~8 GB (Option A doesn't use in-container GPU); a **pip cache
mount** so a build retry after a Docker Desktop/WSL2 hiccup resumes from downloaded wheels. Datasets are
**never in the image** (mounted read-only volumes; training datasets + `physionet/` are `.dockerignore`d);
MedGemma (~5 GB) is pulled into the host Ollama, not the image. The one code change enabling this is
`generate_report_medgemma`'s env-configurable `host` (above). Full runbook + the A/B/C option comparison +
troubleshooting: **`DEPLOY.md`** and `results/docker_deployment_guide.pdf`; the Docker Hub overview to hand to
anyone who `docker pull`s it (what to provide beyond the image): **`DOCKERHUB_README.md`**. Scope: local Docker
+ Docker Hub push, **no public deployment** (see the guide's "real walls" section).

### Data layout
- `kaggle/archive/ptb-xl-.../` — full PTB-XL dataset as downloaded from Kaggle (`ptbxl_database.csv` is the
  record index; signals live under `records100/` and `records500/`, keyed by the `filename_lr`/`filename_hr`
  columns). This is large reference data, not something to regenerate or edit.
- `ACDC/database/{training,testing}/patientXXX/` — ACDC MRI dataset: `Info.cfg` (ED/ES frame numbers,
  pathology group), `patientXXX_4d.nii.gz` (full cine sequence), `patientXXX_frameNN[_gt].nii.gz` (single
  phase + its segmentation mask). Also large reference data, not something to regenerate or edit.
- `physionet/ptb-xl-.../` — the official PhysioNet PTB-XL distribution (same internal layout as the Kaggle
  copy above), used to verify the Kaggle republish is unmodified - see `notebooks/test_verification.ipynb`
  ("zero differences" result below).
- `RTLP/` — shallow clone of github.com/danikiyasseh/RTLP (`translations/en_df_round4.csv` is the only file
  used). Large reference data, not something to regenerate or edit.
- `uci_heart_disease/` — UCI Heart Disease / Cleveland CSV (`processed_cleveland.csv`) + `variables_metadata.csv`,
  the patient-data agent's local source (fetched once via `ucimlrepo`, then read locally). Reference data.
- `genecg_downloaded/` — local cache of GenECG images (PTB-XL rendered as 12-lead ECG images, `edcci/GenECG`
  on HuggingFace), fetched on demand via `src/genecg_fixtures.py`. Regenerable cache, not source. The
  image↔PTB-XL mapping lives in `src/ptbxl_with_genecg_paths.csv` (built by `src/ecg_image.ipynb`); the ECG-IMAGE
  branch + classifier consume these (see the "ECG IMAGE branch" architecture section).
- `rag_sources/` — curated cardiology PDFs ingested into the RAG index. Only the 3 top-level PDFs are in the
  production glob; subfolders (`dataset_docs/`, `epidemiology/`, `guidelines_raw/`, `New/`) hold sources deliberately
  excluded after curation. Reference data. `rag_sources/README.md` documents the KB sources (production index vs
  excluded subfolders, per-source chunk counts, provenance/usage).
- `results/evaluation_brief/` — `report_evaluation.py`'s current largest/most-cited run (brief-mode). The
  earlier narrative-only and +embedding runs (`evaluation/`, `evaluation_with_embedding/`), and the early
  per-record pipeline figures (`records/`), were moved to `archive/` in the 2026-08-22 cleanup - regenerable
  build output, not read by anything.
- `results/flag_validation/` — `flag_validation.py` outputs; `results/patient_data_flag_validation_report.md`
  + `results/patient_data_flag_validation_results.csv` — patient-data agent validation table/CSV. (The one
  real end-to-end demo run, `patient_data_agent_test/`, was moved to `archive/`.)
- `results/rag_knowledge_base/` — the saved RAG index (`chunks.json`, `embeddings.npy`, `dropped_chunks.json`);
  `results/rag_ab_test/` — RAG-utility and G2/model A/B outputs (`rag_utility_ab_synthesis.md` is the synthesis,
  `*_ab_result.json` the raw runs, `groq_backend_setup.md` the Groq config note).
- `results/ecg_embedding/` — `ecg_embedding.py` training outputs (unsuffixed = quick subset, `_full` = leaky
  full run, `_stratfold` = leakage-free, cite this one).
- `results/_backups/` — manual backups made before large documentation rewrites (this project has no git).
- `results/report_outline.md` — internship-report section plan + **source map** (where each section's material
  already lives, keyed to the `.md`/code); `results/synthetic_test_cases.md` — reproducible cross-modality test
  cases. Report-prep aids, not analysis output. A top-level **`README.md`** (public-facing scope note +
  quickstart) also exists at the project root.

### Notebooks
- `notebooks/ecg_preprocessing.ipynb` — main exploratory notebook. Loads PTB-XL records via `wfdb.rdsamp`,
  has an older inline exploration at the top, then a "structured workflow" section further down that
  imports `ecg_preprocessing` and `ecg_agent` and runs the real pipeline end-to-end. When editing pipeline
  behavior, prefer changing the `.py` modules — the notebook imports them rather than duplicating logic
  (though the early cells still contain redundant inline versions from before the module existed).
- `notebooks/mri_preprocessing.ipynb` — MRI counterpart: audits the ACDC dataset (150 patients, no CSV index
  like PTB-XL), walks through `mri_preprocessing.preprocess_patient` on one example patient followed by
  `mri_agent.generate_mri_contribution`, then runs preprocessing across all 150 patients into a feature
  table saved to `results/mri_features_all_patients.csv`.
- `notebooks/ecg_embedding_training.ipynb` — trains `ECGEmbeddingNet` on a subset; see `ecg_embedding.py`
  above for the confirmed run results. Outputs under `results/ecg_embedding/`.
- `notebooks/sep_agent_demo.ipynb` — runs one real PTB-XL ECG record (`00001_hr`) and one real ACDC MRI
  patient (`patient001`) through their respective agents, then `sep_agent.generate_sep_agent_decision` on
  both, then `report_pdf.generate_case_report_pdf`. Saves all outputs under `results/sep_agent_demo/`
  (`ecg_contribution.json`, `mri_contribution.json`, `sep_agent_decision.json`, `case_report.pdf`) - re-running
  recreates that folder; the last saved run was moved to `archive/sep_agent_demo/` in the 2026-08-22 cleanup.
  Explicitly labeled as a synthetic ECG+MRI pairing. **Last confirmed run's summary contained a clinical
  characterization error** (described the DCM patient's markedly enlarged 295.5 mL LV end-diastolic volume
  as "low" instead of elevated/dilated), even though every traceability entry correctly pointed to a real
  finding - the traceability mechanism guarantees a cited finding is real, not that its narrative
  characterization is clinically correct.
- `notebooks/test_verification.ipynb` — exhaustive comparison of the official PhysioNet PTB-XL distribution
  against the Kaggle republish (advertised as "optimized/processed" with no changelog): a fast
  `SHA256SUMS.txt` manifest diff, then an exhaustive per-file SHA256 comparison (~87,000 files, ~3.2 GB,
  ~20-30 min). Last confirmed result: **zero differences** - every file byte-for-byte identical.

## Working conventions
- Docstrings and comments in the modules are written in English with occasional French section-header
  comments (e.g. "Suppression du baseline wander"); match this style if editing those files.
- Sampling rate (`fs`) is always threaded explicitly through functions rather than assumed — PTB-XL provides
  both 100 Hz (`records100`) and 500 Hz (`records500`) variants, so never hardcode a sample rate.
- Standalone validation/evaluation modules (`report_evaluation.py`, `flag_validation.py`) are written to
  never modify the agent/pipeline modules they check - real fixes discovered through them (the tachycardia
  threshold, the embedding integration) get applied directly in the relevant agent module, not worked around
  in the validation module.
