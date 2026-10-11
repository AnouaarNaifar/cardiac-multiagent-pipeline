"""
Streamlit page — Case aggregation (Sep Agent).

Collects whatever modality contributions have accumulated in the shared case
store (ECG from the home page, patient-data / MRI from their pages), runs the
Sep Agent to synthesize a cross-modality impression with traceability, and
exports the case-report PDF. Runs sep_agent / report_pdf unchanged.

The Sep Agent's opt-in layers (RAG / Chain-of-Thought / knowledge graph) are
exposed as toggles, defaulting off (same as the underlying function).
"""

import os
from datetime import datetime

import streamlit as st

import ui_common
from sep_agent import (
    generate_sep_agent_decision, build_sep_agent_prompt,
    parse_sep_agent_response, SepAgentParseError,
)
from report_pdf import generate_case_report_pdf

ui_common.init_case_store(st)
role = ui_common.current_role(st)
_is_patient = role == "patient"

if _is_patient:
    st.title("📋 Your case report")
    st.caption("Combines the checks you ran into one plain-language summary you can download "
               "and discuss with a doctor.")
else:
    st.title("📋 Case aggregation — Sep Agent")
    st.caption("Combines the modality contributions collected so far into one "
               "integrated impression with full traceability. Runs `sep_agent` unchanged.")

# LLM backend: clinicians/researchers pick it in the sidebar; the patient view
# just uses the local default (no technical selector shown).
if _is_patient:
    backend = ui_common.get_llm_backends()["MedGemma 4B (local)"]
else:
    with st.sidebar:
        st.header("LLM backend")
        backend = ui_common.backend_selector(st, True, key_prefix="sep")

contributions_map = st.session_state.get("contributions", {})
context_lines = st.session_state.get("patient_context_lines", {})
figure_paths_map = st.session_state.get("figure_paths", {})

# --- Show what's in the case -----------------------------------------------
st.subheader("Modalities in this case")
KNOWN = {"ecg": "ECG agent", "mri": "MRI agent", "patient": "Patient-data agent"}
cols = st.columns(len(KNOWN))
for col, (agent_id, label) in zip(cols, KNOWN.items()):
    present = agent_id in contributions_map
    n = len(contributions_map[agent_id].findings) if present else 0
    col.metric(label, "✓ present" if present else "— absent",
               f"{n} findings" if present else None)

if not contributions_map:
    st.info("No contribution yet. Produce at least one on the **ECG** (home) or "
            "**Patient data** page — it will appear here automatically.")
    st.stop()

contributions = list(contributions_map.values())


def _build_case_pdf():
    """Generate the case-report PDF and return its path + filename. Shared by the
    patient and clinician views."""
    note = None
    if len(set(contributions_map.keys())) > 1:
        note = ("This case combines modalities from datasets with no real shared patients "
                "(PTB-XL ECG / ACDC MRI / UCI patient data) — a synthetic pairing for "
                "architecture validation, not a real joint clinical case.")
    os.makedirs(ui_common.RESULTS_UI_DIR, exist_ok=True)
    tag = st.session_state.get("sep_config", "standard")
    name = f"case_report_{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(ui_common.RESULTS_UI_DIR, name)
    generate_case_report_pdf(
        path, st.session_state["sep_decision"], contributions,
        figure_paths=list(figure_paths_map.values()) or None,
        synthetic_pairing_note=note, patient_context_lines=context_lines or None,
    )
    return path, name


# --- Patient view: a simplified, plain-language final report ----------------
# The patient gets a report to take to a doctor, WITHOUT the clinical-decision
# internals (RAG/CoT/KG toggles, convergent-diagnosis block, traceability table,
# prompt lab) — consistent with the "support, not diagnosis" contract the patient
# chat also keeps.
if _is_patient:
    st.warning("**This is a research/demo tool on public data.** "
               "It summarizes the findings from your checks in plain terms; always discuss the "
               "results with a qualified doctor.")
    gen = st.button("Generate my report", type="primary", use_container_width=True)
    if gen:
        st.session_state["sep_ran"] = True
    if not st.session_state.get("sep_ran", False):
        st.stop()
    if gen or st.session_state.get("sep_decision") is None:
        with st.spinner("Preparing your report…"):
            try:
                # Privileged patient access (deliberate design choice): the patient
                # report runs the full enrichment stack - CoT for reasoning quality,
                # RAG for guideline grounding, and the knowledge graph for
                # cross-modality convergence. CoT's `reasoning` field is still hidden
                # (only decision.summary is rendered); RAG/KG enrich the summary the
                # patient sees. Note: with KG/RAG on, the summary MAY reference
                # possible conditions, so the "no diagnosis" wording is relaxed below.
                decision = generate_sep_agent_decision(
                    contributions, llm_call=backend["call"], record_id="ui_case",
                    model_name=backend["model_name"], max_retries=2,
                    use_cot=True, use_rag=True, use_kg=True)
            except Exception as exc:  # noqa: BLE001
                # Two common causes: the local model isn't reachable, OR the model
                # produced text that the consistency check kept rejecting (often
                # transient — a fresh run usually succeeds). Don't blame Ollama alone.
                consistency = "consistency check" in str(exc).lower()
                if consistency:
                    st.warning("Could not finalize the report on this attempt — the summary the "
                               "model wrote didn't pass an internal consistency check. This is "
                               "usually transient: please click **Generate my report** again.")
                else:
                    st.error("Report generation failed — is the local model (Ollama "
                             f"`medgemma:4b`) running?\n\nDetails: {exc}")
                decision = None
            st.session_state["sep_decision"] = decision
            st.session_state["sep_config"] = "patient_cot_rag_kg"
    decision = st.session_state.get("sep_decision")
    if decision is None:
        st.stop()

    st.subheader("Summary of your results")
    st.write(decision.summary)
    st.caption("This report is generated by a research/demo AI system and may reference "
               "possible conditions based on your results. It is informational only and "
               "must be reviewed with a qualified doctor.")

    # Cross-modality convergence, shown to the patient only when >= 2 modalities
    # were provided and the knowledge graph found agreement between them.
    _conv = decision.trace.get("kg_convergent_diagnoses") if isinstance(decision.trace, dict) else None
    if len(contributions) >= 2 and _conv:
        st.subheader("Findings that point the same way")
        st.write("More than one of your tests point toward the same possible condition:")
        for _d in _conv:
            st.markdown(f"- {_d.replace('_', ' ')}")
        st.caption("This agreement across several of your tests is highlighted to help "
                   "your doctor. It is not a confirmed diagnosis.")

    try:
        pdf_path, pdf_name = _build_case_pdf()
        with open(pdf_path, "rb") as fh:
            st.download_button("⬇️ Download my report (PDF)", fh.read(),
                               file_name=pdf_name, mime="application/pdf", type="primary")
    except Exception as exc:  # noqa: BLE001
        st.error(f"PDF generation failed: {exc}")
    st.stop()

# --- Options ----------------------------------------------------------------
st.subheader("Sep Agent options")
o1, o2, o3 = st.columns(3)
use_rag = o1.toggle("RAG knowledge", value=False,
                    help="Inject retrieved cardiology-guideline chunks as a 'knowledge' agent.")
use_cot = o2.toggle("Chain-of-Thought", value=False,
                    help="Add an explicit reasoning pass before the summary.")
use_kg = o3.toggle("Knowledge graph", value=False,
                   help="Surface cross-modality convergent diagnoses computed by the clinical KG.")
run_btn = st.button("Aggregate (run Sep Agent)", type="primary", use_container_width=True)

# Persist "has run": a button is only True on its own click, so without this the
# prompt-lab button would rerun with run_btn=False and short-circuit the page.
if run_btn:
    st.session_state["sep_ran"] = True
ran = st.session_state.get("sep_ran", False)
if not ran:
    st.stop()

# --- Run the Sep Agent (only on an explicit click; else reuse the result) ---
if run_btn:
    with st.spinner(f"Aggregating with {backend['model_name']}…"):
        try:
            decision = generate_sep_agent_decision(
                contributions,
                llm_call=backend["call"],
                record_id="ui_case",
                model_name=backend["model_name"],
                max_retries=2,
                use_rag=use_rag,
                use_cot=use_cot,
                use_kg=use_kg,
            )
        except Exception as exc:  # noqa: BLE001
            if "consistency check" in str(exc).lower():
                hint = ("The model's summary kept failing the directional/citation consistency "
                        "check (not a backend problem — often transient; re-run, or try another "
                        "backend/toggle combination).")
            else:
                hint = ("Is `GROQ_API_KEY` set and valid?" if backend["cloud"]
                        else "Is the Ollama server running with `medgemma:4b` pulled?")
            st.error(f"Sep Agent failed ({backend['model_name']}). {hint}\n\nDetails: {exc}")
            decision = None
        st.session_state["sep_decision"] = decision
        if decision is not None:
            # Config tag (used for the PDF name AND the comparison columns).
            _active = [n for n, on in [("rag", use_rag), ("cot", use_cot), ("kg", use_kg)] if on]
            cfg = "_".join(_active) if _active else "standard"
            st.session_state["sep_config"] = cfg
            st.session_state.setdefault("sep_runs", {})[cfg] = decision

decision = st.session_state.get("sep_decision")
if decision is None:
    st.warning("Aggregation not available — click **Aggregate** to run the Sep Agent.")
    st.stop()
config_tag = st.session_state.get("sep_config", "standard")
st.caption(f"Backend used: `{backend['model_name']}` · configuration: `{config_tag}`")

# --- Configuration comparison (accumulates one column per run config) -------
runs = st.session_state.get("sep_runs", {})
if len(runs) >= 2:
    with st.container(border=True):
        h1, h2 = st.columns([4, 1])
        h1.subheader("Configuration comparison")
        if h2.button("Clear", key="sep_clear_runs"):
            st.session_state["sep_runs"] = {}
            st.rerun()
        cols = st.columns(len(runs))
        for col, (label, dec) in zip(cols, runs.items()):
            with col:
                st.markdown(f"**{label.replace('_', '+')}**")
                st.caption(", ".join(f"`{fl}`" for fl in dec.flags_for_review) or "no flags")
                st.write(dec.summary)
                conv = dec.trace.get("kg_convergent_diagnoses") if isinstance(dec.trace, dict) else None
                if conv:
                    st.caption("KG convergent: " + ", ".join(f"`{d}`" for d in conv))
    st.caption("Run **Aggregate** with different RAG / CoT / KG toggles to add columns here "
               "and compare the resulting impressions side by side.")

# --- Integrated impression --------------------------------------------------
st.subheader("Integrated clinical impression")
st.write(decision.summary)

st.markdown("**Flags for review (deterministic union):** " + (
    ", ".join(f"`{fl}`" for fl in decision.flags_for_review) or "_none_"))

if use_cot and decision.reasoning:
    with st.expander("Reasoning (Chain-of-Thought — kept out of the summary)"):
        st.write(decision.reasoning)

# --- KG convergent diagnoses (deterministic, from the graph) ---------------
conv = decision.trace.get("kg_convergent_diagnoses") if isinstance(decision.trace, dict) else None
conv_cat = decision.trace.get("kg_convergent_categories") if isinstance(decision.trace, dict) else None
_kg_ran = isinstance(decision.trace, dict) and "kg_convergent_diagnoses" in decision.trace
if conv or conv_cat:
    st.subheader("Cross-modality convergent evidence (computed by the KG)")
    if conv:
        st.markdown("**Diagnoses supported by ≥2 different modalities:**")
        for d in conv:
            st.markdown(f"- `{d}`")
    if conv_cat:
        st.markdown("**Superclass-level convergences:** " + ", ".join(f"`{c}`" for c in conv_cat))
elif _kg_ran:
    st.info("**Knowledge graph ran, but found no cross-modality convergent diagnosis** for this case. "
            "Convergence requires findings from **≥2 different modalities** whose flags meet at the same "
            "diagnosis in the graph — a single-modality case, or flags that don't overlap on a shared "
            "diagnosis, yields none. Add another modality (e.g. ECG + patient-data) with related flags.")

# --- Traceability -----------------------------------------------------------
st.subheader("Traceability")
if decision.traceability:
    st.dataframe([{
        "statement": t.statement, "agent_id": t.agent_id, "feature": t.feature,
    } for t in decision.traceability], use_container_width=True, hide_index=True)
    st.caption("Note: traceability guarantees a cited finding is real, NOT that its "
               "narrative characterization is clinically correct.")
else:
    st.markdown("_No traceability entries._")

# --- PDF export -------------------------------------------------------------
st.subheader("Case report PDF")
note = None
present = set(contributions_map.keys())
if len(present) > 1:
    note = ("This case combines modalities from datasets with no real shared patients "
            "(PTB-XL ECG / ACDC MRI / UCI patient data) — a synthetic pairing for "
            "architecture validation, not a real joint clinical case.")

os.makedirs(ui_common.RESULTS_UI_DIR, exist_ok=True)
pdf_name = f"case_report_{config_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
pdf_path = os.path.join(ui_common.RESULTS_UI_DIR, pdf_name)
try:
    generate_case_report_pdf(
        pdf_path, decision, contributions,
        figure_paths=list(figure_paths_map.values()) or None,
        synthetic_pairing_note=note,
        patient_context_lines=context_lines or None,
    )
    with open(pdf_path, "rb") as fh:
        st.download_button(f"⬇️ Download case report ({pdf_name})", fh.read(),
                           file_name=pdf_name, mime="application/pdf",
                           type="primary")
except Exception as exc:  # noqa: BLE001
    st.error(f"PDF generation failed: {exc}")

# --- Prompt lab (experimental, shared helper) ------------------------------
st.divider()
_valid_pairs = {(c.agent_id, f.feature) for c in contributions for f in c.findings}
_sep_prompt = build_sep_agent_prompt(contributions, use_cot=use_cot)


def _sep_handle_raw(raw):
    try:
        summary, traceability, reasoning = parse_sep_agent_response(
            raw, _valid_pairs, require_reasoning=use_cot)
        st.success("Parser accepted the response.")
        st.markdown("**Summary:**")
        st.write(summary)
        if reasoning:
            with st.expander("Reasoning"):
                st.write(reasoning)
        st.markdown(f"**Traceability entries:** {len(traceability)} "
                    "(each validated against a real (agent_id, feature) pair).")
    except SepAgentParseError as exc:
        st.error(f"Parser REJECTED the response (the citation guard still works): {exc}")


ui_common.prompt_lab(st, "sep", _sep_prompt, backend, _sep_handle_raw,
                     extra_note="Here it rejects any traceability citation that isn't a real "
                                "(agent_id, feature) pair from the inputs.")
