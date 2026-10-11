"""
Streamlit page — Patient-data agent (Form B).

A data-entry form for the 13 UCI Heart Disease (Cleveland) clinical fields.
Unlike ECG/MRI there is no file to upload - the input IS tabular - so a form
is the natural entry point. It runs the existing patient_data_agent unchanged
(generate_patient_contribution) and stores the resulting AgentContribution in
the shared case store for the Case aggregation page.

Sex-encoding caveat handled here: UCI encodes sex as 1=male / 0=female, the
OPPOSITE of PTB-XL (1=female), so the demographic context line is built
directly from the UCI code here rather than via agent_common.
build_patient_context_line (which assumes PTB-XL's encoding).
"""

import streamlit as st

import ui_common
from patient_data_agent import (
    compute_feature_dict, detect_flags_patient, generate_patient_contribution,
    build_patient_clinical_summary, build_patient_findings_prompt,
)
from agent_common import (
    build_patient_context_line, parse_findings_response, FindingsParseError,
)

ui_common.init_case_store(st)

_is_patient = ui_common.current_role(st) == "patient"

st.title("🧾 Patient-data agent" if not _is_patient else "🧾 My clinical data")
if _is_patient:
    st.caption("Fill in what you have; you will then get your report on the "
               "**Case aggregation** page and can ask questions on **Discuss**.")
else:
    st.caption("13 UCI Heart Disease (Cleveland) fields → structured findings. "
               "Runs `patient_data_agent` unchanged.")

# LLM config lives in the sidebar on every text agent (consistent placement).
if _is_patient:
    run_llm = True
    backend = ui_common.get_llm_backends()["MedGemma 4B (local)"]
else:
    with st.sidebar:
        st.header("LLM backend")
        run_llm = st.checkbox("Generate LLM findings", value=True)
        backend = ui_common.backend_selector(st, run_llm, key_prefix="patient")

# --- UCI categorical encodings (must match patient_data_agent's label maps) --
SEX = {"male": 1, "female": 0}
CP = {"typical angina": 1, "atypical angina": 2, "non-anginal pain": 3, "asymptomatic": 4}
RESTECG = {"normal": 0, "ST-T abnormality": 1, "probable LVH": 2}
SLOPE = {"upsloping": 1, "flat": 2, "downsloping": 3}
THAL = {"normal": 3, "fixed defect": 6, "reversible defect": 7}
_UNK = "unknown"  # every field defaults to this / to empty -> None

with st.form("patient_form"):
    st.caption("Everything defaults to **unknown** — fill in only the fields you have. "
               "Anything left blank / unknown is reported as missing (the related flag is "
               "skipped, never guessed).")
    st.subheader("Demographics & vitals")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age (years)", 1, 120, value=None, placeholder="unknown")
    sex_label = c2.selectbox("Sex", [_UNK] + list(SEX.keys()), index=0)
    trestbps = c3.number_input("Resting BP (mmHg)", 60, 260, value=None, placeholder="unknown")
    chol = c1.number_input("Cholesterol (mg/dl)", 80, 700, value=None, placeholder="unknown")
    thalach = c2.number_input("Max heart rate achieved (bpm)", 60, 240, value=None, placeholder="unknown")
    fbs = c3.selectbox("Fasting blood sugar > 120 mg/dl", [_UNK, "no", "yes"], index=0)

    st.subheader("Symptoms & ECG")
    d1, d2, d3 = st.columns(3)
    cp_label = d1.selectbox("Chest pain type", [_UNK] + list(CP.keys()), index=0)
    restecg_label = d2.selectbox("Resting ECG result", [_UNK] + list(RESTECG.keys()), index=0)
    exang = d3.selectbox("Exercise-induced angina", [_UNK, "no", "yes"], index=0)

    st.subheader("Exercise test & imaging")
    e1, e2, e3 = st.columns(3)
    oldpeak = e1.number_input("ST depression / oldpeak (mm)", 0.0, 8.0, value=None,
                              step=0.1, placeholder="unknown")
    slope_label = e2.selectbox("ST-segment slope", [_UNK] + list(SLOPE.keys()), index=0)
    ca_choice = e3.selectbox("Major vessels colored (0-3)", [_UNK, "0", "1", "2", "3"], index=0)
    thal_label = e1.selectbox("Thallium result", [_UNK] + list(THAL.keys()), index=0)

    st.divider()
    submitted = st.form_submit_button(
        "Add my data to my case" if _is_patient else "Run patient-data agent",
        type="primary")

# --- Build the raw UCI row (unknown / blank -> None, never guessed) ----------
# .get() returns None for the "unknown" label (not a key in these dicts), and the
# number_inputs return None when left empty. detect_flags_patient guards None on
# every field, so any subset can be missing.
row = {
    "age": age, "sex": SEX.get(sex_label), "cp": CP.get(cp_label), "trestbps": trestbps,
    "chol": chol, "fbs": {"yes": 1, "no": 0}.get(fbs), "restecg": RESTECG.get(restecg_label),
    "thalach": thalach, "exang": {"yes": 1, "no": 0}.get(exang), "oldpeak": oldpeak,
    "slope": SLOPE.get(slope_label),
    "ca": None if ca_choice == _UNK else int(ca_choice), "thal": THAL.get(thal_label),
}

# Context line from the canonical sex LABEL (None if unknown). UCI encodes
# 1=male/0=female (opposite of PTB-XL); passing the label, not the raw int,
# sidesteps the encoding difference (build_patient_context_line accepts labels).
_sex_label = None if row["sex"] is None else ("male" if row["sex"] == 1 else "female")
context_line = build_patient_context_line(age=row["age"], sex=_sex_label)

# Deterministic preview (always, no LLM)
feature_dict = compute_feature_dict(row)
decision = detect_flags_patient(feature_dict)

if not _is_patient:
    left, right = st.columns(2)
    with left:
        st.subheader("Feature dict")
        st.dataframe(
            [{"feature": k, "value": ("—" if v is None else v)} for k, v in feature_dict.items()],
            use_container_width=True, hide_index=True, height=430,
        )
    with right:
        st.subheader("Deterministic flags")
        if decision.deterministic_flags:
            for flag in decision.deterministic_flags:
                st.markdown(f"- ⚠️ `{flag}`")
        else:
            st.markdown("_No deterministic flag raised._")

# Persist "has run" across reruns (a form-submit button is only True on its own
# click; without this, the prompt-lab button would rerun with submitted=False and
# short-circuit the page).
if submitted:
    st.session_state["patient_ran"] = True
ran = st.session_state.get("patient_ran", False)

if not ran:
    st.info("Fill the form and click **Add my data to my case**." if _is_patient
            else "Fill the form and click **Run patient-data agent**.")
    st.stop()

if run_llm:
    if not _is_patient:
        st.subheader("Structured findings")
    if all(v is None for v in row.values()):
        # Nothing was entered - don't generate or save an empty contribution.
        st.warning("All fields are unknown — nothing to analyze, so **no contribution is saved** "
                   "to the case. Enter at least one field and re-run.")
        st.session_state.pop("patient_contribution_obj", None)
        st.session_state.get("contributions", {}).pop("patient", None)
    else:
        if submitted:  # only (re)generate on an explicit submit
            with st.spinner(f"Generating findings with {backend['model_name']}…"):
                try:
                    contribution = generate_patient_contribution(
                        row, llm_call=backend["call"], record_id="ui_patient",
                        model_name=backend["model_name"], patient_info=context_line,
                        max_retries=2)
                except Exception as exc:  # noqa: BLE001
                    hint = ("Is `GROQ_API_KEY` set and valid?" if backend["cloud"]
                            else "Is the Ollama server running with `medgemma:4b` pulled?")
                    st.error(f"LLM call failed ({backend['model_name']}). {hint}\n\nDetails: {exc}")
                    contribution = None
            st.session_state["patient_contribution_obj"] = contribution
            if contribution is not None:
                st.session_state["contributions"]["patient"] = contribution
                st.session_state["patient_context_lines"]["patient"] = context_line

        contribution = st.session_state.get("patient_contribution_obj")
        if contribution is not None and _is_patient:
            st.success("✓ Your data has been added to your case. Open the **Case "
                       "aggregation** page for your report, or **Discuss** to ask questions.")
            st.stop()
        if contribution is not None:
            st.success("Saved to the case → available on the **Case aggregation** page.")
            if contribution.findings:
                st.dataframe([{
                    "feature": f.feature, "value": str(f.value), "confidence": f.confidence,
                    "evidence_ref": f.evidence_ref, "rationale": f.rationale,
                } for f in contribution.findings], use_container_width=True, hide_index=True)
            else:
                st.markdown("_No findings returned._")
            st.markdown("**Merged flags for review:** " + (
                ", ".join(f"`{fl}`" for fl in contribution.flags_for_review) or "_none_"))
else:
    st.info("LLM findings off — deterministic features/flags above are ready, but the Case "
            "aggregation needs a full contribution (enable the checkbox and re-run).")

# --- Prompt lab (experimental, shared helper) ------------------------------
if _is_patient:
    st.stop()  # patient never sees the technical prompt lab
st.divider()
_summary = build_patient_clinical_summary(feature_dict, context_line)
_prompt = build_patient_findings_prompt(_summary, feature_dict, context_line)


def _patient_handle_raw(raw):
    try:
        findings, flags = parse_findings_response(raw, set(feature_dict.keys()))
        st.success(f"Parser accepted the response: {len(findings)} findings.")
        if findings:
            st.dataframe([{
                "feature": f.feature, "value": str(f.value), "confidence": f.confidence,
                "evidence_ref": f.evidence_ref, "rationale": f.rationale,
            } for f in findings], use_container_width=True, hide_index=True)
        st.markdown("**Flags proposed:** " + (
            ", ".join(f"`{fl}`" for fl in flags) or "_none_"))
    except FindingsParseError as exc:
        st.error(f"Parser REJECTED the response (the guard still works): {exc}")


ui_common.prompt_lab(st, "patient", _prompt, backend, _patient_handle_raw,
                     extra_note="Here it validates each `evidence_ref` against the feature keys.")
