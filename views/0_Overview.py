"""
Overview / landing page (shown to every role).

Orients the user, shows the current case, and lets them save/reset it. Page
navigation is provided by the role-aware sidebar built in Home.py (st.navigation).
"""

import streamlit as st

import ui_common

ui_common.init_case_store(st)
role = ui_common.current_role(st) or "researcher"

st.title("🫀 Cardiac Multi-Agent Clinical Pipeline")

if role == "patient":
    st.error("**Research / demo tool on public data — this is NOT a medical diagnosis.** "
             "It produces structured *findings* meant to be discussed with a doctor. "
             "Always consult a qualified clinician.")
    st.markdown("Use the pages on the left to run a check on an **ECG**, an **MRI**, or a short "
                "**clinical form**, then **chat** with the assistant about the results — in plain language.")
elif role == "doctor":
    st.caption("Clinical view — run each modality, aggregate into one traceable impression, export the "
               "case report, and revisit saved cases. Every statement links back to a real finding.")
else:
    st.caption("Researcher view — the full pipeline plus the Models & Validation references, the prompt lab, "
               "backend selection and configuration comparison.")

# --- Current case + reset --------------------------------------------------
st.divider()
head_l, head_reset = st.columns([4, 1])
head_l.subheader("Current case")
contributions = st.session_state.get("contributions", {})
if head_reset.button("🗑️ Reset", disabled=not contributions, use_container_width=True,
                     help="Clear all modalities, Sep-Agent runs and cached results — start a fresh case."):
    ui_common.reset_case(st)
    st.rerun()

LABELS = {"ecg": "ECG", "mri": "MRI", "patient": "Patient data"}
cols = st.columns(len(LABELS))
for col, (agent_id, label) in zip(cols, LABELS.items()):
    present = agent_id in contributions
    n = len(contributions[agent_id].findings) if present else 0
    col.metric(label, "✓ in case" if present else "— empty", f"{n} findings" if present else None)
if not contributions:
    st.info("No modality in the case yet — start on any page in the left sidebar.")

st.divider()
st.caption("Runs locally (MedGemma via Ollama by default). Public research data (PTB-XL / ACDC / UCI); "
           "cross-modality cases are synthetic pairings for architecture validation. The role selector is a "
           "**mode**, not real authentication.")
