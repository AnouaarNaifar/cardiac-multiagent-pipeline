"""
Streamlit entry — role-based router.

Shows a login gate (role + a FICTITIOUS demo password), then builds a role-aware
page list with st.navigation:
  - patient:   overview + simplified check-up + chat
  - doctor:     overview + modalities + case aggregation + history + chat
  - researcher: everything (+ Models, Validation, prompt lab, backend selection)

The role selector is a UX MODE, not real authentication, and everything runs on
public/synthetic research data. Run: streamlit run Home.py
"""

import streamlit as st

import ui_common

st.set_page_config(page_title="Cardiac Multi-Agent Pipeline", page_icon="🫀", layout="wide")

# Edge/IE add their OWN password-reveal button (::-ms-reveal) on top of Streamlit's
# eye icon -> a duplicate eye on password fields. Hide the browser-native one so only
# Streamlit's remains.
st.markdown(
    "<style>input::-ms-reveal, input::-ms-clear { display: none; }</style>",
    unsafe_allow_html=True,
)

# --- Login gate -------------------------------------------------------------
if not ui_common.current_role(st):
    st.title("🫀 Cardiac Multi-Agent Clinical Pipeline")
    st.caption("Demo access — pick a role and enter its password. This is a role **switch** for a research "
               "prototype on public/synthetic data, **not** real authentication or data security.")
    labels = {v["label"]: k for k, v in ui_common.ROLES.items()}
    chosen = st.selectbox("Role", list(labels.keys()))
    role_key = labels[chosen]
    pwd = st.text_input("Password", type="password")
    st.caption("Demo passwords — patient: `patient2026` · doctor: `doctor2026` · researcher: `researcher2026`")
    if st.button("Enter", type="primary"):
        if pwd == ui_common.ROLES[role_key]["password"]:
            st.session_state["role"] = role_key
            st.rerun()
        else:
            st.error("Wrong password for this role.")
    st.stop()

role = ui_common.current_role(st)

# --- Pages (st.Page over the existing files) --------------------------------
overview = st.Page("views/0_Overview.py", title="Overview", icon="🏠", default=True)
ecg = st.Page("views/1_ECG.py", title="ECG", icon="🫀")
patient_data = st.Page("views/2_Patient_data.py", title="Patient data", icon="🧾")
mri = st.Page("views/3_MRI.py", title="MRI", icon="🖼️")
aggregation = st.Page("views/4_Case_aggregation.py", title="Case aggregation", icon="📋")
validation = st.Page("views/5_Validation.py", title="Validation", icon="📊")
models = st.Page("views/6_Models.py", title="Models", icon="🧠")
chat = st.Page("views/8_Chat.py", title="Discuss", icon="💬")

if role == "researcher":
    pages = [overview, ecg, patient_data, mri, aggregation, chat, validation, models]
elif role == "doctor":
    pages = [overview, ecg, patient_data, mri, aggregation, chat]
else:  # patient — enter their own data (ECG/MRI/form), get a plain final report, then chat
    pages = [overview, ecg, patient_data, mri, aggregation, chat]

# --- Run the selected page FIRST (its sidebar widgets render at the top) -----
st.navigation(pages).run()

# --- Role badge + logout, pushed to the BOTTOM of the sidebar ----------------
# Pages call st.stop() before finishing, so rendering this after pg.run() is not
# reliable; instead it is rendered here (into the sidebar) and CSS flex-order
# moves this last-added block below the page's own sidebar controls.
st.markdown(
    """<style>
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"]{
        display:flex; flex-direction:column; min-height: 82vh;}
    #role-footer{ order: 999; margin-top: auto; }
    </style>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("<div id='role-footer'></div>", unsafe_allow_html=True)
    st.divider()
    info = ui_common.ROLES[role]
    st.caption(f"{info['icon']} Signed in as **{info['label']}**")
    if st.button("🔓 Logout", use_container_width=True):
        # Total reset: drop the role AND the whole case (contributions, Sep runs,
        # cached per-page results, chat history, uploaded-image state) so nothing
        # from one role's session leaks into the next (e.g. lingering MRI results
        # after switching patient -> doctor). Cached models (@st.cache_resource)
        # live outside session_state and are preserved, so this is cheap.
        st.session_state.clear()
        st.rerun()
