"""
Shared helpers for the Streamlit UI (Home.py router + views/).

Each Streamlit page runs as its own script, so this module centralizes the
things every page needs - sys.path wiring to reach src/, dataset paths, the
selectable LLM backends, and the cross-page "case store" - instead of each
page duplicating them. Importing it has the side effect of putting src/ on
sys.path (safe/idempotent).

Design note: this is UI plumbing only. No clinical/analysis logic lives here;
it just wires the existing src/ functions into the pages.
"""

import os
import sys

# --- Paths: project root -> src on sys.path (idempotent) -------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "kaggle",
    "archive",
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
)
DATABASE_CSV = os.path.join(DATASET_DIR, "ptbxl_database.csv")
EMBEDDING_CHECKPOINT = os.path.join(
    PROJECT_ROOT, "results", "ecg_embedding", "embedding_stratfold.best.pt"
)
ACDC_DIR = os.path.join(PROJECT_ROOT, "ACDC", "database")
RESULTS_UI_DIR = os.path.join(PROJECT_ROOT, "results", "ui_sessions")

# Standard PTB-XL 12-lead order (verified against a real header).
PTBXL_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"]

# --- Selectable LLM backends -----------------------------------------------
# The multi-lead + embedding feature set produces ~32 numeric features, so the
# model emits ~32 findings - more than the default budget (2048) can hold,
# truncating the JSON. Give the findings call a larger budget.
LLM_MAX_TOKENS = 4096


def get_llm_backends():
    """Return the ordered dict of selectable findings backends.

    MedGemma runs locally (no key, data stays on the machine); GPT goes through
    Groq's free cloud API (needs GROQ_API_KEY - the prompt leaves the machine,
    fine for public PTB-XL/ACDC research data, NOT for private data).
    gpt-oss-120b is a reasoning model whose hidden reasoning tokens spend the
    completion budget, so it needs reasoning_effort="low" + a larger max_tokens.
    """
    from agent_common import generate_report_medgemma, generate_report_groq
    return {
        "MedGemma 4B (local)": {
            "cloud": False,
            "model_name": f"ollama:medgemma:4b (num_predict={LLM_MAX_TOKENS})",
            # num_ctx raised to 16384 so heavy aggregations (RAG+CoT+KG, 3 modalities)
            # do not truncate the JSON; still small enough to stay on a 6 GB GPU.
            # Requires the Ollama server default to allow it: OLLAMA_CONTEXT_LENGTH=16384.
            "call": lambda p: generate_report_medgemma(p, num_predict=LLM_MAX_TOKENS, num_ctx=16384),
        },
        "GPT (gpt-oss-120b · Groq)": {
            "cloud": True,
            "model_name": "groq:openai/gpt-oss-120b",
            "call": lambda p: generate_report_groq(
                p, model="openai/gpt-oss-120b",
                max_tokens=LLM_MAX_TOKENS, reasoning_effort="low",
            ),
        },
    }


# --- Roles / access (fictitious demo passwords — a role SWITCH, not security) --
ROLES = {
    "patient":    {"password": "patient2026",    "label": "Patient",    "icon": "🧑"},
    "doctor":     {"password": "doctor2026",      "label": "Doctor",     "icon": "🩺"},
    "researcher": {"password": "researcher2026",  "label": "Researcher", "icon": "🔬"},
}


def current_role(st):
    return st.session_state.get("role")


def is_researcher(st):
    return st.session_state.get("role") == "researcher"


def role_is(st, *roles):
    return st.session_state.get("role") in roles


def init_case_store(st):
    """Initialize the shared cross-page case store on st.session_state."""
    st.session_state.setdefault("contributions", {})          # agent_id -> AgentContribution
    st.session_state.setdefault("patient_context_lines", {})  # agent_id -> str
    st.session_state.setdefault("figure_paths", {})           # agent_id -> PNG path


def reset_case(st):
    """Clear the whole case: all modality contributions, the Sep-Agent runs, the
    cached per-page results, and the 'has run' flags — so every page returns to
    its initial 'click Run' state and the case store is empty."""
    for k in ("contributions", "patient_context_lines", "figure_paths"):
        st.session_state[k] = {}
    for k in ("sep_runs", "sep_decision", "sep_config",
              "ecg_contribution_obj", "patient_contribution_obj", "mri_contribution_obj",
              "ecg_ran", "patient_ran", "mri_ran", "sep_ran",
              "chat_history", "mri_ed_b64"):
        st.session_state.pop(k, None)


def prompt_lab(st, key_prefix, built_prompt, backend, handle_raw, extra_note=""):
    """Shared 'prompt lab' expander for every text agent (ECG / patient / Sep).

    Shows the EXACT prompt that would be sent; an edit toggle lets the user change
    it and run it via the selected backend. The agent-specific parsing + result
    display is delegated to `handle_raw(raw_text)` - which must run the SAME
    validating parser the agent uses (so a fabricated citation is still rejected)
    and display the outcome; it should handle its own parse errors. Results here
    are experimental and NEVER saved to the case. The guard lives in the parser,
    not the prompt, so editing the prompt cannot disable it.
    """
    with st.expander("🧪 Prompt lab — see / edit the exact prompt (experimental, isolated)"):
        st.caption("Transparency + a testbench for the model's limits. The anti-hallucination "
                   "**parser stays ON**, and results here are **NOT** saved as a canonical "
                   "contribution. " + extra_note)
        edit = st.toggle("Edit mode", value=False, key=f"{key_prefix}_lab_edit")
        if edit:
            fp = str(hash(built_prompt))
            if st.session_state.get(f"{key_prefix}_lab_fp") != fp:  # re-seed on prompt change
                st.session_state[f"{key_prefix}_lab_text"] = built_prompt
                st.session_state[f"{key_prefix}_lab_fp"] = fp
            prompt_text = st.text_area("Prompt (editable)", key=f"{key_prefix}_lab_text", height=320)
            if st.button("↺ Reset to generated prompt", key=f"{key_prefix}_lab_reset"):
                st.session_state[f"{key_prefix}_lab_text"] = built_prompt
                st.rerun()
        else:
            st.code(built_prompt)
            prompt_text = built_prompt

        if st.button("▶ Run this prompt", key=f"{key_prefix}_lab_run", type="primary"):
            raw = None
            with st.spinner(f"Running with {backend['model_name']}…"):
                try:
                    raw = backend["call"](prompt_text)
                except Exception as exc:  # noqa: BLE001 - backend/transport failure
                    st.error(f"Call failed: {exc}")
            if raw is not None:
                edited = prompt_text.strip() != built_prompt.strip()
                st.info("🧪 Lab run — isolated, **not saved to the case**. "
                        + ("Prompt was **edited**." if edited
                           else "Prompt **unchanged** from the generated one."))
                handle_raw(raw)
                with st.expander("Raw model response"):
                    st.code(raw)


def backend_selector(st, run_enabled, key_prefix=""):
    """Render the LLM-backend selectbox + cloud caption/warning; return the
    chosen backend dict. Shared by every page that calls an LLM."""
    backends = get_llm_backends()
    name = st.selectbox(
        "LLM backend", list(backends.keys()), index=0,
        disabled=not run_enabled, key=f"{key_prefix}_backend",
        help="MedGemma runs locally (no key, data stays on the machine). "
             "GPT uses Groq's free cloud API (needs GROQ_API_KEY; the prompt "
             "leaves the machine — fine for public research data).",
    )
    backend = backends[name]
    if run_enabled and backend["cloud"]:
        if os.environ.get("GROQ_API_KEY"):
            st.caption("☁️ Cloud backend — the prompt leaves the machine "
                       "(OK for public research data).")
        else:
            st.warning("`GROQ_API_KEY` not set — this cloud backend will fail. "
                       "Set the env var or pick MedGemma (local).")
    return backend
