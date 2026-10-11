"""
Streamlit UI — MVP (Phase P0) for the clinical multi-agent pipeline.

Single-page ECG explorer: pick a real PTB-XL record, run the preprocessing
pipeline and the ECG agent, and inspect the numeric features, deterministic
flags, the 5-stage pipeline figure, and (optionally) the LLM-generated
structured findings.

Design principle (see the UI plan): this is a *presentation / orchestration*
layer only. It calls the existing src/ functions unchanged - it never
recomputes or duplicates any validated logic.

This is a page of the multipage app; launch the whole app from the project root:
    streamlit run Home.py

Requires:
    pip install streamlit          # the UI itself
    pip install numpy scipy pywavelets matplotlib wfdb pandas
    pip install requests           # local Ollama backend (LLM findings, optional)
    pip install torch              # learned embedding (optional toggle)
An Ollama server with `medgemma:4b` pulled is needed ONLY for the LLM-findings
step; the deterministic analysis (features + flags + figure) works without it.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Shared UI plumbing (adds src/ to sys.path on import, dataset paths, backends).
import ui_common
from ui_common import (
    DATASET_DIR, DATABASE_CSV, EMBEDDING_CHECKPOINT, PTBXL_LEAD_ORDER,
)

# ---------------------------------------------------------------------------
# Imports from the project (src/ is on sys.path via ui_common).
# ---------------------------------------------------------------------------
import wfdb
import ecg_ingest
from ecg_preprocessing import ecg_preprocessing_pipeline
from ecg_agent import (
    generate_ecg_contribution, compute_feature_dict, detect_flags,
    build_clinical_summary, build_findings_prompt,
)
from agent_common import (
    build_patient_context_line, parse_findings_response, FindingsParseError,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_database():
    """Load the PTB-XL record index (a few columns only), cached once."""
    cols = ["ecg_id", "patient_id", "age", "sex", "height", "weight",
            "report", "scp_codes", "filename_lr", "filename_hr"]
    df = pd.read_csv(DATABASE_CSV, usecols=cols)
    return df


@st.cache_resource(show_spinner="Loading the learned ECG embedding model…")
def load_embedding_model():
    """Load the patient-disjoint ECGEmbeddingNet checkpoint (heavy, once)."""
    import torch  # local import so torch is only required when the toggle is on
    from ecg_embedding import ECGEmbeddingNet, load_model_checkpoint
    model = ECGEmbeddingNet()
    load_model_checkpoint(model, EMBEDDING_CHECKPOINT, device="cpu")
    model.eval()
    return model


# Image branch: the GenECG classifier checkpoint + the local multimodal VLMs
# usable for reading the image (the cloud Groq backend is text-only, so it
# cannot see the image and is intentionally NOT offered here).
CLASSIFIER_CHECKPOINT = os.path.join(
    ui_common.PROJECT_ROOT, "results", "ecg_image_pipeline", "genecg_classifier.pt")
_IMAGE_VLM_TAGS = {
    "qwen2.5vl:7b (recommended)": "qwen2.5vl:7b",
    "medgemma:4b (base)": "medgemma:4b",
    "medgemma-ecg fine-tune": "hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M",
}
_SUPERCLASS_NAMES = {
    "CD": "conduction disturbance", "HYP": "hypertrophy", "MI": "myocardial infarction",
    "NORM": "normal ECG", "STTC": "ST-T changes / ischemia",
}


@st.cache_resource(show_spinner="Loading the GenECG image classifier…")
def load_image_classifier():
    """Load the trained GenECG image classifier (heavy, once)."""
    from genecg_classifier_train import load_genecg_classifier
    return load_genecg_classifier(CLASSIFIER_CHECKPOINT, device="cpu")


@st.cache_data(show_spinner=False)
def load_record_signal(filename_rel):
    """Read one PTB-XL record via wfdb; returns (signal (n_samples, 12), fs, lead_names)."""
    record_path = os.path.join(DATASET_DIR, filename_rel)
    signal, fields = wfdb.rdsamp(record_path)
    return signal, float(fields["fs"]), list(fields["sig_name"])


def build_pipeline_figure(results, fs, t_start=0.0, t_end=3.0):
    """Reproduce the 5-stage pipeline figure and RETURN it (for st.pyplot),
    instead of saving/plt.show like ecg_preprocessing.plot_pipeline does."""
    t = np.arange(len(results["raw"])) / fs
    mask = (t >= t_start) & (t <= t_end)
    stages = [
        ("raw", "Raw ECG signal"),
        ("baseline_removed", "After baseline-wander removal"),
        ("powerline_removed", "After powerline-interference removal"),
        ("wavelet_denoised", "After wavelet denoising"),
        ("normalized", "Normalized + R-peaks"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
    for ax, (key, title) in zip(axes, stages):
        ax.plot(t[mask], results[key][mask], linewidth=0.8, color="#0E7C86")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel("mV" if key != "normalized" else "z")
        ax.grid(alpha=0.15)
    r_peaks = results["r_peaks"]
    in_range = r_peaks[(r_peaks / fs >= t_start) & (r_peaks / fs <= t_end)]
    axes[-1].plot(in_range / fs, results["normalized"][in_range], "v",
                  color="#B4322A", markersize=6, label="R-peaks")
    axes[-1].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def _ingest_wfdb_uploads(files):
    """Write uploaded .hea/.dat to a temp dir and load via ecg_ingest.load_wfdb.
    WFDB needs both files together with a matching basename."""
    tmp = tempfile.mkdtemp(prefix="ecg_upload_")
    hea_name = None
    for f in files:
        with open(os.path.join(tmp, f.name), "wb") as fh:
            fh.write(f.getbuffer())
        if f.name.lower().endswith(".hea"):
            hea_name = f.name
    if hea_name is None:
        raise ValueError("No .hea file among the uploads (WFDB needs .hea + .dat).")
    return ecg_ingest.load_wfdb(os.path.join(tmp, hea_name[:-4]))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
# page config is set once by the Home.py router (st.navigation)
st.title("🫀 ECG")
st.caption("Presentation layer over the existing `src/` pipeline — no logic is recomputed here.")

# Shared cross-page "case store": modality contributions accumulate here (ECG
# here, patient-data / MRI on their own pages) and are consumed by the Case
# aggregation page (Sep Agent).
ui_common.init_case_store(st)

# Patient role gets a simplified flow: import data, get a confirmation, then the
# report (Case aggregation) and chat. No technical panels (preprocessing output,
# features, flags, embedding, LLM findings) are shown.
_is_patient = ui_common.current_role(st) == "patient"
if _is_patient:
    st.info("Add your ECG below. You will then get your report on the **Case aggregation** "
            "page and can ask questions on the **Discuss** page.")

if not os.path.exists(DATABASE_CSV):
    st.error(f"PTB-XL index not found at:\n`{DATABASE_CSV}`\n\n"
             "Check that the kaggle/ dataset is present.")
    st.stop()

df = load_database()

# --- Sidebar: source, selection & options ----------------------------------
with st.sidebar:
    st.header("ECG source")
    # The patient imports their own ECG (no PTB-XL dataset browsing).
    _src_opts = (["Upload", "Upload image"] if _is_patient
                 else ["PTB-XL dataset", "Upload", "Upload image"])
    source = st.radio("Source", _src_opts, index=0, label_visibility="collapsed")

    if source == "PTB-XL dataset":
        ecg_id = st.selectbox("PTB-XL record (ecg_id)", options=df["ecg_id"].tolist(),
                              index=0, help="Pick a record from the PTB-XL index.")
        resolution = st.radio("Resolution", ["lr (100 Hz)", "hr (500 Hz)"], index=0,
                              help="lr = records100 (faster); hr = records500.")
    elif source == "Upload":
        up_format = st.radio("File format", ["WFDB (.hea + .dat)", "CSV (samples)"])
        if up_format.startswith("WFDB"):
            up_files = st.file_uploader("Upload the .hea AND .dat", type=["hea", "dat"],
                                        accept_multiple_files=True)
        else:
            up_files = st.file_uploader("Upload a CSV of samples", type=["csv"])
            up_fs = st.number_input("Sampling frequency (Hz)", 1, 5000, 500,
                                    help="A CSV carries no header for this — required.")
            up_unit = st.selectbox("Amplitude unit", ["mV", "uV"])
            up_orient = st.selectbox(
                "Orientation",
                ["auto (longer axis = samples)", "samples in rows", "leads in rows"])
            up_order = st.checkbox(
                "12 columns are in PTB-XL lead order (I, II, III, aVR, aVL, aVF, V1-V6)",
                value=True)
        st.caption("Optional demographics (context only):")
        up_age = st.number_input("Age", 0, 120, 0, help="0 = unknown")
        up_sex = st.selectbox("Sex", ["unknown", "male", "female"])

    else:  # Upload image
        img_file = st.file_uploader("Upload an ECG image", type=["png", "jpg", "jpeg"])
        if _is_patient:
            # Simplified: sensible defaults, no technical toggles.
            img_vlm_label = list(_IMAGE_VLM_TAGS)[0]
            img_use_classifier = True
            img_run_llm = True
            img_ecg_gate = True
            img_age = st.number_input("Age", 0, 120, 0, help="0 = unknown", key="img_age")
            img_sex = st.selectbox("Sex", ["unknown", "male", "female"], key="img_sex")
            run_img_btn = st.button("Add my ECG image to my case", type="primary",
                                    use_container_width=True)
        else:
            img_vlm_label = st.selectbox(
                "Vision model (local, multimodal)", list(_IMAGE_VLM_TAGS),
                help="The image branch needs a multimodal model. The cloud Groq backend "
                     "is text-only and cannot see the image, so it is not offered here.")
            img_use_classifier = st.toggle(
                "GenECG image classifier (probabilities)", value=True,
                help="Adds calibrated pathology_<CLASS>_probability findings + flags "
                     "(discriminative model, validated macro F1 0.64, MI 0.62 - real "
                     "probabilities, unlike the VLM's self-reported confidence).")
            img_run_llm = st.toggle(
                "VLM structured findings", value=True,
                help="Read the image with the vision model. Slower. The classifier "
                     "probabilities + quality report are shown even without it.")
            img_ecg_gate = st.toggle(
                "Check it is really an ECG (CLIP gate)", value=True,
                help="Zero-shot CLIP content check run BEFORE anything else. A non-ECG "
                     "image (photo, MRI, diagram, document) is rejected instead of being "
                     "'read' as an ECG. On by default here (a human may upload anything); "
                     "uncheck to force analysis. First use downloads the CLIP model (~0.6 GB).")
            st.caption("Optional demographics (context only):")
            img_age = st.number_input("Age", 0, 120, 0, help="0 = unknown", key="img_age")
            img_sex = st.selectbox("Sex", ["unknown", "male", "female"], key="img_sex")
            run_img_btn = st.button("Analyze image", type="primary", use_container_width=True)

    if source != "Upload image":
        if _is_patient:
            analysis_lead = PTBXL_LEAD_ORDER[1]  # default to lead II; no technical control shown
        else:
            analysis_lead = st.selectbox(
                "Single-lead pipeline lead", PTBXL_LEAD_ORDER, index=1,
                help="The core pipeline is single-lead. Lead II is the usual default "
                     "(falls back to the first lead if an upload lacks it).")

        if _is_patient:
            # Simplified: no technical toggles; always produce a contribution with
            # the local model so the report can be generated. The learned embedding
            # is on by default for a richer report (auto-disabled if the upload is
            # not 12 leads in PTB-XL order or the checkpoint is missing).
            use_embedding = True
            run_llm = True
            backend = ui_common.get_llm_backends()["MedGemma 4B (local)"]
            run_btn = st.button("Add my ECG to my case", type="primary",
                                use_container_width=True)
        else:
            st.divider()
            st.header("Analysis options")
            use_embedding = st.toggle(
                "Learned embedding (torch)", value=False,
                help="Diagnostic-superclass probabilities from ECGEmbeddingNet. Needs all "
                     "12 leads in PTB-XL order; auto-disabled otherwise on uploads.")
            run_llm = st.toggle(
                "LLM structured findings", value=False,
                help="Generate structured findings with an LLM. Slower. "
                     "Deterministic analysis works without it.")
            backend = ui_common.backend_selector(st, run_llm, key_prefix="ecg")
            run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ===========================================================================
# Image branch (source = "Upload image"): self-contained, goes DIRECTLY through
# the VLM image agent + GenECG classifier. Touches NONE of the signal pipeline
# (no ecg_preprocessing / embedding / 5-stage figure) and st.stop()s at the end,
# so the signal code below is never reached for an image upload.
# ===========================================================================
if source == "Upload image":
    if run_img_btn:
        st.session_state["ecg_image_ran"] = True
    if not st.session_state.get("ecg_image_ran", False):
        st.info("Upload an ECG image (PNG/JPG) and click **Analyze image** in the sidebar.")
        st.stop()
    if not img_file:
        st.warning("Upload an image first.")
        st.stop()

    from adaptive_ecg_pipeline import process_ecg_image
    import ecg_image_agent

    # persist the upload to a temp file (the agent + classifier take a path)
    _suffix = os.path.splitext(img_file.name)[1] or ".png"
    _tmp_dir = tempfile.mkdtemp(prefix="ecg_img_")
    img_path = os.path.join(_tmp_dir, "upload" + _suffix)
    with open(img_path, "wb") as fh:
        fh.write(img_file.getbuffer())

    context_line = build_patient_context_line(
        age=img_age if img_age > 0 else None,
        sex=None if img_sex == "unknown" else img_sex,
    )
    st.image(img_path, caption=f"Uploaded: {img_file.name}", use_container_width=True)
    if context_line:
        st.caption(f"**Context** · {context_line}")

    # --- Content gate (optional): reject a non-ECG image BEFORE any real work --
    if img_ecg_gate:
        try:
            from ecg_image_gate import is_ecg_image
            _is_ecg, _gate_info = is_ecg_image(img_path)
        except Exception as exc:  # noqa: BLE001 - fail-open, never block a real ECG
            _is_ecg, _gate_info = True, {"gate_error": str(exc), "skipped": True}
        _p = _gate_info.get("ecg_probability")
        if not _is_ecg:
            st.error(
                f"🚫 This image does not look like an ECG "
                f"(P(ECG) = {_p:.2f} < 0.50). It was **not** analyzed as an ECG. "
                "Upload a 12-lead ECG image, or turn off the CLIP gate if this is a "
                "genuine ECG the check misread.")
            st.stop()
        if _gate_info.get("skipped"):
            st.warning(f"ECG content gate skipped (model unavailable): {_gate_info.get('gate_error')}. "
                       "Proceeding without the check.")
        else:
            st.caption(f"✅ ECG content gate passed · P(ECG) = {_p:.2f}")

    # --- Deterministic panel (always): adaptive preprocessing + classifier -----
    try:
        _b64, quality_report = process_ecg_image(img_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the image: {exc}")
        st.stop()
    quality_features = ecg_image_agent.compute_image_quality_features(quality_report)
    image_decision = ecg_image_agent.detect_image_flags(quality_features)

    classifier_model, pathology_features, pathology_flags = None, {}, []
    if img_use_classifier:
        if not os.path.exists(CLASSIFIER_CHECKPOINT):
            st.warning(f"Classifier checkpoint not found at `{CLASSIFIER_CHECKPOINT}` — "
                       "run `python src/genecg_classifier_train.py` first; continuing without it.")
        else:
            try:
                classifier_model = load_image_classifier()
                pathology_features = ecg_image_agent.compute_pathology_features(
                    img_path, classifier_model)
                pathology_flags = ecg_image_agent.detect_pathology_flags(pathology_features)
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not run the image classifier — continuing without it. ({exc})")
                classifier_model = None

    # Patient: run the full image analysis, save it, confirm, and stop before any
    # technical panel (classifier table, quality report, VLM findings).
    if _is_patient:
        ecg_image_agent.DEFAULT_IMAGE_MODEL_TAG = _IMAGE_VLM_TAGS[img_vlm_label]
        with st.spinner("Reading your ECG image…"):
            try:
                contribution = ecg_image_agent.generate_image_ecg_contribution(
                    img_path, record_id="upload_image", patient_info=context_line,
                    classifier_model=classifier_model, ecg_gate=img_ecg_gate, max_retries=2)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not analyze the image: {exc}")
                st.stop()
        if contribution is not None and "not_an_ecg_image" not in contribution.flags_for_review:
            st.session_state["contributions"]["ecg_image"] = contribution
            st.session_state["patient_context_lines"]["ecg_image"] = context_line
            st.success("✓ Your ECG image has been added to your case. Open the **Case "
                       "aggregation** page for your report, or **Discuss** to ask questions.")
        else:
            st.warning("This image was not recognised as an ECG. Please upload a clear "
                       "photo or scan of a 12-lead ECG.")
        st.stop()

    icol, fcol = st.columns(2)
    with icol:
        st.subheader("Classifier probabilities")
        if pathology_features:
            prob_df = pd.DataFrame([
                {"class": f"{c} ({_SUPERCLASS_NAMES[c]})",
                 "probability": pathology_features[f"pathology_{c}_probability"]}
                for c in _SUPERCLASS_NAMES
            ]).sort_values("probability", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True,
                         column_config={"probability": st.column_config.ProgressColumn(
                             "probability", min_value=0.0, max_value=1.0, format="%.3f")})
            st.caption("Real per-class probabilities (sigmoid) — not the VLM's self-reported "
                       "confidence.")
        else:
            st.markdown("_Classifier off or unavailable._")
    with fcol:
        st.subheader("Deterministic flags")
        _all_flags = sorted(set(image_decision.deterministic_flags) | set(pathology_flags))
        if _all_flags:
            for fl in _all_flags:
                st.markdown(f"- ⚠️ `{fl}`")
        else:
            st.markdown("_No deterministic flag raised._")

    with st.expander("Image quality report (adaptive preprocessing)"):
        st.json(quality_report)

    # --- VLM structured findings (optional) + save to case ---------------------
    if img_run_llm:
        st.subheader("VLM structured findings")
        _tag = _IMAGE_VLM_TAGS[img_vlm_label]
        if run_img_btn:
            ecg_image_agent.DEFAULT_IMAGE_MODEL_TAG = _tag  # default llm_call attaches the image
            with st.spinner(f"Reading the image with {_tag}"
                            f"{' + classifier' if classifier_model is not None else ''}…"):
                try:
                    contribution = ecg_image_agent.generate_image_ecg_contribution(
                        img_path, record_id="upload_image", patient_info=context_line,
                        classifier_model=classifier_model, max_retries=2)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"VLM call failed ({_tag}). Is Ollama running with that model "
                             f"pulled?\n\nDetails: {exc}")
                    contribution = None
            st.session_state["ecg_image_contribution_obj"] = contribution
            if contribution is not None:
                st.session_state["contributions"]["ecg_image"] = contribution
                st.session_state["patient_context_lines"]["ecg_image"] = context_line

        contribution = st.session_state.get("ecg_image_contribution_obj")
        if contribution is not None:
            st.caption(f"Model used: `{contribution.trace.get('model', _tag)}`")
            st.success("Saved to the case → available on the **Case aggregation** page.")
            if contribution.findings:
                st.dataframe(pd.DataFrame([{
                    "feature": f.feature, "value": str(f.value), "confidence": f.confidence,
                    "evidence_ref": f.evidence_ref, "rationale": f.rationale,
                } for f in contribution.findings]), use_container_width=True, hide_index=True)
            else:
                st.markdown("_No findings returned by the VLM (the classifier panel above still "
                            "stands)._")
            st.markdown("**Merged flags for review:** " + (
                ", ".join(f"`{fl}`" for fl in contribution.flags_for_review) or "_none_"))
    else:
        st.caption("Enable **VLM structured findings** to also read the image with the vision "
                   "model and save a contribution to the case.")

    st.stop()


# Persist "has run" across reruns: a button is only True on its own click, so
# without this, clicking any OTHER widget (a toggle, the prompt-lab button) would
# rerun the script with run_btn=False and short-circuit the whole page.
if run_btn:
    st.session_state["ecg_ran"] = True
ran = st.session_state.get("ecg_ran", False)

# --- Acquire the signal (dataset or upload) --------------------------------
report = None  # ecg_ingest.IngestReport for uploads (guardrails)

if source == "PTB-XL dataset":
    row = df[df["ecg_id"] == ecg_id].iloc[0]
    res_key = "filename_lr" if resolution.startswith("lr") else "filename_hr"
    filename_rel = row[res_key]
    context_line = build_patient_context_line(
        age=row["age"] if pd.notna(row["age"]) else None,
        sex=int(row["sex"]) if pd.notna(row["sex"]) else None,
        height=row["height"] if pd.notna(row["height"]) else None,
        weight=row["weight"] if pd.notna(row["weight"]) else None,
    )
    meta_cols = st.columns([2, 2, 3])
    meta_cols[0].metric("ecg_id", int(ecg_id))
    meta_cols[1].metric("patient_id", int(row["patient_id"]) if pd.notna(row["patient_id"]) else "—")
    meta_cols[2].markdown(f"**Context** · {context_line}")
    with st.expander("Cardiologist report & SCP codes (ground truth)"):
        st.write("**Report:**", row["report"] if pd.notna(row["report"]) else "—")
        st.write("**scp_codes:**", row["scp_codes"])

    if not ran:
        st.info("Set the options in the sidebar and click **Run analysis**.")
        st.stop()
    try:
        signal, fs, lead_names = load_record_signal(filename_rel)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the record signal (`{filename_rel}`): {exc}")
        st.stop()
    lead_names = [str(n).upper() for n in lead_names]
    embedding_allowed, multilead_enabled = True, True
    record_id = str(ecg_id)

else:  # Upload
    st.info("**Expectations & limits** — WFDB (.hea + .dat) is the clean case: fs, "
            "units and lead order come from the header. For CSV you must set fs + unit "
            "+ lead order. The core pipeline is single-lead; the embedding and "
            "voltage criteria need all 12 leads in PTB-XL order and real mV.")
    if not ran:
        st.stop()
    if not up_files:
        st.warning("Upload a file first.")
        st.stop()
    try:
        if up_format.startswith("WFDB"):
            ingested = _ingest_wfdb_uploads(up_files)
        else:
            orient = {"auto (longer axis = samples)": "auto",
                      "samples in rows": "samples_rows",
                      "leads in rows": "leads_rows"}[up_orient]
            ingested = ecg_ingest.load_csv(up_files, up_fs, unit=up_unit,
                                           orientation=orient,
                                           leads_in_ptbxl_order=up_order)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the upload: {exc}")
        st.stop()

    report = ecg_ingest.validate(ingested)
    if not report.ok:
        for e in report.hard_errors:
            st.error(f"⛔ {e}")
        st.stop()
    for w in report.warnings:
        st.warning(f"⚠️ {w}")
    with st.expander("Ingestion details"):
        for n in ingested.notes:
            st.write("•", n)

    signal, fs, lead_names = ingested.signal, ingested.fs, ingested.lead_names
    embedding_allowed = report.enabled["embedding"]
    multilead_enabled = report.enabled["multilead"]
    context_line = build_patient_context_line(
        age=up_age if up_age > 0 else None,
        sex=None if up_sex == "unknown" else up_sex,
    )
    st.caption(f"**Context** · {context_line or 'age/sex not provided'} · "
               f"{signal.shape[0]} samples × {signal.shape[1]} leads @ {fs:g} Hz")
    record_id = "upload"

# --- Map the analysis lead to a column (fallback to the first) -------------
try:
    lead_idx = lead_names.index(analysis_lead.upper())
except ValueError:
    lead_idx = 1 if len(lead_names) > 1 else 0
single_lead = signal[:, lead_idx]
# Canonical multi-lead shape for compute_feature_dict / the agent is
# (n_samples, 12), NOT transposed (the embedding transposes internally). Pass it
# only when the guardrails confirm 12 leads in PTB-XL order.
multilead = signal if multilead_enabled else None

# --- Run the preprocessing pipeline ----------------------------------------
with st.spinner("Running the preprocessing pipeline…"):
    results = ecg_preprocessing_pipeline(single_lead, fs)

# --- Deterministic features & flags (always, no LLM) ------------------------
embedding_model = None
if use_embedding and not embedding_allowed:
    st.info("Learned embedding requested but **disabled for this input** — it needs "
            "all 12 leads in PTB-XL order (see the ingestion warnings above).")
elif use_embedding:
    if not os.path.exists(EMBEDDING_CHECKPOINT):
        st.warning(f"Embedding checkpoint not found at `{EMBEDDING_CHECKPOINT}` — "
                   "continuing without it.")
    else:
        try:
            embedding_model = load_embedding_model()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not load the embedding model — continuing without it. ({exc})")
            embedding_model = None

feature_dict = compute_feature_dict(results, fs, multilead, embedding_model)
decision = detect_flags(feature_dict)

if not _is_patient:
    left, right = st.columns(2)
    with left:
        st.subheader("Numeric features")
        feat_df = pd.DataFrame(
            [(k, v) for k, v in feature_dict.items()], columns=["feature", "value"]
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True, height=430)
    with right:
        st.subheader("Deterministic flags")
        if decision.deterministic_flags:
            for flag in decision.deterministic_flags:
                st.markdown(f"- ⚠️ `{flag}`")
        else:
            st.markdown("_No deterministic flag raised._")
        if decision.skip_llm_call:
            st.error(f"LLM step would be skipped: {decision.reason}")

# --- Pipeline figure --------------------------------------------------------
# The 5-stage signal-processing figure is a technical view; hide it for the
# patient role (kept for doctor/researcher).
if ui_common.current_role(st) != "patient":
    st.subheader("Preprocessing pipeline")
    fig = build_pipeline_figure(results, fs, t_start=0.0, t_end=3.0)
    st.pyplot(fig, use_container_width=True)

# --- LLM structured findings (optional) ------------------------------------
if run_llm:
    if not _is_patient:
        st.subheader("LLM structured findings")
    if decision.skip_llm_call:
        st.info("Skipped — not enough signal in this record (see flag above).")
    else:
        # Only (re)generate on an explicit "Run analysis" click; on other reruns
        # (a toggle, the prompt-lab button) reuse the stored contribution so the
        # LLM isn't called again.
        if run_btn:
            with st.spinner(f"Generating findings with {backend['model_name']}…"):
                try:
                    contribution = generate_ecg_contribution(
                        results, fs,
                        llm_call=backend["call"],
                        record_id=record_id,
                        model_name=backend["model_name"],
                        patient_info=context_line,
                        multilead_signal=multilead,
                        embedding_model=embedding_model,
                        max_retries=2,
                    )
                except Exception as exc:  # noqa: BLE001
                    hint = ("Is `GROQ_API_KEY` set and valid?" if backend["cloud"]
                            else "Is the Ollama server running with `medgemma:4b` pulled?")
                    st.error(f"LLM call failed ({backend['model_name']}). {hint}\n\nDetails: {exc}")
                    contribution = None
            st.session_state["ecg_contribution_obj"] = contribution
            if contribution is not None:
                st.session_state["contributions"]["ecg"] = contribution
                st.session_state["patient_context_lines"]["ecg"] = context_line

        contribution = st.session_state.get("ecg_contribution_obj")
        if contribution is not None and _is_patient:
            st.success("✓ Your ECG has been added to your case. Open the **Case "
                       "aggregation** page for your report, or **Discuss** to ask questions.")
            st.stop()
        if contribution is not None:
            st.caption(f"Backend used: `{contribution.trace.get('model', backend['model_name'])}`")
            st.success("Saved to the case → available on the **Case aggregation** page.")
            if contribution.findings:
                st.dataframe(pd.DataFrame([{
                    "feature": f.feature, "value": str(f.value), "confidence": f.confidence,
                    "evidence_ref": f.evidence_ref, "rationale": f.rationale,
                } for f in contribution.findings]), use_container_width=True, hide_index=True)
            else:
                st.markdown("_No findings returned._")
            st.markdown("**Merged flags for review:** " + (
                ", ".join(f"`{fl}`" for fl in contribution.flags_for_review) or "_none_"))
            with st.expander("Raw contribution trace"):
                st.json(contribution.trace)
else:
    st.caption("Enable **LLM structured findings** in the sidebar to also generate the "
               "MedGemma findings for this record.")

# --- Prompt lab (experimental, shared helper) ------------------------------
st.divider()
_clinical_summary = build_clinical_summary(results, fs, context_line, multilead, embedding_model)
_built_prompt = build_findings_prompt(_clinical_summary, feature_dict, context_line)


def _ecg_handle_raw(raw):
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


ui_common.prompt_lab(st, "ecg", _built_prompt, backend, _ecg_handle_raw,
                     extra_note="Here it validates each `evidence_ref` against the feature keys.")
