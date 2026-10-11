"""
Streamlit page — MRI agent (ACDC patient or raw NIfTI upload).

Two things happen here:
  1. From a raw volume ALONE (no expert mask), the trained YOLO+U-Net predicts a
     segmentation mask over the full ED and ES stacks -> chamber volumes +
     ejection fraction (via ui_mri) + a pathology-classifier read. This is the
     real-deployment path (a real MRI has no expert mask), and it gives the EF
     that mri_agent needs, so an upload produces a full MRI contribution.
  2. "Pipeline-utility check": if an expert mask is also available (always for
     ACDC, optional on upload), predicted EF is compared to expert EF and the
     per-structure Dice is shown - does our pipeline actually reconstruct the
     mask well enough to trust its EF?

Backend note: the LLM findings step attaches the image, so it uses local
MedGemma (multimodal) only - the Groq cloud backend is text-only.
"""

import base64
import os
import re
import tempfile

import numpy as np
import streamlit as st

import ui_common
import ui_mri
from mri_preprocessing import load_patient_info, load_nifti_volume
from mri_agent import generate_mri_contribution
from agent_common import build_patient_context_line

# page config is set once by the Home.py router (st.navigation)
st.title("🖼️ MRI")
st.caption("Predict the mask from the raw volume (YOLO+U-Net) → EF + pathology; "
           "optionally compare predicted vs expert mask to check the pipeline.")
ui_common.init_case_store(st)


@st.cache_resource(show_spinner="Loading YOLO + U-Net segmentation models…")
def load_seg_models():
    import torch
    import mri_pipeline_validation as mpv
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model, unet_model = mpv._load_models(device)
    return yolo_model, unet_model, device


@st.cache_resource(show_spinner="Loading pathology classifier…")
def load_classifier(device):
    import torch
    from mri_classifier_train import PathologyClassifier
    model = PathologyClassifier().to(device)
    model.load_state_dict(torch.load("results/mri_classifier_training/best.pt", map_location=device))
    model.eval()
    return model


@st.cache_data(show_spinner="Segmenting the stack with YOLO + U-Net…")
def cached_predict_mask(volume, _yolo, _unet, _device):
    """Cache the (expensive) per-slice YOLO+U-Net mask prediction by the input
    volume, so re-running on navigation is instant. The models are passed with a
    leading underscore so Streamlit does NOT try to hash them (only `volume` is
    hashed, which is what actually determines the output)."""
    return ui_mri.predict_mask_volume(volume, _yolo, _unet, _device)


def _load_nifti_upload(uploaded):
    """Write an uploaded NIfTI to a temp file and load it -> (volume, voxel)."""
    suffix = ".nii.gz" if uploaded.name.lower().endswith(".gz") else ".nii"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.close()
    return load_nifti_volume(tmp.name)


ACDC_TEST = os.path.join(ui_common.ACDC_DIR, "testing")

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("MRI source")
    source = st.radio("Source", ["ACDC patient", "Upload NIfTI"], label_visibility="collapsed")

    ed_up = es_up = ed_mask_up = es_mask_up = None
    if source == "ACDC patient":
        patients = sorted(d for d in os.listdir(ACDC_TEST)
                          if d.startswith("patient")
                          and os.path.isdir(os.path.join(ACDC_TEST, d))) if os.path.isdir(ACDC_TEST) else []
        patient = st.selectbox("ACDC patient", patients, index=0) if patients else None
    else:
        st.caption("Volumes (required for EF — need both phases):")
        ed_up = st.file_uploader("ED volume (.nii/.nii.gz)", type=["nii", "gz"])
        es_up = st.file_uploader("ES volume (.nii/.nii.gz)", type=["nii", "gz"])
        st.caption("Expert masks (optional — to compare):")
        ed_mask_up = st.file_uploader("ED mask", type=["nii", "gz"], key="edm")
        es_mask_up = st.file_uploader("ES mask", type=["nii", "gz"], key="esm")
        info_up = st.file_uploader(
            "Info.cfg (optional — auto-extracts patient ID, pathology group, ED/ES, height/weight)",
            type=["cfg"], key="infocfg")
        up_age = st.number_input("Age (optional)", 0, 120, 0)
        up_sex = st.selectbox("Sex (optional)", ["unknown", "male", "female"])

    if ui_common.current_role(st) == "patient":
        run_llm = True
        run_btn = st.button("Add my MRI to my case", type="primary", use_container_width=True)
    else:
        run_llm = st.toggle("LLM findings (MedGemma) + add to case", value=False,
                            help="Also run the MRI agent (image-grounded findings) and store "
                                 "the contribution for the Case aggregation page. Slower.")
        run_btn = st.button("Run MRI analysis", type="primary", use_container_width=True)

_is_patient = ui_common.current_role(st) == "patient"

# Persist "has run" across reruns: a button is only True on its own click, so
# without this, navigating away and back (or any other widget) would rerun with
# run_btn=False and blank the page. The heavy segmentation is cached below, so
# re-displaying on return is instant (no re-run of YOLO+U-Net).
if run_btn:
    st.session_state["mri_ran"] = True
mri_ran = st.session_state.get("mri_ran", False)
if not mri_ran:
    st.info("Pick an ACDC patient or upload ED + ES volumes, then **Add my MRI to my case**."
            if _is_patient else
            "Pick an ACDC patient or upload ED + ES volumes, then **Run MRI analysis**.")
    st.stop()

# --- Acquire volumes + optional expert masks --------------------------------
gt_ed = gt_es = None
try:
    if source == "ACDC patient":
        if not patient:
            st.error("No ACDC patients found.")
            st.stop()
        pdir = os.path.join(ACDC_TEST, patient)
        info = load_patient_info(os.path.join(pdir, "Info.cfg"))
        ed_vol, voxel = load_nifti_volume(os.path.join(pdir, f"{patient}_frame{info['ED']:02d}.nii.gz"))
        es_vol, voxel_es = load_nifti_volume(os.path.join(pdir, f"{patient}_frame{info['ES']:02d}.nii.gz"))
        gt_ed, _ = load_nifti_volume(os.path.join(pdir, f"{patient}_frame{info['ED']:02d}_gt.nii.gz"))
        gt_es, _ = load_nifti_volume(os.path.join(pdir, f"{patient}_frame{info['ES']:02d}_gt.nii.gz"))
        patient_id = patient
        info_for_result = info
        context_line = build_patient_context_line(height=info.get("Height"), weight=info.get("Weight"))
        st.caption(f"ACDC ground-truth group: **{info.get('Group', '—')}**")
    else:
        if ed_up is None or es_up is None:
            st.warning("Upload BOTH the ED and ES volumes (EF needs both phases).")
            st.stop()
        ed_vol, voxel = _load_nifti_upload(ed_up)
        es_vol, voxel_es = _load_nifti_upload(es_up)
        if ed_mask_up is not None and es_mask_up is not None:
            gt_ed, _ = _load_nifti_upload(ed_mask_up)
            gt_es, _ = _load_nifti_upload(es_mask_up)
        # --- Auto-extraction (mirrors the ECG upload reading its header) --------
        # (1) patient ID from the uploaded filename (ACDC-style "patientXXX").
        _fname = getattr(ed_up, "name", "") or getattr(es_up, "name", "")
        _m = re.search(r"patient\d+", _fname, re.IGNORECASE)
        patient_id = _m.group(0).lower() if _m else "upload"
        # (2) metadata from an optional Info.cfg: pathology group, ED/ES, height/weight.
        up_info, up_group = {}, None
        if info_up is not None:
            _cfg = tempfile.NamedTemporaryFile(delete=False, suffix=".cfg")
            _cfg.write(info_up.getbuffer()); _cfg.close()
            try:
                up_info = load_patient_info(_cfg.name)
                up_group = up_info.get("Group")
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not parse Info.cfg: {exc}")
        info_for_result = {"ED": up_info.get("ED", "ED"), "ES": up_info.get("ES", "ES")}
        # Prefer Info.cfg height/weight for context; else the manual age/sex fields.
        if up_info.get("Height") or up_info.get("Weight"):
            context_line = build_patient_context_line(
                height=up_info.get("Height"), weight=up_info.get("Weight"))
        else:
            context_line = build_patient_context_line(
                age=up_age if up_age > 0 else None,
                sex=None if up_sex == "unknown" else up_sex)
        st.caption(
            f"Auto-extracted — ID: **{patient_id}**"
            + (f" · pathology group (from Info.cfg): **{up_group}**" if up_group
               else " · pathology group: _not in the upload — the classifier predicts it below_"))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the MRI volume(s): {exc}")
    st.stop()

# --- Canonicalize axes (robust slice-axis detection) + diagnostics ----------
# The rest of the pipeline assumes axis 2 is the short-axis slice stack (true for
# ACDC), but an arbitrary upload may be 4-D (a cine) or have permuted axes, which
# would slice the WRONG plane -> a striped, non-anatomical image and garbage
# masks. Detect the real slice axis from the voxel spacing and reorient.
raw_ed_shape, raw_es_shape = ed_vol.shape, es_vol.shape
try:
    ed_vol, voxel, sa_ed, changed = ui_mri.to_canonical(ed_vol, voxel)
    es_vol, voxel_es, sa_es, _ = ui_mri.to_canonical(es_vol, voxel_es)
    if gt_ed is not None:
        gt_ed = ui_mri.move_slice_to_last(gt_ed, sa_ed)
        gt_es = ui_mri.move_slice_to_last(gt_es, sa_es)
except ValueError as exc:
    st.error(f"⛔ {exc}")
    st.stop()

if not _is_patient:
    with st.expander("🔎 Volume diagnostics (shape / spacing / slice axis)"):
        st.write(f"Raw ED shape: `{raw_ed_shape}` · ES shape: `{raw_es_shape}`")
        st.write(f"ED voxel spacing (mm): `{tuple(round(v, 2) for v in voxel)}`")
        st.write(f"Detected slice axis (ED): `{sa_ed}` → "
                 f"{'reoriented' if changed else 'already canonical'} to `{ed_vol.shape}` "
                 f"= (X, Y, {ed_vol.shape[2]} slices)")
        if ed_vol.shape[2] > 40 or ed_vol.shape[2] < 3:
            st.warning(f"Unusual slice count ({ed_vol.shape[2]}). A short-axis cardiac stack is "
                       "typically ~6–20 slices — check the volume is a single-phase short-axis frame.")

# --- Predict masks (U-Net) + features --------------------------------------
yolo_model, unet_model, device = load_seg_models()
pred_ed = cached_predict_mask(ed_vol, yolo_model, unet_model, device)
pred_es = cached_predict_mask(es_vol, yolo_model, unet_model, device)
pred_features = ui_mri.features_from_masks(pred_ed, pred_es, voxel)
ed_idx, raw_ed_uint8, ed_b64 = ui_mri.representative_ed(ed_vol, pred_ed)
st.session_state["mri_ed_b64"] = ed_b64  # so the Discuss (chat) page can attach the image

# Patient: generate + save the MRI contribution, show a friendly confirmation, and
# stop before all technical panels (mask overlay, slice-validity, EF metrics, Dice,
# findings table).
if _is_patient:
    if run_btn:
        classifier_model = load_classifier(device)
        preprocessing_result = {
            "patient_id": patient_id, "info": info_for_result, "features": pred_features,
            "images": {"short_axis_mid_ed": ed_b64}, "slice_indices": {"ed": ed_idx},
        }
        with st.spinner("Analysing your MRI…"):
            try:
                contribution = generate_mri_contribution(
                    preprocessing_result, record_id=patient_id, patient_info=context_line,
                    image_mode="ed_only", raw_ed_slice_uint8=raw_ed_uint8,
                    yolo_model=yolo_model, unet_model=unet_model,
                    classifier_model=classifier_model, device=device, max_retries=2)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not analyse the MRI: {exc}")
                contribution = None
        st.session_state["mri_contribution_obj"] = contribution
        if contribution is not None:
            st.session_state["contributions"]["mri"] = contribution
            st.session_state["patient_context_lines"]["mri"] = context_line
    contribution = st.session_state.get("mri_contribution_obj")
    if contribution is not None:
        st.image(base64.b64decode(ed_b64), caption="Your cardiac MRI (end-diastole)", width=360)
        st.success("✓ Your MRI has been added to your case. Open the **Case "
                   "aggregation** page for your report, or **Discuss** to ask questions.")
    st.stop()

# --- Case metadata (mirrors the ECG page's header) -------------------------
_group = info_for_result.get("Group") if isinstance(info_for_result, dict) else None
_m1, _m2, _m3 = st.columns([2, 2, 3])
_m1.metric("patient_id", patient_id)
_m2.metric("ACDC group", _group or "—")
_m3.markdown(f"**Context** · {context_line or 'age/sex not available'}")

# --- Display: raw slice + predicted overlay, side by side ------------------
st.subheader("Heart cross-section (end-diastole)")
st.caption("A cardiac MRI is a stack of 2-D slices through the heart. This is the short-axis slice at "
           "**end-diastole** (heart relaxed / filled), auto-picked as the one with the most heart tissue.")
_overlay = ui_mri.overlay_mask_on_slice(raw_ed_uint8, pred_ed[:, :, ed_idx].T)
_img_l, _img_r = st.columns(2)
with _img_l:
    st.image(base64.b64decode(ed_b64), use_container_width=True, caption=f"Raw slice — ED #{ed_idx}")
with _img_r:
    st.image(_overlay, use_container_width=True, caption="Predicted mask overlay")

# Colour legend (swatches) - readable up close, unlike a small caption.
st.markdown(
    '<div style="display:flex;gap:24px;flex-wrap:wrap;font-size:15px;margin:4px 0 8px">'
    '<span style="display:inline-flex;align-items:center;gap:7px">'
    '<span style="width:15px;height:15px;background:#4682FF;border-radius:3px;display:inline-block"></span>'
    '<b>RV</b>&nbsp;— right ventricle</span>'
    '<span style="display:inline-flex;align-items:center;gap:7px">'
    '<span style="width:15px;height:15px;background:#46DC82;border-radius:3px;display:inline-block"></span>'
    '<b>Myocardium</b></span>'
    '<span style="display:inline-flex;align-items:center;gap:7px">'
    '<span style="width:15px;height:15px;background:#FF5050;border-radius:3px;display:inline-block"></span>'
    '<b>LV</b>&nbsp;— left ventricle</span>'
    '</div>', unsafe_allow_html=True)

# Slice-validity metric — ANATOMICAL, not just "is there tissue". A real short-axis
# slice has a COMPACT mask (myocardium ring + LV blob = a few big connected regions).
# A striped / wrong-axis image yields a mask fragmented into many thin pieces even
# though "tissue" exists, so we check connected-component structure too.
from scipy import ndimage  # noqa: E402
_sl = pred_ed[:, :, ed_idx]  # ACDC labels: 1=RV, 2=myocardium, 3=LV
_fg = 100.0 * np.count_nonzero(_sl) / _sl.size
_present = [n for n, lab in (("RV", 1), ("Myo", 2), ("LV", 3)) if np.any(_sl == lab)]
_fg_mask = _sl > 0
_labeled, _ncomp = ndimage.label(_fg_mask)
_total = int(_fg_mask.sum())
_largest_frac = (float(ndimage.sum(_fg_mask, _labeled, range(1, _ncomp + 1)).max()) / _total
                 if _total else 0.0)
_anatomical_ok = (_fg >= 1.0 and len(_present) >= 2 and _ncomp <= 8 and _largest_frac >= 0.35)
_metrics = (f"coverage **{_fg:.1f}%** · structures **{', '.join(_present) or 'NONE'}** · "
            f"connected pieces **{_ncomp}** · largest-piece share **{_largest_frac:.0%}**")
if _anatomical_ok:
    st.success(f"✅ Slice looks anatomically valid — {_metrics}.")
else:
    st.error(f"⚠️ Slice looks WRONG (fragmented / not heart-shaped) — {_metrics}. "
             "Many small pieces or a low largest-piece share is the striped-image signature. "
             "Open **Volume diagnostics** above and check the detected slice axis / slice count.")

# --- Predicted EF & volumes ------------------------------------------------
st.subheader("Predicted EF & volumes")
_e1, _e2 = st.columns(2)
_e1.metric("LV ejection fraction", f"{pred_features['lv_ejection_fraction_pct']:.1f} %")
_e2.metric("RV ejection fraction", f"{pred_features['rv_ejection_fraction_pct']:.1f} %")
st.dataframe(
    [{"feature": k, "value": round(v, 1)} for k, v in pred_features.items() if "volume" in k],
    use_container_width=True, hide_index=True)

# --- Pipeline-utility comparison (predicted vs expert) ----------------------
if gt_ed is not None and gt_es is not None:
    st.subheader("Pipeline-utility check — predicted vs expert mask")
    exp_features = ui_mri.features_from_masks(gt_ed, gt_es, voxel)
    dice = ui_mri.dice_all(pred_ed, gt_ed)
    cmp = [
        {"metric": "LV EF (%)", "predicted": round(pred_features["lv_ejection_fraction_pct"], 1),
         "expert": round(exp_features["lv_ejection_fraction_pct"], 1),
         "Δ": round(pred_features["lv_ejection_fraction_pct"] - exp_features["lv_ejection_fraction_pct"], 1)},
        {"metric": "RV EF (%)", "predicted": round(pred_features["rv_ejection_fraction_pct"], 1),
         "expert": round(exp_features["rv_ejection_fraction_pct"], 1),
         "Δ": round(pred_features["rv_ejection_fraction_pct"] - exp_features["rv_ejection_fraction_pct"], 1)},
        {"metric": "LV vol ED (mL)", "predicted": round(pred_features["lv_volume_ed_ml"], 1),
         "expert": round(exp_features["lv_volume_ed_ml"], 1),
         "Δ": round(pred_features["lv_volume_ed_ml"] - exp_features["lv_volume_ed_ml"], 1)},
    ]
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    dcols = st.columns(3)
    for col, (k, v) in zip(dcols, dice.items()):
        col.metric(f"Dice {k}", f"{v:.3f}")
    st.caption("Dice = overlap of the predicted vs expert mask (ED). Close EFs + high Dice "
               "⇒ the predicted mask is a trustworthy stand-in for the expert one.")
else:
    st.info("Add an expert ED + ES mask to compare predicted vs expert EF (pipeline-utility check).")

# --- Optional: LLM findings + case contribution -----------------------------
if run_llm:
    st.subheader("MRI agent — structured findings")
    if run_btn:  # only (re)generate on an explicit Run click; reuse on navigation
        classifier_model = load_classifier(device)
        preprocessing_result = {
            "patient_id": patient_id, "info": info_for_result, "features": pred_features,
            "images": {"short_axis_mid_ed": ed_b64}, "slice_indices": {"ed": ed_idx},
        }
        with st.spinner("Generating image-grounded findings with MedGemma…"):
            try:
                contribution = generate_mri_contribution(
                    preprocessing_result, record_id=patient_id, patient_info=context_line,
                    image_mode="ed_only", raw_ed_slice_uint8=raw_ed_uint8,
                    yolo_model=yolo_model, unet_model=unet_model,
                    classifier_model=classifier_model, device=device, max_retries=2)
            except Exception as exc:  # noqa: BLE001
                st.error("MRI agent failed. Is Ollama running with `medgemma:4b`?\n\n"
                         f"Details: {exc}")
                contribution = None
        st.session_state["mri_contribution_obj"] = contribution
        if contribution is not None:
            st.session_state["contributions"]["mri"] = contribution
            st.session_state["patient_context_lines"]["mri"] = context_line

    contribution = st.session_state.get("mri_contribution_obj")
    if contribution is not None:
        st.success("Saved to the case → available on the **Case aggregation** page.")
        if contribution.findings:
            st.dataframe([{
                "feature": f.feature, "value": str(f.value), "confidence": f.confidence,
                "evidence_ref": f.evidence_ref, "rationale": f.rationale,
            } for f in contribution.findings], use_container_width=True, hide_index=True)
        st.markdown("**Merged flags for review:** " + (
            ", ".join(f"`{fl}`" for fl in contribution.flags_for_review) or "_none_"))
else:
    st.caption("Enable **LLM findings** in the sidebar to also run the MRI agent and add this "
               "modality to the case.")
