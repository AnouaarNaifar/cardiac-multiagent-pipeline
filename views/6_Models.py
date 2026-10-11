"""
Streamlit page — Models (architectures, training curves, metrics).

A read-only reference: every model in the pipeline, its architecture, its
training curve (read LIVE from the saved per-epoch logs), and its headline
metric — so the modelling choices can be shown and justified at a glance.
No training happens here; it only plots the logs already on disk.
"""

import os

import pandas as pd
import streamlit as st

import ui_common

# page config is set once by the Home.py router (st.navigation)
st.title("🧠 Models — architectures, training & metrics")
st.caption("Every model in the pipeline: what it is, how it trained, and its result. "
           "Curves are read live from the saved epoch logs — nothing is retrained here.")

ROOT = ui_common.PROJECT_ROOT


@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)


# --- Inventory --------------------------------------------------------------
st.subheader("Full inventory")
st.dataframe([
    {"model": "ECGEmbeddingNet", "kind": "1-D CNN (from scratch)", "base": "—",
     "role": "ECG superclass probs", "metric": "acc 84.9-89.1%"},
    {"model": "YOLO26n", "kind": "detector (fine-tuned)", "base": "yolo26n / COCO",
     "role": "heart localization (MRI)", "metric": "mAP50 0.976"},
    {"model": "U-Net", "kind": "seg (fine-tuned)", "base": "ResNet18 / ImageNet",
     "role": "MRI segmentation", "metric": "val Dice 0.906"},
    {"model": "PathologyClassifier", "kind": "CNN (fine-tuned)", "base": "ResNet18 / ImageNet",
     "role": "MRI pathology", "metric": "RV F1 0.80"},
    {"model": "MedGemma 4B", "kind": "VLM (used)", "base": "Google MedGemma",
     "role": "findings + MRI vision + synthesis", "metric": "default backend"},
    {"model": "GPT-OSS 120B", "kind": "LLM (used)", "base": "OpenAI OSS / Groq",
     "role": "stronger text backend (opt-in)", "metric": "cloud"},
    {"model": "all-MiniLM-L6-v2", "kind": "embedder (used)", "base": "sentence-transformers",
     "role": "RAG retrieval (384-dim)", "metric": "—"},
    {"model": "distilbert-base-uncased", "kind": "embedder (used)", "base": "DistilBERT",
     "role": "BERTScore evaluation", "metric": "—"},
], use_container_width=True, hide_index=True)
st.caption("Four models are learned here (one from scratch, three fine-tuned from a pretrained backbone); "
           "the rest are pretrained models the pipeline calls as-is.")

st.divider()
st.subheader("Learned models — architecture + training curves")


def learned_model(title, kicker, spec, csv_rel, charts, metric, extra_img=None):
    st.markdown(f"#### {title}")
    st.caption(kicker)
    left, right = st.columns([1, 1.25])
    with left:
        st.code(spec, language="text")
        st.success(metric)
    with right:
        path = os.path.join(ROOT, csv_rel)
        if os.path.exists(path):
            df = load_csv(path)
            idx = df.columns[0]
            ccols = st.columns(len(charts))
            for col, (label, cols) in zip(ccols, charts):
                cols = [c for c in cols if c in df.columns]
                with col:
                    st.caption(label)
                    if cols:
                        st.line_chart(df.set_index(idx)[cols], height=190)
            st.caption(f"Source: `{csv_rel}` · {len(df)} epochs")
        else:
            st.info(f"No epoch log at `{csv_rel}`.")
    if extra_img and os.path.exists(os.path.join(ROOT, extra_img)):
        with st.expander("Per-class evaluation"):
            st.image(os.path.join(ROOT, extra_img), use_container_width=True)
    st.divider()


learned_model(
    "🫀 ECGEmbeddingNet", "1-D CNN over the 12-lead signal — trained from scratch on PTB-XL",
    "Conv1d(12->32,k7)-BN-ReLU-MaxPool(2)\n"
    "Conv1d(32->64,k5)-BN-ReLU-MaxPool(2)\n"
    "Conv1d(64->64,k3)-BN-ReLU\n"
    "AdaptiveAvgPool1d(1) -> emb(64)\n"
    "Linear(64 -> 5)  # CD HYP MI NORM STTC",
    "results/ecg_embedding/epoch_log_stratfold.csv",
    [("Loss (train vs val)", ["train_loss", "val_loss"])],
    "Final: accuracy 84.9–89.1% (patient-disjoint stratfold split)")

learned_model(
    "🎯 YOLO26n", "Ultralytics YOLO26-nano — fine-tuned from COCO to localize the heart",
    "Base: yolo26n.pt (COCO)\n"
    "Task: single-class detection ('heart')\n"
    "epochs=50 · imgsz=320 · ACDC\n"
    "Output: bbox -> crop for the U-Net",
    "results/mri_yolo_training/final/results.csv",
    [("mAP (val)", ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]),
     ("Box loss (train vs val)", ["train/box_loss", "val/box_loss"])],
    "Final: mAP50 0.976")

learned_model(
    "🧩 U-Net (ResNet18 encoder)", "segmentation_models_pytorch — fine-tuned from ImageNet",
    "smp.Unet(\n"
    "  encoder_name='resnet18',\n"
    "  encoder_weights='imagenet',\n"
    "  in_channels=1, classes=4)\n"
    "# bg · RV · myocardium · LV",
    "results/mri_unet_training/epoch_log.csv",
    [("Train loss", ["train_loss"]), ("Val Dice", ["val_dice"])],
    "Final: val Dice 0.906")

learned_model(
    "🔬 PathologyClassifier", "ResNet18 reading image + predicted mask — fine-tuned from ImageNet",
    "encoder = smp resnet18 (imagenet),\n"
    "          in_channels=5  # img+mask\n"
    "AdaptiveAvgPool2d(1) -> flatten\n"
    "Linear(512 -> 5)  # DCM HCM MINF NOR RV",
    "results/mri_classifier_training/epoch_log.csv",
    [("Train loss", ["train_loss"]), ("Val patient accuracy", ["val_patient_accuracy"])],
    "Final: RV F1 0.80 (its strongest class)")

# --- Evaluation on the held-out test set -----------------------------------
st.subheader("Evaluation (held-out test set)")
st.caption("Beyond the training curves above — where the errors actually are, from the saved test-set "
           "evaluation artifacts.")

def _safe_image(path, caption=None, width=None, centered=False):
    """Show an image full-width, or a clean message on any failure (so a broken
    image never leaks a raw traceback that mentions the router)."""
    if not os.path.exists(path):
        st.info(f"Not found: `{os.path.relpath(path, ROOT)}`")
        return
    try:
        if width is None:
            st.image(path, use_container_width=True, caption=caption)
        elif centered:
            _, image_col, _ = st.columns([1, 2, 1])
            with image_col:
                st.image(path, width=width, caption=caption)
        else:
            st.image(path, width=width, caption=caption)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not display `{os.path.basename(path)}` ({exc}).")


with st.expander("🫀 ECGEmbeddingNet — per-class accuracy"):
    _safe_image(os.path.join(ROOT, "results/ecg_embedding/eval_per_class_accuracy_stratfold.png"),
                width=700, centered=True)

with st.expander("🎯 YOLO26n — precision-recall / F1 / confusion (heart detection)"):
    st.caption("Single-class heart detector, evaluated on the held-out val split (384 images). "
               "Peak F1 ≈ 0.97 at confidence ≈ 0.36; mAP@0.5 = 0.976. In the **confusion matrix** "
               "(counted at the default conf ≥ 0.25): **TP** = hearts correctly detected, **FN** = missed "
               "hearts (only 4), **FP** = extra boxes called 'heart' — mostly low-confidence duplicate "
               "detections, which is why precision at the operating point stays 0.98 despite the FP count.")
    ydir = os.path.join(ROOT, "results/mri_yolo_training/final")
    _yolo_imgs = [("pretty_pr_curve.png", "Precision–Recall"),
                  ("pretty_f1_curve.png", "F1 vs confidence"),
                  ("pretty_confusion.png", "Confusion matrix")]
    for _col, (_img, _cap) in zip(st.columns(3), _yolo_imgs):
        with _col:
            st.markdown(f"**{_cap}**")
            _safe_image(os.path.join(ydir, _img))

with st.expander("🧩 U-Net — segmentation Dice by structure (test set)"):
    p = os.path.join(ROOT, "results/mri_pipeline_validation/segmentation_dice_test_set.csv")
    if os.path.exists(p):
        d = load_csv(p)
        by_struct = d.groupby("structure")["dice"].mean().reindex(["rv", "myocardium", "lv"])
        st.bar_chart(by_struct, height=240)
        st.caption("Mean Dice per structure on the held-out test set — the thin **myocardial ring** is the "
                   "hardest (~0.74), the blood-pool chambers (LV/RV ~0.87) segment better. "
                   "(These per-structure test means differ from the 0.906 overall val-Dice during training.)")
    else:
        st.info("segmentation_dice_test_set.csv not found")

with st.expander("🔬 PathologyClassifier — confusion matrix + per-class metrics (test)"):
    cm_p = os.path.join(ROOT, "results/mri_classifier_validation/test_confusion_matrix.csv")
    pc_p = os.path.join(ROOT, "results/mri_classifier_validation/test_per_class_metrics.csv")
    if os.path.exists(cm_p):
        cm = load_csv(cm_p).set_index("true_group")
        st.caption("Confusion matrix — rows = true group, columns = predicted:")
        try:
            st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)
        except Exception:  # noqa: BLE001 - matplotlib/jinja2 optional
            st.dataframe(cm, use_container_width=True)
    if os.path.exists(pc_p):
        st.caption("Per-class precision / recall / F1:")
        st.dataframe(load_csv(pc_p), use_container_width=True, hide_index=True)

# --- Flag metrics + backends pointers --------------------------------------
st.subheader("Deterministic flags & text evaluation")
st.markdown("Precision / recall / F1 per deterministic flag and the threshold sweeps are on the "
            "**Validation** page (recomputed live). BLEU / ROUGE / BERTScore for the ECG narrative live in "
            "`results/evaluation_brief/`.")

st.subheader("Language / vision-language backends (used, not trained)")
st.markdown(
    "- **MedGemma 4B — VLM (default):** generates every agent's structured findings, **reads the MRI image**, "
    "and writes the Sep Agent's impression. Local via Ollama.\n"
    "- **GPT-OSS 120B — LLM (opt-in):** stronger text backend via Groq's free cloud.\n"
    "- **all-MiniLM-L6-v2:** RAG knowledge-base embedder. "
    "- **distilbert-base-uncased:** BERTScore backbone.")
