"""
Streamlit page — Validation metrics (read-only).

Surfaces the project's already-computed deterministic-flag validation results
(no recomputation of the expensive full-dataset runs): precision/recall/F1 per
flag computed on the fly from the saved per-record CSVs, the threshold-sweep
curves, and the written validation reports. Pure presentation of existing
`results/` artifacts.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

import ui_common

# page config is set once by the Home.py router (st.navigation)
st.title("📊 Deterministic-flag validation")
st.caption("Precision / recall / F1 of the agents' deterministic flags against ground truth "
           "(PTB-XL scp_codes, ACDC groups, UCI target). Read-only view of saved `results/`.")

ROOT = ui_common.PROJECT_ROOT


def _p(*parts):
    return os.path.join(ROOT, *parts)


def _as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


@st.cache_data(show_spinner=False)
def metrics_from_csv(csv_path):
    """Compute precision/recall/F1 per flag from a *_predicted / *_ground_truth CSV."""
    df = pd.read_csv(csv_path)
    flags = sorted({c[:-len("_predicted")] for c in df.columns if c.endswith("_predicted")})
    rows = []
    for flag in flags:
        pred = _as_bool(df[f"{flag}_predicted"])
        gt = _as_bool(df[f"{flag}_ground_truth"])
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * prec * rec / (prec + rec)) if (tp + fp) and (tp + fn) and (prec + rec) else np.nan
        rows.append({"flag": flag, "precision": prec, "recall": rec, "F1": f1,
                     "TP": tp, "FP": fp, "FN": fn, "GT positives": tp + fn})
    return pd.DataFrame(rows), len(df)


def show_modality(title, csv_path, report_path, note=None):
    st.subheader(title)
    if note:
        st.caption(note)
    if os.path.exists(csv_path):
        table, n = metrics_from_csv(csv_path)
        st.caption(f"Computed from {n} records · `{os.path.relpath(csv_path, ROOT)}`")
        st.dataframe(
            table.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "F1": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info(f"No per-record CSV at `{os.path.relpath(csv_path, ROOT)}`.")
    if os.path.exists(report_path):
        with st.expander("Full validation report"):
            with open(report_path, encoding="utf-8") as fh:
                st.markdown(fh.read())


tab_ecg, tab_mri, tab_patient, tab_sweeps = st.tabs(
    ["ECG flags", "MRI flags", "Patient-data flags", "Threshold sweeps"])

with tab_ecg:
    show_modality(
        "ECG deterministic flags (vs PTB-XL scp_codes)",
        _p("results", "flag_validation", "flag_validation_results.csv"),
        _p("results", "flag_validation", "flag_validation_report.md"),
        note="Full dataset (21 799 records). Only wide_qrs performs well absolutely; the rate/rhythm "
             "flags fire on ordinary physiological variation — thresholds are recall-favoring on purpose "
             "(these feed a human-review alert, not a final classification).",
    )

with tab_mri:
    show_modality(
        "MRI deterministic flags (vs ACDC pathology group)",
        _p("results", "mri_flag_validation", "mri_flag_validation_results.csv"),
        _p("results", "mri_flag_validation", "mri_flag_validation_report.md"),
        note="All 150 ACDC patients. low_lv_ejection_fraction is nearly perfect (F1 ≈ 0.975); "
             "low_rv is naturally more variable.",
    )

with tab_patient:
    show_modality(
        "Patient-data deterministic flags (vs UCI heart_disease_present)",
        _p("results", "patient_data_flag_validation_results.csv"),  # may not exist
        _p("results", "patient_data_flag_validation_report.md"),
        note="303 Cleveland patients. Best flags: abnormal thallium (F1 0.739), multiple vessels (F1 0.702).",
    )

with tab_sweeps:
    st.subheader("Threshold sweeps (recalibration evidence)")
    st.caption("Precision / recall / F1 vs candidate threshold — how the recall-favoring cutoffs were chosen.")
    st.info("**Only these two ECG sweeps were saved as CSV**, so only they are charted here — it does NOT "
            "mean only two flags were swept:\n"
            "- **Swept & shown** (saved CSV): tachycardia, irregular rhythm.\n"
            "- **Swept but printed only** (no CSV): patient-data resting-BP, cholesterol, oldpeak, "
            "vessels, max-HR formula; MRI high-LV-EF (62–75%).\n"
            "- **Deliberately NOT swept**: bradycardia (SBRAD never occurs → no ground truth) and "
            "wide-QRS (already the best ECG flag, left untouched).")

    # --- Final adopted thresholds for EVERY flag (live from the detect_flags code) ---
    st.markdown("#### Final adopted thresholds — every flag")
    st.caption("Read **live** from the `detect_flags*` function signatures, so this always matches what the "
               "app actually uses — shown for all flags, including those with no saved sweep CSV.")
    import inspect

    def _defaults(fn):
        return {p: v.default for p, v in inspect.signature(fn).parameters.items()
                if v.default is not inspect._empty}

    rows = []
    try:
        from ecg_agent import detect_flags
        e = _defaults(detect_flags)
        rows += [
            ("ECG", "Tachycardia", f"> {e['tachycardia_bpm']:.0f} bpm", "recalibrated 100 → 120 (recall-favoring)"),
            ("ECG", "Bradycardia", f"< {e['bradycardia_bpm']:.0f} bpm", "not swept — SBRAD never occurs (no ground truth)"),
            ("ECG", "Wide QRS", f"≥ {e['wide_qrs_ms']:.0f} ms", "not swept — already the best ECG flag"),
            ("ECG", "Sokolow-Lyon LVH", f"≥ {e['lvh_sokolow_lyon_mv']} mV", "textbook 3.5 mV, on the F1 plateau"),
            ("ECG", "Inferior pathological Q", f"≥ {e['inferior_q_min_leads']} inferior leads", "clinical consensus rule"),
            ("ECG", "Embedding superclass", f"prob ≥ {e['embedding_probability_threshold']}", "shared with the learned model"),
        ]
    except Exception as exc:  # noqa: BLE001
        st.caption(f"(ECG thresholds unavailable: {exc})")
    try:
        from mri_agent import detect_flags_mri
        m = _defaults(detect_flags_mri)
        rows += [
            ("MRI", "Low LV EF", f"< {m['low_lv_ef_pct']:.0f}%", "nearly perfect (F1 0.975), no recalibration"),
            ("MRI", "Low RV EF", f"< {m['low_rv_ef_pct']:.0f}%", "moderate (RV naturally variable)"),
            ("MRI", "High LV EF", f"≥ {m['high_lv_ef_pct']:.0f}%", "added from a 62–75% sweep (HCM)"),
            ("MRI", "Pathology class", f"prob ≥ {m['pathology_classifier_probability_threshold']}", "shared with the classifier"),
        ]
    except Exception as exc:  # noqa: BLE001
        st.caption(f"(MRI thresholds unavailable: {exc})")
    try:
        from patient_data_agent import detect_flags_patient
        p = _defaults(detect_flags_patient)
        rows += [
            ("Patient", "High resting BP", f"≥ {p['resting_bp_mmHg_threshold']:.0f} mmHg", "clinical cutoff (weak sweep, kept honest)"),
            ("Patient", "High cholesterol", f"≥ {p['cholesterol_mgdl_threshold']:.0f} mg/dl", "clinical cutoff (weak sweep)"),
            ("Patient", "ST depression (oldpeak)", f"≥ {p['oldpeak_threshold']} mm", "recalibrated 1.0 → 0.8 (recall-favoring)"),
            ("Patient", "Reduced exercise capacity",
             f"< {p['max_hr_pct_threshold'] * 100:.0f}% of {str(p['max_hr_formula']).title()} max HR",
             "F1-max (0.669); higher pct = more positives, so 91% beats 90% on recall too (0.705 vs 0.683)"),
            ("Patient", "Multiple vessels", f"≥ {p['n_vessels_threshold']} vessel(s)", "fluoroscopy-colored vessels"),
            ("Patient", "Abnormal thallium / exercise angina", "categorical (no numeric threshold)",
             "raised on the abnormal category"),
        ]
    except Exception as exc:  # noqa: BLE001
        st.caption(f"(Patient thresholds unavailable: {exc})")
    if rows:
        st.dataframe([{"modality": r[0], "flag": r[1], "adopted threshold": r[2], "note": r[3]}
                      for r in rows], use_container_width=True, hide_index=True)
    st.divider()

    st.markdown("#### Saved sweep curves")
    sweeps = {
        "Tachycardia (bpm)": _p("results", "flag_validation", "tachycardia_threshold_sweep.csv"),
        "Irregular rhythm (RR CV)": _p("results", "flag_validation", "irregular_rhythm_threshold_sweep.csv"),
    }
    for label, path in sweeps.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            st.markdown(f"**{label}**")
            st.line_chart(df.set_index("threshold")[["precision", "recall", "f1"]], height=260)
        else:
            st.info(f"No sweep CSV at `{os.path.relpath(path, ROOT)}`.")
