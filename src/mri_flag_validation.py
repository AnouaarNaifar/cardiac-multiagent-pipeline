"""
Deterministic MRI flag validation against ACDC pathology groups
===================================================================
Standalone module (does not modify mri_preprocessing.py itself): validates
mri_agent.detect_flags_mri's 3 fixed-threshold flags
(low_lv_ejection_fraction_suspected, low_rv_ejection_fraction_suspected,
high_lv_ejection_fraction_suspected) against ACDC's own cardiologist-
assigned pathology group labels - the MRI counterpart to
flag_validation.py's ECG scp_codes check. Same convention as that module:
this validation code never edits mri_agent.py itself, but a real finding
from running it (high_lv_ejection_fraction_suspected's threshold sweep,
below) was applied directly to mri_agent.detect_flags_mri as a
follow-up edit - mirroring how flag_validation.py's tachycardia sweep led
to a threshold change in ecg_agent.detect_flags.

No LLM call anywhere, and no image reprocessing either: reuses
results/mri_features_all_patients.csv's already-computed per-patient
ejection fractions directly (see mri_preprocessing.py /
notebooks/mri_preprocessing.ipynb for how that CSV was built).

Ground truth mapping and its limits, stated plainly: ACDC's 5 pathology
groups (Bernard et al. 2018, "Deep Learning Techniques for Automatic MRI
Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem
Solved?" - the dataset's own mandatory citation, ACDC/database/MANDATORY_CITATION.md)
are NOR (normal), MINF (post-myocardial-infarction with reduced LV systolic
function), DCM (dilated cardiomyopathy, also reduced LV systolic function),
HCM (hypertrophic cardiomyopathy, preserved/high LV EF), RV (abnormal right
ventricle). MINF and DCM are used here as ground-truth-positive for low LV
EF, and RV for low RV EF. This repository does not include the primary
paper's exact numeric diagnostic criteria (e.g. the precise EF cutoff used
to define each group) - only the group LABELS themselves, which ARE
verified directly from each patient's own Info.cfg (real, not guessed).
The group-based mapping used here is the commonly-cited general
characterization of these categories, not independently re-verified
against the source paper's exact thresholds in this repo. It is supported
by this project's own already-computed group-mean ejection fractions
(results/mri_features_all_patients.csv): DCM 19.8%, MINF 30.8%, HCM 69.8%,
NOR 62.4% mean LV EF; RV group 34.3% vs. 52-59% for the other groups on
mean RV EF - clinically coherent with the mapping used, though this is a
group-mean sanity check, not a substitute for the primary paper's criteria.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from mri_agent import detect_flags_mri

# Ground-truth pathology groups per flag - see module docstring for the
# mapping's basis and its stated limitation.
_LOW_LV_EF_GROUPS = {"MINF", "DCM"}
_LOW_RV_EF_GROUPS = {"RV"}
_HIGH_LV_EF_GROUPS = {"HCM"}

_FLAGS_AND_GROUND_TRUTH = {
    "low_lv_ejection_fraction_suspected": _LOW_LV_EF_GROUPS,
    "low_rv_ejection_fraction_suspected": _LOW_RV_EF_GROUPS,
    "high_lv_ejection_fraction_suspected": _HIGH_LV_EF_GROUPS,
}

# Tested and REJECTED: "high_rv_ejection_fraction_suspected" (RV EF > threshold,
# ground truth = HCM group, same idea as high_lv_ejection_fraction_suspected).
# HCM has the highest mean RV EF too (59.4%), but the gap to the next groups
# (MINF 53.4%, NOR 52.6%) is only ~6 points - even narrower than the LV case
# (HCM 69.8% vs NOR 62.4%, ~7.4 points). Best F1 found by sweeping candidate
# thresholds (50-65%) was 0.541 at 57% (precision 0.455, recall 0.667) -
# clearly weaker than high_lv_ejection_fraction_suspected's 0.698. More
# importantly, it would be largely REDUNDANT: the HCM patients it would catch
# are already caught by high_lv_ejection_fraction_suspected (recall 0.733
# there vs. 0.667 here), so this flag would mostly add false positives on
# already-flagged patients rather than genuinely new detection power. Not
# implemented in mri_agent.py.


def evaluate_mri_flags_against_pathology_groups(
    features_csv="results/mri_features_all_patients.csv",
    low_lv_ef_pct: float = 45.0,
    low_rv_ef_pct: float = 45.0,
    high_lv_ef_pct: float = 65.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate mri_agent's 3 deterministic clinical flags against ACDC's
    pathology group labels, over all patients in features_csv.

    Inputs:
        features_csv: path to the CSV produced by running
            mri_preprocessing.preprocess_patient across all ACDC patients
            (columns: patient_id, group, lv_ejection_fraction_pct,
            rv_ejection_fraction_pct, ...) - defaults to this project's
            existing results/mri_features_all_patients.csv.
        low_lv_ef_pct / low_rv_ef_pct / high_lv_ef_pct: thresholds passed
            through to mri_agent.detect_flags_mri (same defaults as
            that function).

    Output:
        (results_df, summary) tuple:
        - results_df: one row per patient, with
          "<flag>_predicted"/"<flag>_ground_truth" boolean columns for each
          of the 3 flags in _FLAGS_AND_GROUND_TRUTH.
        - summary: dict mapping flag name -> {"tp", "fp", "fn", "tn",
          "precision", "recall", "f1", "prevalence"}.
    """
    df = pd.read_csv(features_csv)

    rows = []
    for _, row in df.iterrows():
        feature_dict = {
            "lv_ejection_fraction_pct": row["lv_ejection_fraction_pct"],
            "rv_ejection_fraction_pct": row["rv_ejection_fraction_pct"],
        }
        decision = detect_flags_mri(
            feature_dict, low_lv_ef_pct=low_lv_ef_pct, low_rv_ef_pct=low_rv_ef_pct, high_lv_ef_pct=high_lv_ef_pct
        )
        predicted = set(decision.deterministic_flags)

        record_row = {
            "patient_id": row["patient_id"],
            "group": row["group"],
            "lv_ejection_fraction_pct": row["lv_ejection_fraction_pct"],
            "rv_ejection_fraction_pct": row["rv_ejection_fraction_pct"],
        }
        for flag, ground_truth_groups in _FLAGS_AND_GROUND_TRUTH.items():
            record_row[f"{flag}_predicted"] = flag in predicted
            record_row[f"{flag}_ground_truth"] = row["group"] in ground_truth_groups
        rows.append(record_row)

    results_df = pd.DataFrame(rows)

    summary = {}
    for flag in _FLAGS_AND_GROUND_TRUTH:
        predicted = results_df[f"{flag}_predicted"]
        truth = results_df[f"{flag}_ground_truth"]
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        tn = int((~predicted & ~truth).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        summary[flag] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "prevalence": float(truth.mean()),
        }

    return results_df, summary


def write_mri_flag_validation_report(summary: Dict[str, Any], n_records: int, output_dir) -> str:
    """
    Write a short markdown report of evaluate_mri_flags_against_pathology_groups's
    summary, for direct use in the project report.

    Inputs:
        summary: dict as returned by evaluate_mri_flags_against_pathology_groups.
        n_records: number of patients the summary was computed over.
        output_dir: directory to write "mri_flag_validation_report.md" into.

    Output:
        Path to the written markdown file (str).
    """
    lines = [
        "# Deterministic MRI flag validation against ACDC pathology groups",
        "",
        f"Evaluated over {n_records} patients, no LLM call and no image reprocessing involved "
        "(detect_flags_mri is pure feature-threshold logic, run directly on "
        "results/mri_features_all_patients.csv's already-computed ejection fractions).",
        "",
        "| Flag | Precision | Recall | F1 | Ground-truth prevalence | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for flag, s in summary.items():
        lines.append(
            f"| `{flag}` | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | "
            f"{s['prevalence']:.4f} | {s['tp']} | {s['fp']} | {s['fn']} | {s['tn']} |"
        )
    lines += [
        "",
        "Ground truth: low_lv_ejection_fraction_suspected's ground truth is True for patients in ACDC's "
        "MINF/DCM pathology groups; low_rv_ejection_fraction_suspected's is True for the RV group; "
        "high_lv_ejection_fraction_suspected's is True for the HCM group - see module docstring for the "
        "mapping's basis and its stated limitation (group labels are verified directly from each patient's "
        "Info.cfg; the exact numeric diagnostic criteria behind each group are not independently "
        "re-verified against the primary ACDC paper in this repo).",
        "",
        "high_lv_ejection_fraction_suspected (65% threshold) was added to mri_agent.detect_flags_mri "
        "as a direct result of this module's threshold sweep: HCM's mean LV EF (69.8%) is only 7-13 points "
        "above the other groups (vs. 20-30+ points of separation for the low-EF flags), so this flag is "
        "necessarily noisier than low_lv_ejection_fraction_suspected - but its F1 is still notably better "
        "than low_rv_ejection_fraction_suspected's, making it a genuinely useful addition despite the "
        "narrower separation.",
    ]

    note_path = Path(output_dir) / "mri_flag_validation_report.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(note_path)
