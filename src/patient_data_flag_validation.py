"""
Deterministic flag validation for the structured patient data agent
========================================================================
Standalone validation module, mirroring flag_validation.py's (ECG) and
mri_flag_validation.py's (MRI) role and methodology for the third modality.
Does NOT modify patient_data_agent.py - it only calls its existing public
functions and reports what it finds; any resulting threshold recalibration
is applied by hand afterward, in patient_data_agent.py itself, same
convention as flag_validation.py's tachycardia_bpm 100->120 story.

Ground truth: the UCI Heart Disease dataset's "num" column (0-4 severity)
binarized to heart_disease_present = (num > 0) - the standard treatment for
this dataset (every published benchmark on it uses this binarization; the
raw 5-level severity scale is not what any of this agent's flags claim to
predict). No LLM call is involved anywhere in this module
(detect_flags_patient is pure feature-threshold logic) - runs on all
303 patients instantly, unlike the LLM-bottlenecked contribution pipeline.

Threshold sweep covers 5 of the 7 deterministic flags (resting BP,
cholesterol, ST depression, vessel count, and the joint max-heart-rate
formula+percentage pair) - not assumed correct from their literature-cited
starting values in patient_data_agent.detect_flags_patient's
docstring, same "validate before trusting" discipline already applied once
to ecg_agent.tachycardia_bpm. exercise_induced_angina_present and
abnormal_thallium_result_suspected are direct pass-throughs of an
already-categorical/binary UCI field (no threshold to sweep) - still
evaluated against ground truth below, just not swept.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from patient_data_agent import compute_feature_dict

_DEFAULT_CSV = "uci_heart_disease/processed_cleveland.csv"

# All 7 deterministic flags this module validates.
_ALL_FLAGS = [
    "high_resting_bp_suspected",
    "high_cholesterol_suspected",
    "reduced_exercise_capacity_suspected",
    "significant_st_depression_suspected",
    "exercise_induced_angina_present",
    "multiple_vessels_involved_suspected",
    "abnormal_thallium_result_suspected",
]


def _prf1(predicted: pd.Series, truth: pd.Series) -> Dict[str, Any]:
    """Shared precision/recall/F1/prevalence computation from two boolean Series."""
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "prevalence": float(truth.mean()),
    }


def compute_features_and_ground_truth(csv_path=_DEFAULT_CSV) -> pd.DataFrame:
    """
    Compute, once per patient, this agent's feature dict plus the binarized
    ground truth, cached in one DataFrame so sweep_threshold and
    sweep_max_hr_formula_and_threshold can try many candidate thresholds
    cheaply afterward without recomputing features per candidate.

    Inputs:
        csv_path: path to the locally-saved UCI Cleveland CSV (see
            patient_data_agent.py's module docstring for how it was
            obtained/saved).

    Output:
        DataFrame with one row per patient: "row_index", all 13
        patient_data_agent.compute_feature_dict keys, and
        "heart_disease_present" (bool ground truth, num > 0).
    """
    df = pd.read_csv(csv_path)
    rows = []
    for idx, row in df.iterrows():
        feature_dict = compute_feature_dict(row)
        record = {"row_index": idx, **feature_dict}
        record["heart_disease_present"] = bool(row["num"] > 0)
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate_flags(features_df: pd.DataFrame, **decision_kwargs) -> Dict[str, Any]:
    """
    Call the REAL detect_flags_patient (not a reimplementation) once
    per patient and report precision/recall/F1/prevalence for all 7 flags
    in _ALL_FLAGS against heart_disease_present - deliberately calling the
    production decision function directly (feasible at only 303 rows, no
    performance reason to avoid it) so this evaluation can never silently
    drift from what patient_data_agent.py actually does, unlike
    sweep_threshold/sweep_max_hr_formula_and_threshold below, which DO
    reimplement the comparison logic on cached raw values - justified
    there only because trying hundreds of candidate thresholds by calling
    the real function each time would be wasteful, mirroring
    flag_validation.sweep_threshold's identical tradeoff.

    Inputs:
        features_df: DataFrame as returned by compute_features_and_ground_truth.
        **decision_kwargs: forwarded as-is to detect_flags_patient
            (e.g. oldpeak_threshold=0.8) - omit to evaluate at
            detect_flags_patient's own current defaults.

    Output:
        dict mapping flag name -> {"tp", "fp", "fn", "tn", "precision",
        "recall", "f1", "prevalence"}.
    """
    from patient_data_agent import detect_flags_patient

    truth = features_df["heart_disease_present"]
    predicted_flags_per_row = [
        set(detect_flags_patient(row, **decision_kwargs).deterministic_flags)
        for _, row in features_df.iterrows()
    ]

    summary = {}
    for flag in _ALL_FLAGS:
        predicted = pd.Series([flag in flags for flags in predicted_flags_per_row], index=features_df.index)
        summary[flag] = _prf1(predicted, truth)
    return summary


def build_per_record_results_df(features_df: pd.DataFrame, **decision_kwargs) -> pd.DataFrame:
    """
    Build a per-record predicted/ground-truth table for every flag, mirroring the
    ECG/MRI "*_results.csv" layout the Validation UI recomputes precision/recall/F1
    from (one <flag>_predicted and one <flag>_ground_truth column per flag). Unlike
    those modalities, the ground truth here is a single binary target
    (heart_disease_present), so every flag's *_ground_truth column carries that same
    value - each of the 7 flags is a candidate predictor of the SAME outcome.

    Inputs:
        features_df: DataFrame as returned by compute_features_and_ground_truth.
        **decision_kwargs: forwarded to detect_flags_patient (defaults if omitted).

    Output:
        DataFrame, one row per patient: "record_id", "heart_disease_present", and
        "<flag>_predicted" / "<flag>_ground_truth" for each flag in _ALL_FLAGS.
    """
    from patient_data_agent import detect_flags_patient

    rows = []
    for _, feat in features_df.iterrows():
        fired = set(detect_flags_patient(feat, **decision_kwargs).deterministic_flags)
        gt = bool(feat["heart_disease_present"])
        record = {"record_id": f"cleveland_{int(feat['row_index'])}", "heart_disease_present": gt}
        for flag in _ALL_FLAGS:
            record[f"{flag}_predicted"] = flag in fired
            record[f"{flag}_ground_truth"] = gt
        rows.append(record)
    return pd.DataFrame(rows)


def sweep_threshold(
    features_df: pd.DataFrame, feature_col: str, candidate_thresholds, direction: str = "greater"
) -> pd.DataFrame:
    """
    Compute precision/recall/F1 against heart_disease_present at several
    candidate threshold values for a single continuous feature column -
    mirrors flag_validation.sweep_threshold's role, extended with a
    "greater_equal" direction option (needed for oldpeak, whose real
    detect_flags_patient rule is >=, not >).

    Inputs:
        features_df: DataFrame as returned by compute_features_and_ground_truth.
        feature_col: column to threshold (e.g. "resting_bp_mmHg").
        candidate_thresholds: iterable of threshold values to try.
        direction: "greater" (>), "greater_equal" (>=), or "less" (<).

    Output:
        DataFrame with one row per threshold: "threshold", "precision",
        "recall", "f1", "tp", "fp", "fn".
    """
    valid = features_df.dropna(subset=[feature_col])
    truth = valid["heart_disease_present"]

    rows = []
    for threshold in candidate_thresholds:
        if direction == "greater":
            predicted = valid[feature_col] > threshold
        elif direction == "greater_equal":
            predicted = valid[feature_col] >= threshold
        else:
            predicted = valid[feature_col] < threshold
        stats = _prf1(predicted, truth)
        rows.append({"threshold": threshold, **stats})
    return pd.DataFrame(rows)


def sweep_max_hr_formula_and_threshold(
    features_df: pd.DataFrame, formulas=("220_minus_age", "tanaka"), pct_thresholds=None
) -> pd.DataFrame:
    """
    Jointly sweep the age-predicted-max-heart-rate FORMULA and the
    PERCENTAGE threshold together for "reduced_exercise_capacity_suspected"
    - the two are not independent (a different formula shifts what counts
    as "85% of predicted max" for the same patient), so sweeping either one
    alone while holding the other fixed could miss the actually-best
    combination. Requested explicitly rather than assuming either formula
    is correct upfront.

    Inputs:
        features_df: DataFrame as returned by compute_features_and_ground_truth
            (needs "age_years", "max_heart_rate_achieved_bpm",
            "heart_disease_present").
        formulas: iterable of formula names, "220_minus_age" or "tanaka".
        pct_thresholds: iterable of candidate percentage thresholds
            (e.g. [0.75, 0.80, ..., 0.95]). Defaults to 0.70-0.95 in 0.01
            steps if not given.

    Output:
        DataFrame with one row per (formula, pct_threshold) combination:
        "formula", "pct_threshold", "precision", "recall", "f1", "tp",
        "fp", "fn" - sorted by F1 descending, best combination first.
    """
    if pct_thresholds is None:
        pct_thresholds = [round(0.70 + 0.01 * i, 2) for i in range(26)]  # 0.70 .. 0.95

    valid = features_df.dropna(subset=["age_years", "max_heart_rate_achieved_bpm"])
    truth = valid["heart_disease_present"]

    rows = []
    for formula in formulas:
        if formula == "tanaka":
            predicted_max_hr = 208.0 - 0.7 * valid["age_years"]
        else:
            predicted_max_hr = 220.0 - valid["age_years"]
        for pct in pct_thresholds:
            predicted = valid["max_heart_rate_achieved_bpm"] < pct * predicted_max_hr
            stats = _prf1(predicted, truth)
            rows.append({"formula": formula, "pct_threshold": pct, **stats})

    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)


def write_patient_flag_validation_report(summary: Dict[str, Any], n_records: int, output_dir) -> str:
    """
    Write a short markdown report of evaluate_flags's summary, mirroring
    flag_validation.write_flag_validation_report's format.

    Inputs:
        summary: dict as returned by evaluate_flags.
        n_records: number of patients the summary was computed over.
        output_dir: directory to write "patient_data_flag_validation_report.md" into.

    Output:
        Path to the written markdown file (str).
    """
    lines = [
        "# Deterministic flag validation for the structured patient data agent",
        "",
        f"Evaluated over {n_records} patients (UCI Heart Disease, Cleveland), ground truth = "
        "heart_disease_present (num > 0). No LLM call involved.",
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
        "## Interpretation",
        "",
        "Unlike the ECG deterministic flags (whose target diagnoses are rare, so precision is "
        "swamped by false positives), the target here - `heart_disease_present` - is roughly "
        "balanced (prevalence ~0.46), so precision above ~0.46 is a genuine signal rather than the "
        "base rate. Precision here means: of the patients this flag fired on, what fraction actually "
        "have heart disease.",
        "",
        "- **Strongest flags:** `abnormal_thallium_result_suspected` (F1 ~0.74) and "
        "`multiple_vessels_involved_suspected` (F1 ~0.70) - a positive thallium scan and any "
        "fluoroscopy-colored vessel are both directly ischaemia-related, so they track the outcome well.",
        "- **Two thresholds were recalibrated by sweep** (same discipline as the ECG tachycardia "
        "100→120 recalibration): `oldpeak` 1.0→0.8 mm (Pareto-improves recall at similar precision), "
        "and max-HR `220-age @85%` → **Tanaka `208-0.7×age` @90%** (90% is one step below the F1-max at "
        "91%, chosen to favor recall).",
        "- **`high_resting_bp_suspected` (140 mmHg) and `high_cholesterol_suspected` (240 mg/dl) were "
        "kept at their clinically meaningful cutoffs despite weak sweeps** - both are simply weak CAD "
        "predictors in this dataset at ANY threshold (precision barely clears the base rate), so the "
        "standard clinical cutoff was kept and the weakness reported honestly rather than chasing a "
        "statistically marginal, clinically meaningless gain.",
        "",
        "Recall-favoring choices throughout: these flags feed a human-review alert, not a final "
        "classification, so a missed case is treated as worse than an extra reviewed false positive.",
    ]

    note_path = Path(output_dir) / "patient_data_flag_validation_report.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(note_path)


if __name__ == "__main__":
    features_df = compute_features_and_ground_truth()
    print(f"Loaded {len(features_df)} patients.")

    print("\n=== Baseline (starting thresholds from patient_data_agent.py) ===")
    baseline_summary = evaluate_flags(features_df)
    for flag, s in baseline_summary.items():
        print(f"  {flag}: precision={s['precision']:.3f} recall={s['recall']:.3f} "
              f"f1={s['f1']:.3f} prevalence={s['prevalence']:.3f}")

    print("\n=== Sweep: resting_bp_mmHg_threshold ===")
    print(sweep_threshold(features_df, "resting_bp_mmHg", range(110, 171, 5), "greater").to_string(index=False))

    print("\n=== Sweep: cholesterol_mgdl_threshold ===")
    print(sweep_threshold(features_df, "cholesterol_mgdl", range(180, 301, 10), "greater").to_string(index=False))

    print("\n=== Sweep: oldpeak_threshold ===")
    print(sweep_threshold(features_df, "st_depression_oldpeak_mm", [round(0.1 * i, 1) for i in range(0, 41)], "greater_equal").to_string(index=False))

    print("\n=== Sweep: n_vessels_threshold ===")
    print(sweep_threshold(features_df, "n_major_vessels_colored", [1, 2, 3], "greater_equal").to_string(index=False))

    print("\n=== Joint sweep: max_hr_formula x pct_threshold (top 10 by F1) ===")
    print(sweep_max_hr_formula_and_threshold(features_df).head(10).to_string(index=False))

    report_path = write_patient_flag_validation_report(baseline_summary, len(features_df), "results")
    print("\nBaseline report written to:", report_path)

    results_df = build_per_record_results_df(features_df)
    results_csv = "results/patient_data_flag_validation_results.csv"
    results_df.to_csv(results_csv, index=False)
    print("Per-record results CSV written to:", results_csv, f"({len(results_df)} rows)")
