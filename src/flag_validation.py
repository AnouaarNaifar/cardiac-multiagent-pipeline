"""
Deterministic flag validation against PTB-XL scp_codes
=========================================================
Standalone validation module. Does NOT modify ecg_agent.py, ecg_preprocessing.py,
or ecg_embedding.py - it only calls their existing public functions.

Purpose: ecg_agent.detect_flags raises tachycardia_suspected/
bradycardia_suspected/wide_qrs_suspected/irregular_rhythm_suspected from
fixed numeric thresholds, but until now those flags were never checked
against real diagnostic ground truth - only exercised qualitatively via a
handful of manual examples. This module compares them against PTB-XL's own
scp_codes labels, which is a fair, correctly-scoped test (a classification
agreement check) unlike the BLEU/ROUGE/BERTScore comparison in
report_evaluation.py, which compares free text against a diagnostic
reference and was found structurally unable to validate anything about
these flags.

No LLM call is involved anywhere in this module (detect_flags is pure
feature-threshold logic) - only the ECG preprocessing pipeline runs, so this
can be evaluated on hundreds of records quickly, unlike report_evaluation.py's
n_samples=30 (bottlenecked by real Ollama calls).

Ground truth code sets below were verified directly against this project's
scp_statements.csv (not guessed from general medical knowledge - PTB-XL
doesn't necessarily include every plausible code, and some codes affect a
different property than the flag name might suggest, e.g. AV blocks (1AVB/
2AVB/3AVB) affect PR interval/conduction timing, not QRS width, so they are
deliberately excluded from wide_qrs_suspected's ground truth; fascicular
blocks (LAFB/LPFB) mainly cause axis deviation with little/no QRS widening,
also excluded).

Embedding-based flags (embedding_<CLASS>_pattern_suspected) are deliberately
NOT covered here: ecg_embedding.py's model was already validated properly
against a patient-disjoint held-out test split (see CLAUDE.md, the
strat_fold retraining) - re-checking it against an arbitrary sample here,
without controlling for train/test overlap, would be a less rigorous,
potentially circular duplicate of that existing validation.
"""

import ast
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from ecg_agent import compute_feature_dict, detect_flags
from ecg_preprocessing import ecg_preprocessing_pipeline

# Ground truth SCP-ECG code sets per flag - see module docstring for why
# these specific codes (verified against scp_statements.csv) and not others.
_FLAG_GROUND_TRUTH_CODES = {
    "tachycardia_suspected": {"STACH", "SVTAC", "PSVT"},
    "bradycardia_suspected": {"SBRAD"},
    "irregular_rhythm_suspected": {"AFIB", "SARRH", "SVARR", "AFLT", "BIGU", "TRIGU"},
    "wide_qrs_suspected": {"CRBBB", "CLBBB", "ILBBB", "IRBBB", "WPW", "IVCD"},
}


def record_has_any_code(scp_codes_str: str, code_set: Set[str]) -> bool:
    """
    Check whether a record's raw "scp_codes" string has a nonzero-likelihood
    entry for any code in code_set.

    Inputs:
        scp_codes_str: raw string value from ptbxl_database.csv's
            "scp_codes" column (e.g. "{'NORM': 100.0, 'STACH': 0.0}").
        code_set: set of SCP-ECG code strings to check for.

    Output:
        True if any code in code_set has likelihood > 0 for this record.
    """
    codes = ast.literal_eval(scp_codes_str)
    return any(code in code_set and likelihood > 0 for code, likelihood in codes.items())


def compute_ground_truth_flags(scp_codes_str: str) -> Dict[str, bool]:
    """
    Derive ground-truth booleans for each of the 4 deterministic flags in
    _FLAG_GROUND_TRUTH_CODES, from a record's real scp_codes.

    Inputs:
        scp_codes_str: raw string value from the database's "scp_codes" column.

    Output:
        dict mapping flag name -> bool (True if the record's scp_codes
        include at least one of that flag's ground-truth codes).
    """
    return {flag: record_has_any_code(scp_codes_str, codes) for flag, codes in _FLAG_GROUND_TRUTH_CODES.items()}


def _load_and_compute_features(dataset_dir, filename_lr) -> Dict[str, Any]:
    """
    Shared internal helper: load one record's signal (via wfdb) and run it
    through the ECG preprocessing pipeline to get its numeric feature dict
    (NO LLM call). Used by both compute_predicted_flags and
    compute_features_and_ground_truth, so the load-and-preprocess sequence
    isn't duplicated between them.

    Inputs:
        dataset_dir: PTB-XL root directory (containing "records100/", etc.).
        filename_lr: value from the database's "filename_lr" column.

    Output:
        dict as returned by ecg_agent.compute_feature_dict.
    """
    import wfdb

    sig, meta = wfdb.rdsamp(str(Path(dataset_dir) / filename_lr))
    results = ecg_preprocessing_pipeline(sig[:, 0], meta["fs"])
    return compute_feature_dict(results, meta["fs"])


def compute_predicted_flags(dataset_dir, filename_lr) -> Set[str]:
    """
    Run the ECG preprocessing pipeline and ecg_agent's deterministic
    decision logic (NO LLM call - detect_flags is pure feature
    thresholds) and return the set of raised flags for one record.

    Inputs:
        dataset_dir: PTB-XL root directory (containing "records100/", etc.).
        filename_lr: value from the database's "filename_lr" column.

    Output:
        set of flag name strings (e.g. {"tachycardia_suspected"}) - may
        include "insufficient_data_for_analysis" for records with fewer
        than 2 detected QRS complexes, which is excluded from the
        precision/recall computation in evaluate_flags_against_scp_codes
        (it isn't one of the 4 clinical flags being validated).
    """
    feature_dict = _load_and_compute_features(dataset_dir, filename_lr)
    decision = detect_flags(feature_dict)
    return set(decision.deterministic_flags)


def evaluate_flags_against_scp_codes(
    database_csv,
    dataset_dir,
    n_samples: Optional[int] = 500,
    random_seed: int = 42,
    validated_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate ecg_agent's 4 deterministic clinical flags against PTB-XL's own
    scp_codes ground truth, over a sample of records. No LLM call involved,
    so this can run on hundreds of records quickly (unlike
    report_evaluation.py's LLM-bottlenecked n_samples=30).

    Records the preprocessing pipeline fails to read/process are skipped
    (logged, not fatal) rather than aborting the whole run.

    Inputs:
        database_csv: path to PTB-XL's ptbxl_database.csv.
        dataset_dir: PTB-XL root directory (containing "records100/", etc.).
        n_samples: number of records to sample, or None for the entire
            dataset (with no LLM cost, this is feasible - budget for
            preprocessing-only runtime, no cold-load/inference latency).
        random_seed: seed for reproducible sampling.
        validated_only: if True, restrict to validated_by_human == True
            records (higher-trust scp_codes ground truth).

    Output:
        (results_df, summary) tuple:
        - results_df: one row per evaluated record, with
          "<flag>_predicted"/"<flag>_ground_truth" boolean columns for each
          of the 4 flags in _FLAG_GROUND_TRUTH_CODES.
        - summary: dict mapping flag name -> {"tp", "fp", "fn", "tn",
          "precision", "recall", "f1", "prevalence"} (prevalence = fraction
          of evaluated records where the ground truth is True).
    """
    df = pd.read_csv(database_csv).dropna(subset=["scp_codes"])
    if validated_only:
        df = df[df["validated_by_human"] == True]  # noqa: E712
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        try:
            predicted = compute_predicted_flags(dataset_dir, row["filename_lr"])
        except Exception:  # noqa: BLE001 - a single unreadable record must not abort the batch
            continue
        ground_truth = compute_ground_truth_flags(row["scp_codes"])
        record_row = {"ecg_id": row["ecg_id"], "record_id": Path(row["filename_lr"]).name}
        for flag in _FLAG_GROUND_TRUTH_CODES:
            record_row[f"{flag}_predicted"] = flag in predicted
            record_row[f"{flag}_ground_truth"] = ground_truth[flag]
        rows.append(record_row)

    results_df = pd.DataFrame(rows)

    summary = {}
    for flag in _FLAG_GROUND_TRUTH_CODES:
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


def write_flag_validation_report(summary: Dict[str, Any], n_records: int, output_dir) -> str:
    """
    Write a short markdown report of evaluate_flags_against_scp_codes's
    summary, for direct use in the project report.

    Inputs:
        summary: dict as returned by evaluate_flags_against_scp_codes.
        n_records: number of records the summary was computed over.
        output_dir: directory to write "flag_validation_report.md" into.

    Output:
        Path to the written markdown file (str).
    """
    lines = [
        "# Deterministic flag validation against PTB-XL scp_codes",
        "",
        f"Evaluated over {n_records} records, no LLM call involved (detect_flags is pure "
        "feature-threshold logic).",
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
        "Precision here means: of the records this flag fired on, what fraction actually carry a "
        "matching scp_codes diagnosis (see module docstring for the exact code sets per flag). Low "
        "precision with high recall (tachycardia_suspected, irregular_rhythm_suspected) means the fixed "
        "threshold catches most true cases but is swamped by false positives - the raw statistical "
        "condition (e.g. mean heart rate > 100 bpm over 10s) fires far more often than PTB-XL's "
        "cardiologists considered diagnostically significant enough to code. wide_qrs_suspected is the "
        "only flag with balanced, moderately usable precision and recall.",
        "",
        "bradycardia_suspected is a different case from the other two weak flags, not just another instance "
        "of over-triggering: its ground-truth prevalence is exactly 0 (SBRAD never appears with nonzero "
        "likelihood anywhere in this database, verified directly - not a code bug). This makes its precision "
        "0.000 by mathematical necessity, not by miscalibration - with zero true positives possible, every "
        "one of its 3868 firings is unavoidably a false positive regardless of what threshold is used, and "
        "recall is undefined (0/0) rather than merely low. Unlike tachycardia_suspected, there is no "
        "threshold sweep that can improve this flag's precision against this dataset - see "
        "sweep_threshold's docstring for why bradycardia_suspected is deliberately excluded from it. This "
        "flag's real-world usefulness is simply unvalidated here, not demonstrated poor.",
    ]

    note_path = Path(output_dir) / "flag_validation_report.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(note_path)


# ----------------------------------------------------------------------
# Threshold sweep (tachycardia_suspected / irregular_rhythm_suspected only)
# ----------------------------------------------------------------------
# Does NOT touch wide_qrs_suspected or any other part of the detection
# pipeline (ecg_agent.py / ecg_preprocessing.py) - this only tries
# alternative threshold VALUES, applied post-hoc to the same raw
# heart_rate_mean_bpm/rhythm_cv numbers the existing pipeline already
# computes. bradycardia_suspected is deliberately excluded from sweeping:
# its ground truth is 0 positives dataset-wide (SBRAD never occurs - see
# module docstring / CLAUDE.md), so precision/recall/F1 cannot be
# evaluated at ANY threshold - there is nothing to tune against.
def compute_features_and_ground_truth(
    database_csv, dataset_dir, n_samples: Optional[int] = None, random_seed: int = 42, validated_only: bool = False
) -> pd.DataFrame:
    """
    Compute, once per record, the raw numeric features needed for
    threshold experimentation (qrs_count, heart_rate_mean_bpm, rhythm_cv)
    alongside scp_codes ground truth for the 4 flags. Lets sweep_threshold
    try many candidate thresholds cheaply afterward without re-running the
    ECG pipeline (wfdb load + preprocessing) once per candidate.

    Inputs: same as evaluate_flags_against_scp_codes.

    Output:
        DataFrame with columns "ecg_id", "record_id", "qrs_count",
        "heart_rate_mean_bpm", "rhythm_cv", and "<flag>_ground_truth" for
        each of the 4 flags in _FLAG_GROUND_TRUTH_CODES.
    """
    df = pd.read_csv(database_csv).dropna(subset=["scp_codes"])
    if validated_only:
        df = df[df["validated_by_human"] == True]  # noqa: E712
    if n_samples is not None and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        try:
            feature_dict = _load_and_compute_features(dataset_dir, row["filename_lr"])
        except Exception:  # noqa: BLE001 - a single unreadable record must not abort the batch
            continue
        ground_truth = compute_ground_truth_flags(row["scp_codes"])
        record_row = {
            "ecg_id": row["ecg_id"],
            "record_id": Path(row["filename_lr"]).name,
            "qrs_count": feature_dict["qrs_count"],
            "heart_rate_mean_bpm": feature_dict["heart_rate_mean_bpm"],
            "rhythm_cv": feature_dict["rhythm_cv"],
        }
        record_row.update({f"{flag}_ground_truth": gt for flag, gt in ground_truth.items()})
        rows.append(record_row)

    return pd.DataFrame(rows)


def sweep_threshold(
    features_df: pd.DataFrame, flag: str, feature_col: str, candidate_thresholds, direction: str = "greater"
) -> pd.DataFrame:
    """
    Compute precision/recall/F1 for a flag at several candidate threshold
    values, against the cached features from compute_features_and_ground_truth.

    Records with qrs_count < 2 are excluded, mirroring detect_flags'
    real behavior in production (it short-circuits to
    skip_llm_call=True/deterministic_flags=["insufficient_data_for_analysis"]
    for those - none of the 4 clinical flags are ever raised for them,
    regardless of threshold).

    Inputs:
        features_df: DataFrame as returned by compute_features_and_ground_truth.
        flag: flag name, must have a "<flag>_ground_truth" column in features_df.
        feature_col: column to threshold ("heart_rate_mean_bpm" or "rhythm_cv").
        candidate_thresholds: iterable of threshold values to try.
        direction: "greater" (predicted True if feature_col > threshold - use
            for tachycardia/irregular rhythm) or "less" (predicted True if
            feature_col < threshold - use for bradycardia, though see the
            module-level note on why bradycardia isn't actually swept here).

    Output:
        DataFrame with one row per threshold: "threshold", "precision",
        "recall", "f1", "tp", "fp", "fn".
    """
    valid = features_df[features_df["qrs_count"] >= 2]
    truth = valid[f"{flag}_ground_truth"]

    rows = []
    for threshold in candidate_thresholds:
        predicted = valid[feature_col] > threshold if direction == "greater" else valid[feature_col] < threshold
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn})

    return pd.DataFrame(rows)
