"""
Validate the Sokolow-Lyon LVH flag (ecg_multilead.py) against PTB-XL scp_codes.

Mirrors flag_validation.py's approach on the new multi-lead feature: a pure
feature-threshold classification-agreement check (precision/recall/F1), no LLM.
Ground truth = the `LVH` scp_code (left ventricular hypertrophy) specifically -
NOT the whole HYP superclass, since Sokolow-Lyon is a LEFT-ventricular voltage
criterion and does not target RVH / atrial enlargement / septal hypertrophy.

The Sokolow-Lyon index per record is cached to a CSV once, so thresholds can be
swept cheaply without re-loading signals (same discipline as flag_validation.py).
"""
import ast
import glob
import os

import numpy as np
import pandas as pd
import wfdb

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_multilead import compute_sokolow_lyon_mv, SOKOLOW_LYON_LVH_THRESHOLD_MV

LVH_GROUND_TRUTH_CODE = "LVH"


def _find_ptbxl():
    db = glob.glob("kaggle/**/ptbxl_database.csv", recursive=True)[0]
    scp = glob.glob("kaggle/**/scp_statements.csv", recursive=True)[0]
    root = db[: db.index("ptbxl_database.csv")]
    return db, scp, root


def compute_index_and_truth(max_records=None, resolution="lr", cache_path=None):
    """
    Compute the Sokolow-Lyon index + LVH ground truth for every record (or the
    first max_records). Returns a DataFrame [ecg_id, sokolow_lyon_index_mv,
    lvh_true]. Cached to cache_path if given.
    """
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    db, _, root = _find_ptbxl()
    df = pd.read_csv(db, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    if max_records:
        df = df.iloc[:max_records]

    fname_col = "filename_lr" if resolution == "lr" else "filename_hr"
    fs = 100 if resolution == "lr" else 500
    rows = []
    for ecg_id, row in df.iterrows():
        try:
            sig, _ = wfdb.rdsamp(root + row[fname_col])
            sl = compute_sokolow_lyon_mv(sig, fs)["sokolow_lyon_index_mv"]
        except Exception:
            sl = None
        rows.append({
            "ecg_id": ecg_id,
            "sokolow_lyon_index_mv": sl,
            "lvh_true": bool(row.scp_codes.get(LVH_GROUND_TRUTH_CODE, 0) > 0),
        })
    out = pd.DataFrame(rows)
    if cache_path:
        out.to_csv(cache_path, index=False)
    return out


def score(df, threshold_mv):
    """precision/recall/F1 for the flag `index >= threshold` vs lvh_true."""
    d = df.dropna(subset=["sokolow_lyon_index_mv"])
    pred = d["sokolow_lyon_index_mv"] >= threshold_mv
    true = d["lvh_true"]
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"threshold_mv": threshold_mv, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "n": len(d),
            "prevalence": round(float(true.mean()), 4)}


def sweep(df, thresholds):
    return pd.DataFrame([score(df, t) for t in thresholds])


if __name__ == "__main__":
    max_records = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cache = "results/ecg_multilead_validation/sokolow_lyon_index_cache.csv"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tag = f"_{max_records}" if max_records else "_full"
    cache = cache.replace(".csv", tag + ".csv")

    print(f"Computing Sokolow-Lyon index for {'all' if not max_records else max_records} records...", flush=True)
    df = compute_index_and_truth(max_records=max_records, cache_path=cache)
    print(f"done: {len(df)} records, {int(df.lvh_true.sum())} LVH-coded "
          f"({100*df.lvh_true.mean():.2f}% prevalence), "
          f"{int(df.sokolow_lyon_index_mv.isna().sum())} uncomputable\n")

    print("=== default threshold (3.5 mV) ===")
    print(score(df, SOKOLOW_LYON_LVH_THRESHOLD_MV))
    print("\n=== threshold sweep ===")
    sw = sweep(df, np.round(np.arange(2.0, 5.01, 0.25), 2))
    print(sw.to_string(index=False))
    sw.to_csv(cache.replace("index_cache", "threshold_sweep"), index=False)
