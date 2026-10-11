"""
Validate the NEW multi-lead features (ST deviation -> STTC, R/S in V1 -> RVH)
against PTB-XL scp_codes, same discipline as ecg_multilead_validation.py.
Caches all features + ground truths per record so thresholds sweep cheaply.
"""
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_multilead import compute_multilead_features


def _load_superclass_map(scp_csv):
    df = pd.read_csv(scp_csv, index_col=0)
    diag = df[df.diagnostic == 1.0]
    return diag.diagnostic_class.to_dict()  # code -> superclass


def compute_cache(max_records=None, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    db = glob.glob("kaggle/**/ptbxl_database.csv", recursive=True)[0]
    scp = glob.glob("kaggle/**/scp_statements.csv", recursive=True)[0]
    root = db[: db.index("ptbxl_database.csv")]
    supermap = _load_superclass_map(scp)

    df = pd.read_csv(db, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    if max_records:
        df = df.iloc[:max_records]

    rows = []
    for ecg_id, row in df.iterrows():
        codes = row.scp_codes
        supers = {supermap.get(c) for c in codes}
        try:
            sig, _ = wfdb.rdsamp(root + row.filename_lr)
            f = compute_multilead_features(sig, 100)
        except Exception:
            f = {}
        rows.append({
            "ecg_id": ecg_id,
            "max_st_elevation_mv": f.get("max_st_elevation_mv"),
            "max_st_depression_mv": f.get("max_st_depression_mv"),
            "rs_ratio_v1": f.get("rs_ratio_v1"),
            "right_axis_deviation": f.get("right_axis_deviation"),
            "sttc_true": "STTC" in supers,
            "rvh_true": bool(codes.get("RVH", 0) > 0),
        })
    out = pd.DataFrame(rows)
    if cache_path:
        out.to_csv(cache_path, index=False)
    return out


def score(pred, true):
    pred, true = pred.astype(bool), true.astype(bool)
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return dict(precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3), tp=tp, fp=fp, fn=fn)


if __name__ == "__main__":
    max_records = int(sys.argv[1]) if len(sys.argv) > 1 else None
    tag = f"_{max_records}" if max_records else "_full"
    cache = f"results/ecg_multilead_validation/multilead_features_cache{tag}.csv"
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    print(f"Computing multi-lead features for {max_records or 'all'} records...", flush=True)
    df = compute_cache(max_records=max_records, cache_path=cache)
    n = len(df)
    print(f"done: {n} records | STTC prev {df.sttc_true.mean():.3f} | RVH prev {df.rvh_true.mean():.4f}\n")

    # --- STTC: ST elevation OR depression beyond a threshold ---
    d = df.dropna(subset=["max_st_elevation_mv", "max_st_depression_mv"])
    print(f"=== STTC flag: (ST elevation>=T or ST depression>=T)  n={len(d)} ===")
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pred = (d.max_st_elevation_mv >= t) | (d.max_st_depression_mv >= t)
        print(f"  T={t:.2f} mV: {score(pred, d.sttc_true)}")

    # --- RVH: dominant R in V1 ---
    d2 = df.dropna(subset=["rs_ratio_v1"])
    print(f"\n=== RVH flag: R/S in V1 >= T  n={len(d2)} ===")
    for t in [0.7, 1.0, 1.3, 1.6, 2.0]:
        pred = d2.rs_ratio_v1 >= t
        print(f"  T={t:.1f}: {score(pred, d2.rvh_true)}")
    # RVH with right-axis support
    d3 = df.dropna(subset=["rs_ratio_v1", "right_axis_deviation"])
    pred = (d3.rs_ratio_v1 >= 1.0) & (d3.right_axis_deviation.astype(bool))
    print(f"  R/S>=1.0 AND right-axis: {score(pred, d3.rvh_true)}")

    # NOTE: bundle-branch block (CD) was validated here too and did NOT separate
    # (a homemade multi-lead QRS width could not reliably delineate the QRS - see
    # ecg_multilead.py's removal note and the validation report). The single-lead
    # wide_qrs_suspected flag (flag_validation.py, F1 0.44) already covers CD.
