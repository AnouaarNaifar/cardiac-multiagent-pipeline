"""
Validate territorial MI features (ecg_territorial.py) against PTB-XL scp_codes,
no LLM, cached + threshold sweep (mirrors flag_validation.py / ecg_multilead_*).

Two ground truths, per wall:
  - pathological Q (qr_ratio_<wall>) vs the big CHRONIC MI codes of that wall
  - ST elevation (st_elev_<wall>_mv) vs the ACUTE injury/STE codes of that wall
"""
import ast
import glob
import os
import sys

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ecg_territorial import WALLS, compute_territorial_features

MI_WALL = {
    "inferior": {"IMI", "ILMI", "IPMI", "IPLMI"},
    "anterior": {"ASMI", "AMI", "ALMI"},
    "lateral": {"LMI", "ALMI", "ILMI", "IPLMI"},
}
INJ_WALL = {
    "inferior": {"INJIN", "INJIL"},
    "anterior": {"INJAS", "INJAL"},
    "lateral": {"INJLA", "INJIL", "STE_"},
}


def build_cache(n, cache_path):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    db = glob.glob("kaggle/**/ptbxl_database.csv", recursive=True)[0]
    root = db[: db.index("ptbxl_database.csv")]
    df = pd.read_csv(db, index_col="ecg_id"); df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    if n:
        df = df.iloc[:n]
    rows = []
    for eid, row in df.iterrows():
        codes = set(k for k, v in row.scp_codes.items())
        try:
            f = compute_territorial_features(wfdb.rdsamp(root + row.filename_lr)[0], 100)
        except Exception:
            f = {}
        rec = {"ecg_id": eid}
        for w in WALLS:
            rec[f"qr_{w}"] = f.get(f"qr_ratio_{w}")
            rec[f"qcount_{w}"] = f.get(f"qcount_{w}")
            rec[f"stelev_{w}"] = f.get(f"st_elev_{w}_mv")
            rec[f"strecip_{w}"] = f.get(f"st_recip_{w}_mv")
            rec[f"mi_{w}"] = bool(codes & MI_WALL[w])
            rec[f"inj_{w}"] = bool(codes & INJ_WALL[w])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(cache_path, index=False)
    return out


def sweep(df, feat_col, truth_col, thresholds, recip_col=None, recip_max=None):
    d = df.dropna(subset=[feat_col])
    best = None
    res = []
    for t in thresholds:
        pred = d[feat_col] >= t
        if recip_col is not None:  # require reciprocal ST depression too
            pred = pred & (d[recip_col].fillna(0) <= recip_max)
        true = d[truth_col].astype(bool)
        tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        res.append((t, p, r, f1, tp, fp, fn))
        if best is None or f1 > best[3]:
            best = (t, p, r, f1, tp, fp, fn)
    return best, res, len(d), int(df[truth_col].sum())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    cache = f"results/ecg_territorial_validation/territorial_cache_{n or 'full'}.csv"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    print(f"Computing territorial features for {n or 'all'} records...", flush=True)
    df = build_cache(n, cache)
    print(f"done: {len(df)} records\n")

    print("=== (2a) Pathological Q, MAX Q/R in wall  vs  chronic MI codes ===")
    for w in WALLS:
        best, _, nn, npos = sweep(df, f"qr_{w}", f"mi_{w}", np.round(np.arange(0.2, 1.01, 0.1), 2))
        print(f"  {w:9s} (MI prev {npos}/{nn}): best F1 {best[3]:.3f} @Q/R>={best[0]} "
              f"(P={best[1]:.3f} R={best[2]:.3f})")

    print("\n=== (2b) Territorial CONSENSUS: >= k leads with pathological Q  vs MI ===")
    for w in WALLS:
        best, _, nn, npos = sweep(df, f"qcount_{w}", f"mi_{w}", [1, 2, 3, 4])
        print(f"  {w:9s}: best F1 {best[3]:.3f} @>={int(best[0])} leads "
              f"(P={best[1]:.3f} R={best[2]:.3f})")

    print("\n=== (1) ST elevation vs acute injury/STE codes, per wall ===")
    for w in WALLS:
        best, _, nn, npos = sweep(df, f"stelev_{w}", f"inj_{w}", np.round(np.arange(0.05, 0.31, 0.05), 2))
        print(f"  {w:9s} (INJ prev {npos}/{nn}): best F1 {best[3]:.3f} @ST>={best[0]}mV "
              f"(P={best[1]:.3f} R={best[2]:.3f})")

    print("\n=== (1b) ST elevation + reciprocal depression (inferior & lateral only) ===")
    for w in ["inferior", "lateral"]:
        best, _, nn, npos = sweep(df, f"stelev_{w}", f"inj_{w}",
                                  np.round(np.arange(0.05, 0.31, 0.05), 2),
                                  recip_col=f"strecip_{w}", recip_max=-0.05)
        print(f"  {w:9s}: best F1 {best[3]:.3f} @ST>={best[0]}mV + recip<=-0.05 "
              f"(P={best[1]:.3f} R={best[2]:.3f})")
