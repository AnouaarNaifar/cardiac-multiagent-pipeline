"""
BBB-via-QRS-delineation experiment (NOT shipped - documented negative result).

Tests whether NeuroKit2's wavelet QRS delineation gives a usable bundle-branch-block
(CRBBB|CLBBB) flag where our homemade amplitude heuristic failed. Explores the
aggregation of per-lead widths across the 12 leads, which turned out to be the crux.
Full write-up + numbers: results/ecg_multilead_validation/neurokit2_delineation_notes.md.

Usage:
    python src/ecg_bbb_neurokit_experiment.py balanced   # aggregation comparison (60 BBB + 130 NORM)
    python src/ecg_bbb_neurokit_experiment.py natural 450 # honest F1 at real prevalence (first N records)

Requires: pip install neurokit2  (delineation only; not used by any agent).
"""
import ast
import glob
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk

FS = 500  # 500 Hz records (filename_hr) for delineation resolution


def _ptbxl():
    db = glob.glob("kaggle/**/ptbxl_database.csv", recursive=True)[0]
    root = db[: db.index("ptbxl_database.csv")]
    df = pd.read_csv(db, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    return df, root


def _is_bbb(codes):
    return bool(codes.get("CRBBB", 0) > 0 or codes.get("CLBBB", 0) > 0)


def per_lead_widths(sig12, fs=FS):
    """(12, n_beats) matrix of per-lead QRS widths (ms) from dwt delineation, using
    ONE shared set of R-peaks (lead II). NaN where a lead/beat failed."""
    try:
        c_ii = nk.ecg_clean(sig12[:, 1], sampling_rate=fs)
        _, ri = nk.ecg_peaks(c_ii, sampling_rate=fs)
        rp = ri["ECG_R_Peaks"]
    except Exception:
        return None
    if len(rp) < 3:
        return None
    onsets = np.full((12, len(rp)), np.nan)
    offsets = np.full((12, len(rp)), np.nan)
    for l in range(12):
        try:
            c = nk.ecg_clean(sig12[:, l], sampling_rate=fs)
            _, w = nk.ecg_delineate(c, rp, sampling_rate=fs, method="dwt")
            on = np.array(w["ECG_R_Onsets"], float)
            off = np.array(w["ECG_R_Offsets"], float)
            m = min(len(on), len(rp))
            onsets[l, :m] = on[:m]
            offsets[l, :m] = off[:m]
        except Exception:
            continue
    return onsets, offsets


def aggregations(onsets, offsets, fs=FS):
    """Record-level QRS width (ms) under 3 aggregations of the per-lead delineation."""
    per_lead = (offsets - onsets) / fs * 1000.0
    global_extreme = (np.nanmax(offsets, 0) - np.nanmin(onsets, 0)) / fs * 1000.0  # min-onset/max-offset
    median_across = np.nanmedian(per_lead, 0)   # robust to a bad lead
    p75_across = np.nanpercentile(per_lead, 75, 0)

    def med(a):
        a = a[np.isfinite(a)]
        return float(np.median(a)) if len(a) else np.nan

    return {"global_extreme": med(global_extreme),
            "median_across": med(median_across),
            "p75_across": med(p75_across)}


def _score(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn


def run(mode, n):
    df, root = _ptbxl()
    if mode == "balanced":
        bbb = df[df.scp_codes.apply(_is_bbb)].index[:60]
        norm = df[df.scp_codes.apply(lambda c: "NORM" in c)].index[:130]
        ids = list(bbb) + list(norm)
    else:  # natural
        ids = list(df.index[:n])

    rows = []
    for i in ids:
        row = df.loc[i]
        res = per_lead_widths(wfdb.rdsamp(root + row.filename_hr)[0])
        if res is None:
            continue
        a = aggregations(*res)
        a["bbb"] = _is_bbb(row.scp_codes)
        rows.append(a)
    d = pd.DataFrame(rows)
    out = "results/ecg_multilead_validation/bbb_neurokit_12lead_cache.csv"
    d.to_csv(out, index=False)
    print(f"mode={mode} n={len(d)} BBB={int(d.bbb.sum())} prev={d.bbb.mean():.3f} -> {out}\n")

    for agg in ["global_extreme", "median_across", "p75_across"]:
        dd = d.dropna(subset=[agg])
        best = (0, 0, 0, 0)
        for t in range(90, 251, 10):
            p, r, f1, tp, fp, fn = _score(dd[agg] >= t, dd.bbb)
            if f1 > best[1]:
                best = (t, f1, p, r)
        print(f"  {agg:15s} NORM {dd[~dd.bbb][agg].median():.0f} / BBB {dd[dd.bbb][agg].median():.0f} ms "
              f"| best F1 {best[1]:.3f} @T={best[0]} (P={best[2]:.3f} R={best[3]:.3f})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "natural"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 450
    run(mode, n)
