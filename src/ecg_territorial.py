"""
Territorial ECG features for myocardial infarction (MI superclass), additive and
standalone like ecg_multilead.py. Built on the anatomical logic that each cardiac
wall is "watched" by specific leads, with reciprocal leads on the opposite wall:

    Wall       Leads                Reciprocal leads
    inferior   II, III, aVF         I, aVL
    anterior   V1, V2, V3, V4       (posterior V7-V9, not in 12-lead)
    lateral    I, aVL, V5, V6       II, III, aVF

Two complementary MI signals, in ONE shared framework:
  1. ST elevation + reciprocal depression (acute STEMI) -> ground truth INJ*/STE_.
  2. Pathological Q waves / loss of R (chronic/old MI)   -> ground truth the big
     territorial MI codes (IMI, ASMI, ...). In PTB-XL (resting ECGs) MI is mostly
     chronic, so (2) is expected to carry far more signal than (1) - see the
     validation module.

Measured on the RAW 12-lead signal in mV (like Sokolow-Lyon). Reuses
ecg_multilead's per-lead helpers. Validated before any wiring.
"""
import numpy as np

from ecg_multilead import (
    PTBXL_LEAD_ORDER,
    lead_index,
    measure_beat_amplitudes,
    measure_st_level,
)
from ecg_preprocessing import pan_tompkins_rpeaks

_RHYTHM_LEAD = "II"

WALLS = {
    "inferior": {"leads": ["II", "III", "AVF"], "reciprocal": ["I", "AVL"]},
    "anterior": {"leads": ["V1", "V2", "V3", "V4"], "reciprocal": []},
    "lateral": {"leads": ["I", "AVL", "V5", "V6"], "reciprocal": ["II", "III", "AVF"]},
}


def measure_q_over_r(lead_signal, r_peaks, fs, pre_window_ms=50.0, half_window_ms=60.0):
    """
    Median Q/R ratio in one lead: per beat, the depth of the initial negative
    deflection (Q wave, in the window just before the local R) divided by the R
    amplitude. A pathological Q (old MI) has Q/R > ~0.25, or a QS complex (loss of
    R) drives R->0 so the ratio is large. Returns (median_q_over_r, median_r_amp).
    """
    baseline = float(np.median(lead_signal))
    pre = int(round(pre_window_ms / 1000.0 * fs))
    half = int(round(half_window_ms / 1000.0 * fs))
    n = len(lead_signal)
    ratios = []
    for peak in r_peaks:
        a, b = max(0, peak - half), min(n, peak + half + 1)
        if b - a < 3:
            continue
        # local R = max in the full QRS window; Q = min in the pre-R window
        r_amp = max(1e-6, float(np.max(lead_signal[a:b])) - baseline)
        qa, qb = max(0, peak - pre), min(n, peak + 1)
        q_depth = max(0.0, baseline - float(np.min(lead_signal[qa:qb])))
        ratios.append(q_depth / r_amp)
    return (float(np.median(ratios)) if ratios else None)


def _wall_mean_st(sig, wall_leads, r_peaks, fs):
    vals = [measure_st_level(sig[:, lead_index(l)], r_peaks, fs) for l in wall_leads]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def compute_territorial_st(sig, fs, r_peaks):
    """
    Per-wall ST elevation (mean ST level across the wall's leads) and reciprocal
    depression (mean ST across the reciprocal leads, expected negative in a real
    STEMI). Keys: st_elev_<wall>_mv, st_recip_<wall>_mv (None where no reciprocal).
    """
    out = {}
    for wall, cfg in WALLS.items():
        out[f"st_elev_{wall}_mv"] = _wall_mean_st(sig, cfg["leads"], r_peaks, fs)
        out[f"st_recip_{wall}_mv"] = (
            _wall_mean_st(sig, cfg["reciprocal"], r_peaks, fs) if cfg["reciprocal"] else None
        )
    return out


# Per-lead pathological-Q threshold: Q at least 25% of R (the classic clinical rule).
_PATHOLOGICAL_Q_OVER_R = 0.25


def compute_territorial_q(sig, fs, r_peaks):
    """
    Per-wall pathological-Q features:
      - qr_ratio_<wall>: MAX Q/R ratio across the wall's leads (sensitive).
      - qcount_<wall>: COUNT of the wall's leads with a pathological Q
        (Q/R >= 0.25). Territorial CONSENSUS (>=2 leads) is far more specific than
        a single lead - a real MI shows Q in several contiguous leads of a wall,
        whereas an isolated septal q is physiological.
    """
    out = {}
    for wall, cfg in WALLS.items():
        ratios = [measure_q_over_r(sig[:, lead_index(l)], r_peaks, fs) for l in cfg["leads"]]
        ratios = [r for r in ratios if r is not None]
        out[f"qr_ratio_{wall}"] = float(np.max(ratios)) if ratios else None
        out[f"qcount_{wall}"] = int(sum(r >= _PATHOLOGICAL_Q_OVER_R for r in ratios)) if ratios else None
    return out


def compute_territorial_features(multilead_signal, fs, r_peaks=None):
    """Entry point: all territorial ST + Q features in one dict (R-peaks detected
    once on lead II if not supplied)."""
    sig = np.asarray(multilead_signal, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != len(PTBXL_LEAD_ORDER):
        raise ValueError(f"expected (n_samples, 12) raw signal, got {sig.shape}")
    if r_peaks is None:
        r_peaks = pan_tompkins_rpeaks(sig[:, lead_index(_RHYTHM_LEAD)], fs)
    feats = {}
    if r_peaks is None or len(r_peaks) < 2:
        for wall in WALLS:
            feats[f"st_elev_{wall}_mv"] = feats[f"st_recip_{wall}_mv"] = None
            feats[f"qr_ratio_{wall}"] = feats[f"qcount_{wall}"] = None
        return feats
    feats.update(compute_territorial_st(sig, fs, r_peaks))
    feats.update(compute_territorial_q(sig, fs, r_peaks))
    return feats
