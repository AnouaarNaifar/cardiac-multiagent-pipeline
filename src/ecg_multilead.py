"""
Multi-lead ECG features (B9, proof-of-concept: Sokolow-Lyon LVH voltage index).

Rationale: the deterministic pipeline in ecg_preprocessing.py analyses a SINGLE
lead end-to-end (documented limitation), so it can never compute the clinical
features that need several leads at once - axis, bundle-branch morphology, and
voltage-criteria hypertrophy. report_evaluation.py identified exactly this as the
project's diagnostic ceiling ("generated text is quantitative-only, no diagnostic
content"), and it is the standing interpretation of the HYP-class stagnation.

This module is ADDITIVE and standalone (mirrors how the learned embedding and the
raw-MRI pipeline were added): it does NOT modify ecg_preprocessing.py. It consumes
the SAME raw 12-lead array (n_samples, 12) already threaded into ecg_agent as
`multilead_signal`, and computes features from it directly. Amplitudes are measured
on the RAW signal in millivolts (NOT the z-scored / wavelet-denoised pipeline
output), because a voltage criterion is only meaningful in real mV.

First feature: the Sokolow-Lyon index for left ventricular hypertrophy (LVH):
    SL = S_amplitude(V1) + max(R_amplitude(V5), R_amplitude(V6))
LVH is suspected when SL >= 3.5 mV (the classic 35 mm at 10 mm/mV). This is a
deliberately simple, well-established, deterministic criterion so it can be
validated against PTB-XL scp_codes the same way flag_validation.py validates the
single-lead flags. It targets the HYP superclass specifically.
"""
import numpy as np

from ecg_preprocessing import pan_tompkins_rpeaks

# PTB-XL / wfdb standard 12-lead order (verified against a real record header).
PTBXL_LEAD_ORDER = ["I", "II", "III", "AVR", "AVL", "AVF",
                    "V1", "V2", "V3", "V4", "V5", "V6"]

# Rhythm lead used to locate beats (lead II is the conventional rhythm strip).
_RHYTHM_LEAD = "II"

# Classic Sokolow-Lyon LVH threshold: 3.5 mV (35 mm at 10 mm/mV).
SOKOLOW_LYON_LVH_THRESHOLD_MV = 3.5


def lead_index(name):
    """Column index of a lead name (case-insensitive) in a PTB-XL signal array."""
    return PTBXL_LEAD_ORDER.index(name.upper())


def measure_beat_amplitudes(lead_signal, r_peaks, fs, half_window_ms=60.0):
    """
    Per-beat R (max positive) and S (max negative depth) amplitudes in ONE lead,
    measured in the same units as `lead_signal` (mV for PTB-XL), relative to an
    isoelectric baseline estimated as the median of the whole lead (a decent proxy
    over a 10 s strip that is mostly baseline between beats).

    Inputs:
        lead_signal: 1D raw signal for a single lead (mV), length n_samples.
        r_peaks: sample indices of detected beats (from the rhythm lead - QRS is
            near-simultaneous across leads, so a window around each R-peak location
            contains that lead's QRS too).
        fs: sampling frequency in Hz.
        half_window_ms: half-width of the search window around each R-peak; the
            full window (~120 ms default) spans a whole QRS complex.

    Output:
        (r_amplitudes, s_amplitudes): two numpy arrays (one value per usable beat),
        both non-negative. Empty arrays if no beat has a usable window.
    """
    baseline = float(np.median(lead_signal))
    half_w = int(round(half_window_ms / 1000.0 * fs))
    n = len(lead_signal)
    r_amps, s_amps = [], []
    for peak in r_peaks:
        start, end = max(0, peak - half_w), min(n, peak + half_w + 1)
        if end - start < 3:
            continue
        window = lead_signal[start:end]
        r_amps.append(max(0.0, float(np.max(window)) - baseline))   # upward deflection
        s_amps.append(max(0.0, baseline - float(np.min(window))))   # downward deflection depth
    return np.array(r_amps), np.array(s_amps)


def measure_net_qrs(lead_signal, r_peaks, fs, half_window_ms=60.0):
    """
    Median SIGNED net QRS deflection in one lead (mV): per beat, the algebraic
    sum of the upward and downward deflections around the R-peak
    (max(window)-baseline) - (baseline-min(window)). Positive = net-upward QRS,
    negative = net-downward. Used for cardiac-axis estimation.
    """
    baseline = float(np.median(lead_signal))
    half_w = int(round(half_window_ms / 1000.0 * fs))
    n = len(lead_signal)
    nets = []
    for peak in r_peaks:
        start, end = max(0, peak - half_w), min(n, peak + half_w + 1)
        if end - start < 3:
            continue
        window = lead_signal[start:end]
        up = float(np.max(window)) - baseline
        down = baseline - float(np.min(window))
        nets.append(up - down)
    return float(np.median(nets)) if nets else None


def compute_cardiac_axis(multilead_signal, fs, r_peaks):
    """
    Mean QRS frontal-plane axis (degrees) from the net QRS of leads I (0 deg) and
    aVF (+90 deg): axis = atan2(net_aVF, net_I). Normal axis is -30..+90 deg; left
    axis deviation (LAD) < -30, right axis deviation (RAD) > +90. Returns
    {cardiac_axis_deg, left_axis_deviation (bool), right_axis_deviation (bool)},
    all None if not computable.
    """
    sig = np.asarray(multilead_signal, dtype=np.float64)
    net_i = measure_net_qrs(sig[:, lead_index("I")], r_peaks, fs)
    net_avf = measure_net_qrs(sig[:, lead_index("AVF")], r_peaks, fs)
    if net_i is None or net_avf is None or (net_i == 0 and net_avf == 0):
        return {"cardiac_axis_deg": None, "left_axis_deviation": None,
                "right_axis_deviation": None}
    axis = float(np.degrees(np.arctan2(net_avf, net_i)))
    return {"cardiac_axis_deg": axis,
            "left_axis_deviation": bool(-90 <= axis < -30),
            "right_axis_deviation": bool(90 < axis <= 180)}


def measure_st_level(lead_signal, r_peaks, fs, st_offset_ms=100.0):
    """
    Median ST-segment level in one lead (mV) relative to the isoelectric baseline
    (median of the lead), sampled at st_offset_ms after each R-peak (~J-point+60ms
    at a typical QRS width). Positive = ST elevation, negative = ST depression.
    """
    baseline = float(np.median(lead_signal))
    offset = int(round(st_offset_ms / 1000.0 * fs))
    n = len(lead_signal)
    levels = [float(lead_signal[p + offset]) - baseline
              for p in r_peaks if 0 <= p + offset < n]
    return float(np.median(levels)) if levels else None


# Leads used for ST deviation (aVR excluded - its ST direction is inverted and
# clinically read differently).
_ST_LEADS = ["I", "II", "III", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def compute_st_deviation(multilead_signal, fs, r_peaks):
    """
    Largest ST elevation and largest ST depression (both as non-negative mV
    magnitudes) across the standard leads (see _ST_LEADS). Targets the STTC
    (ST/T change) superclass. Returns {max_st_elevation_mv, max_st_depression_mv},
    None if not computable.
    """
    sig = np.asarray(multilead_signal, dtype=np.float64)
    levels = [measure_st_level(sig[:, lead_index(l)], r_peaks, fs) for l in _ST_LEADS]
    levels = [x for x in levels if x is not None]
    if not levels:
        return {"max_st_elevation_mv": None, "max_st_depression_mv": None}
    return {"max_st_elevation_mv": max(0.0, max(levels)),
            "max_st_depression_mv": max(0.0, -min(levels))}


def compute_rvh_features(multilead_signal, fs, r_peaks):
    """
    Right ventricular hypertrophy voltage marker: the R/S amplitude ratio in V1
    (a dominant R in V1, R/S > 1, is a classic RVH sign - and, combined with right
    axis deviation, completes the HYP superclass that Sokolow-Lyon's left-sided
    criterion does not cover). Returns {rs_ratio_v1}, None if not computable.
    """
    sig = np.asarray(multilead_signal, dtype=np.float64)
    r_v1, s_v1 = measure_beat_amplitudes(sig[:, lead_index("V1")], r_peaks, fs)
    if len(r_v1) == 0 or len(s_v1) == 0:
        return {"rs_ratio_v1": None}
    r, s = float(np.median(r_v1)), float(np.median(s_v1))
    return {"rs_ratio_v1": r / s if s > 1e-6 else None}


# NOTE - bundle-branch block (CD) was attempted here (a multi-lead QRS-width +
# V1-terminal-morphology criterion) and REMOVED: reliable QRS onset/offset
# delineation is a sub-project of its own (ecg_preprocessing.compute_qrs_durations
# is likewise documented as NOT a validated clinical delineator). Two width
# heuristics were tried and neither separated NORM from CRBBB/CLBBB (over- then
# under-estimation; validation precision stuck at the ~2% base rate). The existing
# single-lead `wide_qrs_suspected` flag (validated at F1 0.44 against the wide-QRS
# code set incl. CRBBB/CLBBB, see flag_validation.py) already covers the CD case,
# so no worse multi-lead width was shipped. See ecg_multilead_validation.md.


def compute_sokolow_lyon_mv(multilead_signal, fs, r_peaks=None):
    """
    Sokolow-Lyon LVH voltage index in mV from a raw 12-lead PTB-XL signal.

    Inputs:
        multilead_signal: raw numpy array of shape (n_samples, 12), wfdb.rdsamp's
            output shape, in mV (NOT normalized) - the SAME array ecg_agent already
            holds as `multilead_signal`.
        fs: sampling frequency in Hz.
        r_peaks: optional precomputed beat locations; if None, detected on lead II.

    Output:
        dict with the index and its components (all mV, or None if not computable):
            sokolow_lyon_index_mv, sl_s_v1_mv, sl_r_v5_mv, sl_r_v6_mv.
        Per-beat amplitudes are aggregated with the MEDIAN (robust to a noisy beat).
    """
    sig = np.asarray(multilead_signal, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != len(PTBXL_LEAD_ORDER):
        raise ValueError(
            f"expected (n_samples, {len(PTBXL_LEAD_ORDER)}) raw signal, got {sig.shape}")

    if r_peaks is None:
        r_peaks = pan_tompkins_rpeaks(sig[:, lead_index(_RHYTHM_LEAD)], fs)

    none_result = {"sokolow_lyon_index_mv": None, "sl_s_v1_mv": None,
                   "sl_r_v5_mv": None, "sl_r_v6_mv": None}
    if r_peaks is None or len(r_peaks) < 2:
        return none_result

    _, s_v1 = measure_beat_amplitudes(sig[:, lead_index("V1")], r_peaks, fs)
    r_v5, _ = measure_beat_amplitudes(sig[:, lead_index("V5")], r_peaks, fs)
    r_v6, _ = measure_beat_amplitudes(sig[:, lead_index("V6")], r_peaks, fs)
    if len(s_v1) == 0 or len(r_v5) == 0 or len(r_v6) == 0:
        return none_result

    s_v1_mv = float(np.median(s_v1))
    r_v5_mv = float(np.median(r_v5))
    r_v6_mv = float(np.median(r_v6))
    return {
        "sokolow_lyon_index_mv": s_v1_mv + max(r_v5_mv, r_v6_mv),
        "sl_s_v1_mv": s_v1_mv,
        "sl_r_v5_mv": r_v5_mv,
        "sl_r_v6_mv": r_v6_mv,
    }


def sokolow_lyon_lvh_suspected(sokolow_lyon_index_mv,
                               threshold_mv=SOKOLOW_LYON_LVH_THRESHOLD_MV):
    """True if the Sokolow-Lyon index meets the LVH threshold; False if below;
    None if the index itself is None (not computable)."""
    if sokolow_lyon_index_mv is None:
        return None
    return sokolow_lyon_index_mv >= threshold_mv


def compute_multilead_features(multilead_signal, fs, r_peaks=None):
    """
    Public entry point mirroring ecg_agent.compute_feature_dict's pattern: returns
    a flat dict of ALL multi-lead-derived numeric features, meant to be merged into
    the ECG feature dict. R-peaks are detected ONCE (lead II) and shared across the
    per-feature computations. Currently:
        - sokolow_lyon_index_mv (+ components)  -> HYP (LVH)
        - cardiac_axis_deg, left/right_axis_deviation
        - max_st_elevation_mv, max_st_depression_mv  -> STTC
        - rs_ratio_v1  -> HYP (RVH), with right axis
    """
    sig = np.asarray(multilead_signal, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] != len(PTBXL_LEAD_ORDER):
        raise ValueError(
            f"expected (n_samples, {len(PTBXL_LEAD_ORDER)}) raw signal, got {sig.shape}")
    if r_peaks is None:
        r_peaks = pan_tompkins_rpeaks(sig[:, lead_index(_RHYTHM_LEAD)], fs)

    features = compute_sokolow_lyon_mv(sig, fs, r_peaks=r_peaks)
    if r_peaks is None or len(r_peaks) < 2:
        # not enough beats: sokolow already returned Nones; fill the rest with Nones
        features.update({"cardiac_axis_deg": None, "left_axis_deviation": None,
                         "right_axis_deviation": None, "max_st_elevation_mv": None,
                         "max_st_depression_mv": None, "rs_ratio_v1": None})
        return features

    features.update(compute_cardiac_axis(sig, fs, r_peaks))
    features.update(compute_st_deviation(sig, fs, r_peaks))
    features.update(compute_rvh_features(sig, fs, r_peaks))
    return features
