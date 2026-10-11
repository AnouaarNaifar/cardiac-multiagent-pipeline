"""
ECG image digitization (prototype) - recover a 1-D signal from an ECG IMAGE.

Scope of this prototype: the bottom RHYTHM STRIP (lead II, full width ~10 s) of a
standard 3x4+1 layout, which is the lead the single-lead pipeline already uses.
This proves the core chain works - grid-scale detection -> trace extraction ->
mV/ms calibration -> resampled signal - without the harder 12-lead layout parsing.
Validated on GenECG (images rendered from PTB-XL), where the original signal is the
ground truth (see ecg_digitization_validation.py).

Standard ECG paper: 25 mm/s (time), 10 mm/mV (amplitude). We recover pixels-per-mm
from the grid, then convert. Colour cue: the grid is pink/red, the trace is black.

Dependencies: numpy, opencv-python, Pillow (all already in the image branch).
"""
import numpy as np
from PIL import Image

PAPER_SPEED_MM_S = 25.0      # standard
PAPER_GAIN_MM_MV = 10.0      # standard


def _load_rgb(image_path):
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def estimate_px_per_mm(rgb):
    """
    Estimate pixels-per-millimetre from the pink grid, via autocorrelation of a
    'grid-ness' column projection. The grid is reddish (R clearly > G,B); its
    vertical lines are periodic at 1 mm (small square). The first autocorrelation
    peak after lag 0 is that period.
    """
    r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
    gridness = ((r - np.maximum(g, b)) > 18).astype(float)   # reddish grid pixels
    col_profile = gridness.mean(axis=0)                      # over rows -> per-column
    col_profile = col_profile - col_profile.mean()
    ac = np.correlate(col_profile, col_profile, mode="full")[len(col_profile) - 1:]
    # search a plausible small-square range: 3..40 px per mm
    lo, hi = 3, min(40, len(ac) - 1)
    if hi <= lo:
        return None
    peak = lo + int(np.argmax(ac[lo:hi]))
    return float(peak)


def extract_trace(strip_rgb, ink_thresh=120):
    """
    Column-wise trace extraction on a cropped strip: for each x, the trace's y is
    the DARKEST ink pixel (min grey), which keeps sharp R-peaks that a centroid
    would smear. Grid pixels (reddish) are excluded from the ink mask so the pink
    grid doesn't contaminate the trace. Columns with no ink are NaN (interpolated
    by the caller). Returns a float array y(x) in pixels (image coords, y down).
    """
    r = strip_rgb[:, :, 0].astype(int)
    g = strip_rgb[:, :, 1].astype(int)
    b = strip_rgb[:, :, 2].astype(int)
    gray = strip_rgb.mean(axis=2)
    reddish_grid = (r - np.maximum(g, b)) > 18
    ink = (gray < ink_thresh) & (~reddish_grid)
    gray_masked = np.where(ink, gray, np.inf)          # non-ink -> +inf
    ys = np.argmin(gray_masked, axis=0).astype(float)  # darkest ink row per column
    ys[~np.isfinite(gray_masked.min(axis=0))] = np.nan  # columns with no ink
    return ys


def _interp_nans(y):
    idx = np.arange(len(y))
    good = ~np.isnan(y)
    if good.sum() < 2:
        return None
    return np.interp(idx, idx[good], y[good])


def _clean_trace(ys, px_per_mm, dev_mm=4.0, smooth_px=3):
    """
    Fix A - remove isolated spurious spikes from the pixel trace (the artifacts
    that later break the R-peak detector's adaptive threshold). Two steps:
      1. Reject columns whose darkest-pixel y deviates by more than dev_mm from a
         ~1.5 mm local MEDIAN envelope (a real R-peak is tracked by the median, so
         it is NOT flagged; only 1-few-column jumps to grid / text / an adjacent
         lead are), then interpolate them.
      2. A light median filter to polish any single-pixel residue.
    """
    from scipy.ndimage import median_filter
    filled = _interp_nans(ys)
    if filled is None:
        return None
    env_win = max(3, int(round(1.5 * px_per_mm)) | 1)   # odd window ~1.5 mm
    env = median_filter(filled, size=env_win)
    spike = np.abs(filled - env) > dev_mm * px_per_mm
    filled[spike] = np.nan
    filled = _interp_nans(filled)
    if filled is None:
        return None
    return median_filter(filled, size=(smooth_px | 1))


def robust_r_peaks(signal, fs, refractory_s=0.30, k=4.0):
    """
    Fix B - R-peak detection robust to residual outlier spikes, for the digitized
    signal only (NOT a change to the shared Pan-Tompkins). Threshold = k x robust
    sigma (1.4826 x MAD), which a few giant values do not inflate - unlike
    Pan-Tompkins' max-adaptive threshold - plus a refractory minimum RR.
    """
    from scipy.signal import find_peaks
    x = np.asarray(signal, dtype=float)
    x = x - np.median(x)
    sigma = 1.4826 * (np.median(np.abs(x)) + 1e-9)
    peaks, _ = find_peaks(x, height=k * sigma, distance=max(1, int(refractory_s * fs)))
    return peaks


def digitize_rhythm_strip(image_path, fs_out=100.0,
                          strip_top_frac=0.72, strip_bot_frac=0.95,
                          left_crop_frac=0.06):
    """
    Digitize the bottom rhythm strip of a GenECG-style ECG image into a 1-D signal.

    Inputs:
        image_path: path to the ECG image.
        fs_out: output sampling frequency (Hz) after resampling.
        strip_top/bot_frac: vertical band (fraction of height) holding the strip.
        left_crop_frac: drop the left margin (lead label + calibration pulse).

    Output:
        (signal_mv, fs_out, meta) - signal in mV at fs_out; meta has px_per_mm,
        px_per_s, seconds, n_ink_cols. Returns (None, fs_out, meta) on failure.
    """
    rgb = _load_rgb(image_path)
    H, W = rgb.shape[:2]
    px_per_mm = estimate_px_per_mm(rgb)
    meta = {"px_per_mm": px_per_mm, "img_shape": (H, W)}
    if not px_per_mm:
        return None, fs_out, meta

    y0, y1 = int(H * strip_top_frac), int(H * strip_bot_frac)
    x0 = int(W * left_crop_frac)
    strip = rgb[y0:y1, x0:]
    ys = extract_trace(strip)
    meta["n_ink_cols"] = int(np.sum(~np.isnan(ys)))
    ys = _clean_trace(ys, px_per_mm)   # Fix A: despike (continuity) + median polish
    if ys is None:
        return None, fs_out, meta

    # calibrate: y (px, downward) -> mV (up positive); baseline = median
    px_per_mv = px_per_mm * PAPER_GAIN_MM_MV
    signal_mv = -(ys - np.median(ys)) / px_per_mv

    # time axis: x px -> seconds; resample to fs_out
    px_per_s = px_per_mm * PAPER_SPEED_MM_S
    seconds = len(signal_mv) / px_per_s
    n_out = max(2, int(round(seconds * fs_out)))
    t_src = np.arange(len(signal_mv)) / px_per_s
    t_dst = np.arange(n_out) / fs_out
    signal_out = np.interp(t_dst, t_src, signal_mv)

    meta.update({"px_per_s": px_per_s, "seconds": seconds, "n_out": n_out})
    return signal_out.astype(np.float32), fs_out, meta


def measure_rhythm_from_image(image_path, fs_out=100.0, cv_threshold=0.15):
    """
    High-level entry point for the ECG image agent: digitize the rhythm strip and
    return the VALIDATED numeric measurements (rate + rhythm only - heart rate is
    validated to ~1 bpm on GenECG; morphology/voltage from a single strip is NOT
    validated and is deliberately not exposed here). R-peaks via the robust
    detector (Fix B, highest success rate).

    Output:
        dict of "digitized_*" numeric features, or None if digitization / peak
        detection fails.
    """
    sig, fs, meta = digitize_rhythm_strip(image_path, fs_out=fs_out)
    if sig is None:
        return None
    # Combiner: Pan-Tompkins first (tightest errors when it detects, MAE ~0.9),
    # fall back to the robust MAD detector (highest success) when it finds < 2.
    peaks = None
    try:
        from ecg_preprocessing import pan_tompkins_rpeaks
        pt = pan_tompkins_rpeaks(np.asarray(sig, dtype=float), fs)
        if len(pt) >= 2:
            peaks = pt
    except Exception:  # noqa: BLE001
        peaks = None
    if peaks is None or len(peaks) < 2:
        peaks = robust_r_peaks(sig, fs)
    if len(peaks) < 2:
        return None
    rr = np.diff(peaks) / fs                      # seconds
    hr = 60.0 / float(np.median(rr))
    cv = float(np.std(rr) / (np.mean(rr) + 1e-9))
    return {
        "digitized_heart_rate_bpm": round(hr, 1),
        "digitized_rr_std_ms": round(float(np.std(rr) * 1000), 1),
        "digitized_rhythm_cv": round(cv, 3),
        "digitized_rhythm_irregular": bool(cv > cv_threshold),
        "digitized_n_beats": int(len(peaks)),
        "digitized_seconds": round(float(meta.get("seconds", 0.0)), 1),
        "digitized_px_per_mm": meta.get("px_per_mm"),
    }
