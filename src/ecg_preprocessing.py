"""
ECG signal preprocessing pipeline
================================
Steps:
    1. Baseline wander removal       (high-pass filter, 0.5 Hz)
    2. Powerline interference removal (notch filter, 50/60 Hz)
    3. Wavelet denoising              (db6, 5 levels, universal soft thresholding)
    4. Statistical normalization     (zero mean, unit variance)
    5. R-peak detection               (Pan-Tompkins algorithm)
    6. QRS duration estimation       (amplitude-threshold heuristic, not a validated
                                       clinical onset/offset detector)
    7. Rhythm irregularity detection (RR interval coefficient of variation)

Only a single lead is processed per call - the whole pipeline operates on one 1D
array end to end (see CLAUDE.md for the project-wide implications of this).

Dependencies: numpy, scipy, pywavelets, matplotlib
    pip install numpy scipy pywavelets matplotlib
"""

import os

try:
    import numpy as np
    import pywt
    from scipy.signal import butter, iirnotch, filtfilt
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc}. Install the required packages with: "
        "pip install numpy scipy pywavelets matplotlib wfdb"
    ) from exc


# ----------------------------------------------------------------------
# 1. Baseline wander removal
# ----------------------------------------------------------------------
def remove_baseline_wander(sig, fs, cutoff=0.5, order=4):
    """
    Butterworth high-pass filter to remove low-frequency drift
    (patient movement, respiration, poor electrode contact).
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, sig)


# ----------------------------------------------------------------------
# 2. Powerline interference removal
# ----------------------------------------------------------------------
def remove_powerline_interference(sig, fs, freq=50.0, bandwidth=2.0):
    """
    Notch filter centered at 50 Hz (Europe/Africa/Asia)
    or 60 Hz (North America), with a narrow bandwidth (~2 Hz).
    """
    Q = freq / bandwidth  # quality factor
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, sig)


# ----------------------------------------------------------------------
# 3. Wavelet denoising (Daubechies 6)
# ----------------------------------------------------------------------
def wavelet_denoise(sig, wavelet="db6", level=5):
    """
    5-level wavelet decomposition with soft thresholding,
    based on the universal threshold: threshold = sigma * sqrt(2 * log(N))
    where sigma is estimated from the level-1 detail coefficients
    (robust estimator based on MAD).
    """
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    detail_coeffs_lvl1 = coeffs[-1]

    # Robust noise estimation (Mean Absolute Deviation)
    sigma = np.median(np.abs(detail_coeffs_lvl1)) / 0.6745

    N = len(sig)
    universal_threshold = sigma * np.sqrt(2 * np.log(N))

    # Soft thresholding applied only to detail coefficients
    new_coeffs = [coeffs[0]]  # approximation unchanged
    for detail in coeffs[1:]:
        new_coeffs.append(pywt.threshold(detail, universal_threshold, mode="soft"))

    denoised = pywt.waverec(new_coeffs, wavelet)
    return denoised[:N]  # adjust length (waverec may add one sample)


# ----------------------------------------------------------------------
# 4. Statistical normalization
# ----------------------------------------------------------------------
def normalize_signal(sig):
    """Zero mean and unit variance (z-score)."""
    return (sig - np.mean(sig)) / np.std(sig)


# ----------------------------------------------------------------------
# 5. R-peak detection — Pan-Tompkins algorithm
# ----------------------------------------------------------------------
def pan_tompkins_rpeaks(sig, fs):
    """
    Steps: band-pass filtering -> differentiation -> squaring ->
            moving-window integration -> adaptive thresholding

    Returns the indices (samples) of the detected R peaks.
    """
    # --- band-pass filtering (5-15 Hz, typical QRS band) ---
    nyquist = 0.5 * fs
    low, high = 5 / nyquist, 15 / nyquist
    b, a = butter(2, [low, high], btype="band")
    filtered = filtfilt(b, a, sig)

    # --- differentiation (enhances the steep QRS slope) ---
    diff = np.diff(filtered, prepend=filtered[0])

    # --- squaring (makes values positive and emphasizes high frequencies) ---
    squared = diff ** 2

    # --- moving-window integration (~150 ms) ---
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode="same")

    # --- adaptive thresholding ---
    fs_i = int(fs)
    r_peaks = []
    refractory_samples = int(0.2 * fs)  # refractory period ~200 ms
    threshold = 0.3 * np.max(integrated)
    i = 0
    while i < len(integrated):
        if integrated[i] > threshold:
            # find the local maximum within a window around the threshold crossing
            window_end = min(i + refractory_samples, len(integrated))
            local_max_idx = i + np.argmax(integrated[i:window_end])
            r_peaks.append(local_max_idx)
            # adaptive threshold update (simple moving average)
            threshold = 0.3 * np.max(integrated[max(0, local_max_idx - fs_i):local_max_idx + fs_i])
            i = local_max_idx + refractory_samples
        else:
            i += 1

    return np.array(r_peaks)


def compute_rr_intervals(r_peaks, fs):
    """RR intervals in seconds, derived from the R-peak indices."""
    return np.diff(r_peaks) / fs


# ----------------------------------------------------------------------
# 6. QRS duration (amplitude-threshold heuristic)
# ----------------------------------------------------------------------
def compute_qrs_durations(sig, r_peaks, fs, search_half_width_ms=100.0, amplitude_fraction=0.1):
    """
    Estimate QRS complex duration (in ms) around each detected R peak.

    Method: starting from each R peak, walk outward in both directions until
    the signal amplitude drops below `amplitude_fraction` of the peak's
    amplitude, bounded by a maximum search half-width. This is a simple
    amplitude-threshold heuristic, not a validated clinical QRS onset/offset
    detector - it gives a rough morphology feature, not a precise measurement.

    Inputs:
        sig: 1D signal to measure on (typically the normalized signal from
            ecg_preprocessing_pipeline).
        r_peaks: array of R-peak sample indices.
        fs: sampling frequency in Hz.
        search_half_width_ms: maximum distance (ms) to search on each side of
            the peak.
        amplitude_fraction: fraction of the peak's absolute amplitude used as
            the onset/offset threshold.

    Output:
        1D numpy array of estimated QRS durations in ms, one per R peak
        (empty array if r_peaks is empty).
    """
    half_window = int(search_half_width_ms / 1000.0 * fs)
    durations_ms = np.zeros(len(r_peaks))

    for i, r in enumerate(r_peaks):
        peak_val = abs(sig[r])
        threshold = amplitude_fraction * peak_val

        left = r
        limit_left = max(0, r - half_window)
        while left > limit_left and abs(sig[left]) > threshold:
            left -= 1

        right = r
        limit_right = min(len(sig) - 1, r + half_window)
        while right < limit_right and abs(sig[right]) > threshold:
            right += 1

        durations_ms[i] = (right - left) / fs * 1000.0

    return durations_ms


# ----------------------------------------------------------------------
# 7. Rhythm irregularity (coefficient of variation of RR intervals)
# ----------------------------------------------------------------------
def detect_rhythm_irregularity(rr_intervals, cv_threshold=0.15):
    """
    Flag whether the rhythm appears irregular, based on the coefficient of
    variation (std / mean) of the RR intervals. This is a coarse heuristic
    meant to surface a review flag, not a clinical arrhythmia classification.

    Inputs:
        rr_intervals: array of RR intervals in seconds (as returned by
            compute_rr_intervals).
        cv_threshold: coefficient-of-variation value above which the rhythm
            is flagged as irregular.

    Output:
        (is_irregular, coefficient_of_variation) tuple. is_irregular is False
        and coefficient_of_variation is None when fewer than 2 RR intervals
        are available.
    """
    if len(rr_intervals) < 2:
        return False, None
    cv = float(np.std(rr_intervals) / np.mean(rr_intervals))
    return cv > cv_threshold, cv


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
def ecg_preprocessing_pipeline(raw_signal, fs, powerline_freq=50.0):
    """
    Apply the full preprocessing pipeline and return a dictionary
    containing each intermediate step, the detected R peaks, RR intervals,
    per-beat QRS duration estimates, and a rhythm-irregularity flag.
    """
    step1 = remove_baseline_wander(raw_signal, fs)
    step2 = remove_powerline_interference(step1, fs, freq=powerline_freq)
    step3 = wavelet_denoise(step2)
    step4 = normalize_signal(step3)
    r_peaks = pan_tompkins_rpeaks(step4, fs)
    rr_intervals = compute_rr_intervals(r_peaks, fs)
    qrs_durations_ms = compute_qrs_durations(step4, r_peaks, fs)
    rhythm_irregular, rhythm_cv = detect_rhythm_irregularity(rr_intervals)

    return {
        "raw": raw_signal,
        "baseline_removed": step1,
        "powerline_removed": step2,
        "wavelet_denoised": step3,
        "normalized": step4,
        "r_peaks": r_peaks,
        "rr_intervals": rr_intervals,
        "qrs_durations_ms": qrs_durations_ms,
        "rhythm_irregular": rhythm_irregular,
        "rhythm_cv": rhythm_cv,
    }


# ----------------------------------------------------------------------
# Visualization (reproduces the figure from the document)
# ----------------------------------------------------------------------
def plot_pipeline(results, fs, t_start=1.0, t_end=3.0, output_dir=None, file_name="ecg_pipeline_result"):
    t = np.arange(len(results["raw"])) / fs
    mask = (t >= t_start) & (t <= t_end)

    fig, axes = plt.subplots(5, 1, figsize=(9, 12), sharex=True)

    axes[0].plot(t[mask], results["raw"][mask])
    axes[0].set_title("Raw ECG Signal")
    axes[0].set_ylabel("Amplitude (mV)")

    axes[1].plot(t[mask], results["baseline_removed"][mask])
    axes[1].set_title("After Baseline Wander Removal")
    axes[1].set_ylabel("Amplitude (mV)")

    axes[2].plot(t[mask], results["powerline_removed"][mask])
    axes[2].set_title("After Powerline Interference Removal")
    axes[2].set_ylabel("Amplitude (mV)")

    axes[3].plot(t[mask], results["wavelet_denoised"][mask])
    axes[3].set_title("After Wavelet Denoising")
    axes[3].set_ylabel("Amplitude (mV)")

    axes[4].plot(t[mask], results["normalized"][mask])
    r_peaks_in_range = results["r_peaks"][
        (results["r_peaks"] / fs >= t_start) & (results["r_peaks"] / fs <= t_end)
    ]
    axes[4].plot(r_peaks_in_range / fs, results["normalized"][r_peaks_in_range], "kv", label="QRS")
    axes[4].set_title("Normalized ECG Signal + R-peaks")
    axes[4].set_ylabel("Amplitude")
    axes[4].set_xlabel("Time (s)")
    axes[4].legend()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name}.png")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Figure saved to: {output_path}")
    plt.show()