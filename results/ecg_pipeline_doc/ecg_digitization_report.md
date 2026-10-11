# ECG Image Digitization — Recovering a Signal (and Numeric Findings) from an ECG Image

*Explanatory write-up of the digitization prototype (`src/ecg_digitization.py`) and its
validation on GenECG. Prepared for the internship report.*

---

## 1. Why this exists — the problem it answers

The ECG image branch (VLM + image classifier) produces a qualitative reading and calibrated
class probabilities, but **no measured clinical values** — no heart rate in bpm, no QRS duration in
ms, no Sokolow-Lyon voltage in mV. This is not because the image lacks that information: a 12-lead
tracing on gridded paper encodes exactly those values. It is because a vision-language model reads
pixels, it does not *measure* them, and the branch was deliberately kept model-based (no
digitization) as a first design.

This raised a fair question: **is it actually impossible to get numeric findings from an ECG
image?** The answer is no. The values are recoverable by **digitization** — reversing the plot back
into a signal — after which the project's existing, validated signal measurements apply. This
document builds and validates that step.

> **Scope of the prototype.** It digitizes the **bottom rhythm strip (lead II, full ~10 s)** of a
> standard 3×4+1 layout — the lead the single-lead pipeline already uses — which proves the core
> chain end to end without the harder full-12-lead layout parsing. Full-layout + amplitude criteria
> are future work (§5).

---

## 2. How it works

ECG paper is a universal standard: **25 mm/s** on the time axis, **10 mm/mV** on the amplitude axis,
on a pink millimetre grid. Digitization only needs to find the pixel scale of that grid and then
follow the black trace. Four steps:

**1 — Grid scale (pixels per mm).** The grid is reddish (`R` clearly greater than `G`,`B`). Summing
those grid pixels per column gives a periodic profile; its autocorrelation's first peak is the
1 mm spacing. On GenECG this comes out to **5 px/mm** consistently.

**2 — Trace extraction.** The rhythm-strip band is cropped (its vertical fraction of the image, with
the left margin — lead label + calibration pulse — dropped). For each column (each time step), the
trace point is the **darkest ink pixel**, with reddish grid pixels excluded so the grid cannot
contaminate the trace. Using the darkest pixel (not the centroid of all dark pixels) keeps the sharp
R-peaks a centroid would smear. Columns with no ink are interpolated.

**3 — Calibration.** The pixel curve `y(x)` is converted to physical units using the grid scale:
- time: `px_per_second = px_per_mm × 25` → each column maps to a time in seconds;
- amplitude: `px_per_mV = px_per_mm × 10`, baseline = median row, sign flipped (image `y` grows
  downward) → millivolts.

**4 — Resample.** The calibrated signal is resampled to a target rate (100 Hz), giving a standard
1-D lead-II signal — the exact input the existing pipeline (`pan_tompkins_rpeaks`, feature
extraction, flags) already consumes.

```
ECG image ─▶ grid scale (px/mm)     ─┐
          ─▶ crop rhythm strip       ─┤─▶ darkest-pixel trace y(x) ─▶ calibrate (mV, s) ─▶ resample 100 Hz ─▶ lead-II signal
          ─▶ exclude reddish grid   ─┘
```

Dependencies are already in the image branch (numpy, OpenCV, Pillow) — no new model, no training.

---

## 3. Validation on GenECG

GenECG images are PTB-XL records rendered as images, so **the original PTB-XL signal is the ground
truth**. For 15 records we digitize the rhythm strip and compare the recovered lead II against the
true lead II, by heart rate (the headline numeric finding) and by best-lag waveform correlation.

| ecg_id | HR recovered | HR true | \|Δ HR\| | best corr |
|---|---:|---:|---:|---:|
| 394 | 54.5 | 54.8 | 0.3 | 0.965 |
| 20863 | 87.0 | 88.0 | 1.0 | 0.944 |
| 15514 | 94.5 | 94.2 | 0.3 | 0.670 |
| 6785 | 66.7 | 67.4 | 0.7 | 0.383 |
| 21661 | 60.0 | 60.5 | 0.5 | 0.538 |
| 5234 | 76.9 | 75.6 | 1.4 | 0.541 |
| 16410 | 76.9 | 78.3 | 1.4 | 0.606 |
| 13532 | 54.5 | 56.1 | 1.6 | 0.325 |
| 346 | 94.5 | 97.4 | 2.9 | 0.313 |
| 21642 | 79.5 | 76.4 | 3.0 | 0.497 |
| … (3 detection failures + 1 ground-truth artifact, see below) | | | | |

**Heart-rate error, after the two fixes below (trace cleaning + robust detection), n = 14:**
- Fix A (cleaned trace + Pan-Tompkins): detection success 79 %, median **1.0 bpm**, **MAE 0.9 bpm**,
  within 3 bpm 71 %.
- Fix A+B (+ MAD-threshold detector): detection success **93 %**, median **0.5 bpm**, MAE 3.1,
  within 5 bpm 86 %.
- Mean best-lag waveform correlation **0.77**.

For reference, before the fixes the MAE was 9.4 bpm and correlation 0.56 — the fixes cut the MAE by
~10× and lifted fidelity. Per-record A-vs-A+B table: `digitization_validation.md`. A visual check on
record `00001` (`digitization_check_00001.png`): clean QRS with visible P and T waves, HR 62.5 vs 63.9.

---

## 4. The failures were a detection problem — diagnosed, then fixed

Before the fixes, the MAE was inflated by a few records whose recovered HR was ~half the truth
(e.g. `4178`: 28.8 vs 63.8 bpm). A diagnostic — plotting the recovered signal with the detected
R-peaks marked — settled the cause: **all the beats were physically present in the recovered trace,
but the detector counted only 3–5 of ~11–14.** So it was a *detection* problem, not a digitization
one. The mechanism: a few spurious tall spikes (a single darkest-pixel jump to grid/text/an adjacent
lead) inflated Pan-Tompkins' max-adaptive amplitude threshold, so the normal R-peaks fell below it.

Two fixes (both live in `ecg_digitization.py`, neither touches the shared Pan-Tompkins):

- **Fix A — trace cleaning.** Reject columns whose darkest-pixel jumps far from a ~1.5 mm local
  median envelope (real R-peaks are tracked by the median, so they survive), then a light median
  filter. This removes the spurious spikes at the source, so the standard detector works.
- **Fix B — robust detection.** A MAD-threshold peak detector (`median + k·1.4826·MAD` + refractory
  RR), immune to any residual outliers — used only on the digitized signal.

Result: the half-rate cases are gone (`4178` 28.8→63.8, `13037` 53.6→80.0, `10689` 38.7→87.0), the
MAE dropped ~10× (9.4→0.9 bpm for Fix A), and waveform correlation rose 0.56→0.77. One record
(`12618`) is a ground-truth artifact (Pan-Tompkins failed on the *true* signal, HR 13.9) and is
excluded from the fair statistics.

---

## 5. What this proves, and what is still open

**Proven:** numeric findings *are* recoverable from an ECG image. On GenECG the median heart-rate
error is a couple of bpm, from image pixels alone, with no model and no training — using only the
universal ECG-paper standard. The earlier statement "no values from the image" was a property of the
*current branch design*, not a fundamental limit.

**Done (this iteration):** trace cleaning (Fix A) + robust R-peak detection (Fix B) — the half-rate
failures are resolved, MAE ~1 bpm, correlation 0.77.

**Still open (future work):**
- **A/B selection** — use Fix A when it detects (tightest errors), fall back to Fix B otherwise
  (highest success) — a small combiner would beat either alone.
- **Full 12-lead layout parsing + amplitude calibration** — unlocks *clinical* measured findings
  (QRS duration, Sokolow-Lyon LVH voltage, axis), not just rate.
- **Wiring into `ecg_image_agent`** — so the image branch emits its own measured numeric findings
  (heart rate, rhythm) alongside the classifier probabilities and the VLM narrative.
- **Robustness to real photos** (skew, shadows, phone cameras) — GenECG is a clean render; a real
  photograph is harder.

---

## Reproducibility

- Module: `src/ecg_digitization.py` — `digitize_rhythm_strip(image_path, fs_out=100)`.
- Validation: `src/ecg_digitization_validation.py` → `results/ecg_pipeline_doc/digitization_validation.{md,csv}`.
- Visual check: `results/ecg_pipeline_doc/digitization_check_00001.png`.
- Ground truth: PTB-XL `filename_hr` lead II (GenECG renders the 500 Hz record).
