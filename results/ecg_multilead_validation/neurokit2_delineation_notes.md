# NeuroKit2 for QRS delineation — why, config, and what it gave for BBB

## Why we tried it

Bundle-branch block (CD superclass) detection needs a reliable **QRS width** (complete
BBB requires ≥120 ms), which in turn needs proper **QRS onset/offset delineation**. Our
homemade amplitude heuristics failed (validation precision stuck at the ~2% base rate —
see `ecg_multilead_validation_report.md`). **NeuroKit2** is a well-known, validated ECG
library that provides wavelet-based delineation (`ecg_delineate`), so it was the natural
"proper tool" to test the hypothesis: *was the gap our measurement tool, or something
deeper?*

## Configuration

- **Install:** `pip install neurokit2` (installed version 0.2.13).
- **Dependency caveat:** NeuroKit2 requires `pandas < 3`, so installing it **downgraded
  pandas 3.0.1 → 2.3.3** in this environment. The project's pandas usage is conservative
  and works on 2.3.x; noted here only so the change isn't a surprise.
- **Optional / experimental only:** NeuroKit2 is NOT wired into any agent or the
  production pipeline (the test below did not validate). It is a documented experiment,
  like the Groq backend — nothing imports it outside the scratchpad test.
- **Resolution matters:** delineation was run on PTB-XL's **500 Hz** records
  (`filename_hr`, 2 ms/sample) rather than 100 Hz (10 ms/sample), since QRS width needs
  fine temporal resolution.

## How it is used (the delineation call)

```python
import neurokit2 as nk
cleaned = nk.ecg_clean(ecg_1lead, sampling_rate=500)          # filter
_, rinfo = nk.ecg_peaks(cleaned, sampling_rate=500)           # R-peaks
rpeaks = rinfo["ECG_R_Peaks"]
_, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=500, method="dwt")
qrs_width_ms = (np.array(waves["ECG_R_Offsets"]) - np.array(waves["ECG_R_Onsets"])) / 500 * 1000
```

- Only `method="dwt"` (and `"cwt"`) return `ECG_R_Onsets`/`ECG_R_Offsets`; `method="peak"`
  returns only P/Q/S/T peaks (no onsets/offsets) — it gave `None` for width here.

## What it gave for our case (n=300 records, BBB = CRBBB|CLBBB)

| Metric | Homemade heuristic | **NeuroKit2 dwt** |
|---|---|---|
| Best F1 (BBB) | ~0.04 (base rate) | **0.149** (at a recalibrated 220 ms) |
| Median QRS width, NORM-ish | (unusable) | **178 ms** |
| Median QRS width, BBB | (unusable) | **234 ms** |

**Verdict: better (~4×) but still NOT a usable flag.** Two problems remain:
1. **`dwt` over-estimates width systematically** — a normal QRS reads ~178 ms (should be
   ~90 ms), so the clinical 120 ms threshold is meaningless here (everything exceeds it);
   only a recalibrated ~220 ms threshold gives the best F1.
2. **Heavy distribution overlap** — despite separated medians (178 vs 234 ms), many NORM
   records read as wide, so precision stays ~0.05–0.09.

The single-lead-II test above under-samples the QRS (one lead misses the widest portion),
so the natural next step was the *proper* clinical measurement: delineate all 12 leads and
combine them. That was done — and the AGGREGATION across leads turned out to be the crux.

## 12-lead follow-up — aggregation is the crux (`src/ecg_bbb_neurokit_experiment.py`)

Delineated all 12 leads (one shared set of R-peaks from lead II) and combined the per-lead
widths three ways. On a **balanced** sample (60 BBB + 130 NORM, so the aggregations are
comparable without base-rate noise):

| Cross-lead aggregation | NORM / BBB median width | Best F1 (balanced) |
|---|---|---|
| `min(onset)`→`max(offset)` (global extent) | 292 / 298 ms | 0.480 (= base rate, useless) |
| median of per-lead widths | 170 / 190 ms | 0.512 |
| **75th-percentile of per-lead widths** | 194 / 220 ms | **0.562** |

**Key insight — the clinically "correct" global extent is the WORST aggregation.** Taking
`min(onset)` and `max(offset)` across 12 leads picks the *single most-erroneous* lead: one
lead whose dwt onset lands too early (or offset too late) inflates the width for the whole
beat. So the global extreme *amplifies* delineation error instead of averaging it — every
record reads ~290–300 ms and nothing separates. **Robust aggregations** (median, p75) that
*ignore* the outlier lead recover real separation (NORM 194 vs BBB 220 ms with p75).

## The honest number — at NATURAL prevalence (not the balanced sample)

The balanced F1 0.562 is inflated by the artificial 31.6% prevalence. Re-run on a natural
sample (first 449 records, BBB prevalence 3.1%, the setting comparable to Sokolow-Lyon):

| Approach | Best F1 (natural prevalence) |
|---|---|
| Homemade amplitude heuristic | ~0.04 |
| NeuroKit2, single lead II | 0.149 |
| **NeuroKit2, 12-lead + p75 robust aggregation** | **0.190** (P 0.114, R 0.571 @230 ms) |
| *(reference: Sokolow-Lyon LVH)* | *0.55* |
| *(reference: single-lead `wide_qrs`)* | *0.44* |

Best across all attempts, but **still not usable** (0.19 ≪ 0.44).

## Why the F1 stays low — measurement quality, NOT lack of data

The bottleneck is **precision** (recall is fine, 0.57): at the best threshold, 8 true
positives vs 62 false positives — ~25% of NORM records read as "wide" because dwt
over-reads, so NORM (196 ms) and BBB (235 ms) overlap heavily.

**This is not a data-quantity problem**, proven by the in-project counter-example: LVH is
*also* rare (9.5% prevalence) yet Sokolow-Lyon reaches F1 0.55 — because its measurement
(voltage in mV) is *clean and specific*. BBB has the noisier measurement (dwt over-reads the
QRS boundary), not less data. More records would only tighten the *estimate* (the 14-positive
natural sample is noisy), not raise the F1 (the balanced 60-positive sample already fixed
specificity at ~0.75). The base-rate effect (rare target → any false-positive rate crushes
precision) then amplifies the specificity shortfall — but that too is beaten by a *specific*
measurement, as LVH shows.

## Two clarifications (common questions)

- **The T-wave is not "noise" that preprocessing removes.** The signals ARE cleaned
  (`nk.ecg_clean`, a band-pass), not raw — but P and T waves are *wanted* signal, not
  artifact. The over-reading is the delineator confusing where the QRS ends and the T-wave
  begins (a morphology/boundary problem), which no denoising fixes. (The "raw mV" note
  earlier concerns only the amplitude features like Sokolow-Lyon, a different module, where
  real millivolts are needed.)
- **`min`/`max` across leads is exactly the fragile global-extent aggregation** — it is the
  clinically correct concept only with *perfect* delineation; with imperfect per-lead onsets/
  offsets, one bad lead dominates it (see table above).

## Verdict

NeuroKit2 + 12-lead + robust aggregation is the best BBB attempt (F1 0.19, ~5× the homemade
heuristic) and confirms part of the gap was the tool — but a shippable BBB flag needs a
**better-calibrated delineator (or per-beat/per-lead quality control) + intra-QRS morphology**
(rSR' in V1, notching in V6), a genuine sub-project. **Not shipped.** The single-lead
`wide_qrs_suspected` flag (F1 0.44, `flag_validation.py`) remains the CD-relevant flag.

Reproduce: `python src/ecg_bbb_neurokit_experiment.py balanced` (aggregation comparison) and
`python src/ecg_bbb_neurokit_experiment.py natural 450` (honest F1). Per-record cache:
`results/ecg_multilead_validation/bbb_neurokit_12lead_cache.csv`.
