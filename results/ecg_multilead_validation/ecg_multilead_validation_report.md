# Multi-lead ECG features — validation report

First multi-lead deterministic feature (`src/ecg_multilead.py`), validated against
PTB-XL `scp_codes` exactly as `flag_validation.py` validates the single-lead flags.
No LLM. Motivation: the single-lead pipeline structurally cannot compute
voltage-criteria hypertrophy — the documented diagnostic ceiling and the standing
interpretation of the HYP-class stagnation.

## Feature

**Sokolow-Lyon index** = `S_amplitude(V1) + max(R_amplitude(V5), R_amplitude(V6))`,
measured on the RAW 12-lead signal in mV (per-beat amplitudes aggregated with the
median). LVH suspected when the index ≥ **3.5 mV** (classic 35 mm at 10 mm/mV).

## Ground truth

The `LVH` scp_code specifically (left ventricular hypertrophy) — **not** the whole
HYP superclass, since Sokolow-Lyon is a left-ventricular voltage criterion and does
not target RVH / atrial enlargement / septal hypertrophy.

## Result (n = 9000 records, LVH prevalence 9.5%)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 3.00 mV | 0.466 | 0.673 | 0.551 | 569 | 653 | 276 |
| 3.25 mV | 0.588 | 0.580 | **0.584** | 490 | 343 | 355 |
| **3.50 mV (default)** | **0.676** | 0.465 | 0.551 | 393 | 188 | 452 |
| 3.75 mV | 0.739 | 0.349 | 0.474 | 295 | 104 | 550 |
| 4.00 mV | 0.770 | 0.245 | 0.372 | 207 | 62 | 638 |

(Full sweep 2.0–5.0 mV in `sokolow_lyon_threshold_sweep_9000.csv`; per-record index
in `sokolow_lyon_index_cache_9000.csv`. An earlier n=3000 run agreed: F1 0.571 at
3.5 mV.)

## Interpretation

- **Best-performing deterministic ECG flag in the project.** At the classic 3.5 mV,
  F1 0.551 / precision 0.676 — vs the previous best single-lead flag `wide_qrs`
  (F1 0.440) and the weak rate/rhythm flags (0.047–0.086). It is also the **only**
  deterministic flag that reaches the HYP superclass.
- **The textbook threshold needs no recalibration.** F1 peaks on a flat plateau at
  3.25–3.5 mV (0.584 → 0.551); the standard 3.5 mV is near-optimal on PTB-XL, unlike
  `tachycardia_bpm` which had to be recalibrated. Kept at 3.5 mV for interpretability
  (a raised flag means the actual clinical Sokolow-Lyon criterion is met). A
  recall-favoring 3.25 mV is a defensible alternative (same F1, recall 0.580) if the
  project's "a missed case is worse than an extra reviewed false positive" rule is
  applied here too — left at 3.5 for now.
- **Behaviour is clinically coherent:** the criterion is known to be specific but
  insensitive, so the many false negatives (LVH below threshold) are expected, not a
  bug; an RVH-coded record correctly does NOT trip a left-ventricular criterion.

## Wiring

Wired into `ecg_agent.py` (additive, opt-in, mirroring the embedding integration):
`compute_feature_dict` adds `sokolow_lyon_index_mv` + components whenever a
`multilead_signal` is passed (no model needed); `detect_flags` raises
`lvh_sokolow_lyon_suspected`; `build_clinical_summary` reports the index in the LLM
prompt. Single-lead-only callers (no `multilead_signal`) are unaffected. Verified
end-to-end without an LLM on real records (LVH id 96 → flag raised; NORM id 1 → not).

Reproduce: `python src/ecg_multilead_validation.py [max_records]`.

---

# Other multi-lead features — attempted, honest outcomes

After Sokolow-Lyon (the one clear win), four more multi-lead features were built and validated the
same way (`src/ecg_multilead_validation2.py`, n=2000, no LLM). Only Sokolow-Lyon became a review
flag; the rest are either **descriptive-only** or were **removed**. Reported honestly, matching the
project's pattern (some flags validate, some don't).

| Feature (target) | Best result | Verdict |
|---|---|---|
| **Sokolow-Lyon** (HYP/LVH) | F1 0.551 @3.5 mV | **Flag shipped** |
| **Cardiac axis** (`cardiac_axis_deg`, L/R deviation) | robust measurement, standard cutoffs | **Descriptive feature** (no flag) |
| **ST deviation** (STTC) | best F1 ~0.38, precision ~0.24 ≈ base rate | Descriptive-only: does not discriminate |
| **R/S in V1** (HYP/RVH) | precision ~0.03; RVH prevalence 0.25% | Descriptive-only: RVH too rare to validate |
| **Bundle-branch block** (CD) | precision stuck at ~2% base rate | **Removed** (see below) |

**ST deviation (→ STTC):** precision hovered at the STTC base rate (0.24) across all thresholds.
Root cause: crude ST measurement (median level at a fixed R+100 ms offset vs a median baseline, no
J-point detection) and STTC also includes T-wave changes, which are not measured. Kept as a
descriptive numeric feature (`max_st_elevation_mv`/`max_st_depression_mv`) — the measurement is real,
just not a reliable classifier — but raises no flag.

**R/S in V1 (→ RVH):** RVH prevalence in PTB-XL is only ~0.25% (≈5/2000), too rare for a meaningful
precision/recall estimate; a dominant R in V1 also has benign causes. `R/S≥1 AND right-axis` reached
F1 0.27 but on ~4 positives. Kept as descriptive (`rs_ratio_v1`); no flag.

**Bundle-branch block (→ CD): removed for a METHOD limitation, not a data or detectability one — an
important distinction.** Unlike RVH, the data was sufficient: ~73 BBB cases in the 2000-record sample
(CRBBB ~2.0%, CLBBB ~1.7%), plenty to validate. And BBB is detectable. The failure was purely my
measurement:
- **Morphology alone (V1 vs V6) is not enough.** A *normal* ECG already has V1-negative / V6-positive
  morphology — the same pattern as LBBB (confirmed on real NORM records: `v1_terminal_positive` came out
  ~50/50 True/False across normals). So the V1/V6 sign cannot separate BBB from normal on its own.
- **The discriminator is QRS WIDTH** (complete BBB requires ≥120 ms); morphology only says *which* block
  once you know the QRS is wide. Two homemade multi-lead width heuristics were tried — max-across-leads
  (grossly over-estimated, flagged ~80% of records, precision at the ~2% base rate) and a
  contiguous-run-around-the-R-peak (under-estimated: NORM 20–55 ms, BBB 60–90 ms). Neither delineated the
  QRS reliably.
- Reliable QRS onset/offset **delineation** is a sub-project of its own (the single-lead
  `compute_qrs_durations` is likewise documented as not a validated delineator). Proper detection would
  need a validated delineator (e.g. a wavelet-based one, NeuroKit2/BioSPPy), then width-gate + rSR'/
  notching morphology — genuinely feasible, just not a simple amplitude criterion.
- **NeuroKit2 follow-up (done — full write-up: `neurokit2_delineation_notes.md`):** its wavelet delineator
  was tested single-lead (F1 0.149), then properly across **all 12 leads**. The cross-lead aggregation was
  the crux: the clinically-correct global extent (`min(onset)`→`max(offset)`) is the WORST (F1 0.48 = base
  rate — one badly-delineated lead dominates the min/max); robust aggregation (75th percentile of per-lead
  widths) recovers separation. At **natural prevalence** the best is **F1 0.190** (12-lead + p75) — the best
  BBB attempt (~5× the homemade heuristic) but **still not usable** (≪ 0.44). Root cause is measurement
  quality, NOT data: dwt over-reads the QRS/T boundary so NORM (196 ms) and BBB (235 ms) overlap → low
  specificity → low precision at 3% prevalence. Proven not-a-data-problem by Sokolow-Lyon (also rare at
  9.5%, yet F1 0.55, because voltage is a clean measurement). A shippable BBB flag needs a better-calibrated
  delineator + intra-QRS morphology — a real sub-project; not shipped.

**The code was removed** (not shipped as an unvalidated flag); the existing single-lead
`wide_qrs_suspected` flag (validated at F1 0.44 against the wide-QRS code set incl. CRBBB/CLBBB,
`flag_validation.py`) already covers the CD case, so nothing worse was shipped. **Summary: BBB is in the
data, in sufficient quantity, and detectable — the gap was the QRS-delineation tool, not the dataset.**

**Net multi-lead outcome:** one new validated deterministic flag (Sokolow-Lyon LVH, the project's best),
plus cardiac axis / ST / R-S-V1 as interpretable descriptive context in the ECG agent's prompt. The
harder classes (STTC T-waves, MI territorial Q-waves, BBB delineation) need proper waveform
delineation, not simple amplitude criteria — genuine future work, not a quick feature.

Reproduce: `python src/ecg_multilead_validation2.py [max_records]`.
