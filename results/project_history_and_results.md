# Project history & detailed results (archive)

Verbose result tables and run histories moved out of `CLAUDE.md` to keep that file (reloaded every turn)
small. Nothing here is needed to *work on* the code day-to-day — it is the audit trail behind the numbers
`CLAUDE.md` now cites in one line. Authoritative per-run artifacts live under the `results/` subfolders noted
in each section.

---

## ECG embedding — training history (`src/ecg_embedding.py`)

Three runs, in order, each correcting a limitation of the last. Outputs under `results/ecg_embedding/`
(unsuffixed = quick subset, `_full` = leaky full run, `_stratfold` = leakage-free — **cite `_stratfold`**).

| Run | Data | Epochs | Split | Result |
|---|---|---|---|---|
| Quick subset | 4000 records | 10 | random, 500 held out | Held-out accuracy 83.8-90.4%; HYP at trivial baseline |
| Full (`_full`) | ~19.8k records | 25 | **random per-record**, 2000 held out | Accuracy 84.0-89.1%, real gains over quick subset - **but leaked** |
| Stratfold (`_stratfold`) | 17,418 train | 25 | **`strat_fold`, patient-disjoint** | Accuracy 84.9-89.1% - the numbers to cite |

- **Full run interpretation**: scaling 4k→19.8k records and 10→25 epochs improved every class except HYP
  (lift stuck at +0.4→+1.7 pts despite 5x more data/2.5x more epochs) - unlikely a data-volume problem given
  every other class improved; more likely the single-lead limitation (LVH is diagnosed from voltage criteria
  across *multiple* leads, so a single-lead model may structurally lack the signal).
- **Patient-level leakage found in the full run, then corrected**: the full split was random over *records*,
  not *patients*. PTB-XL has 18,869 patients / 21,799 records; verified 421 of 1,960 eval-set patients (22.3%
  of eval records) also appeared in training - inflating reported accuracy via partial memorization. Fix:
  re-split via `strat_fold` (folds 1-8 train / 9 val / 10 test, 0 patient overlap), retrain from scratch.
  Train/val/test sizes 17,418 / 2,183 / 2,198.

  | Class | Accuracy (leakage-free) | Prevalence | Lift | Accuracy (previous, leaky) |
  |---|---|---|---|---|
  | NORM | 84.9% | 43.8% | +28.8 pts | 89.1% |
  | MI | 85.8% | 25.0% | +10.8 pts | 86.4% |
  | CD | 87.8% | 22.6% | +10.4 pts | 89.0% |
  | STTC | 87.9% | 23.0% | +10.9 pts | 84.0% |
  | HYP | 89.1% | 11.9% | +1.0 pt | 89.0% |

  Most classes did not drop sharply once leakage was removed (NORM largest real decline, MI/CD stable, STTC
  improved) - expected, since `strat_fold` is a *different partition*, not the same test set with leaky
  patients removed (so not a precise leakage ablation). HYP's lift stays at the trivial baseline in both runs,
  reinforcing the multi-lead-limitation interpretation. Validated on an RTX 4050 (6.4GB VRAM); works on CPU.

---

## ECG report evaluation — result history & findings (`src/report_evaluation.py`)

Per-run interpretation is written to `results/evaluation*/evaluation_interpretation.md`
(`evaluation_brief/` holds the current largest/most-cited run).

**Reference-pool construction detail**: built from RTLP's English translations, not language detection - a
full-dataset cross-check found a **13.3% false-positive rate** for `langdetect` (of 6050 records it called
English, 807 were misclassified non-English, e.g. ecg_id `5186`'s German report). RTLP's
`en_df_round4.csv` covers 16247/21799 records (74.5%). `is_administrative_placeholder_report` excludes pure
admin notes ("trace only requested.") - discovered via `16360_lr` scoring BLEU 0.65/BERTScore 0.654,
meaningless. **Final pool: 12040 records** (16247 → 12279 after `validated_by_human` → 12040 after 239
placeholders), vs the old lang-detect pool of 7702. Caveat: most references now machine-translated from
German (added scoring noise).

**Result history** (chronological; later runs supersede earlier as "the" number):

| Run | n | Pool | Style | BLEU | ROUGE-1 | BERTScore F1 | Contrastive gap |
|---|---|---|---|---|---|---|---|
| 1 | 30 | 8047 (leaky lang-detect) | narrative | 0.97±0.60 | 0.053±0.038 | 0.705±0.039 | +0.002 |
| 2 | 80 | 8047 | narrative | 0.96±0.46 | 0.064±0.042 | 0.709±0.029 | +0.000 |
| 3 | 30 | 7702 (placeholder-filtered) | narrative | 1.02±0.67 | 0.064±0.047 | 0.706±0.034 | +0.000 |
| 4 | 30 | 12040 (RTLP-primary) | narrative | 0.80±0.55 | 0.055±0.040 | 0.708±0.028 | -0.003 |
| 5 | 30 | 12040 | brief + embedding | 0.63±0.33 | 0.067±0.039 | 0.725±0.024 | +0.003 |
| 6 | 150 | 12040 | brief + embedding | 4.36±5.40 | 0.260±0.193 | 0.820±0.046 | +0.002 |
| 7 | **300** | 12040 | brief + embedding | **4.29±5.60** | **0.262±0.207** | **0.821±0.049** | **+0.002** |

Runs 1-3 used the superseded lang-detect pool; run 4 confirmed the RTLP switch didn't change the story; runs
5-7 confirm the brief-style BLEU/ROUGE gain is real and reproducible at 10x sample. **Run 7 (n=300) is the
current citable number** - per-class BLEU/BERTScore F1: CD 2.93/0.822, HYP 2.93/0.819, MI 1.74/0.793, NORM
11.14/0.873 (NORM's outsized BLEU is a real class effect), STTC 2.65/0.798. Contrastive gap +0.002.

**Three findings, in discovery order:**
1. **BERTScore doesn't discriminate here at all** - across 5 runs the contrastive gap (real minus mismatched
   pairs) stayed within ±0.003 of zero (+0.002, +0.000, +0.000, -0.003, +0.002), incl. n=150/300. A structural
   property of raw (non-rescaled) BERTScore with this backbone on short clinical text: ~0.70-0.82 is the
   compressed baseline floor, not semantic alignment. With BLEU/ROUGE near-zero (telegraphic references vs
   synthesized narrative), **this setup cannot make a quantitative semantic-faithfulness claim** - a legitimate
   conclusion, not a failure. Untried: BERTScore baseline-rescaling, `roberta-large`, or non-reference-based
   scoring against the JSON findings.
2. **Reference = diagnostic interpretation, generated text = quantitative description only, by design.** Every
   text inspected confirmed it (e.g. `06783_lr` ref "p wave vector suggests respiratory disease partial RBBB"
   vs generated text restating HR/QRS/RR numbers). Not a bug - the prompt forbids diagnosis. This category
   mismatch motivated two done follow-ups: `flag_validation.py` and wiring the embedding into `ecg_agent.py`.
   Ceiling remains: diagnoses needing axis/morphology the single-lead pipeline never computes (axis deviation,
   BBB, fascicular block, voltage-criteria hypertrophy) - same root as HYP embedding stagnation; per-beat
   chunking is a candidate direction (needs its own dataset + training run).
3. **A brief, number-free style gives a real BLEU/ROUGE win, but BERTScore's rise is a trap.**
   `generate_brief_ecg_summary` improved BLEU/ROUGE via real lexical overlap ("sinus rhythm normal ecg"), but
   the BERTScore rise alongside is not a quality signal - the contrastive gap barely moved. NORM's outsized
   BLEU is just "normal" cases sharing more literal vocabulary with references.

---

## ECG deterministic-flag validation — full result & sweep (`src/flag_validation.py`)

Table also in `results/flag_validation/flag_validation_report.md`. Full-dataset result (21799 records), before
and after the tachycardia recalibration:

| Flag | Precision | Recall | F1 | Ground-truth prevalence |
|---|---|---|---|---|
| `tachycardia_suspected` (100→**120** bpm) | 0.015→**0.045** | 0.893→0.786 | 0.029→**0.086** | 0.13% (28 records) |
| `bradycardia_suspected` (60 bpm, unchanged) | 0.000 | n/a | n/a | 0% (`SBRAD` never appears with nonzero likelihood, confirmed) |
| `irregular_rhythm_suspected` (CV 0.15, unchanged) | 0.024 | 0.609 | 0.047 | 0.59% (128 records) |
| `wide_qrs_suspected` (120 ms, untouched) | **0.376** | 0.531 | **0.440** | 14.3% (3118 records) |

Only `wide_qrs_suspected` performs well absolutely. Original finding (not a logic bug): fixed numeric
thresholds fired on ordinary physiological rate/rhythm variation PTB-XL cardiologists didn't code as
diagnostic - a statistically elevated 10s heart rate ≠ a clinical "sinus tachycardia". Matters beyond the
experiment: these 4 flags feed real `flags_for_review` in every `AgentContribution`.

Sweep/recalibration (tachycardia + irregular rhythm only): Tachycardia F1 peaked at 165 bpm (0.213) but recall
collapsed to 0.286; **chose 120 bpm** (F1 0.086, recall 0.786), recall-favoring since `flags_for_review` is a
human-review alert not a classifier. Irregular rhythm F1 only crept 0.047→0.056 (0.15→0.26) while recall
dropped 0.609→0.339 - **kept at 0.15**, not fixable by threshold tuning alone.

---

## MRI deterministic-flag validation & raw-MRI extension (`src/mri_agent.py`)

Table also in `results/mri_flag_validation/mri_flag_validation_report.md`. Validated against ACDC pathology
group labels, all 150 patients (reuses `results/mri_features_all_patients.csv`, no LLM/reprocessing):

| Flag | Precision | Recall | F1 | Ground-truth prevalence |
|---|---|---|---|---|
| `low_lv_ejection_fraction_suspected` (45%) | **0.967** | 0.983 | **0.975** | 40% (60 patients) |
| `low_rv_ejection_fraction_suspected` (45%) | 0.424 | 0.833 | 0.562 | 20% (30 patients) |
| `high_lv_ejection_fraction_suspected` (65%) | 0.667 | 0.733 | 0.698 | 20% (30 patients) |

Much better calibrated than the ECG flags: `low_lv` is nearly perfect (59 TP / 2 FP / 1 FN) - the 45%
threshold cleanly separates DCM/MINF (mean EF 19.8%/30.8%) from HCM/NOR (69.8%/62.4%). `low_rv` is moderate
(34 FP) - RV function is naturally more variable. No recalibration needed for the original 2 flags.
**New flag `high_lv_ejection_fraction_suspected`** (HCM-associated) added from a 62-75% sweep (F1 peaks 65% at
0.698); applied as an `elif` after the low-EF check (mutually exclusive), validated end-to-end on `patient021`
(HCM, LV EF 67.6%).

**Raw-MRI pipeline extension** (full write-up: `results/report_package/05_future_work_raw_mri_pipeline/raw_mri_pipeline_proposal.md`).
Replaces ACDC's expert mask (a real patient's MRI has none): `mri_yolo_train.py` (YOLO26n, mAP50 0.976) +
`mri_unet_train.py` (`segmentation_models_pytorch` U-Net resnet18, val Dice 0.906); `mri_pipeline_validation.py`
confirms the predicted-mask pipeline reproduces the EF flags on 50 held-out patients. `mri_classifier_train.py`
adds a 5-class pathology classifier over image + predicted-mask (RV class F1 0.80, its strongest, vs the weak
`low_rv` flag; 62% per-patient accuracy, chance 20%). Wired into `mri_agent.py` via
`compute_pathology_classifier_features` (lazily imported) + optional params on
`build_mri_clinical_summary`/`detect_flags_mri`/`generate_mri_contribution` mirroring ECG's embedding
conventions. Validated end-to-end on `patient021` (HCM): `high_lv_ejection_fraction_suspected` +
`pathology_HCM_pattern_suspected` (99.0%) raised, all 5 pathology findings self-cited, no hallucinated evidence.
