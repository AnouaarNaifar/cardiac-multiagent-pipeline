## Structured patient data agent (3rd modality)

**Status: built and validated.** Same contract as `ecg_agent.py`/`mri_agent.py` -
produces structured findings only, no diagnosis; the Sep Agent combines all
three modalities.

**Data source**: UCI Heart Disease, Cleveland subset, 303 patients (UCI repo
id=45). Fetched once via `ucimlrepo` (authoritative column names/types),
saved locally at `uci_heart_disease/processed_cleveland.csv` +
`variables_metadata.csv` - the agent reads this local file, never a live
network fetch at runtime (`ucimlrepo` doesn't cache; confirmed empirically
two consecutive fetches took the same ~1.7s each).

### Files
- `src/patient_data_agent.py` - the agent (`compute_feature_dict`,
  `detect_flags_patient`, `generate_patient_contribution`, `agent_id="patient"`).
- `src/patient_data_flag_validation.py` - no-LLM validation against the
  binary target, mirrors `flag_validation.py`.
- `results/patient_data_flag_validation_report.md` - full precision/recall/F1 table.
- `results/patient_data_agent_test/` - one real end-to-end run (real MedGemma call).

### Feature dict (13 fields, no namespace prefix)
Confirmed directly in code that neither `ecg_agent.py` nor `mri_agent.py`
actually namespaces its keys (an earlier docstring note suggesting future
agents should was never implemented) - and it would be redundant here too,
since every citation is already `(agent_id, feature)`, not a bare feature
name. Direct reads: `age_years`, `sex`, `chest_pain_type`, `fasting_blood_sugar_elevated`,
`resting_ecg_result`, `exercise_induced_angina`, `st_segment_slope`,
`n_major_vessels_colored`, `thalassemia_result`. Qualitatively interpreted
(via the deterministic flags below, never guessed): `resting_bp_mmHg`,
`cholesterol_mgdl`, `max_heart_rate_achieved_bpm`, `st_depression_oldpeak_mm`.
Missing values (`ca`/`thal` only, 4/2 patients) -> `None`, omitted
gracefully from the summary, never guessed.

### Deterministic flags - validated against `heart_disease_present = (num > 0)`, no LLM, all 303 patients

| Flag | Precision | Recall | F1 | Change from starting value |
|---|---|---|---|---|
| `abnormal_thallium_result_suspected` | 0.745 | 0.734 | **0.739** | none (categorical, no threshold) |
| `multiple_vessels_involved_suspected` | 0.748 | 0.662 | 0.702 | none - confirmed optimal at ≥1 vessel |
| `significant_st_depression_suspected` | 0.654 | 0.719 | 0.685 | **oldpeak 1.0 → 0.8 mm** |
| `reduced_exercise_capacity_suspected` | 0.646 | 0.683 | 0.664 | **formula 220-age → Tanaka; 85% → 90%** |
| `exercise_induced_angina_present` | 0.768 | 0.547 | 0.639 | none (binary, no threshold) |
| `high_cholesterol_suspected` | 0.526 | 0.576 | 0.550 | none - see note below |
| `high_resting_bp_suspected` | 0.591 | 0.281 | 0.380 | none - see note below |

**Two flags recalibrated** via threshold sweep (same discipline as
`ecg_agent.tachycardia_bpm`'s 100→120 story):
- `oldpeak_threshold`: 1.0 → 0.8 mm - Pareto-dominates 1.0 on both recall and F1.
- `max_hr_formula` + `max_hr_pct_threshold`: 220-age@85% → **Tanaka (208-0.7×age) @90%** -
  Tanaka beat 220-age at essentially every matched percentage in a joint sweep;
  90% chosen one step below the F1-max (91%), favoring recall per this
  project's established "a missed case is worse than an extra reviewed
  false positive" philosophy.

**Two flags kept unchanged despite a weak-looking sweep** (`high_resting_bp_suspected`,
`high_cholesterol_suspected`): precision stays close to the dataset's base
prevalence (0.459) at nearly every threshold tested - F1 only rises by
flagging almost everyone, a statistically marginal, clinically meaningless
gain. Same pattern already documented for ECG's `irregular_rhythm_suspected`:
kept at the clinically meaningful cutoff (140 mmHg / 240 mg/dl) rather than
chasing that gain. Resting BP/cholesterol alone are simply weak CAD
predictors in this dataset, at any threshold - reported honestly, not forced.

### Validated end-to-end
Real MedGemma call, patient `cleveland_0`: 13/13 findings generated, all
`evidence_ref` valid (no fabricated citations), first-attempt parse
success, 3 flags correctly raised and merged
(`abnormal_thallium_result_suspected`, `high_resting_bp_suspected`,
`significant_st_depression_suspected`) - this specific patient's ground
truth is `num=0` (no disease), a genuine expected false positive given the
flags' ~0.6-0.75 precision, not a bug.

No changes made to `ecg_agent.py`, `mri_agent.py`, or `sep_agent.py`.
