# Deterministic flag validation for the structured patient data agent

Evaluated over 303 patients (UCI Heart Disease, Cleveland), ground truth = heart_disease_present (num > 0). No LLM call involved.

| Flag | Precision | Recall | F1 | Ground-truth prevalence | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| `high_resting_bp_suspected` | 0.591 | 0.281 | 0.380 | 0.4587 | 39 | 27 | 100 | 137 |
| `high_cholesterol_suspected` | 0.526 | 0.576 | 0.550 | 0.4587 | 80 | 72 | 59 | 92 |
| `reduced_exercise_capacity_suspected` | 0.636 | 0.705 | 0.669 | 0.4587 | 98 | 56 | 41 | 108 |
| `significant_st_depression_suspected` | 0.654 | 0.719 | 0.685 | 0.4587 | 100 | 53 | 39 | 111 |
| `exercise_induced_angina_present` | 0.768 | 0.547 | 0.639 | 0.4587 | 76 | 23 | 63 | 141 |
| `multiple_vessels_involved_suspected` | 0.748 | 0.662 | 0.702 | 0.4587 | 92 | 31 | 47 | 133 |
| `abnormal_thallium_result_suspected` | 0.748 | 0.727 | 0.737 | 0.4587 | 101 | 34 | 38 | 130 |

## Interpretation

Unlike the ECG deterministic flags (whose target diagnoses are rare, so precision is swamped by false positives), the target here - `heart_disease_present` - is roughly balanced (prevalence ~0.46), so precision above ~0.46 is a genuine signal rather than the base rate. Precision here means: of the patients this flag fired on, what fraction actually have heart disease.

- **Strongest flags:** `abnormal_thallium_result_suspected` (F1 ~0.74) and `multiple_vessels_involved_suspected` (F1 ~0.70) - a positive thallium scan and any fluoroscopy-colored vessel are both directly ischaemia-related, so they track the outcome well.
- **Two thresholds were recalibrated by sweep** (same discipline as the ECG tachycardia 100→120 recalibration): `oldpeak` 1.0→0.8 mm (Pareto-improves recall at similar precision), and max-HR `220-age @85%` → **Tanaka `208-0.7×age` @91%** (the F1-maximum, F1 0.669; because the flag fires when `max_hr < pct×predicted`, a higher pct fires more positives, so 91% also **dominates 90% on recall** — 0.705 vs 0.683 — there is no recall/F1 trade-off here, 91% simply wins both. 92% gives marginally more recall, 0.712, but F1 drops to 0.651).
- **`high_resting_bp_suspected` (140 mmHg) and `high_cholesterol_suspected` (240 mg/dl) were kept at their clinically meaningful cutoffs despite weak sweeps** - both are simply weak CAD predictors in this dataset at ANY threshold (precision barely clears the base rate), so the standard clinical cutoff was kept and the weakness reported honestly rather than chasing a statistically marginal, clinically meaningless gain.

Recall-favoring choices throughout: these flags feed a human-review alert, not a final classification, so a missed case is treated as worse than an extra reviewed false positive.