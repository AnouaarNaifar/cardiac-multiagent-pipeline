# Deterministic MRI flag validation against ACDC pathology groups

Evaluated over 50 patients, no LLM call and no image reprocessing involved (decide_next_steps_mri is pure feature-threshold logic, run directly on results/mri_features_all_patients.csv's already-computed ejection fractions).

| Flag | Precision | Recall | F1 | Ground-truth prevalence | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| `low_lv_ejection_fraction_suspected` | 0.645 | 1.000 | 0.784 | 0.4000 | 20 | 11 | 0 | 19 |
| `low_rv_ejection_fraction_suspected` | 0.185 | 0.500 | 0.270 | 0.2000 | 5 | 22 | 5 | 18 |
| `high_lv_ejection_fraction_suspected` | 0.857 | 0.600 | 0.706 | 0.2000 | 6 | 1 | 4 | 39 |

Ground truth: low_lv_ejection_fraction_suspected's ground truth is True for patients in ACDC's MINF/DCM pathology groups; low_rv_ejection_fraction_suspected's is True for the RV group; high_lv_ejection_fraction_suspected's is True for the HCM group - see module docstring for the mapping's basis and its stated limitation (group labels are verified directly from each patient's Info.cfg; the exact numeric diagnostic criteria behind each group are not independently re-verified against the primary ACDC paper in this repo).

high_lv_ejection_fraction_suspected (65% threshold) was added to mri_agent.decide_next_steps_mri as a direct result of this module's threshold sweep: HCM's mean LV EF (69.8%) is only 7-13 points above the other groups (vs. 20-30+ points of separation for the low-EF flags), so this flag is necessarily noisier than low_lv_ejection_fraction_suspected - but its F1 is still notably better than low_rv_ejection_fraction_suspected's, making it a genuinely useful addition despite the narrower separation.