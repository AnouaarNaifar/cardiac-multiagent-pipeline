# Synthetic test cases — reporting/pipeline capabilities + expected results

Quick cases to exercise the Sep Agent, the knowledge graph (convergence + hierarchy), the deterministic
flags, and the directional-consistency check. All are **synthetic** (PTB-XL/ACDC/UCI share no real patients) —
architecture validation, not real joint clinical cases.

---

# Ready-to-run cases in the Streamlit UI (real records + exact Form B inputs)

These are the **exact patients to select on each page** and the **exact values to type into Form B**
(Patient data), so you can reproduce the convergence cases end-to-end through the app. Every flag below was
**verified by actually running the agents** on these records (ECG on PTB-XL `lr`, MRI on ACDC, patient on the
UCI Cleveland row). Reminder: the three datasets share no real patients, so a multi-modality case is a
**synthetic pairing** for architecture validation.

> **Split provenance — all demos use held-out records.** The splits are: PTB-XL ECG `strat_fold` **1–8 =
> train, 9 = val, 10 = test/held-out** (the learned embedding); ACDC MRI `training/` = **patient001–100**
> (train+val) vs `testing/` = **patient101–150 = held-out** (the YOLO/U-Net/pathology-classifier); UCI
> Cleveland has **no split** — its 7 flags are deterministic (validated on all 303 patients, nothing learned).
> For strictness, **every real record below is drawn from the held-out set** (PTB-XL fold 10 / ACDC
> patient101–150). Note that a learned-model demo *must* be held-out (otherwise it scores a model on its own
> training data — e.g. the HCM pathology-classifier case uses held-out `patient138`), whereas
> **deterministic flags are split-agnostic** — Sokolow-Lyon LVH, territorial
> inferior-Q, the EF-threshold flags and the rate/rhythm flags learn nothing, so their fold is irrelevant; we
> still use held-out records for them purely for uniformity.

### Where to enter each modality
- **ECG page** → source *PTB-XL dataset* → type the `ecg_id` below (resolution `lr` is fine).
- **MRI page** → the held-out ACDC patients below (patient101–150) are in ACDC's `testing/` folder, which the
  *ACDC patient* selector does **not** list (it only offers the training patients 001–100). So load a held-out
  patient via source **_Upload NIfTI_**: upload its `patientXXX_frame{ED}.nii.gz` (ED volume) and
  `patientXXX_frame{ES}.nii.gz` (ES volume) from `ACDC/database/testing/patientXXX/` (optionally the two
  `_gt.nii.gz` masks to also see the predicted-vs-expert check). For the pathology flag, enable the **pathology
  classifier** toggle. **Note:** the UI runs the classifier on **one representative ED slice**, so its
  per-class probabilities differ from the per-patient multi-slice aggregate reported in
  `results/mri_classifier_validation/`; the numbers below are the **single-slice UI** values (what you will
  actually see).
- **Patient data page (Form B)** → fill the 13 fields exactly as the table says (leave nothing "unknown" for
  these cases).
- **Case aggregation page** → tick **`use_kg`** (and optionally `use_rag`/`use_cot`) → Run.

## Case A — Hypertensive LV hypertrophy (3-modality convergence)
| Modality | Select | Flag it raises (verified) |
|---|---|---|
| ECG | PTB-XL **ecg_id 299** (LVH) *[fold 10 — held-out]* | `lvh_sokolow_lyon_suspected` (Sokolow-Lyon **7.37 mV** ≥ 3.5) |
| MRI | ACDC **patient138** (HCM) *[ACDC testing — held-out; load via Upload NIfTI]* | `high_lv_ejection_fraction_suspected` (EF ≈ 69.0% expert mask / ≈73.1% predicted-mask pipeline, both ≥ 65) + `pathology_HCM_pattern_suspected` (single-slice UI HCM ≈ 0.79, needs the classifier toggle) |
| Patient (Form B) | values below | `high_resting_bp_suspected` (+ high_cholesterol, multiple_vessels, …) |

**Form B — Case A** (real UCI Cleveland row, target `num=2` = disease present):
| Field | Value to enter |
|---|---|
| Age (years) | 67 |
| Sex | male |
| Chest pain type | asymptomatic |
| Resting BP (mmHg) | 160 |
| Cholesterol (mg/dl) | 286 |
| Fasting blood sugar > 120 mg/dl | No |
| Resting ECG result | probable_lvh |
| Max heart rate achieved (bpm) | 108 |
| Exercise-induced angina | Yes |
| ST depression / oldpeak (mm) | 1.5 |
| ST segment slope | flat |
| Major vessels colored (ca) | 3 |
| Thallium result (thal) | normal |

**Expected (use_kg=True):** `kg_convergent_diagnoses` = **left ventricular hypertrophy** (ECG+MRI) **+
hypertensive heart disease** (ECG+MRI+patient); the summary names *"hypertensive heart disease"*. Without the
KG it only lists findings — that A/B gap is the point.

## Case B — Coronary artery disease / ischaemia (ECG + patient convergence)
| Modality | Select | Flag it raises (verified) |
|---|---|---|
| ECG | PTB-XL **ecg_id 1737** (IMI) *[fold 10 — held-out]* | `pathological_q_inferior_suspected` (Q in **2** inferior leads) |
| MRI | *(none — ACDC has no scar/LGE, correctly absent)* | — |
| Patient (Form B) | values below | `abnormal_thallium_result_suspected`, `multiple_vessels_involved_suspected`, `significant_st_depression_suspected`, `exercise_induced_angina_present` |

**Form B — Case B** (real UCI Cleveland row, target `num=1` = disease present):
| Field | Value to enter |
|---|---|
| Age (years) | 67 |
| Sex | male |
| Chest pain type | asymptomatic |
| Resting BP (mmHg) | 120 |
| Cholesterol (mg/dl) | 229 |
| Fasting blood sugar > 120 mg/dl | No |
| Resting ECG result | probable_lvh |
| Max heart rate achieved (bpm) | 129 |
| Exercise-induced angina | Yes |
| ST depression / oldpeak (mm) | 2.6 |
| ST segment slope | flat |
| Major vessels colored (ca) | 2 |
| Thallium result (thal) | reversible_defect |

**Expected (use_kg=True):** `kg_convergent_diagnoses` = **coronary artery disease / myocardial ischaemia**
(ECG+patient), via the hierarchy *Q inferior → old inferior MI → MI superclass → CAD*; the summary names
*"coronary artery disease"*.

## Case C — HFrEF, single modality (negative control, NO convergence)
| Modality | Select | Flag it raises (verified) |
|---|---|---|
| MRI | ACDC **patient101** (DCM) *[ACDC testing — held-out]* | `low_lv_ejection_fraction_suspected` (EF ≈ 29.9% expert mask / ≈26.4% predicted-mask pipeline, both < 45) |

**Expected:** `kg_convergent_diagnoses` = **empty** (reached by one modality only → the KG must not over-claim
convergence); the summary must describe the LV ejection fraction as **reduced/low** (the directional check
rejects "normal/preserved/elevated" here). Swap in **patient138** (HCM, high EF — held-out) to test the
opposite direction.

---

## How to run (template)

```python
import sys; sys.path.insert(0, "src")
from functools import partial
from agent_common import AgentContribution, Finding, generate_report_medgemma
from sep_agent import generate_sep_agent_decision

# build contributions per case (see below), then:
dec = generate_sep_agent_decision(contribs, llm_call=partial(generate_report_medgemma, seed=42),
                                  record_id="case", use_kg=True)
print(dec.summary)
print(dec.trace.get("kg_convergent_diagnoses"))     # deterministic convergence (diagnosis level)
print(dec.trace.get("kg_convergent_categories"))    # deterministic convergence (superclass level)
```
A `Finding` is `Finding(feature, value, rationale, confidence, evidence_ref)`; an `AgentContribution` is
`AgentContribution(agent_id, record_id, [findings], [flags_for_review], {})`. The flags in `flags_for_review`
are what the KG reasons over — they must be the real flag names.

---

## Case 1 — Hypertensive LV hypertrophy (3-modality convergence)
- **ECG** flags: `lvh_sokolow_lyon_suspected`
- **MRI** flags: `pathology_HCM_pattern_suspected`
- **patient** flags: `high_resting_bp_suspected`

**Expected (use_kg=True):**
- `kg_convergent_diagnoses` = **left ventricular hypertrophy** (ecg+mri) **+ hypertensive heart disease**
  (ecg+mri+patient).
- Summary explicitly names **"hypertensive heart disease"** (the convergent diagnosis). Without the KG
  (`use_kg=False`) it only lists the findings — that is the A/B difference (robust 9/10 vs 0/10 on the 4B).

## Case 2 — Coronary artery disease / ischaemia (ECG + patient convergence)
- **ECG** flags: `pathological_q_inferior_suspected` (and optionally `embedding_MI_pattern_suspected`)
- **patient** flags: `abnormal_thallium_result_suspected`, `multiple_vessels_involved_suspected`,
  `significant_st_depression_suspected`, `exercise_induced_angina_present`

**Expected:**
- `kg_convergent_diagnoses` = **coronary artery disease / myocardial ischaemia** (ecg+patient).
- Reasoning path uses the hierarchy: *Q inferior → old inferior MI → MI superclass → CAD*.
- Summary names **"coronary artery disease"**. (MRI is correctly absent — scar needs LGE, which ACDC lacks.)

## Case 3 — HFrEF, single modality (NO convergence — negative control)
- **MRI** flags: `low_lv_ejection_fraction_suspected` only

**Expected:**
- `kg_convergent_diagnoses` = **empty** (HFrEF is reached by MRI only → no ≥2-modality convergence). This
  checks the KG does not over-claim convergence from one modality.
- Summary describes the ejection fraction as **reduced/low** (see Case 4).

## Case 4 — Directional-consistency guarantee
- **MRI** flags: `low_lv_ejection_fraction_suspected`, with a finding whose value is a low EF (e.g. "23.7%").

**Expected:**
- The summary must describe the **left ventricular ejection fraction as reduced/low**. If the model instead
  writes "normal"/"preserved"/"elevated" LV ejection fraction, the deterministic directional check rejects it
  and retries (the check is sourced from the KG's `directional` block). Try also
  `high_lv_ejection_fraction_suspected` — then "reduced/low" is what gets rejected. This verifies the
  anti-contradiction layer (independent of traceability, which only checks the finding is real).

## Case 5 — Hierarchy: deterministic ↔ embedding bridge
- **ECG** flags: `pathological_q_inferior_suspected` **and** `embedding_MI_pattern_suspected`

**Expected:**
- `categories_for_findings` rolls **both** up to the **MI superclass** — the rule-based Q flag and the learned
  CNN flag corroborate at the category level. `category_of("old_inferior_mi")` → "myocardial infarction (MI
  superclass)". (No cross-modality convergence here — both are ECG — but the taxonomy link is the point.)

## Case 6 — Real-record deterministic flags (no LLM needed)
Run the agents on real records to check the shipped multi-lead flags fire correctly:
- PTB-XL **ecg_id 299** (LVH) *[fold 10 — held-out]* → `lvh_sokolow_lyon_suspected` raised (Sokolow-Lyon ≈ 7.37 mV ≥ 3.5).
- PTB-XL **ecg_id 1737** (IMI) *[fold 10 — held-out]* → `pathological_q_inferior_suspected` raised (Q in 2 inferior leads).
- ACDC **patient101** (DCM) *[ACDC testing — held-out]* → `low_lv_ejection_fraction_suspected` raised (EF ≈ 29.9%).
- A **NORM** record *(fold 10, e.g. ecg_id 9)* → neither Sokolow-Lyon nor a wrong-direction description.

*(All four bullets here exercise **deterministic** flags only — no learned model — but held-out records are
used throughout for uniformity.)*

(Construct via `ecg_preprocessing_pipeline` + `ecg_agent.compute_feature_dict(..., multilead_signal=sig)` +
`detect_flags`; see `results/ecg_multilead_validation/` for the validated numbers.)

---

**Reproducible references:** `experiments/kg_ab.py` (Case 1), `experiments/kg_robustness.py` (Case 1 across
seeds), `experiments/kg_demo_pdf.py` (Case 1 → case-report PDF). Adapt the contributions for the other cases.
