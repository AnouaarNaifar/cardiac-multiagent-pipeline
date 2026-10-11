"""
End-to-end test of the patient-demographic-context feature (final design:
PDF rendering only, NOT fed into any LLM prompt - the Sep Agent already
only combines findings the modality agents interpreted, it never needs the
raw demographics itself). Case: ECG record 00001_hr (PTB-XL) + MRI
patient021 (ACDC) - the original sep_agent_demo.ipynb pairing.

Confirms:
  1. build_patient_context_line renders correctly for each real source
     (ECG is missing height for this specific record - graceful single-
     field omission; MRI has no age/sex at all - the "not available" case).
  2. Passing that line as patient_info to each single-modality agent's
     OWN prompt requires no code change beyond the call site.
  3. generate_sep_agent_decision's prompt does NOT receive patient context
     at all (reverted - confirmed by signature inspection).
  4. generate_case_report_pdf renders each source's context line under its
     own findings section, with no same-patient/different-patient logic
     needed - each contribution already gets its own section.
"""
import sys

import pandas as pd
import wfdb

sys.path.insert(0, "src")
from agent_common import build_patient_context_line, save_contribution
from ecg_agent import generate_ecg_contribution
from ecg_preprocessing import ecg_preprocessing_pipeline
from mri_agent import generate_mri_contribution
from mri_preprocessing import load_patient_info, preprocess_patient
from report_pdf import generate_case_report_pdf
from sep_agent import generate_sep_agent_decision

PTBXL_DIR = "kaggle/archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ACDC_PATIENT_DIR = "ACDC/database/training/patient021"
OUTPUT_DIR = "results/patient_context_test"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Real demographic lookups ---
db = pd.read_csv(f"{PTBXL_DIR}/ptbxl_database.csv")
row = db[db["filename_hr"].str.contains("00001_hr")].iloc[0]
ecg_context_line = build_patient_context_line(
    age=row["age"], sex=row["sex"],
    height=None if pd.isna(row["height"]) else row["height"],
    weight=None if pd.isna(row["weight"]) else row["weight"],
)
print("ECG context line:", repr(ecg_context_line))
assert ecg_context_line == "Patient: 56-year-old female, 63kg."

mri_info = load_patient_info(f"{ACDC_PATIENT_DIR}/Info.cfg")
mri_context_line = build_patient_context_line(height=mri_info["Height"], weight=mri_info["Weight"])
print("MRI context line:", repr(mri_context_line))
assert mri_context_line == "Patient: age/sex not available, 192cm/110kg."

# --- 2. Single-modality agents: pass the line in as patient_info, no other code change ---
ecg_signal, ecg_meta = wfdb.rdsamp(f"{PTBXL_DIR}/records500/00000/00001_hr")
fs = ecg_meta["fs"]
ecg_results = ecg_preprocessing_pipeline(ecg_signal[:, 0], fs)
ecg_contribution = generate_ecg_contribution(ecg_results, fs, record_id="00001_hr", patient_info=ecg_context_line)
print("\nECG contribution OK. Flags:", ecg_contribution.flags_for_review)

mri_result = preprocess_patient(ACDC_PATIENT_DIR)
mri_contribution = generate_mri_contribution(mri_result, record_id="patient021", patient_info=mri_context_line)
print("MRI contribution OK. Flags:", mri_contribution.flags_for_review)

save_contribution(ecg_contribution, output_dir=OUTPUT_DIR, file_name="ecg_contribution")
save_contribution(mri_contribution, output_dir=OUTPUT_DIR, file_name="mri_contribution")

# --- 3. Confirm the Sep Agent prompt no longer accepts/uses patient context at all ---
import inspect
sig = inspect.signature(generate_sep_agent_decision)
assert "patient_context_lines" not in sig.parameters, "Sep Agent should NOT have this param anymore"
print("\nConfirmed: generate_sep_agent_decision has no patient_context_lines param (reverted).")

decision = generate_sep_agent_decision(
    [ecg_contribution, mri_contribution], record_id="patient_context_test_00001hr_patient021",
)
print("Sep Agent summary:", decision.summary)
print("Sep Agent flags:", decision.flags_for_review)

# --- 4. PDF rendering: patient context appears per-source, under each agent's own section ---
patient_context_lines = {"ecg": ecg_context_line, "mri": mri_context_line}
pdf_path = f"{OUTPUT_DIR}/case_report.pdf"
generate_case_report_pdf(
    output_path=pdf_path,
    sep_decision=decision,
    contributions=[ecg_contribution, mri_contribution],
    patient_context_lines=patient_context_lines,
    synthetic_pairing_note=(
        "This case pairs a PTB-XL ECG record with an ACDC MRI patient. These datasets share no real "
        "patients - this is a synthetic pairing for multi-agent architecture validation, not a real "
        "joint clinical case."
    ),
)
print("\nPDF written to:", pdf_path)
print("(visual confirmation of rendered content done separately via the Read tool)")
print("\nALL CHECKS PASSED.")
