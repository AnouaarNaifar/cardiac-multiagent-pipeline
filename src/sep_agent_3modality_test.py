"""
Architectural validation: the Sep Agent combining THREE modalities.

Until now the Sep Agent has only ever been exercised with two contributions
(ECG + MRI). The whole point of building the structured patient data agent
as a third parallel modality is that the shared AgentContribution schema
should support aggregating an arbitrary number of modalities without any
change to sep_agent.py - this script tests exactly that, end to end, with
real MedGemma calls for all three agents plus the Sep Agent.

Synthetic pairing (same limitation as the existing ECG+MRI demo, now with a
third source): PTB-XL (ECG), ACDC (MRI) and UCI Cleveland (structured
patient data) share no real patients, so this is an architecture-validation
case, not a real joint clinical case.
"""
import sys
from pathlib import Path

import pandas as pd
import wfdb

sys.path.insert(0, "src")
from agent_common import save_contribution
from ecg_agent import generate_ecg_contribution
from ecg_preprocessing import ecg_preprocessing_pipeline
from mri_agent import generate_mri_contribution
from mri_preprocessing import preprocess_patient
from patient_data_agent import generate_patient_contribution
from sep_agent import generate_sep_agent_decision, save_sep_agent_decision

PROJECT_ROOT = Path(".").resolve()
OUT_DIR = PROJECT_ROOT / "results" / "sep_agent_3modality_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. ECG contribution (real PTB-XL record) ---
PTBXL_DIR = PROJECT_ROOT / "kaggle" / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ecg_signal, ecg_meta = wfdb.rdsamp(str(PTBXL_DIR / "records500" / "00000" / "00001_hr"))
ecg_results = ecg_preprocessing_pipeline(ecg_signal[:, 0], ecg_meta["fs"])
ecg_contribution = generate_ecg_contribution(ecg_results, ecg_meta["fs"], record_id="00001_hr")
print("ECG contribution:", ecg_contribution.flags_for_review, flush=True)

# --- 2. MRI contribution (real ACDC patient, expert-mask path) ---
mri_result = preprocess_patient(str(PROJECT_ROOT / "ACDC" / "database" / "training" / "patient001"))
mri_contribution = generate_mri_contribution(mri_result, record_id="patient001", image_mode="ed_only")
print("MRI contribution:", mri_contribution.flags_for_review, flush=True)

# --- 3. Patient data contribution (real UCI Cleveland row) ---
uci_df = pd.read_csv(PROJECT_ROOT / "uci_heart_disease" / "processed_cleveland.csv")
patient_contribution = generate_patient_contribution(uci_df.iloc[0], record_id="cleveland_0")
print("Patient contribution:", patient_contribution.flags_for_review, flush=True)

for c, name in [(ecg_contribution, "ecg"), (mri_contribution, "mri"), (patient_contribution, "patient")]:
    save_contribution(c, output_dir=str(OUT_DIR), file_name=f"{name}_contribution")

# --- 4. Sep Agent over ALL THREE contributions ---
contributions = [ecg_contribution, mri_contribution, patient_contribution]
decision = generate_sep_agent_decision(contributions, record_id="3modality_ecg00001_mriP001_clev0")
save_sep_agent_decision(decision, output_dir=str(OUT_DIR), file_name="sep_agent_decision")

print("\n=== SEP AGENT DECISION (3 modalities) ===")
print("Contributing agents:", decision.contributing_agents)
print("Attempts:", decision.trace["attempts"])
print("Flags for review (union of all 3):", decision.flags_for_review)
print("\nSUMMARY:")
print(decision.summary)
print("\nTRACEABILITY (agent -> feature):")
agents_cited = set()
for e in decision.traceability:
    agents_cited.add(e.agent_id)
    print(f'  [{e.agent_id}.{e.feature}] {e.statement}')
print("\nDistinct agents actually cited in traceability:", sorted(agents_cited))
