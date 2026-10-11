"""
Re-run of notebooks/sep_agent_demo.ipynb's flow, with the newly-wired-in MRI
pathology classifier exercised - validates the full multi-agent pipeline
(ECG agent -> MRI agent (now with YOLO+U-Net+classifier) -> Sep Agent ->
PDF report) still fits together after that integration.

Differences from the original notebook demo (deliberate, both to keep the
ECG side an unchanged control and to keep the MRI side honest):
  - MRI patient is patient101 (DCM), a genuinely HELD-OUT TEST patient for
    the classifier/YOLO/U-Net (patient001, the original demo's choice, is in
    the classifier's own TRAINING split - not an honest demo of the new
    integration).
  - ECG side is left exactly as the original demo (single-lead, no
    embedding) - isolates this run to validating the MRI classifier
    addition specifically, not a broader ECG upgrade.
Saved to a NEW output directory (not overwriting the original demo's
artifacts, this project has no git safety net).
"""
import sys
from pathlib import Path

import torch
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from agent_common import save_contribution
from ecg_agent import generate_ecg_contribution
from ecg_preprocessing import ecg_preprocessing_pipeline
from mri_agent import generate_mri_contribution
from mri_classifier_train import PathologyClassifier
from mri_pipeline_validation import _load_models
from mri_preprocessing import load_nifti_volume, load_patient_info, normalize_slice_to_uint8, preprocess_patient
from report_pdf import generate_case_report_pdf
from sep_agent import generate_sep_agent_decision, save_sep_agent_decision

DEMO_OUTPUT_DIR = PROJECT_ROOT / "results" / "sep_agent_demo_v2_with_mri_classifier"
DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- ECG side (unchanged from the original demo) ---
PTBXL_DIR = PROJECT_ROOT / "kaggle" / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ecg_record_path = PTBXL_DIR / "records500" / "00000" / "00001_hr"
ecg_signal, ecg_meta = wfdb.rdsamp(str(ecg_record_path))
fs = ecg_meta["fs"]
ecg_results = ecg_preprocessing_pipeline(ecg_signal[:, 0], fs)
ecg_contribution = generate_ecg_contribution(ecg_results, fs, record_id="00001_hr")
save_contribution(ecg_contribution, output_dir=str(DEMO_OUTPUT_DIR), file_name="ecg_contribution")
print("ECG contribution done. Flags:", ecg_contribution.flags_for_review)

# --- MRI side (held-out test patient, full YOLO+U-Net+classifier pipeline wired in) ---
ACDC_DIR = PROJECT_ROOT / "ACDC" / "database"
mri_patient_dir = ACDC_DIR / "testing" / "patient101"
mri_result = preprocess_patient(str(mri_patient_dir))

info = load_patient_info(str(mri_patient_dir / "Info.cfg"))
ed_volume, _ = load_nifti_volume(str(mri_patient_dir / f"patient101_frame{info['ED']:02d}.nii.gz"))
ed_slice_idx = mri_result["slice_indices"]["ed"]
raw_ed_slice_uint8 = normalize_slice_to_uint8(ed_volume[:, :, ed_slice_idx].T)

device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model, unet_model = _load_models(device)
classifier_model = PathologyClassifier().to(device)
classifier_model.load_state_dict(torch.load("results/mri_classifier_training/best.pt", map_location=device))
classifier_model.eval()

mri_contribution = generate_mri_contribution(
    mri_result,
    record_id="patient101",
    image_mode="ed_only",
    raw_ed_slice_uint8=raw_ed_slice_uint8,
    yolo_model=yolo_model,
    unet_model=unet_model,
    classifier_model=classifier_model,
    device=device,
)
save_contribution(mri_contribution, output_dir=str(DEMO_OUTPUT_DIR), file_name="mri_contribution")
print("MRI contribution done. Flags:", mri_contribution.flags_for_review)
print("True group: DCM (held out from all MRI training)")

# --- Sep Agent aggregation ---
decision = generate_sep_agent_decision(
    [ecg_contribution, mri_contribution],
    record_id="demo_case_ecg00001hr_mriPatient101_with_classifier",
)
print("\nContributing agents:", decision.contributing_agents)
print("Flags for review (union):", decision.flags_for_review)
print("\nSUMMARY:\n", decision.summary)
print("\nTRACEABILITY:")
for entry in decision.traceability:
    print(f'  "{entry.statement}" <- {entry.agent_id}.{entry.feature}')
save_sep_agent_decision(decision, output_dir=str(DEMO_OUTPUT_DIR), file_name="sep_agent_decision")

# --- PDF case report ---
pdf_path = DEMO_OUTPUT_DIR / "case_report.pdf"
generate_case_report_pdf(
    output_path=pdf_path,
    sep_decision=decision,
    contributions=[ecg_contribution, mri_contribution],
    figure_paths=None,
    synthetic_pairing_note=(
        "This case pairs a PTB-XL ECG record with an ACDC MRI patient. These datasets share no real "
        "patients - this is a synthetic pairing for multi-agent architecture validation, not a real "
        "joint clinical case. The MRI patient (patient101, DCM) is a genuinely held-out test patient for "
        "the newly-integrated YOLO/U-Net/pathology-classifier pipeline - not used in any of that "
        "pipeline's training."
    ),
)
print("\nPDF report saved to:", pdf_path)
