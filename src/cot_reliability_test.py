"""
CoT reliability test: does the reasoning field's effect on the summary hold
across a SAMPLE of cases (not just the one patient021/HCM case already
validated), and does it actually change output vs. being purely cosmetic?

For each of 5 held-out test patients (one per ACDC pathology group - DCM,
HCM, MINF, NOR, RV, all genuinely unseen by YOLO/U-Net/classifier training),
paired with the same fixed real ECG record (00001_hr): runs the Sep Agent
TWICE on the identical inputs - once with use_cot=False, once with
use_cot=True (both with use_rag=True, the intended real configuration) -
and records whether the summary text actually differs, whether parsing
succeeded without retries, and whether the reasoning field was present and
substantive.
"""
import json
import sys
from pathlib import Path

import torch
import wfdb

sys.path.insert(0, "src")
from ecg_agent import generate_ecg_contribution
from ecg_preprocessing import ecg_preprocessing_pipeline
from mri_agent import generate_mri_contribution
from mri_classifier_train import PathologyClassifier
from mri_pipeline_validation import _load_models
from mri_preprocessing import load_nifti_volume, load_patient_info, normalize_slice_to_uint8, preprocess_patient
from sep_agent import generate_sep_agent_decision

PROJECT_ROOT = Path(".").resolve()
TEST_PATIENTS = [
    ("patient101", "DCM"),
    ("patient104", "HCM"),
    ("patient103", "MINF"),
    ("patient102", "NOR"),
    ("patient109", "RV"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ECG contribution (fixed, built once, reused across all 5 cases) ---
PTBXL_DIR = PROJECT_ROOT / "kaggle" / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ecg_record_path = PTBXL_DIR / "records500" / "00000" / "00001_hr"
ecg_signal, ecg_meta = wfdb.rdsamp(str(ecg_record_path))
fs = ecg_meta["fs"]
ecg_results = ecg_preprocessing_pipeline(ecg_signal[:, 0], fs)
ecg_contribution = generate_ecg_contribution(ecg_results, fs, record_id="00001_hr")
print("ECG contribution built. Flags:", ecg_contribution.flags_for_review, flush=True)

# --- Load YOLO/U-Net/classifier once ---
yolo_model, unet_model = _load_models(device)
classifier_model = PathologyClassifier().to(device)
classifier_model.load_state_dict(torch.load("results/mri_classifier_training/best.pt", map_location=device))
classifier_model.eval()

results = []
for patient_id, true_group in TEST_PATIENTS:
    print(f"\n=== {patient_id} ({true_group}) ===", flush=True)
    patient_dir = PROJECT_ROOT / "ACDC" / "database" / "testing" / patient_id
    mri_result = preprocess_patient(str(patient_dir))
    info = load_patient_info(str(patient_dir / "Info.cfg"))
    ed_volume, _ = load_nifti_volume(str(patient_dir / f"{patient_id}_frame{info['ED']:02d}.nii.gz"))
    raw_ed_slice_uint8 = normalize_slice_to_uint8(ed_volume[:, :, mri_result["slice_indices"]["ed"]].T)

    mri_contribution = generate_mri_contribution(
        mri_result, record_id=patient_id, image_mode="ed_only",
        raw_ed_slice_uint8=raw_ed_slice_uint8, yolo_model=yolo_model, unet_model=unet_model,
        classifier_model=classifier_model, device=device,
    )
    print(f"  MRI flags: {mri_contribution.flags_for_review}", flush=True)

    contributions = [ecg_contribution, mri_contribution]

    decision_no_cot = generate_sep_agent_decision(
        contributions, record_id=f"{patient_id}_no_cot", use_rag=True, use_cot=False,
    )
    decision_cot = generate_sep_agent_decision(
        contributions, record_id=f"{patient_id}_cot", use_rag=True, use_cot=True,
    )

    differs = decision_no_cot.summary.strip() != decision_cot.summary.strip()
    print(f"  Summary differs with CoT: {differs}", flush=True)
    print(f"  no_cot attempts={decision_no_cot.trace['attempts']}, cot attempts={decision_cot.trace['attempts']}", flush=True)
    print(f"  reasoning present: {bool(decision_cot.reasoning)}", flush=True)

    results.append({
        "patient_id": patient_id,
        "true_group": true_group,
        "mri_flags": mri_contribution.flags_for_review,
        "summary_no_cot": decision_no_cot.summary,
        "attempts_no_cot": decision_no_cot.trace["attempts"],
        "summary_cot": decision_cot.summary,
        "attempts_cot": decision_cot.trace["attempts"],
        "reasoning": decision_cot.reasoning,
        "summary_differs": differs,
        "traceability_no_cot": [(e.statement, e.agent_id, e.feature) for e in decision_no_cot.traceability],
        "traceability_cot": [(e.statement, e.agent_id, e.feature) for e in decision_cot.traceability],
    })

with open("results/cot_reliability_test.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

n_differs = sum(r["summary_differs"] for r in results)
n_first_try_no_cot = sum(r["attempts_no_cot"] == 1 for r in results)
n_first_try_cot = sum(r["attempts_cot"] == 1 for r in results)
n_reasoning_present = sum(bool(r["reasoning"]) for r in results)

print("\n=== SUMMARY ACROSS 5 CASES ===")
print(f"Summary changed by CoT: {n_differs}/5")
print(f"First-attempt success (no CoT): {n_first_try_no_cot}/5")
print(f"First-attempt success (CoT): {n_first_try_cot}/5")
print(f"Reasoning field present when required: {n_reasoning_present}/5")
