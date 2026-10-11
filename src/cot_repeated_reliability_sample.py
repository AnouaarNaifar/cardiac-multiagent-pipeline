"""
A single 5-case run cannot give a reliable error-rate estimate: MedGemma's
generation is non-deterministic, and patient104's citation-coverage error
(see results/cot_rag_narrative_accuracy_limitation.md) appeared only on a
RE-RUN of a case that had been clean in the original test - proof that
narrative errors are not tied to specific "hard" patients, they can occur
on any generation of any case.

This script measures the real thing: for the same 5 patients (one per ACDC
pathology group), each with a FIXED MRI contribution (generated once, so
the MRI agent's own non-determinism isn't a confound), the Sep Agent
(use_rag=True, use_cot=True - the intended production config) is called
N=5 times per patient on that identical input (25 total Sep Agent
generations), with both post-fix checks active
(_check_directional_consistency + _check_ef_citation_coverage). Reports,
per patient and in aggregate: how often the final ACCEPTED output is clean,
how often a retry fired and then succeeded, and how often the retry budget
was exhausted without ever producing a clean response.
"""
import json
import sys
from pathlib import Path

import torch
import wfdb

sys.path.insert(0, "src")
from agent_common import generate_report_medgemma
from ecg_agent import generate_ecg_contribution
from ecg_preprocessing import ecg_preprocessing_pipeline
from mri_agent import generate_mri_contribution
from mri_classifier_train import PathologyClassifier
from mri_pipeline_validation import _load_models
from mri_preprocessing import load_nifti_volume, load_patient_info, normalize_slice_to_uint8, preprocess_patient
from sep_agent import SepAgentParseError, _check_directional_consistency, _check_ef_citation_coverage, generate_sep_agent_decision

PROJECT_ROOT = Path(".").resolve()
TEST_PATIENTS = [
    ("patient101", "DCM"),
    ("patient104", "HCM"),
    ("patient103", "MINF"),
    ("patient102", "NOR"),
    ("patient109", "RV"),
]
N_REPEATS = 5

RAW_DIR = PROJECT_ROOT / "results" / "cot_repeated_sample_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

PTBXL_DIR = PROJECT_ROOT / "kaggle" / "archive" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ecg_record_path = PTBXL_DIR / "records500" / "00000" / "00001_hr"
ecg_signal, ecg_meta = wfdb.rdsamp(str(ecg_record_path))
fs = ecg_meta["fs"]
ecg_results = ecg_preprocessing_pipeline(ecg_signal[:, 0], fs)
ecg_contribution = generate_ecg_contribution(ecg_results, fs, record_id="00001_hr")
print("ECG contribution built. Flags:", ecg_contribution.flags_for_review, flush=True)

yolo_model, unet_model = _load_models(device)
classifier_model = PathologyClassifier().to(device)
classifier_model.load_state_dict(torch.load("results/mri_classifier_training/best.pt", map_location=device))
classifier_model.eval()


def make_logging_llm_call(log_prefix):
    state = {"attempt": 0}

    def _call(prompt):
        state["attempt"] += 1
        raw = generate_report_medgemma(prompt)
        out_path = RAW_DIR / f"{log_prefix}_attempt{state['attempt']}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW RESPONSE ===\n")
            f.write(raw)
        return raw

    return _call


all_results = []
for patient_id, true_group in TEST_PATIENTS:
    print(f"\n=== {patient_id} ({true_group}) - building fixed MRI contribution ===", flush=True)
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
    all_flags = mri_contribution.flags_for_review + ecg_contribution.flags_for_review

    patient_results = []
    for rep in range(1, N_REPEATS + 1):
        tag = f"{patient_id}_rep{rep}"
        try:
            decision = generate_sep_agent_decision(
                contributions, record_id=tag, use_rag=True, use_cot=True,
                llm_call=make_logging_llm_call(tag), max_retries=2,
            )
            traceability_pairs = [(e.agent_id, e.feature) for e in decision.traceability]
            remaining = (
                _check_directional_consistency(decision.summary, all_flags)
                + _check_directional_consistency(decision.reasoning or "", all_flags)
                + _check_ef_citation_coverage(decision.summary, traceability_pairs)
                + _check_ef_citation_coverage(decision.reasoning or "", traceability_pairs)
            )
            outcome = {
                "rep": rep, "ok": True, "attempts": decision.trace["attempts"],
                "summary": decision.summary, "reasoning": decision.reasoning,
                "remaining_violations": remaining, "clean": not remaining,
            }
            print(f"  rep {rep}: ok, attempts={decision.trace['attempts']}, clean={not remaining}", flush=True)
        except SepAgentParseError as exc:
            outcome = {
                "rep": rep, "ok": False, "attempts": None, "summary": None, "reasoning": None,
                "remaining_violations": None, "clean": False, "error": str(exc),
            }
            print(f"  rep {rep}: FAILED after retries: {exc}", flush=True)
        patient_results.append(outcome)

    all_results.append({
        "patient_id": patient_id, "true_group": true_group,
        "mri_flags": mri_contribution.flags_for_review, "repeats": patient_results,
    })

with open("results/cot_repeated_reliability_sample.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n=== AGGREGATE ACROSS 25 GENERATIONS ===")
total = 0
n_ok = 0
n_clean_first_try = 0
n_clean_after_retry = 0
n_exhausted = 0
for patient in all_results:
    for r in patient["repeats"]:
        total += 1
        if not r["ok"]:
            n_exhausted += 1
            continue
        n_ok += 1
        if r["clean"] and r["attempts"] == 1:
            n_clean_first_try += 1
        elif r["clean"] and r["attempts"] > 1:
            n_clean_after_retry += 1

print(f"Total generations: {total}")
print(f"Succeeded within retry budget: {n_ok}/{total}")
print(f"Clean on first attempt (no retry needed): {n_clean_first_try}/{total}")
print(f"Clean after a retry fired and corrected it: {n_clean_after_retry}/{total}")
print(f"Exhausted retry budget without ever producing a clean response: {n_exhausted}/{total}")
print(f"Overall final-output error rate (exhausted / total): {n_exhausted}/{total} = {100*n_exhausted/total:.1f}%")

print("\nPer-patient breakdown:")
for patient in all_results:
    clean_count = sum(r["ok"] and r["clean"] for r in patient["repeats"])
    retry_count = sum(r["ok"] and r["attempts"] and r["attempts"] > 1 for r in patient["repeats"])
    exhausted_count = sum(not r["ok"] for r in patient["repeats"])
    print(f"  {patient['patient_id']} ({patient['true_group']}): "
          f"{clean_count}/{N_REPEATS} clean, {retry_count}/{N_REPEATS} needed a retry, "
          f"{exhausted_count}/{N_REPEATS} exhausted retries")
