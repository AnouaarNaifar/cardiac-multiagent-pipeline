"""
Re-runs the same 5-case CoT reliability test (cot_reliability_test.py) now
that sep_agent.py has the directional-consistency fix (_check_directional_
consistency + the new unconditional prompt rule), to check whether the two
known failures - patient102's EF-direction inversion ("decreased" for an
elevated 70.9% LV EF) and patient109's self-contradictory summary - are
now caught/corrected, and whether anything else regresses.

Also captures the RAW prompt and RAW LLM response for every attempt (not
just the parsed summary) to results/cot_reliability_raw/<patient>_<cot|no_cot>_
attempt<N>.txt, so they can be read directly rather than only inspecting the
already-parsed JSON - this is what actually shows the retry mechanism
working (or not) on the model's real output text.
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
from sep_agent import SepAgentParseError, _check_directional_consistency, generate_sep_agent_decision

PROJECT_ROOT = Path(".").resolve()
TEST_PATIENTS = [
    ("patient101", "DCM"),
    ("patient104", "HCM"),
    ("patient103", "MINF"),
    ("patient102", "NOR"),
    ("patient109", "RV"),
]

RAW_DIR = PROJECT_ROOT / "results" / "cot_reliability_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

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


def make_logging_llm_call(log_prefix):
    """Wraps generate_report_medgemma to save (prompt, raw_response) per call attempt to disk."""
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
    all_flags = mri_contribution.flags_for_review + ecg_contribution.flags_for_review

    def _run(use_cot, tag):
        """Runs one Sep Agent call; on exhausted-retries failure, records the
        failure instead of crashing the whole test (mirrors report_evaluation.py's
        'per-record failures are caught and logged, not fatal' convention) -
        an exhausted-retries failure IS a real result to report here (did the
        directional check catch something the model couldn't self-correct?),
        not a script bug to hide."""
        try:
            decision = generate_sep_agent_decision(
                contributions, record_id=f"{patient_id}_{tag}", use_rag=True, use_cot=use_cot,
                llm_call=make_logging_llm_call(f"{patient_id}_{tag}"), max_retries=2,
            )
            return {
                "ok": True,
                "summary": decision.summary,
                "reasoning": decision.reasoning,
                "attempts": decision.trace["attempts"],
                "traceability": [(e.statement, e.agent_id, e.feature) for e in decision.traceability],
            }
        except SepAgentParseError as exc:
            print(f"  [{tag}] FAILED after retries: {exc}", flush=True)
            return {"ok": False, "summary": None, "reasoning": None, "attempts": None,
                     "traceability": [], "error": str(exc)}

    r_no_cot = _run(False, "no_cot")
    r_cot = _run(True, "cot")

    differs = (r_no_cot["summary"] or "").strip() != (r_cot["summary"] or "").strip()
    remaining_violations_no_cot = _check_directional_consistency(r_no_cot["summary"] or "", all_flags) if r_no_cot["ok"] else None
    remaining_violations_cot = (
        _check_directional_consistency(r_cot["summary"] or "", all_flags)
        + _check_directional_consistency(r_cot["reasoning"] or "", all_flags)
    ) if r_cot["ok"] else None

    print(f"  Summary differs with CoT: {differs}", flush=True)
    print(f"  no_cot ok={r_no_cot['ok']} attempts={r_no_cot['attempts']}, cot ok={r_cot['ok']} attempts={r_cot['attempts']}", flush=True)
    print(f"  remaining violations in final accepted output: no_cot={remaining_violations_no_cot}, cot={remaining_violations_cot}", flush=True)

    results.append({
        "patient_id": patient_id,
        "true_group": true_group,
        "mri_flags": mri_contribution.flags_for_review,
        "no_cot_ok": r_no_cot["ok"],
        "summary_no_cot": r_no_cot["summary"],
        "attempts_no_cot": r_no_cot["attempts"],
        "error_no_cot": r_no_cot.get("error"),
        "cot_ok": r_cot["ok"],
        "summary_cot": r_cot["summary"],
        "attempts_cot": r_cot["attempts"],
        "error_cot": r_cot.get("error"),
        "reasoning": r_cot["reasoning"],
        "summary_differs": differs,
        "remaining_violations_no_cot": remaining_violations_no_cot,
        "remaining_violations_cot": remaining_violations_cot,
        "traceability_no_cot": r_no_cot["traceability"],
        "traceability_cot": r_cot["traceability"],
    })

with open("results/cot_reliability_test_after_fix.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

n_differs = sum(r["summary_differs"] for r in results)
n_ok_no_cot = sum(r["no_cot_ok"] for r in results)
n_ok_cot = sum(r["cot_ok"] for r in results)
n_retried_no_cot = sum(r["no_cot_ok"] and r["attempts_no_cot"] > 1 for r in results)
n_retried_cot = sum(r["cot_ok"] and r["attempts_cot"] > 1 for r in results)
n_clean_no_cot = sum(r["no_cot_ok"] and not r["remaining_violations_no_cot"] for r in results)
n_clean_cot = sum(r["cot_ok"] and not r["remaining_violations_cot"] for r in results)
n_exhausted = sum((not r["no_cot_ok"]) or (not r["cot_ok"]) for r in results)

print("\n=== SUMMARY ACROSS 5 CASES (AFTER FIX) ===")
print(f"Summary changed by CoT: {n_differs}/5")
print(f"Succeeded within retry budget (no CoT): {n_ok_no_cot}/5")
print(f"Succeeded within retry budget (CoT): {n_ok_cot}/5")
print(f"Cases where a retry was triggered and then succeeded (no CoT): {n_retried_no_cot}/5")
print(f"Cases where a retry was triggered and then succeeded (CoT): {n_retried_cot}/5")
print(f"Final accepted output free of directional violations (no CoT, of those that succeeded): {n_clean_no_cot}/{n_ok_no_cot}")
print(f"Final accepted output free of directional violations (CoT, of those that succeeded): {n_clean_cot}/{n_ok_cot}")
print(f"Cases that exhausted retries without ever producing a directionally-consistent response: {n_exhausted}/5")
