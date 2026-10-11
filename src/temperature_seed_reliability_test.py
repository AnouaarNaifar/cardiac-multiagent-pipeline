"""
Tests whether lowering MedGemma's generation temperature (and adding a
fixed seed) reduces the qualitative-characterization error rate documented
in results/cot_rag_narrative_accuracy_limitation.md - measured the same
way the embedding-model comparison was (before/after on the real task),
not adopted on intuition.

Step 1 (confirmed separately, interactively, before writing this script -
see the conversation/results doc, not repeated here as code):
  - medgemma:4b's own Modelfile sets no PARAMETER temperature or seed -
    only `stop`. So the previous N=25 runs used Ollama's own runtime
    default (~0.8), not a project-chosen value.
  - 3 identical calls with no options set produced 3 different phrasings -
    confirms generation is genuinely non-deterministic at current settings.
  - 2 identical calls with the same explicit seed returned BYTE-IDENTICAL
    text - confirms Ollama/medgemma:4b does support reproducible generation
    via seed.

This script covers steps 2 and 3: re-running the exact same N=25
reliability sample (5 patients x 5 repeats, CoT+RAG, both consistency
checks active, same fixed MRI contribution per patient as
cot_repeated_reliability_sample.py) at:
  - temperature=0.2, no seed (still resampled each repeat, but a much
    sharper distribution over next tokens - chosen as a reasonable
    lower-temperature candidate: not 0.0/greedy, since we still want to
    observe natural repeat-to-repeat variation for the no-seed condition;
    low enough to meaningfully reduce the chance of drifting onto a
    lower-probability, wrong-direction word like "reduced" for an
    elevated-EF case).
  - temperature=0.2 + a DISTINCT fixed seed per patient (not one shared
    seed across all 5 patients, so patients aren't artificially pushed
    toward resembling each other's output - just made individually
    reproducible).

The existing results/cot_repeated_reliability_sample.json (produced by
cot_repeated_reliability_sample.py after the num_ctx fix, Ollama's default
temperature/seed) is reused as the "default settings" baseline for the
three-way comparison - not re-run here, to avoid a redundant N=25 pass.
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
    ("patient101", "DCM", 1001),
    ("patient104", "HCM", 1002),
    ("patient103", "MINF", 1003),
    ("patient102", "NOR", 1004),
    ("patient109", "RV", 1005),
]
N_REPEATS = 5
TEMPERATURE = 0.2

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

print("\n=== Building fixed MRI contributions (shared across both configs) ===", flush=True)
fixed_mri = {}
for patient_id, true_group, seed in TEST_PATIENTS:
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
    print(f"  {patient_id}: MRI flags = {mri_contribution.flags_for_review}", flush=True)
    fixed_mri[patient_id] = mri_contribution


def make_logging_llm_call(log_prefix, raw_dir, temperature, seed):
    raw_dir.mkdir(parents=True, exist_ok=True)
    state = {"attempt": 0}

    def _call(prompt):
        state["attempt"] += 1
        raw = generate_report_medgemma(prompt, temperature=temperature, seed=seed)
        out_path = raw_dir / f"{log_prefix}_attempt{state['attempt']}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"=== CONFIG: temperature={temperature}, seed={seed} ===\n")
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW RESPONSE ===\n")
            f.write(raw)
        return raw

    return _call


def run_config(config_name, temperature, use_per_patient_seed, raw_dir, output_json):
    print(f"\n\n######## CONFIG: {config_name} (temperature={temperature}, "
          f"per_patient_seed={use_per_patient_seed}) ########", flush=True)
    all_results = []
    for patient_id, true_group, seed in TEST_PATIENTS:
        mri_contribution = fixed_mri[patient_id]
        contributions = [ecg_contribution, mri_contribution]
        all_flags = mri_contribution.flags_for_review + ecg_contribution.flags_for_review
        effective_seed = seed if use_per_patient_seed else None

        print(f"\n=== {patient_id} ({true_group}) - seed={effective_seed} ===", flush=True)
        patient_results = []
        for rep in range(1, N_REPEATS + 1):
            tag = f"{patient_id}_rep{rep}"
            try:
                decision = generate_sep_agent_decision(
                    contributions, record_id=tag, use_rag=True, use_cot=True,
                    llm_call=make_logging_llm_call(tag, raw_dir, temperature, effective_seed),
                    max_retries=2,
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
            "patient_id": patient_id, "true_group": true_group, "seed": effective_seed,
            "mri_flags": mri_contribution.flags_for_review, "repeats": patient_results,
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    total = sum(len(p["repeats"]) for p in all_results)
    n_clean_first = sum(r["ok"] and r["clean"] and r["attempts"] == 1 for p in all_results for r in p["repeats"])
    n_clean_retry = sum(r["ok"] and r["clean"] and r["attempts"] > 1 for p in all_results for r in p["repeats"])
    n_exhausted = sum(not r["ok"] for p in all_results for r in p["repeats"])
    unique_summaries_per_patient = {
        p["patient_id"]: len({r["summary"] for r in p["repeats"] if r["ok"]}) for p in all_results
    }

    print(f"\n=== AGGREGATE: {config_name} ===")
    print(f"Clean on first attempt: {n_clean_first}/{total}")
    print(f"Clean after retry: {n_clean_retry}/{total}")
    print(f"Exhausted retry budget: {n_exhausted}/{total}")
    print(f"Final-output error rate: {n_exhausted}/{total} = {100*n_exhausted/total:.1f}%")
    print("Unique summary text count per patient (1 = fully reproducible across all successful repeats):")
    for pid, n in unique_summaries_per_patient.items():
        print(f"  {pid}: {n} distinct summaries")
    return all_results


results_temp_only = run_config(
    "temperature=0.2, no seed", TEMPERATURE, use_per_patient_seed=False,
    raw_dir=PROJECT_ROOT / "results" / "temp_seed_test_raw" / "temp_only",
    output_json="results/temp_seed_test_temp_only.json",
)

results_temp_seed = run_config(
    "temperature=0.2 + per-patient fixed seed", TEMPERATURE, use_per_patient_seed=True,
    raw_dir=PROJECT_ROOT / "results" / "temp_seed_test_raw" / "temp_and_seed",
    output_json="results/temp_seed_test_temp_and_seed.json",
)

print("\n\n=== DONE. Compare against results/cot_repeated_reliability_sample.json (default temp/seed, post num_ctx fix) ===")
