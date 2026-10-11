"""
End-to-end KG demo: a synthetic 3-modality convergent-hypertrophy case ->
Sep Agent with use_kg=True -> case-report PDF showing the cross-modality
convergence section. Architecture demo (synthetic pairing), like sep_agent_demo.
"""
import os, sys
from functools import partial
sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_medgemma
from sep_agent import generate_sep_agent_decision, save_sep_agent_decision
from report_pdf import generate_case_report_pdf

ecg = AgentContribution("ecg", "synthetic_lvh_ecg", [
    Finding("sokolow_lyon_index_mv", "4.20 mV",
            "Sokolow-Lyon voltage index exceeds the 3.5 mV threshold, consistent with left ventricular hypertrophy.",
            0.80, "sokolow_lyon_index_mv"),
    Finding("heart_rate_mean_bpm", "78 bpm", "Heart rate within normal range.", 0.90, "heart_rate_mean_bpm"),
], ["lvh_sokolow_lyon_suspected"], {})

mri = AgentContribution("mri", "synthetic_hcm_mri", [
    Finding("pathology_HCM_probability", "0.95",
            "Learned pathology classifier indicates a hypertrophic cardiomyopathy pattern on the raw MRI.",
            0.90, "pathology_HCM_probability"),
], ["pathology_HCM_pattern_suspected"], {})

patient = AgentContribution("patient", "synthetic_htn_patient", [
    Finding("resting_bp_mmHg", "160 mmHg", "Elevated resting blood pressure.", 0.75, "resting_bp_mmHg"),
], ["high_resting_bp_suspected"], {})

contribs = [ecg, mri, patient]
llm = partial(generate_report_medgemma, seed=42)
dec = generate_sep_agent_decision(contribs, llm_call=llm, record_id="kg_convergence_demo", use_kg=True)

print("SUMMARY:\n", dec.summary)
print("\nDeterministic KG convergence (trace):")
for d, e in dec.trace.get("kg_convergent_diagnoses", {}).items():
    print(f"  {e['label']}: {e['agents']}")

out_dir = "results/kg_integration"
os.makedirs(out_dir, exist_ok=True)
save_sep_agent_decision(dec, output_dir=out_dir, file_name="kg_demo_sep_decision")
pdf = generate_case_report_pdf(
    os.path.join(out_dir, "kg_demo_case_report.pdf"),
    dec, contribs,
    synthetic_pairing_note=("synthetic 3-modality pairing for architecture demonstration - the ECG, MRI, "
                            "and patient-data records are not from the same real patient."),
)
print("\nPDF written:", pdf)
