"""
A/B for KG consumer #3 (cross-modality convergence). Same contributions, same
fixed seed -> the ONLY difference is the KG convergence block (use_kg off vs on).
Synthetic convergent-hypertrophy case (architecture validation, like the existing
3-modality test): ECG LVH + MRI HCM + patient hypertension all converge on LV
hypertrophy / hypertensive heart disease in the clinical KG.
"""
import re, sys
from functools import partial
sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_medgemma
from sep_agent import generate_sep_agent_decision
from knowledge_graph import load_kg, convergent_diagnoses

ecg = AgentContribution("ecg", "synthetic_lvh", [
    Finding("sokolow_lyon_index_mv", "4.20 mV",
            "Sokolow-Lyon voltage index exceeds the 3.5 mV threshold, consistent with left ventricular hypertrophy.",
            0.80, "sokolow_lyon_index_mv"),
    Finding("heart_rate_mean_bpm", "78 bpm", "Heart rate within normal range.", 0.9, "heart_rate_mean_bpm"),
], ["lvh_sokolow_lyon_suspected"], {})

mri = AgentContribution("mri", "synthetic_lvh", [
    Finding("pathology_HCM_probability", "0.95",
            "Learned pathology classifier indicates a hypertrophic cardiomyopathy pattern on the raw MRI.",
            0.90, "pathology_HCM_probability"),
], ["pathology_HCM_pattern_suspected"], {})

patient = AgentContribution("patient", "synthetic_lvh", [
    Finding("resting_bp_mmHg", "160 mmHg", "Elevated resting blood pressure.", 0.75, "resting_bp_mmHg"),
], ["high_resting_bp_suspected"], {})

contribs = [ecg, mri, patient]
all_flags = [f for c in contribs for f in c.flags_for_review]
print("=== KG convergence (what use_kg injects) ===")
for d, e in convergent_diagnoses(load_kg(), all_flags).items():
    print(f"  {e['label']}: {sorted(e['agents'])}")

llm = partial(generate_report_medgemma, seed=42)
CONV = re.compile(r"converg|hypertroph|hypertensive|across (the )?modalit|all three|both", re.I)

for label, use_kg in [("A_no_kg", False), ("B_kg", True)]:
    print(f"\n{'='*60}\n=== {label} (use_kg={use_kg}) ===")
    dec = generate_sep_agent_decision(contribs, llm_call=llm, record_id=label, use_kg=use_kg)
    print("SUMMARY:\n", dec.summary)
    print("mentions convergence/hypertrophy/hypertensive?:", bool(CONV.search(dec.summary)))
