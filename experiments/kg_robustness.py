"""
Robustness of the KG-convergence effect on MedGemma 4B: across several seeds, does
use_kg=True make the 4B name the convergent diagnosis ("hypertensive heart disease")
more often than use_kg=False? Tests the clean case (no EF finding). n = len(seeds)
per condition.
"""
import sys
from functools import partial
sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_medgemma
from sep_agent import generate_sep_agent_decision

ecg = AgentContribution("ecg", "syn", [
    Finding("sokolow_lyon_index_mv", "4.20 mV",
            "Sokolow-Lyon voltage index exceeds the 3.5 mV threshold, consistent with left ventricular hypertrophy.",
            0.80, "sokolow_lyon_index_mv"),
    Finding("heart_rate_mean_bpm", "78 bpm", "Heart rate within normal range.", 0.9, "heart_rate_mean_bpm"),
], ["lvh_sokolow_lyon_suspected"], {})
mri = AgentContribution("mri", "syn", [
    Finding("pathology_HCM_probability", "0.95",
            "Learned pathology classifier indicates a hypertrophic cardiomyopathy pattern on the raw MRI.",
            0.90, "pathology_HCM_probability"),
], ["pathology_HCM_pattern_suspected"], {})
patient = AgentContribution("patient", "syn", [
    Finding("resting_bp_mmHg", "160 mmHg", "Elevated resting blood pressure.", 0.75, "resting_bp_mmHg"),
], ["high_resting_bp_suspected"], {})
contribs = [ecg, mri, patient]

SEEDS = [1, 2, 3, 42, 100]
TARGET = "hypertensive heart"  # the ecg+mri+patient convergent diagnosis


def names_convergent(use_kg, seed):
    dec = generate_sep_agent_decision(
        contribs, llm_call=partial(generate_report_medgemma, seed=seed),
        record_id=f"s{seed}", use_kg=use_kg)
    return TARGET in dec.summary.lower()


a_hits = b_hits = 0
for s in SEEDS:
    a = names_convergent(False, s)
    b = names_convergent(True, s)
    a_hits += a; b_hits += b
    print(f"seed {s:4d}: no_kg names convergent = {a} | kg names convergent = {b}")
n = len(SEEDS)
print(f"\nMedGemma 4B, n={n} seeds, clean case:")
print(f"  use_kg=False names 'hypertensive heart disease': {a_hits}/{n}")
print(f"  use_kg=True  names 'hypertensive heart disease': {b_hits}/{n}")
