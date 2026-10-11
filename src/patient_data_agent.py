"""
Structured patient data agent - structured findings generation
==================================================================
Third modality agent in the multi-agent clinical pipeline, alongside
ecg_agent.py and mri_agent.py. Same contract as those two: this module does
NOT produce a final diagnostic report, it produces a structured
"contribution" object (a list of individual findings, with a short rationale
and a confidence score, plus a deterministic decision trace) for the Sep Agent
to combine with the other modalities. No cross-modality or diagnostic conclusions
are made here.

Data source: UCI Heart Disease dataset (Cleveland subset, 303 patients, 13
input features + a target - https://doi.org/10.24432/C52P4X, UCI repo
id=45). Authoritative column names/types/descriptions were fetched once via
the ucimlrepo package and saved locally (uci_heart_disease/
processed_cleveland.csv, uci_heart_disease/variables_metadata.csv) - this
module reads that local CSV, not a live network fetch, matching this
project's established local-first pattern for reference data (PTB-XL,
ACDC) and avoiding a runtime dependency on UCI's servers (via ucimlrepo)

Feature naming: deliberately NOT namespaced (e.g. "resting_bp_mmHg", not
"patient.resting_bp_mmHg") - confirmed directly in ecg_agent.py/mri_agent.py
that NEITHER existing agent actually namespaces its feature_dict keys.

Pipeline: one patient row (13 raw UCI fields) -> feature dict (categorical
codes translated to labels, continuous values passed through) -> text
summary -> structured-findings prompt (via agent_common's shared scaffold)
-> LLM -> parsed/validated AgentContribution. Unlike ECG/MRI, there is no
separate "preprocessing" step - the raw row already IS the modality's
native representation, no signal/image processing happens here.

Deterministic thresholds: see detect_flags_patient's docstring for the
starting values and which are still pending empirical validation/
recalibration against the binary target via
src/patient_data_flag_validation.py - not assumed correct just
because they're commonly-cited clinical cutoffs (les seuils standards de la littérature médicale)
they were validated against the dataset's binary target (heart_disease_present = num > 0)
and recalibrated where necessary.
***"même si 140 mmHg ou 240 mg/dl sont des seuils médicaux universellement reconnus, on ne les adopte pas aveuglément — on les teste empiriquement sur nos vraies données, parce qu'un seuil optimal pour poser un diagnostic n'est pas forcément optimal pour déclencher une alerte de révision."***

LLM backend: same shared MedGemma 4B backbone as ECG/MRI (agent_common.
generate_report_medgemma), called text-only - no images in this modality.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from agent_common import (
    AgentContribution,
    FindingsParseError,
    build_generic_findings_prompt,
    generate_report_medgemma,
    parse_findings_response,
    save_contribution,
)

__all__ = [
    "compute_feature_dict",
    "build_patient_clinical_summary",
    "PatientDecisionPlan",
    "detect_flags_patient",
    "build_patient_findings_prompt",
    "generate_patient_contribution",
    "save_contribution",
    "generate_report_medgemma",
]


# Label mappings for the UCI-encoded categorical fields (see
# uci_heart_disease/variables_metadata.csv for the authoritative
# descriptions this project fetched via ucimlrepo). Only the numeric ->
# label translation lives here; the underlying question of what number
# means what is UCI's documented encoding, not a judgment call this module
# makes.
_SEX_LABELS = {1: "male", 0: "female"}
_CP_LABELS = {
    1: "typical_angina",
    2: "atypical_angina",
    3: "non_anginal_pain",
    4: "asymptomatic",
}
_RESTECG_LABELS = {0: "normal", 1: "st_t_abnormality", 2: "probable_lvh"}
_SLOPE_LABELS = {1: "upsloping", 2: "flat", 3: "downsloping"}
_THAL_LABELS = {3: "normal", 6: "fixed_defect", 7: "reversible_defect"}


# ----------------------------------------------------------------------
# 1. Feature extraction (the source of valid "evidence_ref" keys)
# ----------------------------------------------------------------------
def compute_feature_dict(row) -> Dict[str, Any]:
    """
    Translate one raw UCI Heart Disease row (13 fields: age, sex, cp,
    trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    into the flat feature dictionary this agent reports on. This is the
    single source of truth for "evidence_ref" values: the LLM may only
    cite keys that exist here.

    Categorical codes are translated to human-readable labels (e.g.
    cp=4 -> "asymptomatic") using UCI's documented encoding - this is a
    direct label lookup, not a clinical judgment call.
    Continuous/count fields (age, trestbps, chol, thalach, oldpeak, ca)
    are passed through as-is;
    any qualitative interpretation of them (e.g. "is 260 mg/dl high
    cholesterol?") happens only in detect_flags_patient's
    deterministic thresholds, never here.

    Missing values (ca/thal only - the two fields UCI documents as having
    missing entries in this file, encoded "?" in the raw .data format,
    NaN once loaded via pandas/ucimlrepo) are reported as None, same
    convention as ecg_agent.compute_feature_dict's "None where not
    computable" - never guessed.

    Inputs:
        row: dict-like with the 13 raw UCI field names as keys.

    Output:
        dict mapping feature name -> value (13 keys; ca/thal may be None).
    """
    def _get(key):
        value = row[key]
        return None if pd.isna(value) else value

    sex_code = _get("sex")
    cp_code = _get("cp")
    restecg_code = _get("restecg")
    slope_code = _get("slope")
    thal_code = _get("thal")
    ca_value = _get("ca")

    return {
        "age_years": _get("age"),
        "sex": _SEX_LABELS.get(int(sex_code)) if sex_code is not None else None,
        "chest_pain_type": _CP_LABELS.get(int(cp_code)) if cp_code is not None else None,
        "resting_bp_mmHg": _get("trestbps"),
        "cholesterol_mgdl": _get("chol"),
        "fasting_blood_sugar_elevated": bool(_get("fbs")) if _get("fbs") is not None else None,
        "resting_ecg_result": _RESTECG_LABELS.get(int(restecg_code)) if restecg_code is not None else None,
        "max_heart_rate_achieved_bpm": _get("thalach"), 
        "exercise_induced_angina": bool(_get("exang")) if _get("exang") is not None else None,
        "st_depression_oldpeak_mm": _get("oldpeak"),
        "st_segment_slope": _SLOPE_LABELS.get(int(slope_code)) if slope_code is not None else None,
        "n_major_vessels_colored": ca_value,
        "thalassemia_result": _THAL_LABELS.get(int(thal_code)) if thal_code is not None else None,
    }


def build_patient_clinical_summary(feature_dict, patient_info=None) -> str:
    """
    Convert the feature dict (see compute_feature_dict) into a readable
    plain-text summary, used as context inside the LLM prompt. Missing
    fields (ca/thal, when present in the source row) are omitted entirely
    rather than printed as "None", same convention as 
    agent_common.build_patient_context_line.

    Inputs:
        feature_dict: dict produced by compute_feature_dict.
        patient_info: optional free-text patient context string. Not used
            by any current caller of this module - age/sex are already
            this agent's own findings with full rationale/evidence_ref,
            unlike ECG/MRI where demographics are external background
            context - kept only for interface consistency with the other
            two agents' summary builders.

    Output:
        Multi-line text summary (str).
    """
    f = feature_dict
    lines = [
        f"- Age: {f['age_years']:.0f} years" if f["age_years"] is not None else "- Age: not available",
        f"- Sex: {f['sex']}" if f["sex"] is not None else "- Sex: not available",
        f"- Chest pain type: {f['chest_pain_type']}" if f["chest_pain_type"] is not None else "- Chest pain type: not available",
        f"- Resting blood pressure: {f['resting_bp_mmHg']:.0f} mmHg" if f["resting_bp_mmHg"] is not None else "- Resting blood pressure: not available",
        f"- Serum cholesterol: {f['cholesterol_mgdl']:.0f} mg/dl" if f["cholesterol_mgdl"] is not None else "- Serum cholesterol: not available",
    ]
    if f["fasting_blood_sugar_elevated"] is not None:
        lines.append(
            f"- Fasting blood sugar > 120 mg/dl: {'yes' if f['fasting_blood_sugar_elevated'] else 'no'}"
        )
    if f["resting_ecg_result"] is not None:
        lines.append(f"- Resting ECG result: {f['resting_ecg_result']}")
    if f["max_heart_rate_achieved_bpm"] is not None:
        lines.append(f"- Maximum heart rate achieved (exercise test): {f['max_heart_rate_achieved_bpm']:.0f} bpm")
    if f["exercise_induced_angina"] is not None:
        lines.append(
            f"- Exercise-induced angina: {'yes' if f['exercise_induced_angina'] else 'no'}"
        )
    if f["st_depression_oldpeak_mm"] is not None:
        lines.append(f"- ST depression induced by exercise (relative to rest): {f['st_depression_oldpeak_mm']:.1f} mm")
    if f["st_segment_slope"] is not None:
        lines.append(f"- Slope of peak exercise ST segment: {f['st_segment_slope']}")
    if f["n_major_vessels_colored"] is not None:
        lines.append(f"- Major vessels colored by fluoroscopy: {f['n_major_vessels_colored']:.0f}")
    else:
        lines.append("- Major vessels colored by fluoroscopy: not available")
    if f["thalassemia_result"] is not None:
        lines.append(f"- Thallium stress test result: {f['thalassemia_result']}")
    else:
        lines.append("- Thallium stress test result: not available")

    if patient_info:
        lines.append(f"- Patient context: {patient_info}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 2. Deterministic decision logic (rule-based, independent of the LLM)
# ----------------------------------------------------------------------
@dataclass
class PatientDecisionPlan:
    """
    Deterministic, threshold-based decision computed directly from the
    feature dict, independent of any LLM judgment - same role as
    ecg_agent.DecisionPlan / mri_agent.MRIDecisionPlan.

    Inputs (fields):
        deterministic_flags: short machine-readable tags raised by fixed
            thresholds (e.g. "high_resting_bp_suspected"), merged into the
            final AgentContribution.flags_for_review alongside the LLM's
            own flags.
        skip_llm_call: always False for this modality, the UCI Cleveland
            dataset has no field that its absence makes the
            record unanalyzable; only ca/thal are ever missing,
            and every other field is always present. Kept for structural
            consistency with the other two agents' decision plans.
        reason: always None (see skip_llm_call above).
    """

    deterministic_flags: List[str]
    skip_llm_call: bool
    reason: Optional[str]


def detect_flags_patient(
    feature_dict: Dict[str, Any],
    resting_bp_mmHg_threshold: float = 140.0,
    cholesterol_mgdl_threshold: float = 240.0,
    oldpeak_threshold: float = 0.8,
    max_hr_formula: str = "tanaka",
    max_hr_pct_threshold: float = 0.91,
    n_vessels_threshold: int = 1,
    #exercise_induced_angina,thalassemia_result are binary / categorical - no thresholds fixed
    
) -> PatientDecisionPlan:
    """
    Inspect the feature dict and decide, deterministically, which review
    flags to raise. Runs independently of any LLM output.

    All defaults below were validated against the binary target
    (heart_disease_present = num > 0) over all 303 Cleveland patients
    via src/patient_data_flag_validation.py, no LLM involved.

    Inputs:
        feature_dict: dict produced by compute_feature_dict.
        resting_bp_mmHg_threshold: resting BP (mmHg) above which
            "high_resting_bp_suspected" is raised. Kept at 140 mmHg
            NOT because it validated well: precision stays close to
            the dataset's base prevalence (0.459) at essentially every
            threshold tested (110-170), and F1 only improves by flagging
            almost everyone (110: F1=0.615, precision=0.470, barely above
            the 0.459 base rate). So the clinically meaningful cutoff was
            kept rather than chasing a statistically marginal, clinically
            meaningless gain.
        cholesterol_mgdl_threshold: serum cholesterol (mg/dl) above which
            "high_cholesterol_suspected" is raised. Kept at 240 mg/dl
            for the identical reason as resting_bp_mmHg_threshold above
            - precision never clears the dataset's base prevalence by much
            at any tested threshold (180-300).
        oldpeak_threshold: ST depression (mm) at or above which
            "significant_st_depression_suspected" is raised. Recalibrated
            from an initial 1.0 to 0.8 mm after sweeping 0.0-4.0 in 0.1
            steps: 0.8 Pareto-dominates the original 1.0 on both
            precision (0.654 vs 0.672 - close) and, more materially,
            recall (0.719 vs 0.662) and F1 (0.685 vs 0.667).
        max_hr_formula: which age-predicted-maximum-heart-rate formula to
            use for "reduced_exercise_capacity_suspected" - "tanaka"
            (208 - 0.7*age, Tanaka et al. 2001) or "220_minus_age"
            (classic). Recalibrated from an initial "220_minus_age" to
            "tanaka" after jointly sweeping both formulas against a
            percentage-threshold range (0.70-0.95): "tanaka" outperforms
            the classic formula at essentially every matched percentage,
            "tanaka" being the more recent, better-validated formula in
            the exercise-physiology literature, not just a better fit to
            this specific sample.
        max_hr_pct_threshold: fraction of the age-predicted maximum heart
            rate below which "reduced_exercise_capacity_suspected" is
            raised. Recalibrated from an initial 0.85 to 0.91 (with the
            "tanaka" formula) - the joint sweep's F1-maximizing point
            (precision 0.636, recall 0.705, F1 0.669), which also has the
            highest recall of any threshold on the surrounding F1 plateau
            (0.89-0.91): because the flag fires when max_hr < pct*predicted,
            a HIGHER pct raises more positives, so 0.91 dominates the nearby
            0.90 (recall 0.683, F1 0.664) on both recall and F1, giving up
            only a little precision (0.636 vs 0.646). Consistent with the
            recall-favoring philosophy already applied to
            ecg_agent.tachycardia_bpm ("a missed true case is a worse
            failure mode than an extra reviewed-and-dismissed false
            positive").
        n_vessels_threshold: number of fluoroscopy-colored major vessels
            at or above which "multiple_vessels_involved_suspected" is
            raised. Confirmed optimal at 1 (any vessel involvement) after
            sweeping {1, 2, 3}: F1 drops sharply at 2 (0.490) and 3
            (0.215) - the original starting choice needed no change.

    Output:
        PatientDecisionPlan instance.
    """
    flags = []
    f = feature_dict

    if f["resting_bp_mmHg"] is not None and f["resting_bp_mmHg"] > resting_bp_mmHg_threshold:
        flags.append("high_resting_bp_suspected")

    if f["cholesterol_mgdl"] is not None and f["cholesterol_mgdl"] > cholesterol_mgdl_threshold:
        flags.append("high_cholesterol_suspected")

    if f["max_heart_rate_achieved_bpm"] is not None and f["age_years"] is not None:
        if max_hr_formula == "tanaka":
            predicted_max = 208.0 - 0.7 * f["age_years"]
        else:
            predicted_max = 220.0 - f["age_years"]
        if f["max_heart_rate_achieved_bpm"] < max_hr_pct_threshold * predicted_max:
            flags.append("reduced_exercise_capacity_suspected")

    if f["st_depression_oldpeak_mm"] is not None and f["st_depression_oldpeak_mm"] >= oldpeak_threshold:
        flags.append("significant_st_depression_suspected")

    if f["exercise_induced_angina"] is True:
        flags.append("exercise_induced_angina_present")

    if f["n_major_vessels_colored"] is not None and f["n_major_vessels_colored"] >= n_vessels_threshold:
        flags.append("multiple_vessels_involved_suspected")

    if f["thalassemia_result"] is not None and f["thalassemia_result"] != "normal":
        flags.append("abnormal_thallium_result_suspected")

    return PatientDecisionPlan(deterministic_flags=flags, skip_llm_call=False, reason=None)


# ----------------------------------------------------------------------
# 3. Structured-findings prompt (via the shared scaffold)
# ----------------------------------------------------------------------
def build_patient_findings_prompt(summary, feature_dict, patient_info=None):
    """
    Build the structured patient-data findings prompt via the shared
    prompt scaffold (agent_common.build_generic_findings_prompt),
    specialized with this agent's role, modality label, and evidence
    keys. The model is explicitly told it is one of several parallel
    agents and must not draw a final or cross-modality conclusion
    (enforced by the shared scaffold) - same "no diagnosis" contract as
    ECG/MRI.

    Inputs:
        summary: text summary produced by build_patient_clinical_summary.
        feature_dict: dict produced by compute_feature_dict; its keys are
            the only valid values for "evidence_ref" in the model's
            output.
        patient_info: optional free-text patient context string (see
            build_patient_clinical_summary's docstring - not used by any
            current caller).

    Output:
        Prompt string ready to send to an LLM backend.
    """
    return build_generic_findings_prompt(
        agent_role="a structured patient data analysis agent",
        modality_label="structured patient data",
        summary=summary,
        evidence_keys=feature_dict.keys(),
        patient_info=patient_info,
    )


# ----------------------------------------------------------------------
# 4. Orchestration - this agent's end-to-end contribution pipeline
# ----------------------------------------------------------------------
def generate_patient_contribution(
    row,
    llm_call: Optional[Callable[[str], str]] = None,
    record_id: Optional[str] = None,
    patient_info: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1,
) -> AgentContribution:
    """
    Run this agent's full contribution pipeline: translate the raw row
    into features, run the deterministic decision logic
    (detect_flags_patient), build the structured-findings prompt,
    call the given LLM backend, and parse/validate the response into an
    AgentContribution. On a parse failure, retries by appending the parse
    error to the prompt and asking the model to correct it. Deterministic
    flags are always merged into the final flags_for_review.

    Inputs:
        row: dict-like with the 13 raw UCI Heart Disease field names as keys.
        llm_call: callable taking a single prompt string and returning the
            raw text response. Defaults to agent_common.
            generate_report_medgemma (free, local, no API key) - requires
            Ollama running locally with a model pulled.
        record_id: optional identifier for this patient row, stored in the
            output for traceability. The source data has no real patient
            ID - callers typically pass a positional identifier (e.g.
            f"cleveland_{row_index}").
        patient_info: optional free-text patient context (see
            build_patient_clinical_summary's docstring).
        model_name: optional string identifying the model/backend used,
            stored in the trace for auditability. Defaults to
            "ollama:medgemma:4b" when llm_call is left at its default.
        max_retries: number of extra attempts if the LLM response fails to
            parse.

    Output:
        AgentContribution instance (agent_id="patient").

    Raises:
        FindingsParseError: if no valid structured response is obtained
        within max_retries + 1 attempts.
    """
    if llm_call is None:
        llm_call = generate_report_medgemma
        if model_name is None:
            model_name = "ollama:medgemma:4b"

    feature_dict = compute_feature_dict(row)
    decision = detect_flags_patient(feature_dict)

    summary = build_patient_clinical_summary(feature_dict, patient_info)
    prompt = build_patient_findings_prompt(summary, feature_dict, patient_info)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw_response = llm_call(prompt)
        try:
            findings, flags = parse_findings_response(raw_response, set(feature_dict.keys()))
            merged_flags = sorted(set(flags) | set(decision.deterministic_flags))
            return AgentContribution(
                agent_id="patient",
                record_id=record_id,
                findings=findings,
                flags_for_review=merged_flags,
                trace={
                    "features_used": feature_dict,
                    "model": model_name,
                    "attempts": attempt + 1,
                },
            )
        except FindingsParseError as exc:
            last_error = exc
            prompt = (
                prompt
                + f"\n\nYour previous response could not be parsed: {exc}. "
                "Return ONLY the corrected JSON object, nothing else."
            )

    raise FindingsParseError(
        f"Failed to obtain valid structured findings after "
        f"{max_retries + 1} attempts: {last_error}"
    )