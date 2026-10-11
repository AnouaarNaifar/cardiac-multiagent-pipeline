"""
ECG agent - structured findings generation
============================================
This module is the ECG-specific agent in a larger multi-agent clinical
pipeline. Alongside ECG, parallel agents will independently analyze MRI data
and structured patient data; a separate aggregating component (the "Sep
Agent") combines all of their outputs into a final clinical decision plus a
traceability summary of each agent's contribution.

Because of this, this module does NOT produce a final diagnostic report. It
produces a structured "contribution" object: a list of individual findings,
each grounded in a specific computed ECG feature with a short rationale and a
confidence score, plus a trace of what fed into it. No cross-modality or
diagnostic conclusions are made here - that is the Sep Agent's job.

Pipeline: ECG pipeline output -> numeric feature dict -> text summary ->
structured-findings prompt -> LLM -> parsed/validated AgentContribution.

Shared building blocks (schema, prompt scaffold, JSON parsing/validation, LLM
backends, contribution saving) now live in agent_common.py, so future
modality agents (MRI, structured patient data) reuse them instead of
duplicating them - see agent_common.py's docstring. This module keeps only
what is genuinely ECG-specific: feature extraction, the clinical-text
summary, deterministic decision thresholds, and the two orchestration modes
below.

LLM backend: free and local by default, via Ollama running MedGemma 4B (a
medical-domain model). generate_ecg_contribution uses generate_report_medgemma
(Ollama) unless a different llm_call is passed in. Requires Ollama installed
and running locally (https://ollama.com/download) with the model pulled
beforehand:
    ollama pull medgemma:4b
    pip install requests
The Anthropic backend (generate_report_anthropic) is kept as an optional, non-default
alternative for later - it needs a paid ANTHROPIC_API_KEY and is not required
for anything in this module to work.

Learned-embedding integration (optional, opt-in): compute_feature_dict can
additionally report ecg_embedding.py's trained ECGEmbeddingNet diagnostic-
superclass probabilities as ordinary numeric features (see
compute_embedding_features) - the only diagnostic-adjacent signal in this
module beyond simple rate/QRS-duration thresholds. This is deliberately just
another probabilistic Finding, not a diagnosis: the model may cite and
comment on a probability, but nothing here asserts a diagnosis, preserving
this module's "findings only" contract. Requires the FULL multi-lead signal
(unlike every other feature here, which is derived from a single lead - see
ecg_preprocessing.py's documented single-lead limitation), since the
embedding model was trained on all 12 leads - only computed if both
multilead_signal and embedding_model are explicitly passed in; omitting
either (the default) skips it entirely, so existing single-lead-only call
sites are unaffected. This adds torch as a hard dependency of this module
(via the ecg_embedding import), not just of ecg_embedding.py itself.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agent_common import (
    AgentContribution,
    FindingsParseError,
    build_generic_findings_prompt,
    generate_report_anthropic,
    generate_report_medgemma,
    parse_findings_response,
    save_contribution,
)
from ecg_embedding import DIAGNOSTIC_SUPERCLASSES, predict_diagnostic_probabilities
from ecg_multilead import (
    SOKOLOW_LYON_LVH_THRESHOLD_MV,
    compute_multilead_features,
)
from ecg_territorial import compute_territorial_features

# Re-exported for backward compatibility: existing code that does
# `from ecg_agent import save_contribution` (e.g. the notebook) keeps working
# unchanged, since save_contribution is fully modality-agnostic and now lives
# in agent_common.py.
__all__ = [
    "compute_embedding_features",
    "compute_multilead_features",
    "compute_feature_dict",
    "build_clinical_summary",
    "DecisionPlan",
    "detect_flags",
    "build_findings_prompt",
    "generate_ecg_contribution",
    "save_contribution",
    "generate_report_medgemma",
    "generate_report_anthropic",
]


# Full names for ecg_embedding.DIAGNOSTIC_SUPERCLASSES's bare abbreviations -
# spelled out in build_clinical_summary's text so the LLM doesn't have to
# guess their meaning. Found necessary empirically: MedGemma, given just
# "HYP" with no expansion, glossed it in its findings rationale as
# "hypokinesis" - a real clinical concept, but the WRONG one (PTB-XL's HYP
# means hypertrophy) - a factual error in generated text that giving the
# expansion up front prevents.
_DIAGNOSTIC_SUPERCLASS_NAMES = {
    "CD": "conduction disturbance",
    "HYP": "hypertrophy",
    "MI": "myocardial infarction",
    "NORM": "normal",
    "STTC": "ST/T change",
}


# ----------------------------------------------------------------------
# 1. Numeric feature extraction (shared by the text summary and the
#    structured findings prompt, and the source of valid "evidence_ref" keys)
# ----------------------------------------------------------------------
def compute_embedding_features(multilead_signal, embedding_model) -> Dict[str, float]:
    """
    Compute PTB-XL diagnostic-superclass probabilities from ecg_embedding.py's
    trained ECGEmbeddingNet, using the FULL multi-lead signal - unlike every
    other feature in this module (derived from a single lead, see
    ecg_preprocessing.py's documented limitation), since the embedding model
    was trained on all 12 leads (see ecg_embedding.PTBXLDataset).

    Normalizes identically to training (per-lead z-score, see PTBXLDataset)
    so the model sees the same input distribution it was trained on.

    Inputs:
        multilead_signal: numpy array of shape (n_samples, n_leads), i.e.
            wfdb.rdsamp's raw output shape (NOT pre-normalized, NOT
            transposed) - the same array a caller would slice with [:, 0] to
            get the single lead used elsewhere in this module.
        embedding_model: a trained ECGEmbeddingNet instance (see
            ecg_embedding.load_model_checkpoint) - typically the
            patient-disjoint "embedding_stratfold.best.pt" checkpoint (see
            CLAUDE.md for why that split, not the earlier leaky one, is the
            one to trust).

    Output:
        dict mapping "embedding_<CLASS>_probability" -> float in [0, 1], one
        per ecg_embedding.DIAGNOSTIC_SUPERCLASSES entry.
    """
    sig = multilead_signal.astype(np.float32).T  # (n_leads, n_samples)
    sig = (sig - sig.mean(axis=1, keepdims=True)) / (sig.std(axis=1, keepdims=True) + 1e-8)
    probabilities = predict_diagnostic_probabilities(sig, embedding_model)
    return {
        f"embedding_{cls}_probability": float(prob)
        for cls, prob in zip(DIAGNOSTIC_SUPERCLASSES, probabilities)
    }


def compute_feature_dict(results, fs, multilead_signal=None, embedding_model=None):
    """
    Compute the flat numeric feature dictionary derived from the ECG
    preprocessing pipeline output. This is the single source of truth for
    "evidence_ref" values: the LLM may only cite keys that exist here.

    Inputs:
        results: dict returned by ecg_preprocessing_pipeline. Uses
            "r_peaks", "rr_intervals", "raw", "qrs_durations_ms",
            "rhythm_irregular", and "rhythm_cv" from that dict.
        fs: Sampling frequency in Hz.
        multilead_signal: optional full multi-lead signal (see
            compute_embedding_features); if given together with
            embedding_model, adds "embedding_<CLASS>_probability" keys.
            Omitted by default - existing single-lead-only callers see no
            change in the returned dict.
        embedding_model: optional trained ECGEmbeddingNet instance (see
            compute_embedding_features).

    Output:
        dict mapping feature name -> numeric value (None where not computable,
        e.g. heart rate with fewer than 2 detected R peaks).
    """
    r_peaks = results["r_peaks"]
    rr = results["rr_intervals"]  # in seconds
    duration_s = len(results["raw"]) / fs
    qrs_durations = results["qrs_durations_ms"]

    if len(rr) >= 1:
        hr_series = 60.0 / rr
        heart_rate = float(np.mean(hr_series))
        hr_std = float(np.std(hr_series)) if len(hr_series) > 1 else 0.0
        rr_mean_ms = float(np.mean(rr) * 1000)
        rr_std_ms = float(np.std(rr) * 1000) if len(rr) > 1 else 0.0
    else:
        heart_rate = hr_std = rr_mean_ms = rr_std_ms = None

    if len(qrs_durations) >= 1:
        qrs_duration_mean_ms = float(np.mean(qrs_durations))
        qrs_duration_std_ms = float(np.std(qrs_durations)) if len(qrs_durations) > 1 else 0.0
    else:
        qrs_duration_mean_ms = qrs_duration_std_ms = None

    feature_dict = {
        "duration_s": duration_s,               # durée du tracé (secondes)
        "qrs_count": len(r_peaks),              # nombre de complexes QRS détectés
        "heart_rate_mean_bpm": heart_rate,      # FC moyenne (battements/minute): frequence
        "heart_rate_std_bpm": hr_std,           # écart-type de la FC
        "rr_mean_ms": rr_mean_ms,               # intervalle RR moyen (ms)
        "rr_std_ms": rr_std_ms,                 # écart-type de l'intervalle RR (ms)
        "qrs_duration_mean_ms": qrs_duration_mean_ms,
        "qrs_duration_std_ms": qrs_duration_std_ms,
        "rhythm_irregularity_flag": results["rhythm_irregular"],
        "rhythm_cv": results["rhythm_cv"],
    }

    # Multi-lead deterministic features (Sokolow-Lyon LVH index, see
    # ecg_multilead.py) - need the full 12-lead RAW signal, but NOT a model, so
    # they are added whenever multilead_signal is given, independently of the
    # embedding. No-op for single-lead-only callers (multilead_signal=None).
    if multilead_signal is not None:
        feature_dict.update(compute_multilead_features(multilead_signal, fs))
        feature_dict.update(compute_territorial_features(multilead_signal, fs))

    if multilead_signal is not None and embedding_model is not None:
        feature_dict.update(compute_embedding_features(multilead_signal, embedding_model))

    return feature_dict


def build_clinical_summary(results, fs, patient_info=None, multilead_signal=None, embedding_model=None):
    """
    Convert the numeric feature dict (see compute_feature_dict) into a
    readable plain-text summary, used as context inside the LLM prompt.

    Inputs:
        results: dict returned by ecg_preprocessing_pipeline.
        fs: Sampling frequency in Hz.
        patient_info: optional free-text patient context string.
        multilead_signal: optional full multi-lead signal (see
            compute_embedding_features); adds a line per diagnostic
            superclass probability if given together with embedding_model.
        embedding_model: optional trained ECGEmbeddingNet instance.

    Output:
        Multi-line text summary (str).
    """
    features = compute_feature_dict(results, fs, multilead_signal, embedding_model)

    lines = [
        f"- Duration of the analyzed trace: {features['duration_s']:.1f} s",
        f"- Number of detected QRS complexes: {features['qrs_count']}",
    ]
    if features["heart_rate_mean_bpm"] is not None:
        lines.append(
            f"- Estimated mean heart rate: {features['heart_rate_mean_bpm']:.1f} bpm "
            f"(std {features['heart_rate_std_bpm']:.1f})"
        )
        lines.append(
            f"- Mean RR interval: {features['rr_mean_ms']:.1f} ms "
            f"(std {features['rr_std_ms']:.1f} ms)"
        )
    else:
        lines.append("- Heart rate: not computable (fewer than 2 R peaks detected)")

    if features["qrs_duration_mean_ms"] is not None:
        lines.append(
            f"- Mean QRS duration: {features['qrs_duration_mean_ms']:.1f} ms "
            f"(std {features['qrs_duration_std_ms']:.1f} ms)"
        )

    if features["rhythm_irregularity_flag"]:
        lines.append(
            f"- Rhythm irregularity suspected "
            f"(RR coefficient of variation {features['rhythm_cv']:.2f})"
        )

    if features.get("sokolow_lyon_index_mv") is not None:
        lines.append(
            f"- Sokolow-Lyon voltage index (multi-lead, for left ventricular "
            f"hypertrophy): {features['sokolow_lyon_index_mv']:.2f} mV "
            f"(S in V1 {features['sl_s_v1_mv']:.2f} + max R in V5/V6 "
            f"{max(features['sl_r_v5_mv'], features['sl_r_v6_mv']):.2f}); "
            f"the classic LVH threshold is {SOKOLOW_LYON_LVH_THRESHOLD_MV:.1f} mV"
        )

    # Descriptive multi-lead context (no review flag - these did not validate as
    # reliable classifiers, see ecg_multilead.py, but the measurements are real).
    if features.get("cardiac_axis_deg") is not None:
        axis_note = ("left axis deviation" if features.get("left_axis_deviation")
                     else "right axis deviation" if features.get("right_axis_deviation")
                     else "normal axis")
        lines.append(
            f"- Mean QRS frontal axis (multi-lead): "
            f"{features['cardiac_axis_deg']:.0f} deg ({axis_note}; normal -30..+90)"
        )
    if features.get("max_st_elevation_mv") is not None:
        lines.append(
            f"- Largest ST deviation across leads (multi-lead): elevation "
            f"{features['max_st_elevation_mv']:.2f} mV, depression "
            f"{features['max_st_depression_mv']:.2f} mV"
        )
    if features.get("qcount_inferior") is not None:
        lines.append(
            f"- Territorial pathological Q waves (multi-lead, for old myocardial "
            f"infarction): inferior leads {features['qcount_inferior']}/3, anterior "
            f"{features.get('qcount_anterior')}/4, lateral {features.get('qcount_lateral')}/4 "
            f"(>=2 contiguous leads suggests an old infarct in that wall)"
        )

    embedding_keys = [f"embedding_{cls}_probability" for cls in DIAGNOSTIC_SUPERCLASSES]
    if any(key in features for key in embedding_keys):
        lines.append("- Learned embedding model diagnostic-superclass probabilities:")
        for cls, key in zip(DIAGNOSTIC_SUPERCLASSES, embedding_keys):
            if key in features:
                cls_label = f"{cls} ({_DIAGNOSTIC_SUPERCLASS_NAMES[cls]})"
                lines.append(f"  - {cls_label}: {features[key]:.0%}")

    if patient_info:
        lines.append(f"- Patient context: {patient_info}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 2. Deterministic decision logic (rule-based, independent of the LLM)
# ----------------------------------------------------------------------
@dataclass
class DecisionPlan:
    """
    Deterministic, threshold-based decision computed directly from the
    numeric feature dict, independent of any LLM judgment. This exists so
    clinically notable conditions are flagged reliably even if the LLM
    misses or misjudges them, and so records with too little signal don't
    waste an LLM call.

    Inputs (fields):
        deterministic_flags: short machine-readable tags raised by fixed
            thresholds (e.g. "tachycardia_suspected"), merged into the final
            AgentContribution.flags_for_review alongside the LLM's own flags.
        skip_llm_call: True if there is not enough signal in the record for
            a meaningful LLM analysis (e.g. fewer than 2 detected QRS
            complexes), in which case no LLM call is made at all.
        reason: short explanation of the skip decision, or None if not skipped.
    """

    deterministic_flags: List[str]
    skip_llm_call: bool
    reason: Optional[str]


def detect_flags(
    feature_dict: Dict[str, Any],
    tachycardia_bpm: float = 120.0,
    bradycardia_bpm: float = 60.0,
    wide_qrs_ms: float = 120.0,
    embedding_probability_threshold: float = 0.5,
    lvh_sokolow_lyon_mv: float = SOKOLOW_LYON_LVH_THRESHOLD_MV,
    inferior_q_min_leads: int = 2,
) -> DecisionPlan:
    """
    Inspect the numeric feature dict and decide, deterministically, which
    review flags to raise and whether calling the LLM is even worthwhile for
    this record. Runs independently of any LLM output.

    Inputs:
        feature_dict: dict produced by compute_feature_dict.
        tachycardia_bpm: heart rate (bpm) above which "tachycardia_suspected"
            is raised. Default raised from the original 100.0 to 120.0 after
            validating against PTB-XL's scp_codes (src/flag_validation.py):
            at 100.0, precision was 0.015 with recall 0.893 (1663 false
            positives for 25 true ones, full dataset) - 120.0 cuts false
            positives by ~72% (to 463) while keeping recall at 0.786, a
            deliberately recall-favoring choice over the F1-maximizing 165.0
            (which drops recall to 0.286) since this flag feeds a
            human-review alert, not a final classification - a missed true
            case (false negative) is a worse failure mode here than an
            extra reviewed-and-dismissed false positive.
        bradycardia_bpm: heart rate (bpm) below which "bradycardia_suspected"
            is raised. NOT validated against scp_codes: SBRAD never occurs
            with nonzero likelihood anywhere in PTB-XL (confirmed directly),
            so precision/recall are undefined at any threshold - kept at its
            original value.
        wide_qrs_ms: QRS duration (ms) above which "wide_qrs_suspected" is
            raised.
        embedding_probability_threshold: probability above which
            "embedding_<CLASS>_pattern_suspected" is raised for a given
            diagnostic superclass. Only checked for classes whose
            "embedding_<CLASS>_probability" key is actually present in
            feature_dict (i.e. only when compute_feature_dict was given a
            multilead_signal and embedding_model) - a no-op otherwise.

    Output:
        DecisionPlan instance.
    """
    if feature_dict["qrs_count"] < 2:
        return DecisionPlan(
            deterministic_flags=["insufficient_data_for_analysis"],
            skip_llm_call=True,
            reason=(
                f"Only {feature_dict['qrs_count']} QRS complex(es) detected; "
                "not enough signal for a meaningful analysis."
            ),
        )

    flags = []

    heart_rate = feature_dict["heart_rate_mean_bpm"]
    if heart_rate is not None:
        if heart_rate > tachycardia_bpm:
            flags.append("tachycardia_suspected")
        elif heart_rate < bradycardia_bpm:
            flags.append("bradycardia_suspected")

    qrs_duration = feature_dict["qrs_duration_mean_ms"]
    if qrs_duration is not None and qrs_duration > wide_qrs_ms:
        flags.append("wide_qrs_suspected")

    if feature_dict["rhythm_irregularity_flag"]:
        flags.append("irregular_rhythm_suspected")

    for cls in DIAGNOSTIC_SUPERCLASSES:
        probability = feature_dict.get(f"embedding_{cls}_probability")
        if probability is not None and probability > embedding_probability_threshold:
            flags.append(f"embedding_{cls}_pattern_suspected")

    # Multi-lead Sokolow-Lyon LVH voltage criterion (see ecg_multilead.py). Only
    # present when compute_feature_dict was given a multilead_signal; a no-op
    # otherwise. Validated against PTB-XL's LVH scp_code (n=9000): at the classic
    # 3.5 mV threshold precision 0.676 / recall 0.465 / F1 0.551 - the best-
    # performing deterministic ECG flag in the project (next is wide_qrs at 0.440),
    # and the only one that reaches the HYP superclass. The textbook 3.5 mV is
    # near the F1-optimal plateau (3.25-3.5 mV) on this dataset, so unlike
    # tachycardia_bpm it needed no recalibration.
    sokolow_lyon = feature_dict.get("sokolow_lyon_index_mv")
    if sokolow_lyon is not None and sokolow_lyon >= lvh_sokolow_lyon_mv:
        flags.append("lvh_sokolow_lyon_suspected")

    # Territorial pathological Q waves (old inferior MI): pathological Q in >= 2
    # contiguous inferior leads (II/III/aVF) - the clinical consensus rule (see
    # ecg_territorial.py). Validated against PTB-XL inferior-MI codes (n=4000):
    # F1 0.328, recall 0.915, precision 0.200 = 2.2x the 9.2% base rate - a real,
    # recall-favoring MI alert, and the ONLY deterministic MI signal in the agent
    # (all other MI coverage is the embedding's probability). Only the inferior
    # wall is flagged: anterior Q barely enriched over base rate and lateral MI is
    # rare, so those territorial features stay descriptive-only (no flag). Acute ST
    # elevation was NOT validatable in PTB-XL (a resting-ECG dataset, almost no
    # acute STEMI) - see the validation report.
    q_inferior = feature_dict.get("qcount_inferior")
    if q_inferior is not None and q_inferior >= inferior_q_min_leads:
        flags.append("pathological_q_inferior_suspected")

    return DecisionPlan(deterministic_flags=flags, skip_llm_call=False, reason=None)


# ----------------------------------------------------------------------
# 3. ECG-specific structured-findings prompt (via the shared scaffold)
# ----------------------------------------------------------------------
def build_findings_prompt(clinical_summary, feature_dict, patient_info=None):
    """
    Build the ECG structured-findings prompt via the shared prompt scaffold
    (agent_common.build_generic_findings_prompt), specialized with the ECG
    agent's role, modality label, and evidence keys. The model is explicitly
    told it is one of several parallel agents and must not draw a final or
    cross-modality conclusion (enforced by the shared scaffold).

    Inputs:
        clinical_summary: text summary produced by build_clinical_summary.
        feature_dict: numeric features dict produced by compute_feature_dict;
            its keys are the only valid values for "evidence_ref" in the
            model's output.
        patient_info: optional free-text patient context string.

    Output:
        Prompt string ready to send to an LLM backend.
    """
    return build_generic_findings_prompt(
        agent_role="an ECG analysis agent",
        modality_label="ECG",
        summary=clinical_summary,
        evidence_keys=feature_dict.keys(),
        patient_info=patient_info,
    )


# ----------------------------------------------------------------------
# 4. Orchestration - the ECG agent's end-to-end contribution pipeline
# ----------------------------------------------------------------------
def generate_ecg_contribution(
    results,
    fs,
    llm_call: Optional[Callable[[str], str]] = None,
    record_id: Optional[str] = None,
    patient_info: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1,
    multilead_signal=None,
    embedding_model=None,
) -> AgentContribution:
    """
    Run the ECG agent's full contribution pipeline: compute features, run the
    deterministic decision logic (detect_flags), then - unless it
    decides to skip - build the structured-findings prompt, call the given
    LLM backend, and parse/validate the response into an AgentContribution. On
    a parse failure, retries by appending the parse error to the prompt and
    asking the model to correct it. Deterministic flags from detect_flags
    are always merged into the final flags_for_review, whether or not the
    LLM is called.

    Inputs:
        results: dict returned by ecg_preprocessing_pipeline.
        fs: sampling frequency in Hz.
        llm_call: callable taking a single prompt string and returning the
            raw text response, e.g. `lambda p: generate_report_medgemma(p)` or
            `lambda p: generate_report_anthropic(p, api_key=...)`. If not provided,
            defaults to the free local Ollama backend (generate_report_medgemma)
            so this works out of the box with no paid API key - requires
            Ollama running locally with a model pulled (see module docstring).
        record_id: optional identifier of the ECG record, stored in the
            output for traceability.
        patient_info: optional free-text patient context.
        model_name: optional string identifying the model/backend used,
            stored in the trace for auditability. Defaults to "ollama:medgemma:4b"
            when llm_call is left at its default.
        max_retries: number of extra attempts if the LLM response fails to
            parse.
        multilead_signal: optional full multi-lead signal (see
            compute_embedding_features); if given together with
            embedding_model, the ECGEmbeddingNet's diagnostic-superclass
            probabilities are added as ordinary features/findings and can
            raise "embedding_<CLASS>_pattern_suspected" flags (see
            detect_flags). Omitted by default - no behavior change for
            existing callers.
        embedding_model: optional trained ECGEmbeddingNet instance (see
            ecg_embedding.load_model_checkpoint).

    Output:
        AgentContribution instance (agent_id="ecg"). If detect_flags
        decides to skip the LLM call (not enough signal in the record),
        findings is an empty list and trace["model"] is None.

    Raises:
        FindingsParseError: if no valid structured response is obtained
        within max_retries + 1 attempts.
    """
    if llm_call is None:
        llm_call = generate_report_medgemma
        if model_name is None:
            model_name = "ollama:medgemma:4b"

    feature_dict = compute_feature_dict(results, fs, multilead_signal, embedding_model)
    decision = detect_flags(feature_dict)

    if decision.skip_llm_call:
        return AgentContribution(
            agent_id="ecg",
            record_id=record_id,
            findings=[],
            flags_for_review=decision.deterministic_flags,
            trace={
                "features_used": feature_dict,
                "model": None,
                "attempts": 0,
                "skipped_reason": decision.reason,
            },
        )

    clinical_summary = build_clinical_summary(results, fs, patient_info, multilead_signal, embedding_model)
    prompt = build_findings_prompt(clinical_summary, feature_dict, patient_info)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw_response = llm_call(prompt)
        try:
            findings, flags = parse_findings_response(
                raw_response, set(feature_dict.keys())
            )
            merged_flags = sorted(set(flags) | set(decision.deterministic_flags))
            return AgentContribution(
                agent_id="ecg",
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


