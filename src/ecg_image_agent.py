"""
ECG image agent - structured findings generation from an ECG IMAGE
====================================================================
Counterpart to ecg_agent.py for the case where the user uploads an ECG
IMAGE (photo/scan/PNG) instead of a WFDB signal. A user provides EITHER
an image OR a signal, never both (see ecg_router.py for the dispatch
logic) - this module handles the image branch only, but returns the
SAME AgentContribution shape as the signal agent, so downstream
consumers (Sep Agent, UI) don't need to know which branch produced it.

Because there is no numeric signal here, this agent cannot compute the
deterministic clinical features ecg_agent.py does (heart rate, QRS
duration, Sokolow-Lyon, territorial Q, embedding probabilities). What it
CAN compute deterministically is the image's own QUALITY diagnostics
(blur, contrast, brightness, corrections applied - see
adaptive_ecg_pipeline.py), which become this agent's numeric evidence
keys, exactly as compute_feature_dict's keys are ecg_agent's. The ECG
image itself is additionally attached to the LLM call as visual
evidence (evidence_type="image_region"), so most clinical findings here
are expected to be image-grounded rather than numeric-feature-grounded
- unlike ecg_agent, where every finding maps to a computed number.

XAI / traceability note: image-grounded evidence_ref values are now
PER-LEAD (ecg_image_I, ecg_image_II, ..., ecg_image_V6 - see LEAD_NAMES/
LEAD_EVIDENCE_KEYS), plus a single ecg_image_overall key for genuinely
whole-tracing observations, instead of one generic key for the entire
image. This lets a human reviewer at least check the cited strip
independently, which a single catch-all reference did not allow -
though it still stops short of true spatial grounding (no bounding
boxes, no saliency map/Grad-CAM); a finding citing "ecg_image_V2" is
still a claim about that lead, not a verified pixel region. If nearly
every image-grounded finding falls back to ecg_image_overall despite
lead-specific keys being available (see the
"image_findings_not_lead_localized" flag in generate_image_ecg_
contribution), that's a signal the model isn't actually localizing its
reading and should be treated with extra scrutiny.

Pipeline: raw image file -> adaptive_ecg_pipeline preprocessing ->
quality feature dict -> text summary -> structured-findings prompt
(image attached) -> LLM -> parsed/validated AgentContribution.

Shared building blocks (schema, prompt scaffold, JSON parsing/
validation, LLM backends, contribution saving) come from
agent_common.py, same as ecg_agent.py - nothing is duplicated here.

LLM backend: free and local via Ollama, but running the ECG-specialized
fine-tune (DEFAULT_IMAGE_MODEL_TAG = "hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M"), NOT base
medgemma:4b (unlike ecg_agent.py's text branch, which stays on the base
model). This checkpoint is not in Ollama's library and must be imported
once - see setup_medgemma_ecginstruct_ollama.sh:
    bash setup_medgemma_ecginstruct_ollama.sh
    pip install requests
To use the base model instead, pass llm_call explicitly, e.g.
`lambda p: generate_report_medgemma(p, model="medgemma:4b", images=[b64_image])`
(you'd need b64_image in scope - simplest is to just change
DEFAULT_IMAGE_MODEL_TAG below).

IMPORTANT - "always respond" contract: this module never lets an
unreadable/corrupted image crash the pipeline silently. If the image
cannot even be loaded, generate_image_ecg_contribution still returns an
AgentContribution (empty findings, an explanatory flag, skip_llm_call
semantics) rather than raising. It DOES still raise FindingsParseError
if the LLM itself never returns valid JSON after all retries - by
design, symmetric with ecg_agent.py's contract. ecg_router.py is the
layer responsible for guaranteeing a response even in that last case
(see its module docstring) - keep that error handling there, not
duplicated here, so this module's contract stays identical to
ecg_agent.py's.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent_common import (
    AgentContribution,
    FindingsParseError,
    build_generic_findings_prompt,
    generate_report_anthropic,
    generate_report_medgemma,
    parse_findings_response,
    save_contribution,
)
from adaptive_ecg_pipeline import (
    BLUR_MILD_THRESHOLD,
    BLUR_SEVERE_THRESHOLD,
    CONTRAST_LOW_THRESHOLD,
    process_ecg_image,
)

__all__ = [
    "IMAGE_EVIDENCE_KEY",
    "LEAD_NAMES",
    "LEAD_EVIDENCE_KEYS",
    "ALL_IMAGE_EVIDENCE_KEYS",
    "compute_image_quality_features",
    "compute_pathology_features",
    "detect_pathology_flags",
    "compute_digitization_features",
    "detect_digitization_flags",
    "build_image_clinical_summary",
    "ImageDecisionPlan",
    "detect_image_flags",
    "build_image_findings_prompt",
    "generate_image_ecg_contribution",
    "save_contribution",
    "generate_report_medgemma",
    "generate_report_anthropic",
]

# Standard 12-lead ordering, used to build one evidence_ref PER LEAD rather
# than a single generic key for the whole image. This is a low-cost XAI
# improvement: it forces the model to say WHICH lead a visual finding comes
# from (e.g. "ST elevation" grounded in "ecg_image_V2", not just "ecg_image"
# in general), which is at least independently checkable by a human looking
# at that specific strip - unlike a single catch-all reference that could
# point to anything in a 12-panel image. It does NOT provide pixel-level
# localization (no bounding boxes/heatmap) - see module docstring's XAI
# limitations note - but it's a meaningful step up for near-zero extra cost
# (no new model, no training, just a richer set of prompt labels).
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# One evidence_ref per lead (e.g. "ecg_image_I", "ecg_image_V1", ...), used
# for findings the model attributes to a specific strip.
LEAD_EVIDENCE_KEYS = [f"ecg_image_{lead}" for lead in LEAD_NAMES]

# Evidence_ref for observations that are inherently global / not tied to a
# single strip (e.g. overall rhythm regularity assessed across the whole
# tracing, general print quality remarks). Kept separate from the per-lead
# keys so "I looked at the whole image" and "I looked at lead V2" remain
# distinguishable in the trace.
IMAGE_EVIDENCE_KEY = "ecg_image_overall"

# All valid image-grounded evidence keys (per-lead + overall), as opposed to
# the numeric quality_features keys from compute_image_quality_features.
ALL_IMAGE_EVIDENCE_KEYS = set(LEAD_EVIDENCE_KEYS) | {IMAGE_EVIDENCE_KEY}

# Ollama tag for the ECG-specialized MedGemma fine-tune (community
# checkpoint convaiinnovations/medgemma-4b-ecginstruct, fine-tuned on 1.15M
# ECG image-text pairs from PULSE-ECG/ECGInstruct - trained specifically to
# read 12-lead ECG images, unlike base medgemma:4b). Must be imported into
# Ollama first (not in Ollama's library like the base model) - see
# setup_medgemma_ecginstruct_ollama.sh. Only the IMAGE branch uses this;
# ecg_agent.py (text/numeric-feature branch) intentionally keeps the base
# "medgemma:4b" default, since ecginstruct's fine-tuning is specific to
# reading the image itself, not to reasoning over numeric ECG features.
DEFAULT_IMAGE_MODEL_TAG = "hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M"

# Fallback used automatically if DEFAULT_IMAGE_MODEL_TAG isn't available in
# Ollama (not imported yet, Ollama not running, model tag typo, etc.) - see
# _call_with_image_fallback below. Set to None to disable the fallback
# entirely and fail loudly instead (e.g. once you're confident ecginstruct
# is always available and prefer a real error over a silent downgrade).
FALLBACK_IMAGE_MODEL_TAG = "medgemma:4b"

# Spelled-out names for the GenECG image classifier's 5 PTB-XL diagnostic
# superclasses, spelled out in the clinical summary so the VLM reads the full
# concept, not just the abbreviation (same reasoning as ecg_agent's embedding
# naming, where a bare "HYP" was once glossed as "hypokinesis"). Insertion
# order = DIAGNOSTIC_SUPERCLASSES order.
_DIAGNOSTIC_SUPERCLASS_NAMES = {
    "CD": "conduction disturbance",
    "HYP": "hypertrophy",
    "MI": "myocardial infarction",
    "NORM": "normal ECG",
    "STTC": "ST-T changes / myocardial ischemia",
}


# ----------------------------------------------------------------------
# 0b. Learned image-classifier features (this agent's equivalent of
#     ecg_agent.compute_embedding_features / mri_agent's pathology
#     classifier - TRUE per-class probabilities from the trained GenECG
#     image classifier, unlike the VLM's uncalibrated self-reported
#     confidence)
# ----------------------------------------------------------------------
def compute_pathology_features(image_path, classifier_model, device="cpu"):
    """
    Run the trained GenECG image classifier (src/genecg_classifier_train.py) on
    the ECG image and return per-class probabilities - the image branch's
    equivalent of ecg_agent.compute_embedding_features. Unlike the VLM (whose
    self-reported confidence is uncalibrated), these are TRUE probabilities from
    a discriminative model validated on a held-out split (macro F1 0.64, MI 0.62
    - see results/ecg_image_pipeline/genecg_classifier_validation.md).

    The classifier is lazily imported so this module stays importable without
    torch/torchvision when the classifier isn't used (mirrors mri_agent.py).

    Inputs:
        image_path: path to the ECG image (the RAW upload, matching how the
            classifier was trained - not the adaptive-preprocessed base64).
        classifier_model: a model from
            genecg_classifier_train.load_genecg_classifier.
        device: "cpu" or "cuda".

    Output:
        dict "pathology_<CLASS>_probability" -> float in [0, 1].
    """
    from genecg_classifier_train import predict_pathology_probabilities
    return predict_pathology_probabilities(image_path, classifier_model, device=device)


def detect_pathology_flags(pathology_features: Dict[str, Any], threshold: float = 0.5) -> List[str]:
    """
    Raise "pathology_<CLASS>_pattern_suspected" for each class whose classifier
    probability exceeds `threshold`. Mirrors ecg_agent's embedding flags and
    mri_agent's pathology flags exactly (same threshold semantics).
    """
    flags = []
    for cls in _DIAGNOSTIC_SUPERCLASS_NAMES:
        key = f"pathology_{cls}_probability"
        prob = pathology_features.get(key)
        if prob is not None and prob > threshold:
            flags.append(f"pathology_{cls}_pattern_suspected")
    return flags


# ----------------------------------------------------------------------
# 0c. Digitization features (opt-in): recover a signal from the rhythm strip
#     and expose the VALIDATED numeric measurements (heart rate + rhythm).
#     This is the ONLY source of real measured NUMBERS in the image branch -
#     the VLM is qualitative, the classifier gives probabilities. See
#     ecg_digitization.py (validated on GenECG, HR MAE ~1 bpm).
# ----------------------------------------------------------------------
def compute_digitization_features(image_path, fs=100.0):
    """
    Digitize the image's rhythm strip and return validated numeric rate/rhythm
    measurements (`digitized_heart_rate_bpm`, `digitized_rhythm_cv`, ...).
    Lazily imports ecg_digitization so this module stays light when unused.
    Returns {} if digitization fails (never raises).
    """
    try:
        from ecg_digitization import measure_rhythm_from_image
        feats = measure_rhythm_from_image(image_path, fs_out=fs)
        return feats or {}
    except Exception:  # noqa: BLE001 - opt-in, never fatal
        return {}


def detect_digitization_flags(features: Dict[str, Any],
                              tachycardia_bpm: float = 120.0,
                              bradycardia_bpm: float = 60.0) -> List[str]:
    """
    Deterministic rate/rhythm flags from the digitized heart rate - the SAME flag
    names and thresholds as ecg_agent.detect_flags (a case is either signal or
    image, so no clash), so they integrate into the Sep Agent / KG identically.
    """
    flags = []
    hr = features.get("digitized_heart_rate_bpm")
    if hr is not None:
        if hr > tachycardia_bpm:
            flags.append("tachycardia_suspected")
        elif hr < bradycardia_bpm:
            flags.append("bradycardia_suspected")
    if features.get("digitized_rhythm_irregular"):
        flags.append("irregular_rhythm_suspected")
    return flags


# ----------------------------------------------------------------------
# 1. Numeric feature extraction from the quality report (this agent's
#    equivalent of ecg_agent.compute_feature_dict - single source of
#    truth for the NUMERIC "evidence_ref" values)
# ----------------------------------------------------------------------
def compute_image_quality_features(quality_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten adaptive_ecg_pipeline.process_ecg_image's quality report into
    the flat numeric/categorical feature dict this agent exposes as valid
    "evidence_ref" values (alongside IMAGE_EVIDENCE_KEY itself).

    Inputs:
        quality_report: the `report` dict returned by
            adaptive_ecg_pipeline.process_ecg_image (has "quality_before",
            "quality_after", "corrections_applied", "residual_issues",
            "source_quality_level").

    Output:
        Flat dict {feature_name: value}. Numeric values are floats/ints;
        "source_quality_level" is a string ("clean"/"corrected"/
        "degraded"); "residual_issue_count" is an int. None values are
        never emitted (mirrors ecg_agent's None-for-uncomputable pattern
        only where genuinely absent - here everything in the report is
        always present).
    """
    before = quality_report["quality_before"]
    after = quality_report["quality_after"]
    corrections = quality_report["corrections_applied"]

    return {
        "image_width_px": after["width"],
        "image_height_px": after["height"],
        "blur_score_before": before["blur_score"],
        "blur_score_after": after["blur_score"],
        "contrast_before": before["contrast"],
        "contrast_after": after["contrast"],
        "brightness_before": before["brightness"],
        "brightness_after": after["brightness"],
        "sharpen_amount_applied": corrections.get("sharpen_amount", 0.0),
        "clahe_clip_limit_applied": corrections.get("clahe_clip_limit", 0.0),
        "gamma_applied": corrections.get("gamma_applied", 1.0),
        "deskew_angle_deg": corrections.get("deskew_angle_deg", 0.0),
        "resize_action": corrections.get("resize_action", "unchanged"),
        "residual_issue_count": len(quality_report.get("residual_issues", [])),
        "source_quality_level": quality_report.get("source_quality_level", "unknown"),
    }


def build_image_clinical_summary(quality_report: Dict[str, Any], patient_info: Optional[str] = None,
                                 pathology_features: Optional[Dict[str, Any]] = None,
                                 digitization_features: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert the quality feature dict into a readable plain-text summary
    for the LLM prompt - this agent's equivalent of
    ecg_agent.build_clinical_summary. Unlike the signal agent, this
    summary describes IMAGE RELIABILITY, not clinical measurements (the
    clinical content is expected to come from the model actually reading
    the attached image).

    Inputs:
        quality_report: dict returned by process_ecg_image.
        patient_info: optional free-text patient context string.

    Output:
        Multi-line text summary (str).
    """
    features = compute_image_quality_features(quality_report)

    lines = [
        f"- Image size after preprocessing: {features['image_width_px']}x{features['image_height_px']} px "
        f"({features['resize_action']})",
        f"- Overall source quality assessment: {features['source_quality_level']}",
    ]

    if features["blur_score_before"] < BLUR_MILD_THRESHOLD:
        severity = "severe" if features["blur_score_before"] < BLUR_SEVERE_THRESHOLD else "mild"
        lines.append(
            f"- Blur detected before correction ({severity}, Laplacian variance "
            f"{features['blur_score_before']:.1f}); sharpening amount applied: "
            f"{features['sharpen_amount_applied']:.2f}"
        )

    if features["contrast_before"] < CONTRAST_LOW_THRESHOLD:
        lines.append(
            f"- Low contrast detected before correction ({features['contrast_before']:.1f}); "
            f"CLAHE clip limit applied: {features['clahe_clip_limit_applied']:.2f}"
        )

    if features["gamma_applied"] != 1.0:
        lines.append(f"- Brightness correction applied (gamma={features['gamma_applied']:.2f})")

    if features["deskew_angle_deg"]:
        lines.append(f"- Image was deskewed by {features['deskew_angle_deg']:.2f} deg")

    if features["residual_issue_count"] > 0:
        issues = ", ".join(quality_report.get("residual_issues", []))
        lines.append(
            f"- WARNING - {features['residual_issue_count']} residual issue(s) remain after "
            f"correction: {issues}. Weight your confidence accordingly and say so explicitly "
            f"if this limits what you can read from the image."
        )

    if pathology_features:
        lines.append(
            "- Learned image-classifier probabilities (trained GenECG ECG-image classifier, "
            "validated on held-out data - these are calibrated per-class probabilities, NOT your "
            "own visual impression):"
        )
        for cls, full_name in _DIAGNOSTIC_SUPERCLASS_NAMES.items():
            key = f"pathology_{cls}_probability"
            if pathology_features.get(key) is not None:
                lines.append(f"  - {cls} ({full_name}): {pathology_features[key] * 100:.1f}%")

    if digitization_features:
        d = digitization_features
        lines.append(
            "- Digitized rhythm-strip measurements (recovered from the image trace by digitization, "
            "so these ARE measured numbers, not a visual estimate):"
        )
        if d.get("digitized_heart_rate_bpm") is not None:
            lines.append(f"  - Heart rate: {d['digitized_heart_rate_bpm']} bpm "
                         f"(from {d.get('digitized_n_beats')} beats over {d.get('digitized_seconds')} s)")
        if d.get("digitized_rhythm_cv") is not None:
            lines.append(f"  - Rhythm variability (RR CV): {d['digitized_rhythm_cv']} "
                         f"({'irregular' if d.get('digitized_rhythm_irregular') else 'regular'})")

    if patient_info:
        lines.append(f"- Patient context: {patient_info}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 2. Deterministic decision logic (independent of the LLM, mirrors
#    ecg_agent.DecisionPlan / detect_flags)
# ----------------------------------------------------------------------
@dataclass
class ImageDecisionPlan:
    """
    Deterministic, threshold-based decision computed directly from the
    image quality feature dict, independent of any LLM judgment. Mirrors
    ecg_agent.DecisionPlan exactly (same field meanings) so callers that
    already handle one can handle both.

    Inputs (fields):
        deterministic_flags: short machine-readable tags raised by fixed
            thresholds (e.g. "image_quality_severely_degraded"), merged
            into the final AgentContribution.flags_for_review alongside
            the LLM's own flags.
        skip_llm_call: True if the image is not usable enough for a
            meaningful LLM analysis (e.g. still severely blurred after
            correction), in which case no LLM call is made at all.
        reason: short explanation of the skip decision, or None if not
            skipped.
    """

    deterministic_flags: List[str]
    skip_llm_call: bool
    reason: Optional[str]


def detect_image_flags(features: Dict[str, Any]) -> ImageDecisionPlan:
    """
    Inspect the image quality feature dict and decide, deterministically,
    which review flags to raise and whether calling the LLM is worthwhile
    for this image. Runs independently of any LLM output - mirrors
    ecg_agent.detect_flags's role for the image branch.

    Inputs:
        features: dict produced by compute_image_quality_features.

    Output:
        ImageDecisionPlan instance.
    """
    flags = []

    if features["source_quality_level"] == "degraded":
        flags.append("image_quality_severely_degraded")

    if features["residual_issue_count"] > 0:
        flags.append("image_quality_residual_issues")

    if features["image_width_px"] < 300 or features["image_height_px"] < 300:
        flags.append("image_resolution_low")

    # Still readable, even if degraded: unlike ecg_agent's "insufficient
    # data" skip (fewer than 2 QRS complexes -> truly nothing to say), a
    # blurry/low-contrast image is still worth an LLM read, with an
    # explicit low-confidence caveat baked into the prompt (see
    # build_image_clinical_summary's WARNING line) - so skip_llm_call
    # stays False here in every case reachable from a successfully
    # preprocessed image. The only skip case for this agent is the image
    # being unreadable in the first place, handled one level up in
    # generate_image_ecg_contribution (before a quality_report even
    # exists to build an ImageDecisionPlan from).
    return ImageDecisionPlan(deterministic_flags=flags, skip_llm_call=False, reason=None)


# ----------------------------------------------------------------------
# 3. Image-specific structured-findings prompt (via the shared scaffold)
# ----------------------------------------------------------------------
def build_image_findings_prompt(clinical_summary: str, quality_features: Dict[str, Any],
                                 patient_info: Optional[str] = None,
                                 pathology_features: Optional[Dict[str, Any]] = None,
                                 digitization_features: Optional[Dict[str, Any]] = None) -> str:
    """
    Build the ECG-image structured-findings prompt via the shared prompt
    scaffold (agent_common.build_generic_findings_prompt), specialized
    with this agent's role, modality label, and evidence keys (the
    quality features PLUS the attached image itself). Mirrors
    ecg_agent.build_findings_prompt.

    Inputs:
        clinical_summary: text summary produced by
            build_image_clinical_summary.
        quality_features: dict produced by compute_image_quality_features;
            its keys, plus ALL_IMAGE_EVIDENCE_KEYS (one per lead, plus the
            "overall" key), are the only valid values for "evidence_ref"
            in the model's output.
        patient_info: optional free-text patient context string.

    Output:
        Prompt string ready to send to an LLM backend.

    XAI note: evidence_keys now includes one entry PER LEAD
    (ecg_image_I, ecg_image_II, ..., ecg_image_V6) instead of a single
    generic "ecg_image" key, and image_labels spells out each lead's key
    explicitly so the model is instructed - not just permitted - to
    localize its visual findings to a specific strip whenever possible,
    falling back to ecg_image_overall only for genuinely global
    observations (e.g. overall rhythm regularity). This does not add any
    model calls or preprocessing cost; it only changes what the prompt
    asks for and what parse_findings_response accepts.
    """
    evidence_keys = list(quality_features.keys())
    if pathology_features:
        evidence_keys += list(pathology_features.keys())
    if digitization_features:
        evidence_keys += list(digitization_features.keys())
    evidence_keys += LEAD_EVIDENCE_KEYS + [IMAGE_EVIDENCE_KEY]

    lead_key_list = ", ".join(LEAD_EVIDENCE_KEYS)
    image_labels = [
        f"The uploaded 12-lead ECG image (preprocessed for readability; "
        f"see quality notes above). It shows the standard 12 leads: "
        f"{', '.join(LEAD_NAMES)}. "
        f"IMPORTANT: whenever a finding is based on something you see in a "
        f"specific lead's strip (e.g. an ST elevation, a wide QRS, a Q wave), "
        f"cite the evidence_ref for THAT lead specifically - one of: "
        f"{lead_key_list}. "
        f"CRITICAL FORMAT RULE: 'evidence_ref' must be EXACTLY ONE of these "
        f"keys - never a list, never multiple keys joined by commas or 'and'. "
        f"If an observation is genuinely supported by MULTIPLE leads (e.g. "
        f"sinus rhythm, visible across most of the tracing), do NOT combine "
        f"them into one evidence_ref string. Instead, repeat the SAME "
        f"'feature' and 'value' in SEPARATE finding objects, one per "
        f"supporting lead - e.g. two finding objects both with "
        f"\"feature\": \"sinus_rhythm\", one with "
        f"\"evidence_ref\": \"ecg_image_II\" and another with "
        f"\"evidence_ref\": \"ecg_image_V5\", rather than a single finding "
        f"citing both. "
        f"Only use '{IMAGE_EVIDENCE_KEY}' for a TRULY global observation you "
        f"cannot attribute to specific leads at all (e.g. overall print "
        f"quality). Do not use it as a shortcut to avoid picking leads."
    ]

    return build_generic_findings_prompt(
        agent_role="an ECG image analysis agent",
        modality_label="ECG image",
        summary=clinical_summary,
        evidence_keys=evidence_keys,
        patient_info=patient_info,
        image_labels=image_labels,
    )


# ----------------------------------------------------------------------
# 4. Orchestration - the ECG image agent's end-to-end contribution
#    pipeline (mirrors ecg_agent.generate_ecg_contribution)
# ----------------------------------------------------------------------
def _call_with_image_fallback(prompt: str, b64_image: str, used_model_holder: Dict[str, Optional[str]]) -> str:
    """
    Try DEFAULT_IMAGE_MODEL_TAG first; on any error (model not pulled/
    imported into Ollama, Ollama not reachable with that tag, etc.), fall
    back once to FALLBACK_IMAGE_MODEL_TAG - unless it's None, in which case
    the original error is re-raised instead of downgrading silently.

    used_model_holder is a small mutable dict (e.g. {"tag": None}) that
    this function fills in with whichever tag actually answered
    ("tag": DEFAULT_IMAGE_MODEL_TAG or FALLBACK_IMAGE_MODEL_TAG), so the
    caller can report the REAL model used in AgentContribution.trace
    instead of just assuming the primary one always worked.
    """
    try:
        response = generate_report_medgemma(
            prompt, model=DEFAULT_IMAGE_MODEL_TAG, images=[b64_image],
            num_predict=3072, json_mode=True,
        )
        used_model_holder["tag"] = DEFAULT_IMAGE_MODEL_TAG
        return response
    except Exception as primary_exc:
        if FALLBACK_IMAGE_MODEL_TAG is None:
            raise
        try:
            response = generate_report_medgemma(
                prompt, model=FALLBACK_IMAGE_MODEL_TAG, images=[b64_image],
                num_predict=3072, json_mode=True,
            )
            used_model_holder["tag"] = FALLBACK_IMAGE_MODEL_TAG
            return response
        except Exception as fallback_exc:
            # Both failed - raise the ORIGINAL error, it's usually the more
            # informative one (e.g. "model not found" for the intended model),
            # but mention the fallback attempt too for debugging.
            raise RuntimeError(
                f"Primary model '{DEFAULT_IMAGE_MODEL_TAG}' failed ({primary_exc}); "
                f"fallback '{FALLBACK_IMAGE_MODEL_TAG}' also failed ({fallback_exc})."
            ) from primary_exc


def _repair_raw_response(raw_response: str) -> str:
    """
    Lenient repair pass, run BEFORE the strict parse_findings_response.

    Two independent problems observed in practice with this small
    quantized vision model (medgemma-ecg-gguf, Q4_K_M), across repeated
    real test runs (see test log / conversation history):

    1. Malformed JSON syntax - missing quotes around property names,
       missing ':' delimiters, etc. ("Expecting property name enclosed
       in double quotes", "Expecting ':' delimiter"). Ollama's
       "format": "json" option (grammar-constrained decoding, see
       json_mode on generate_report_medgemma) was tried first and did
       NOT reliably eliminate this for this model/multimodal combo -
       so instead of depending on that, `json_repair` (a library
       specifically built to fix common LLM JSON mistakes) is used
       here as a deterministic, always-available safety net.

    2. Several valid lead keys joined into a single evidence_ref string
       (e.g. "ecg_image_I, ecg_image_II") instead of one key per
       finding, despite the prompt explicitly asking for one finding
       per lead. Prompt tightening alone didn't reliably fix this
       either, so it's repaired here too: the finding is duplicated
       once per cited key rather than dropped, preserving every lead
       the model actually cited.

    Deliberately lives in ecg_image_agent.py, not agent_common.py: it
    only touches this agent's own output conventions, so ecg_agent.py
    (signal) is completely unaffected.

    Falls back to returning the input unchanged if repair itself fails
    (e.g. the text isn't recoverable JSON at all) - in that case
    parse_findings_response's own error path handles it exactly as
    before this function existed, so this can only make things better,
    never worse.
    """
    try:
        from json_repair import repair_json
        fixed_text = repair_json(raw_response)
        payload = json.loads(fixed_text)
    except Exception:
        return raw_response  # let parse_findings_response's own error path handle it

    findings = payload.get("findings")
    if not isinstance(findings, list):
        return json.dumps(payload)  # still return the syntax-repaired version

    repaired = []
    for f in findings:
        ref = f.get("evidence_ref") if isinstance(f, dict) else None
        if isinstance(ref, str) and "," in ref:
            for key in (part.strip() for part in ref.split(",")):
                if key:
                    repaired.append({**f, "evidence_ref": key})
        else:
            repaired.append(f)

    payload["findings"] = repaired
    return json.dumps(payload)


def generate_image_ecg_contribution(
    image_path: str,
    llm_call: Optional[Callable[[str], str]] = None,
    record_id: Optional[str] = None,
    patient_info: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1,
    classifier_model: Any = None,
    classifier_device: str = "cpu",
    pathology_probability_threshold: float = 0.5,
    digitize: bool = False,
    ecg_gate: bool = False,
    ecg_gate_threshold: float = 0.5,
) -> AgentContribution:
    """
    Run the ECG image agent's full contribution pipeline: preprocess the
    image (adaptive_ecg_pipeline), compute quality features, run the
    deterministic decision logic (detect_image_flags), then build the
    structured-findings prompt WITH the image attached, call the given
    LLM backend, and parse/validate the response into an
    AgentContribution. On a parse failure, retries by appending the
    parse error to the prompt, exactly like ecg_agent.
    generate_ecg_contribution.

    Inputs:
        image_path: path to the uploaded ECG image file (PNG/JPEG).
        llm_call: callable taking a single prompt string and returning
            the raw text response. If not provided, defaults to the free
            local Ollama backend running DEFAULT_IMAGE_MODEL_TAG
            ("hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M") WITH the preprocessed image
            attached (generate_report_medgemma(prompt, model=...,
            images=[b64_image])), so this works out of the box with no
            paid API key - requires Ollama running locally with that tag
            imported (see module docstring /
            setup_medgemma_ecginstruct_ollama.sh). Passing a custom
            llm_call means YOU are responsible for attaching the image
            yourself if you want the model to see it (e.g.
            `lambda p: generate_report_medgemma(p, images=[my_b64])`).
        record_id: optional identifier of the ECG record, stored in the
            output for traceability.
        patient_info: optional free-text patient context.
        model_name: optional string forcing what gets recorded in the
            trace. Leave at None (recommended) when llm_call is also left
            at its default: the trace will then report whichever model
            ACTUALLY answered - DEFAULT_IMAGE_MODEL_TAG normally, or
            FALLBACK_IMAGE_MODEL_TAG (plus an "image_model_fallback_used"
            flag) if the primary model wasn't available - rather than
            just assuming the primary one worked.
        max_retries: number of extra attempts if the LLM response fails
            to parse.
        ecg_gate: if True, run the CLIP zero-shot "is this actually an
            ECG?" content check (ecg_image_gate.is_ecg_image) BEFORE any
            preprocessing/VLM/classifier work. If the image is judged NOT
            to be an ECG, this returns an empty-findings contribution with
            a "not_an_ecg_image" flag instead of confidently "reading" a
            non-ECG image (the failure mode documented as limitation #4 in
            adaptive_ecg_pipeline.py). Opt-in (default False) so existing
            callers are byte-identical; degrades gracefully - if the gate
            model can't be loaded (sentence-transformers missing, offline),
            the check is skipped and the pipeline proceeds as before rather
            than blocking a real ECG.
        ecg_gate_threshold: minimum P(ECG) from the gate to accept the
            image (default 0.5; the ECG-vs-not separation is large, so the
            exact value is not sensitive).

    Output:
        AgentContribution instance (agent_id="ecg_image"). If the image
        cannot even be loaded/decoded, findings is an empty list, a
        single "image_unreadable" flag is set, and trace["model"] is
        None - this function does NOT raise for that case (see module
        docstring's "always respond" contract). If ecg_gate=True and the
        image is judged not to be an ECG, findings is empty, the flag is
        "not_an_ecg_image", and trace carries the gate's P(ECG) score.

    Raises:
        FindingsParseError: if no valid structured response is obtained
        within max_retries + 1 attempts (image loaded fine, but the LLM
        never returned parseable JSON) - symmetric with ecg_agent's
        contract; see ecg_router.py for the layer that catches this.
    """
    # Opt-in content gate: reject non-ECG images before spending any real work
    # on them. Runs first, on the RAW upload. Degrades gracefully - if the gate
    # model can't be loaded (offline / dependency missing), we skip the check
    # and proceed rather than blocking a genuine ECG (fail-open by design: the
    # gate is a convenience guard, not a hard security boundary).
    ecg_gate_info: Optional[Dict[str, Any]] = None
    if ecg_gate:
        try:
            from ecg_image_gate import is_ecg_image
            is_ecg, gate_info = is_ecg_image(image_path, threshold=ecg_gate_threshold)
        except Exception as exc:  # noqa: BLE001 - gate is opt-in, never fatal
            is_ecg, gate_info = True, {"gate_error": str(exc), "skipped": True}
        ecg_gate_info = gate_info
        if not is_ecg:
            return AgentContribution(
                agent_id="ecg_image",
                record_id=record_id,
                findings=[],
                flags_for_review=["not_an_ecg_image"],
                trace={
                    "features_used": None,
                    "model": None,
                    "attempts": 0,
                    "ecg_gate": gate_info,
                    "skipped_reason": (
                        f"Image rejected by the ECG content gate "
                        f"(P(ECG)={gate_info.get('ecg_probability')} < "
                        f"threshold {ecg_gate_threshold})."
                    ),
                },
            )

    try:
        b64_image, quality_report = process_ecg_image(image_path)
    except (FileNotFoundError, ValueError) as exc:
        return AgentContribution(
            agent_id="ecg_image",
            record_id=record_id,
            findings=[],
            flags_for_review=["image_unreadable"],
            trace={
                "features_used": None,
                "model": None,
                "attempts": 0,
                "skipped_reason": f"Could not load/decode image: {exc}",
            },
        )

    quality_features = compute_image_quality_features(quality_report)
    decision = detect_image_flags(quality_features)

    # Optional learned-classifier layer: TRUE per-class probabilities (opt-in,
    # mirrors ecg_agent's embedding_model wiring). Degrades gracefully - a
    # classifier failure must not break the "always respond" contract.
    pathology_features: Dict[str, Any] = {}
    pathology_flags: List[str] = []
    if classifier_model is not None:
        try:
            pathology_features = compute_pathology_features(
                image_path, classifier_model, device=classifier_device
            )
            pathology_flags = detect_pathology_flags(
                pathology_features, threshold=pathology_probability_threshold
            )
        except Exception:  # noqa: BLE001 - classifier is opt-in, never fatal
            pathology_features = {}
            pathology_flags = ["pathology_classifier_failed"]

    # Optional digitization layer: recover the rhythm strip -> MEASURED rate/rhythm
    # numbers + standard rate/rhythm flags. Opt-in, degrades gracefully.
    digitization_features: Dict[str, Any] = {}
    digitization_flags: List[str] = []
    if digitize:
        digitization_features = compute_digitization_features(image_path)
        if digitization_features:
            digitization_flags = detect_digitization_flags(digitization_features)
        else:
            digitization_flags = ["digitization_failed"]

    used_model_holder: Dict[str, Optional[str]] = {"tag": None}
    if llm_call is None:
        def llm_call(p: str) -> str:
            return _call_with_image_fallback(p, b64_image, used_model_holder)
        # model_name is finalized below, AFTER the call, from used_model_holder -
        # so the trace reports whichever model actually answered, not just
        # whichever we hoped would.

    clinical_summary = build_image_clinical_summary(
        quality_report, patient_info, pathology_features=pathology_features or None,
        digitization_features=digitization_features or None,
    )
    prompt = build_image_findings_prompt(
        clinical_summary, quality_features, patient_info,
        pathology_features=pathology_features or None,
        digitization_features=digitization_features or None,
    )

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw_response = llm_call(prompt)
        raw_response = _repair_raw_response(raw_response)
        try:
            findings, flags = parse_findings_response(
                raw_response,
                valid_evidence_keys=(set(quality_features.keys())
                                     | set(pathology_features.keys())
                                     | set(digitization_features.keys())
                                     | ALL_IMAGE_EVIDENCE_KEYS),
                image_evidence_keys=ALL_IMAGE_EVIDENCE_KEYS,
            )
            merged_flags = sorted(
                set(flags) | set(decision.deterministic_flags)
                | set(pathology_flags) | set(digitization_flags)
            )

            # XAI check: if every single image-grounded finding used the
            # generic "overall" key despite lead-specific keys being offered,
            # the model likely isn't actually localizing its observations
            # (or isn't reading the image as closely as it claims to be).
            # This doesn't invalidate the findings, but it's a useful signal
            # for a human reviewer - surfaced as a flag rather than silently
            # dropped, consistent with this module's "always respond, never
            # hide a caveat" philosophy.
            image_grounded = [f for f in findings if f.evidence_ref in ALL_IMAGE_EVIDENCE_KEYS]
            lead_grounded = [f for f in image_grounded if f.evidence_ref != IMAGE_EVIDENCE_KEY]
            if image_grounded and not lead_grounded:
                merged_flags = sorted(set(merged_flags) | {"image_findings_not_lead_localized"})
            resolved_model_name = model_name
            if resolved_model_name is None:
                actual_tag = used_model_holder["tag"] or DEFAULT_IMAGE_MODEL_TAG
                resolved_model_name = f"ollama:{actual_tag}"
                if actual_tag != DEFAULT_IMAGE_MODEL_TAG:
                    merged_flags = sorted(set(merged_flags) | {"image_model_fallback_used"})
            return AgentContribution(
                agent_id="ecg_image",
                record_id=record_id,
                findings=findings,
                flags_for_review=merged_flags,
                trace={
                    "features_used": quality_features,
                    "pathology_features": pathology_features,
                    "digitization_features": digitization_features,
                    "quality_report": quality_report,
                    "ecg_gate": ecg_gate_info,
                    "model": resolved_model_name,
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
