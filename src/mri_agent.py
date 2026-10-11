"""
MRI agent - structured findings generation
============================================
This module is the MRI-specific agent in a larger multi-agent clinical
pipeline, mirroring ecg_agent.py's role for the MRI modality. It consumes
the output of mri_preprocessing.preprocess_patient() (numeric mask-derived
features + base64-encoded ED/ES images) and produces a structured
"contribution" object - not a final diagnostic report, same as the ECG
agent - meant to be consumed by a future aggregating "Sep Agent".

Pipeline: preprocess_patient() output -> text summary -> deterministic
decision -> structured-findings prompt (with attached images) -> LLM ->
parsed/validated AgentContribution (from agent_common.py).

Image-grounded evidence: unlike the ECG agent, some findings here are
grounded in an image rather than a numeric feature. The image registry
(preprocessing_result["images"]) supplies the valid image-type evidence_ref
values, exactly as the numeric feature dict supplies numeric evidence_ref
values - both are merged into one evidence_keys set for the shared prompt
scaffold and for parse_findings_response's validation. evidence_type on
each Finding is derived in code (which registry the cited key came from),
not trusted from the LLM's own output.

LLM backend: same shared MedGemma 4B backend as the ECG agent
(agent_common.generate_report_medgemma), called with images=[...] this
time - MedGemma is natively multimodal (SigLIP-based image encoder), so no
second model is needed for vision; only the prompt and the images argument
change relative to the ECG agent. Requires Ollama installed and running
locally with the model pulled beforehand:
    ollama pull medgemma:4b

Image count, tested empirically (see conversation/report - not yet proven
in this module's automated tests): MedGemma's evaluation is primarily
single-image, with multi-image workflows described as experimental. This
module defaults to image_mode="ed_only" (single image) as the safer
baseline; image_mode="ed_es" (both phases in one call) is supported and can
be selected explicitly once/if its reliability is confirmed acceptable for
a given use case.

Learned pathology classifier integration (mirrors ecg_agent.py's Phase E
embedding integration): compute_pathology_classifier_features runs the
trained YOLO -> U-Net -> classifier chain (see
results/report_package/05_future_work_raw_mri_pipeline/raw_mri_pipeline_proposal.md,
section 9) on a raw, uncropped ED slice to get pathology_<CLASS>_probability
features - a learned complement to the deterministic EF-threshold flags
below, added because end-to-end validation showed low_rv_ejection_fraction_
suspected is comparatively weak and the classifier picks up RV patterns
that flag misses (test-set F1 0.80 vs 0.27-0.56 for the deterministic RV
flag). Only computed if raw_ed_slice_uint8, yolo_model, unet_model, and
classifier_model are all explicitly passed - omitted, this changes nothing
for existing callers, same opt-in convention as the ECG embedding.
Deliberately a LAZY import (inside compute_pathology_classifier_features,
not at module level): unlike ecg_agent.py's embedding integration - which
added torch as a module-level hard dependency, already a project-wide
expectation - wiring in ultralytics/segmentation_models_pytorch/torchvision
at module level here would force every caller of this module (including
mri_flag_validation.py, which only needs the pure deterministic logic) to
have the full ML pipeline stack installed just to import mri_agent at all.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent_common import (
    AgentContribution,
    FindingsParseError,
    build_generic_findings_prompt,
    generate_report_medgemma,
    parse_findings_response,
)

__all__ = [
    "build_mri_clinical_summary",
    "MRIDecisionPlan",
    "detect_flags_mri",
    "build_mri_findings_prompt",
    "compute_pathology_classifier_features",
    "generate_mri_contribution",
]

# Full names for the learned classifier's pathology classes, spelled out in
# the clinical summary - proactively applying the same fix ecg_agent.py
# needed empirically for its own class abbreviations (bare "HYP" was
# misread by MedGemma as "hypokinesis" instead of hypertrophy). Not yet
# independently re-tested for these MRI abbreviations specifically, but the
# failure mode (an LLM guessing at an ambiguous abbreviation) is the same.
_PATHOLOGY_CLASS_NAMES = {
    "DCM": "DCM (dilated cardiomyopathy)",
    "HCM": "HCM (hypertrophic cardiomyopathy)",
    "MINF": "MINF (prior myocardial infarction with reduced function)",
    "NOR": "NOR (normal)",
    "RV": "RV (abnormal right ventricle)",
}


# ----------------------------------------------------------------------
# 1. Human-readable summary of the numeric mask-derived features
# ----------------------------------------------------------------------
def build_mri_clinical_summary(preprocessing_result, patient_info=None, pathology_probabilities=None):
    """
    Convert the numeric feature dict from preprocess_patient() into a
    readable plain-text summary, used as context inside the LLM prompt.

    Inputs:
        preprocessing_result: dict returned by
            mri_preprocessing.preprocess_patient (uses its "features" and
            "info" entries).
        patient_info: optional free-text patient context string.
        pathology_probabilities: optional dict of
            "pathology_<CLASS>_probability" -> float, as returned by
            compute_pathology_classifier_features. When given, each class's
            full name is spelled out (see _PATHOLOGY_CLASS_NAMES) rather
            than the bare abbreviation.

    Output:
        Multi-line text summary (str).
    """
    features = preprocessing_result["features"]
    info = preprocessing_result["info"]

    lines = [
        f"- Cardiac phases analyzed: end-diastole (frame {info['ED']}), "
        f"end-systole (frame {info['ES']})",
        f"- LV volume: {features['lv_volume_ed_ml']:.1f} mL (ED), "
        f"{features['lv_volume_es_ml']:.1f} mL (ES)",
        f"- RV volume: {features['rv_volume_ed_ml']:.1f} mL (ED), "
        f"{features['rv_volume_es_ml']:.1f} mL (ES)",
        f"- Myocardium volume: {features['myocardium_volume_ed_ml']:.1f} mL (ED), "
        f"{features['myocardium_volume_es_ml']:.1f} mL (ES)",
    ]

    if features["lv_ejection_fraction_pct"] is not None:
        lines.append(f"- LV ejection fraction: {features['lv_ejection_fraction_pct']:.1f}%")
    else:
        lines.append("- LV ejection fraction: not computable (zero ED volume)")

    if features["rv_ejection_fraction_pct"] is not None:
        lines.append(f"- RV ejection fraction: {features['rv_ejection_fraction_pct']:.1f}%")
    else:
        lines.append("- RV ejection fraction: not computable (zero ED volume)")

    if pathology_probabilities:
        lines.append("- Learned pathology classifier probabilities (complementary to the ejection-fraction "
                      "flags above, not a diagnosis):")
        for class_abbrev, full_name in _PATHOLOGY_CLASS_NAMES.items():
            key = f"pathology_{class_abbrev}_probability"
            if key in pathology_probabilities:
                lines.append(f"  - {full_name}: {pathology_probabilities[key] * 100:.1f}%")

    if patient_info:
        lines.append(f"- Patient context: {patient_info}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# 2. Deterministic decision logic (rule-based, independent of the LLM)
# ----------------------------------------------------------------------
@dataclass
class MRIDecisionPlan:
    """
    Deterministic, threshold-based decision computed directly from the
    numeric feature dict, independent of any LLM judgment - mirrors ECG's
    DecisionPlan. Ensures clinically notable ejection-fraction impairment
    is flagged reliably even if the LLM misses it, and that a record with
    no computable ejection fraction doesn't waste an LLM call.

    Inputs (fields):
        deterministic_flags: short machine-readable tags raised by fixed
            thresholds (e.g. "low_lv_ejection_fraction_suspected"), merged
            into the final AgentContribution.flags_for_review alongside the
            LLM's own flags.
        skip_llm_call: True if ejection fraction isn't computable at all
            (zero ED volume), in which case no LLM call is made.
        reason: short explanation of the skip decision, or None if not
            skipped.
    """

    deterministic_flags: List[str]
    skip_llm_call: bool
    reason: Optional[str]


def detect_flags_mri(
    feature_dict: Dict[str, Any],
    low_lv_ef_pct: float = 45.0,
    low_rv_ef_pct: float = 45.0,
    high_lv_ef_pct: float = 65.0,
    pathology_classifier_probability_threshold: float = 0.5,
) -> MRIDecisionPlan:
    """
    Inspect the numeric feature dict and decide, deterministically, which
    review flags to raise and whether calling the LLM is even worthwhile for
    this record. Runs independently of any LLM output.

    Inputs:
        feature_dict: "features" dict produced by preprocess_patient.
        low_lv_ef_pct: LV ejection fraction (%) below which
            "low_lv_ejection_fraction_suspected" is raised. 45% is a
            standard clinical cutoff below normal/mildly-reduced range.
        low_rv_ef_pct: RV ejection fraction (%) below which
            "low_rv_ejection_fraction_suspected" is raised.
        high_lv_ef_pct: LV ejection fraction (%) above which
            "high_lv_ejection_fraction_suspected" is raised - a hyperdynamic/
            preserved-high LV, clinically associated with hypertrophic
            cardiomyopathy (HCM). 65.0 is the F1-maximizing threshold found
            by sweeping candidates against ACDC's pathology groups (see
            mri_flag_validation.py): F1 0.698 (precision 0.667, recall
            0.733) against HCM-group membership, over all 150 patients -
            weaker separation than the low-EF flags (HCM's mean LV EF is
            only ~7-13 points above the other groups, vs. 20-30+ points for
            DCM/MINF's low-EF separation), but still a real, useful signal.
        pathology_classifier_probability_threshold: probability above which
            "pathology_<CLASS>_pattern_suspected" is raised for a given
            class - only checked for classes actually present in
            feature_dict (i.e. only when compute_pathology_classifier_features
            was run), mirrors ecg_agent.detect_flags's
            embedding_probability_threshold exactly, including applying it
            uniformly to every class (NOR included) rather than treating
            "normal" specially - the flag name is a generic "this pattern
            was detected with high confidence", not an abnormality claim.

    Output:
        MRIDecisionPlan instance.
    """
    if feature_dict["lv_ejection_fraction_pct"] is None:
        return MRIDecisionPlan(
            deterministic_flags=["insufficient_data_for_analysis"],
            skip_llm_call=True,
            reason="LV ejection fraction not computable (zero ED volume); not enough signal for analysis.",
        )

    flags = []

    if feature_dict["lv_ejection_fraction_pct"] < low_lv_ef_pct:
        flags.append("low_lv_ejection_fraction_suspected")
    elif feature_dict["lv_ejection_fraction_pct"] > high_lv_ef_pct:
        flags.append("high_lv_ejection_fraction_suspected")

    if (
        feature_dict["rv_ejection_fraction_pct"] is not None
        and feature_dict["rv_ejection_fraction_pct"] < low_rv_ef_pct
    ):
        flags.append("low_rv_ejection_fraction_suspected")

    for class_abbrev in _PATHOLOGY_CLASS_NAMES:
        key = f"pathology_{class_abbrev}_probability"
        if key in feature_dict and feature_dict[key] > pathology_classifier_probability_threshold:
            flags.append(f"pathology_{class_abbrev}_pattern_suspected")

    return MRIDecisionPlan(deterministic_flags=flags, skip_llm_call=False, reason=None)


# ----------------------------------------------------------------------
# 3. MRI-specific structured-findings prompt (via the shared scaffold)
# ----------------------------------------------------------------------
def build_mri_findings_prompt(summary, feature_dict, image_keys, patient_info=None):
    """
    Build the MRI structured-findings prompt via the shared prompt scaffold
    (agent_common.build_generic_findings_prompt), specialized with the MRI
    agent's role, modality label, and evidence keys - both numeric feature
    keys and image identifiers. Also tells the model which attached image
    is which, in order, since Ollama does not attach names to images itself.

    Inputs:
        summary: text summary produced by build_mri_clinical_summary.
        feature_dict: numeric features dict ("features" from
            preprocess_patient); its keys are valid numeric evidence_ref
            values.
        image_keys: ordered list of image identifiers (e.g.
            ["short_axis_mid_ed"] or ["short_axis_mid_ed", "short_axis_mid_es"]),
            matching the order of the images actually sent to the LLM
            backend - these are valid image-type evidence_ref values.
        patient_info: optional free-text patient context string.

    Output:
        Prompt string ready to send to an LLM backend.
    """
    evidence_keys = list(feature_dict.keys()) + list(image_keys)
    return build_generic_findings_prompt(
        agent_role="an MRI analysis agent",
        modality_label="MRI",
        summary=summary,
        evidence_keys=evidence_keys,
        patient_info=patient_info,
        image_labels=image_keys,
    )


# ----------------------------------------------------------------------
# 3b. Learned pathology classifier (Phase F, mirrors ecg_agent.py's Phase E)
# ----------------------------------------------------------------------
def compute_pathology_classifier_features(raw_ed_slice_uint8, yolo_model, unet_model, classifier_model, device="cpu"):
    """
    Run the trained YOLO -> U-Net -> classifier chain on a raw, UNCROPPED ED
    slice to get per-pathology-class probabilities - the learned complement
    to the deterministic EF-threshold flags. Deliberately takes the raw
    slice, not preprocess_patient()'s already-cropped display image: that
    crop is computed from ACDC's expert mask, which won't exist at real
    deployment, and the classifier was itself trained on YOLO's own
    (imperfect) crops - feeding it an expert-mask crop would be an
    inconsistent input distribution relative to training.

    Inputs:
        raw_ed_slice_uint8: (H, W) uint8 array - the full, uncropped,
            normalized ED slice (e.g. mri_preprocessing.normalize_slice_to_uint8
            applied to the raw volume at preprocessing_result["slice_indices"]["ed"]).
        yolo_model: loaded Ultralytics YOLO model (heart localization).
        unet_model: loaded segmentation_models_pytorch U-Net (in eval mode).
        classifier_model: loaded mri_classifier_train.PathologyClassifier
            (in eval mode).
        device: torch device string.

    Output:
        dict of "pathology_<CLASS>_probability" -> float, one entry per
        mri_classifier_dataset.PATHOLOGY_CLASSES.
    """
    from mri_classifier_dataset import predict_pathology_probabilities

    return predict_pathology_probabilities(raw_ed_slice_uint8, yolo_model, unet_model, classifier_model, device=device)


# ----------------------------------------------------------------------
# 4. Orchestration - the MRI agent's end-to-end contribution pipeline
# ----------------------------------------------------------------------
def generate_mri_contribution(
    preprocessing_result,
    llm_call: Optional[Callable[..., str]] = None,
    record_id: Optional[str] = None,
    patient_info: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1,
    image_mode: str = "ed_only",
    raw_ed_slice_uint8=None,
    yolo_model=None,
    unet_model=None,
    classifier_model=None,
    device: str = "cpu",
) -> AgentContribution:
    """
    Run the MRI agent's full contribution pipeline: build the deterministic
    decision (detect_flags_mri), then - unless it decides to skip -
    build the structured-findings prompt (with the selected image(s)
    attached), call the given LLM backend, and parse/validate the response
    into an AgentContribution. On a parse failure, retries by appending the
    parse error to the prompt and asking the model to correct it.
    Deterministic flags are always merged into the final flags_for_review,
    whether or not the LLM is called.

    Inputs:
        preprocessing_result: dict returned by
            mri_preprocessing.preprocess_patient.
        llm_call: callable taking (prompt, images=[...]) and returning the
            raw text response. If not provided, defaults to
            agent_common.generate_report_medgemma (free, local, no API key).
        record_id: optional identifier of the record, stored in the output
            for traceability. Defaults to preprocessing_result["patient_id"]
            if not given.
        patient_info: optional free-text patient context.
        model_name: optional string identifying the model/backend used,
            stored in the trace for auditability. Defaults to
            "ollama:medgemma:4b" when llm_call is left at its default.
        max_retries: number of extra attempts if the LLM response fails to
            parse.
        image_mode: which image(s) to attach - "ed_only" (default, the
            safer single-image baseline given MedGemma's evaluation is
            primarily single-image) or "ed_es" (both phases in one call,
            less validated - test its reliability before relying on it).
        raw_ed_slice_uint8, yolo_model, unet_model, classifier_model, device:
            optional - when ALL FOUR of raw_ed_slice_uint8/yolo_model/
            unet_model/classifier_model are given, also computes the learned
            pathology classifier features (compute_pathology_classifier_features)
            and merges them into the feature dict, clinical summary, and
            deterministic flags. Omitted (the default), behaves exactly as
            before this integration - same opt-in convention as ECG's
            embedding wiring.

    Output:
        AgentContribution instance (agent_id="mri"). If
        detect_flags_mri decides to skip the LLM call (ejection
        fraction not computable), findings is an empty list and
        trace["model"] is None.

    Raises:
        FindingsParseError: if no valid structured response is obtained
        within max_retries + 1 attempts.
        ValueError: if image_mode is not "ed_only" or "ed_es".
    """
    if image_mode not in ("ed_only", "ed_es"):
        raise ValueError(f"image_mode must be 'ed_only' or 'ed_es', got {image_mode!r}")

    if record_id is None:
        record_id = preprocessing_result["patient_id"]

    if llm_call is None:
        llm_call = generate_report_medgemma
        if model_name is None:
            model_name = "ollama:medgemma:4b"

    feature_dict = dict(preprocessing_result["features"])

    pathology_probabilities = None
    if raw_ed_slice_uint8 is not None and yolo_model is not None and unet_model is not None and classifier_model is not None:
        pathology_probabilities = compute_pathology_classifier_features(
            raw_ed_slice_uint8, yolo_model, unet_model, classifier_model, device=device
        )
        feature_dict.update(pathology_probabilities)

    decision = detect_flags_mri(feature_dict)

    if decision.skip_llm_call:
        return AgentContribution(
            agent_id="mri",
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

    if image_mode == "ed_only":
        image_keys = ["short_axis_mid_ed"]
    else:
        image_keys = ["short_axis_mid_ed", "short_axis_mid_es"]
    images = [preprocessing_result["images"][key] for key in image_keys]

    summary = build_mri_clinical_summary(preprocessing_result, patient_info, pathology_probabilities)
    prompt = build_mri_findings_prompt(summary, feature_dict, image_keys, patient_info)

    valid_evidence_keys = set(feature_dict.keys()) | set(image_keys)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw_response = llm_call(prompt, images=images)
        try:
            findings, flags = parse_findings_response(
                raw_response, valid_evidence_keys, image_evidence_keys=set(image_keys)
            )
            merged_flags = sorted(set(flags) | set(decision.deterministic_flags))
            return AgentContribution(
                agent_id="mri",
                record_id=record_id,
                findings=findings,
                flags_for_review=merged_flags,
                trace={
                    "features_used": feature_dict,
                    "model": model_name,
                    "attempts": attempt + 1,
                    "image_mode": image_mode,
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
