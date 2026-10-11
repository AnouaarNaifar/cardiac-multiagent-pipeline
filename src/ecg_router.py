"""
ECG router - single entry point, dispatches image vs signal, always responds
================================================================================
A user provides EITHER an ECG image (photo/scan) OR an ECG signal (WFDB
.hea/.dat pair, or a CSV of samples) - never both. This module is the
single entry point the UI/API calls: it decides which branch applies
from the upload itself, runs the matching pipeline
(ecg_ingest + ecg_preprocessing + ecg_agent for a signal;
ecg_image_agent for an image), and returns an AgentContribution either
way - the caller never needs to branch on modality itself.

"Always respond" contract: every failure mode this router can anticipate
(unrecognized file type, ecg_ingest hard validation errors, the image
agent's own FindingsParseError after exhausted retries, or literally any
other unexpected exception) is caught here and turned into a valid
AgentContribution with an explanatory flag - never an unhandled
exception propagating to the caller. This is deliberately centralized
here rather than duplicated in ecg_agent.py / ecg_image_agent.py, which
keep their original "raise on unparseable LLM response" contract (see
ecg_image_agent.py's module docstring) - this router is the one place
that guarantees a caller-facing response no matter what.
"""

import os
from typing import Any, Callable, Optional

from agent_common import AgentContribution, FindingsParseError
from ecg_agent import generate_ecg_contribution
from ecg_image_agent import generate_image_ecg_contribution
from ecg_ingest import load_csv, load_wfdb, validate, PTBXL_LEAD_ORDER
from ecg_preprocessing import ecg_preprocessing_pipeline

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SIGNAL_EXTENSIONS = {".hea", ".dat", ".csv"}

# Rhythm lead used for the single-lead pipeline when available (see
# ecg_multilead.py's _RHYTHM_LEAD - kept consistent with the rest of the
# project). Falls back to the first available lead if "II" isn't present
# (e.g. a non-12-lead upload).
_PREFERRED_SINGLE_LEAD = "II"


def _fallback_contribution(record_id: Optional[str], reason: str,
                            extra_flags: Optional[list] = None) -> AgentContribution:
    """Build the guaranteed-safe AgentContribution used whenever nothing
    else can be returned (see module docstring's "always respond" contract)."""
    flags = ["agent_error"] + (extra_flags or [])
    return AgentContribution(
        agent_id="ecg_router",
        record_id=record_id,
        findings=[],
        flags_for_review=sorted(set(flags)),
        trace={"model": None, "attempts": 0, "error": reason},
    )


def detect_upload_kind(file_path: str) -> str:
    """
    Decide whether an uploaded file is an "image" or a "signal", from its
    extension alone (a WFDB signal is identified by its .hea companion
    file, a CSV by its own extension).

    Inputs:
        file_path: path to the uploaded file. For a WFDB record, pass
            EITHER the .hea or the .dat path - both live alongside each
            other and either is enough to detect the record.

    Output:
        "image", "wfdb", or "csv".

    Raises:
        ValueError: if the extension is not recognized.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in (".hea", ".dat"):
        return "wfdb"
    if ext == ".csv":
        return "csv"
    raise ValueError(f"Unrecognized ECG upload extension '{ext}' (expected one of "
                      f"{sorted(IMAGE_EXTENSIONS | SIGNAL_EXTENSIONS)}).")


def route_ecg_upload(
    file_path: str,
    record_id: Optional[str] = None,
    patient_info: Optional[str] = None,
    llm_call: Optional[Callable[[str], str]] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1,
    csv_fs: Optional[float] = None,
    csv_unit: str = "mV",
    embedding_model: Any = None,
    classifier_model: Any = None,
    digitize: bool = False,
    ecg_gate: bool = False,
    ecg_gate_threshold: float = 0.5,
) -> AgentContribution:
    """
    Single entry point: routes an ECG upload to the matching agent
    (image or signal) and always returns an AgentContribution.

    Inputs:
        file_path: path to the uploaded file (image, or a WFDB .hea/.dat
            path, or a CSV path).
        record_id: optional identifier stored in the output for
            traceability.
        patient_info: optional free-text patient context string.
        llm_call: optional callable overriding the default LLM backend
            for WHICHEVER agent ends up being used. Left at None to use
            each agent's own default (Ollama medgemma:4b, with the image
            attached automatically in the image branch - see
            ecg_image_agent.generate_image_ecg_contribution).
        model_name: optional string identifying the model/backend, passed
            through to the chosen agent.
        max_retries: number of extra LLM attempts on a parse failure,
            passed through to the chosen agent.
        csv_fs: sampling frequency in Hz - REQUIRED if file_path is a
            CSV signal (a CSV carries no header for it). Ignored for
            image/WFDB uploads.
        csv_unit: unit of the CSV signal ("mV" or "uV"), passed to
            ecg_ingest.load_csv. Ignored for image/WFDB uploads.
        embedding_model: optional trained ECGEmbeddingNet instance (see
            ecg_embedding.load_model_checkpoint), passed through to the
            signal agent when the upload has all 12 leads in the right
            order (see ecg_ingest.validate's "embedding" gate). Ignored
            for image uploads (the image branch has its own learned model,
            see classifier_model).
        classifier_model: optional trained GenECG image classifier (see
            genecg_classifier_train.load_genecg_classifier), passed through
            to the IMAGE agent to add calibrated "pathology_<CLASS>_probability"
            findings + "pathology_<CLASS>_pattern_suspected" flags - the image
            branch's equivalent of embedding_model. Ignored for signal uploads.
        ecg_gate: if True, run the CLIP zero-shot "is this actually an ECG?"
            content check on an IMAGE upload before any real work; a non-ECG
            image returns an empty contribution with a "not_an_ecg_image" flag
            (see ecg_image_agent.generate_image_ecg_contribution). Opt-in,
            fail-open, and ignored for signal uploads.
        ecg_gate_threshold: minimum P(ECG) to accept an image upload as an ECG
            (default 0.5). Ignored for signal uploads.

    Output:
        AgentContribution instance. agent_id is "ecg" (signal branch),
        "ecg_image" (image branch), or "ecg_router" (a failure this
        router itself caught - see _fallback_contribution). NEVER
        raises - every anticipated and unanticipated failure mode is
        caught and turned into a valid, explanatory AgentContribution.
    """
    try:
        kind = detect_upload_kind(file_path)
    except ValueError as exc:
        return _fallback_contribution(record_id, str(exc), extra_flags=["unrecognized_upload_type"])

    try:
        if kind == "image":
            return generate_image_ecg_contribution(
                file_path,
                llm_call=llm_call,
                record_id=record_id,
                patient_info=patient_info,
                model_name=model_name,
                max_retries=max_retries,
                classifier_model=classifier_model,
                digitize=digitize,
                ecg_gate=ecg_gate,
                ecg_gate_threshold=ecg_gate_threshold,
            )

        # --- signal branch (wfdb or csv) ---
        if kind == "wfdb":
            record_path = os.path.splitext(file_path)[0]  # strip .hea or .dat
            ingested = load_wfdb(record_path)
        else:  # csv
            if csv_fs is None:
                return _fallback_contribution(
                    record_id, "csv_fs is required for a CSV signal upload but was not provided.",
                    extra_flags=["missing_sampling_frequency"],
                )
            ingested = load_csv(file_path, fs=csv_fs, unit=csv_unit)

        report = validate(ingested)
        if not report.ok:
            return AgentContribution(
                agent_id="ecg",
                record_id=record_id,
                findings=[],
                flags_for_review=sorted(set(report.hard_errors)),
                trace={"model": None, "attempts": 0, "warnings": report.warnings,
                       "skipped_reason": "; ".join(report.hard_errors)},
            )

        lead_names_upper = [n.upper() for n in ingested.lead_names]
        lead_idx = (lead_names_upper.index(_PREFERRED_SINGLE_LEAD)
                    if _PREFERRED_SINGLE_LEAD in lead_names_upper else 0)
        single_lead = ingested.signal[:, lead_idx]

        results = ecg_preprocessing_pipeline(single_lead, ingested.fs)

        multilead_signal = ingested.signal if report.enabled["multilead"] else None
        model_for_embedding = embedding_model if report.enabled["embedding"] else None

        contribution = generate_ecg_contribution(
            results,
            ingested.fs,
            llm_call=llm_call,
            record_id=record_id,
            patient_info=patient_info,
            model_name=model_name,
            max_retries=max_retries,
            multilead_signal=multilead_signal,
            embedding_model=model_for_embedding,
        )
        # Surface ecg_ingest's soft warnings alongside the agent's own trace,
        # without altering the agent's own contract/output shape.
        if report.warnings:
            contribution.trace["ingest_warnings"] = report.warnings
        return contribution

    except FindingsParseError as exc:
        # The chosen agent's LLM never returned parseable JSON within its
        # retry budget - this is the one failure mode ecg_agent.py /
        # ecg_image_agent.py deliberately still raise (see their
        # docstrings); this router is where it finally gets turned into a
        # guaranteed response.
        return _fallback_contribution(record_id, str(exc), extra_flags=["llm_response_unparseable"])
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see module docstring
        return _fallback_contribution(record_id, f"{type(exc).__name__}: {exc}")
