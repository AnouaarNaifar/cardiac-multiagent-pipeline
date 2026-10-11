"""
Shared building blocks for the multi-agent clinical pipeline
==============================================================
This module holds the pieces that are identical across every modality agent
(ECG, MRI, structured patient data, ...): the structured-findings schema, the
JSON parsing/validation logic, the shared prompt scaffold that encodes the
"you are one of several parallel agents, no diagnosis, one JSON shape"
contract, the LLM backend callables, and the contribution-saving helper.

Design principle (Decision 1 - shared model backbone): every agent calls the
SAME underlying model (MedGemma 4B via Ollama). What differs between agents
is not the model, but the prompt (role, modality label, evidence keys) and
the modality-specific preprocessing that happens before the prompt is built.
This module is where that shared contract lives, so a schema or prompt fix
only has to be made once, not once per agent module.

Additional dependencies beyond the rest of the project: requests (for
Ollama) or anthropic (for the optional Claude API backend) - same as before.
"""

import json
import numbers
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# 1. Structured findings schema (contribution object for the Sep Agent)
# ----------------------------------------------------------------------
@dataclass
class Finding:
    """
    A single modality-derived observation with its supporting rationale.

    Inputs (fields):
        feature: Short name of the feature/observation this finding is about
            (e.g. "heart_rate_mean_bpm", or an image identifier for a
            vision-grounded finding).
        value: The value observed for this feature (numeric or string).
        rationale: One short sentence explaining why this observation
            matters, scoped strictly to what this agent's own modality
            supports (no cross-modality or diagnostic claims).
        confidence: Model's confidence in this finding, in [0.0, 1.0].
        evidence_ref: Name of the key (in the agent's numeric feature dict,
            or in its image registry) that this finding is grounded in,
            enabling traceability back to the raw computed value or image.
        evidence_type: Whether evidence_ref points to a numeric feature or
            an image. Defaults to "numeric_feature" for backward
            compatibility with text-only agents that never set it
            explicitly (e.g. the existing ECG agent). Set to "image_region"
            for vision-grounded findings.
    """

    feature: str
    value: Any
    rationale: str
    confidence: float
    evidence_ref: str
    evidence_type: str = "numeric_feature"


@dataclass
class AgentContribution:
    """
    Structured output of a modality agent. Meant to be consumed by a
    downstream aggregating component (the "Sep Agent"), not read directly as
    a final report - it contains no diagnosis and no cross-modality
    conclusion. Shared across all agents (ECG, MRI, ...); agent_id is what
    distinguishes one agent's output from another's.

    Inputs (fields):
        agent_id: Fixed identifier for the agent that produced this
            contribution (e.g. "ecg", "mri").
        record_id: Identifier of the record this contribution refers to,
            supplied by the caller, or None.
        findings: List of Finding objects.
        flags_for_review: Short machine-readable tags for elements that
            warrant human or downstream review.
        trace: Dictionary capturing what fed into this contribution (numeric
            features used, model/backend identifier, retry attempts, and any
            agent-specific extras), so the contribution can be audited.
    """

    agent_id: str
    record_id: Optional[str]
    findings: List[Finding]
    flags_for_review: List[str]
    trace: Dict[str, Any]


class FindingsParseError(Exception):
    """Raised when an LLM response cannot be parsed into valid structured findings."""


# ----------------------------------------------------------------------
# 2. Patient demographic context
# ----------------------------------------------------------------------
_PTBXL_SEX_LABELS = {0: "male", 1: "female"}


def build_patient_context_line(age=None, sex=None, height=None, weight=None) -> str:
    """
    Builds a short, human-readable patient-context line for prepending to a
    modality agent's clinical summary/prompt (fed in as the existing
    `patient_info` free-text parameter already threaded through
    build_generic_findings_prompt and both modality agents' clinical-
    summary builders - this function only decides what that string says,
    it does not change how it's plumbed in).

    Missing fields are omitted gracefully, never guessed - e.g. PTB-XL's
    `height` column is missing for ~68% of records, and this is normal,
    unremarkable sparse data, not worth calling out explicitly. The one
    exception: if BOTH age and sex are missing, the line says so
    explicitly ("age/sex not available") rather than silently omitting
    them.

    Inputs:
        age: patient age in years (int/float), or None if unavailable.
        sex: "male"/"female" (str), or the raw PTB-XL-convention int (0=male,
            1=female) - either is accepted so callers can pass PTB-XL's
            `sex` column directly. None if unavailable, or an unrecognized
            value (neither "male"/"female" nor 0/1) is treated as unavailable.
        height: height in cm (int/float), or None if unavailable.
        weight: weight in kg (int/float), or None if unavailable.

    Output:
        A line like "Patient: 62-year-old male, 178cm/82kg."
        Empty string if every field is unavailable.
    """
    # numbers.Real (not (int, float)) so a numpy int64/float64 straight out
    # of a pandas column (e.g. PTB-XL's `sex`/`age` columns) is recognized
    # too - numpy's integer types are NOT instances of Python's built-in
    # int, so an (int, float) check would silently drop real data here.
    if isinstance(sex, numbers.Real) and int(sex) in _PTBXL_SEX_LABELS:
        sex = _PTBXL_SEX_LABELS[int(sex)]
    elif sex not in ("male", "female"):
        sex = None

    # PTB-XL anonymizes ages > 89 by storing them as 300 (a HIPAA-style
    # de-identification convention); any implausible value (>= 120) gets the
    # same treatment. Render ">89" rather than a fake number - misleading both
    # in the UI and in the LLM prompt.
    age_label = None
    if age is not None:
        age_label = ">89" if age >= 120 else str(int(age))

    parts = []
    if age_label is not None and sex is not None:
        parts.append(f"{age_label}-year-old {sex}")
    elif age_label is not None:
        parts.append(f"{age_label} years old")
    elif sex is not None:
        parts.append(sex)
    elif height is not None or weight is not None:
        # Both age and sex are absent, but we do have SOME demographic
        # data - state the gap explicitly rather than silently dropping it
        # (see docstring: this is the ACDC/MRI case every time).
        parts.append("age/sex not available")

    anthro = []
    if height is not None:
        anthro.append(f"{height:g}cm")
    if weight is not None:
        anthro.append(f"{weight:g}kg")
    if anthro:
        parts.append("/".join(anthro))

    if not parts:
        return ""
    return "Patient: " + ", ".join(parts) + "."


# ----------------------------------------------------------------------
# 3. Shared structured-findings prompt scaffold
# ----------------------------------------------------------------------
def build_generic_findings_prompt(
    agent_role, modality_label, summary, evidence_keys, patient_info=None, image_labels=None
):
    """
    Build the structured-findings prompt shared by every modality agent.
    Encodes the parts of the contract that must stay identical across
    agents (no diagnosis, one JSON shape, the evidence_ref validation rule),
    parameterized only by what's genuinely modality-specific.

    Inputs:
        agent_role: short description of this agent's role, INCLUDING its
            leading article (e.g. "an ECG analysis agent", "an MRI analysis
            agent") since the correct article ("a" vs "an") depends on the
            specific wording and can't be inferred generically.
        modality_label: short label for the modality analyzed, e.g. "ECG"
            (used in "Automatically extracted <modality_label> features:"
            and "information outside the <modality_label> signal").
        summary: text summary of the modality-specific features/findings,
            already built by the calling agent.
        evidence_keys: iterable of valid evidence_ref values the model may
            cite for this record (numeric feature keys and/or image identifiers).
        patient_info: optional free-text patient context string.
        image_labels: optional list of image identifier strings, in the
            same order as the images passed to the LLM backend's "images"
            argument. Ollama attaches no names to images itself,so if
            images are sent, the prompt must spell out which image is which
            so the model can cite the correct evidence_ref. Defaults to None
            (no image section added) - text-only agents are unaffected.

    Output:
        Prompt string ready to send to an LLM backend.
    """
    evidence_keys_str = ", ".join(evidence_keys)
    patient_line = f"\nPatient context: {patient_info}" if patient_info else ""

    image_section = ""
    if image_labels:
        numbered = "\n".join(f"- Image {i + 1}: {label}" for i, label in enumerate(image_labels))
        image_section = f"\nThe following images are attached, in this order:\n{numbered}\n"

    return f"""You are {agent_role}. You are ONE of several parallel
agents (ECG, MRI, structured patient data) whose outputs will be combined by a
separate aggregating component into a final clinical decision. Do NOT produce
a diagnosis, a final report, or any conclusion that depends on information
outside the {modality_label} signal. Your job is only to report structured findings, each
grounded in specific evidence, with a short rationale.

Automatically extracted {modality_label} features:
{summary}
{patient_line}
{image_section}
Respond with ONLY a single JSON object (no markdown fences, no extra text)
with this exact shape:
{{
  "findings": [
    {{
      "feature": "<short name>",
      "value": <number or string>,
      "rationale": "<one short sentence, {modality_label} evidence only>",
      "confidence": <float between 0.0 and 1.0>,
      "evidence_ref": "<one of: {evidence_keys_str}>"
    }}
  ],
  "flags_for_review": ["<short machine-readable tag>", ...]
}}

Rules:
- "evidence_ref" must be one of: {evidence_keys_str}.
- If a finding restates a specific feature listed above, "evidence_ref" must be that exact feature's own
  key - not an image reference, even if an image is attached.
- Only cite an image as "evidence_ref" for a visual observation that is not already captured by one of the
  listed features above.
- Do not include any field or text outside this JSON object.
- If nothing warrants a flag, use an empty list for "flags_for_review".
"""


# ----------------------------------------------------------------------
# 4. Response parsing and validation
# ----------------------------------------------------------------------
def parse_findings_response(raw_text, valid_evidence_keys, image_evidence_keys=None):
    """
    Parse and validate an LLM's raw text response into Finding objects and a
    review-flags list, rejecting responses that are malformed or that cite an
    evidence key the model was not given.

    Inputs:
        raw_text: raw string returned by the LLM backend.
        valid_evidence_keys: set of feature/image names the model was
            allowed to cite as "evidence_ref".
        image_evidence_keys: optional subset of valid_evidence_keys that are
            image identifiers rather than numeric feature keys. Used to set
            each Finding's evidence_type ("image_region" vs
            "numeric_feature") ourselves, from which registry the cited key
            actually came from, rather than trusting the LLM to self-report
            it correctly. Defaults to None (treated as empty - every
            Finding gets evidence_type="numeric_feature", matching the
            existing text-only agents' behavior).

    Output:
        (findings, flags_for_review) tuple: findings is a list of Finding
        instances, flags_for_review is a list of strings.

    Raises:
        FindingsParseError: if the text is not valid JSON, is missing
        required fields, references an unknown evidence key, or has a
        confidence value outside [0.0, 1.0].
    """
    image_evidence_keys = image_evidence_keys or set()

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FindingsParseError(f"Response is not valid JSON: {exc}") from exc

    if "findings" not in payload or "flags_for_review" not in payload:
        raise FindingsParseError(
            "Response JSON is missing 'findings' or 'flags_for_review'"
        )

    numeric_evidence_keys = set(valid_evidence_keys) - set(image_evidence_keys)

    required_fields = {"feature", "value", "rationale", "confidence", "evidence_ref"}
    findings = []
    for item in payload["findings"]:
        missing = required_fields - item.keys()
        if missing:
            raise FindingsParseError(f"Finding is missing fields: {missing}")
        if item["evidence_ref"] not in valid_evidence_keys:
            raise FindingsParseError(
                f"Finding cites unknown evidence_ref '{item['evidence_ref']}'; "
                f"expected one of {sorted(valid_evidence_keys)}"
            )
        confidence = float(item["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise FindingsParseError(
                f"Finding confidence {confidence} is outside the [0.0, 1.0] range"
            )

        evidence_ref = item["evidence_ref"]
        if item["feature"] in numeric_evidence_keys:
            # Models with an attached image were empirically observed to over-cite the image as
            # evidence_ref even for findings that merely restate a specific numeric feature given
            # directly in the text summary. When the finding's own "feature" name exactly matches
            # a known numeric key, trust that over whatever evidence_ref the model chose - safer
            # than relying on prompt instructions alone (which did not fix this).
            evidence_ref = item["feature"]

        evidence_type = "image_region" if evidence_ref in image_evidence_keys else "numeric_feature"
        findings.append(
            Finding(
                feature=item["feature"],
                value=item["value"],
                rationale=item["rationale"],
                confidence=confidence,
                evidence_ref=evidence_ref,
                evidence_type=evidence_type,
            )
        )

    return findings, list(payload["flags_for_review"])


# ----------------------------------------------------------------------
# 5. LLM backends - return raw text for a given prompt
# ----------------------------------------------------------------------
def generate_report_medgemma(
    prompt, model="medgemma:4b",
    host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"), images=None,
    temperature=None, seed=None, num_predict=2048, json_mode=False, num_ctx=8192,
):
    """
    Call a local LLM through Ollama and return its raw text response.

    Requires Ollama installed and running locally (https://ollama.com/download),
    with the model already pulled beforehand:
        ollama pull medgemma:4b
    No API key, no cost, no request limits.

    Inputs:
        prompt: prompt string to send to the model.
        model: Ollama model name.
        host: base URL of the local Ollama server.
        images: optional list of base64-encoded image strings (JPEG/PNG/WebP)
            to send alongside the prompt, for multimodal input via MedGemma's
            image encoder. Defaults to None (text-only), so existing
            text-only callers (the ECG agent) are unaffected - the "images"
            key is only added to the request payload when this is non-empty.
        temperature: optional float overriding Ollama's default sampling
            temperature (confirmed empirically to be ~0.8, NOT set anywhere
            in medgemma:4b's own Modelfile - only `stop` is). None (the
            default) omits it, so existing callers keep the current
            production behavior unless they explicitly opt in - this is
            an experiment/measurement knob
            (src/temperature_seed_reliability_test.py), not yet adopted as
            a new default.
        seed: optional int for reproducible generation - confirmed
            empirically (2 identical calls with the same seed returned
            byte-identical text) that Ollama/medgemma:4b does support this.
            None (the default) means no seed is sent, i.e. current
            production behavior (non-deterministic).
        num_predict: explicit generation cap (Ollama option). Defaults to
            2048 - byte-identical to previous behavior for existing callers.
            Callers that build a prompt asking for many findings (e.g. the ECG
            agent with the multi-lead + embedding feature set wired in, ~32
            numeric features -> ~32 findings) can raise this to avoid the JSON
            being truncated mid-response ("Unterminated string"); the default
            2048 is enough for the single-lead feature set (~10 findings) but
            not the full multi-lead one.
        json_mode: if True, sets Ollama's top-level "format": "json" option,
            which grammar-constrains decoding so the output is
            SYNTACTICALLY guaranteed to be valid JSON (this is enforced at
            the token-sampling level by Ollama itself, not just requested
            via the prompt text - it cannot fix wrong/hallucinated content,
            but it does eliminate malformed-JSON parse failures like
            unquoted keys or stray text). Defaults to False - byte-identical
            behavior for existing callers (ecg_agent.py's text-only calls,
            already reasonably reliable per cot_repeated_reliability_sample.py)
            - opt in explicitly for callers seeing raw JSONDecodeError
            failures (see ecg_image_agent.py's _call_with_image_fallback).

    Output:
        Raw text response from the model (str).
    """
    import requests

    # num_ctx: total token budget shared between the prompt and the
    # response. Without this, Ollama falls back to its own small default
    # (~2-4k tokens) regardless of the model's real 131k-token capacity
    # (medgemma:4b, confirmed via `ollama show`) - with a long prompt (RAG
    # knowledge chunks, CoT instructions, growing retry history), that
    # default leaves too little room for the response and generation gets
    # cut off mid-JSON.
    # num_predict: an explicit generation cap, set
    # generously so it isn't itself the bottleneck. Found via
    # src/cot_repeated_reliability_sample.py: 9/13 retries and both
    # exhausted-retry failures in a 25-generation sample were exactly this
    # truncation pattern ("Unterminated string...").
    options = {"num_ctx": num_ctx, "num_predict": num_predict}
    if temperature is not None:
        options["temperature"] = temperature
    if seed is not None:
        options["seed"] = seed

    payload = {
               "model": model,
               "prompt": prompt,
               "stream": False,
               "options": options
            }
    if images:
        payload["images"] = images

    
    response = requests.post(f"{host}/api/generate", json=payload, timeout=180)
    response.raise_for_status()
    return response.json()["response"]


def generate_report_anthropic(prompt, api_key=None, model="claude-sonnet-5"):
    """
    Call the Anthropic API and return its raw text response.

    Requires a valid Anthropic API key (ANTHROPIC_API_KEY environment
    variable, or passed explicitly) and pip install anthropic.

    Inputs:
        prompt: prompt string to send to the model.
        api_key: Anthropic API key; falls back to the ANTHROPIC_API_KEY
            environment variable if not provided.
        model: Anthropic model identifier.

    Output:
        Raw text response from the model (str).
    """
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def generate_report_groq(
    prompt, api_key=None, model="openai/gpt-oss-120b", temperature=None, seed=None,
    max_tokens=2048, reasoning_effort=None,
    host="https://api.groq.com/openai/v1/chat/completions",
):
    """
    Call Groq's free-tier, OpenAI-compatible API and return the raw text
    response. Optional, non-default backend - a FREE way to test a much larger
    model (e.g. Llama-3.3-70B) than anything that fits locally, to probe whether
    model capability (not prompt) is the ceiling on deep criterion integration
    (G2 - see results/rag_ab_test/). Cloud: the prompt leaves the machine, so use
    only with non-sensitive / synthetic inputs. No new dependency (uses requests).

    Inputs:
        prompt: prompt string to send to the model.
        api_key: Groq API key; falls back to the GROQ_API_KEY environment
            variable if not provided (create a free key at console.groq.com).
        model: Groq model id (default the strong free general model
            "openai/gpt-oss-120b"; e.g. "llama-3.1-8b-instant" is smaller/faster).
        temperature/seed: optional reproducibility/sampling overrides (same role
            as in generate_report_medgemma); None omits them.

    Output:
        Raw text response from the model (str).
    """
    import requests

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No Groq API key: set GROQ_API_KEY or pass api_key.")

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if seed is not None:
        payload["seed"] = seed
    # Reasoning models (e.g. openai/gpt-oss-*) spend part of max_tokens on hidden
    # reasoning; without a larger budget the JSON body gets truncated. Callers can
    # cap reasoning via reasoning_effort ("low"/"medium"/"high") to leave room.
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    response = requests.post(
        host, json=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def call_ollama_chat_with_tools(
    messages, tools, model="qwen3:4b", host="http://localhost:11434", think=True
):
    """
    Call Ollama's /api/chat endpoint with a list of tool schemas and return
    the raw assistant message dict (may contain "content" and/or
    "tool_calls"). Used for LLM-driven tool-calling orchestration - requires
    a model that Ollama reports as supporting "tools" (MedGemma does not).

    Inputs:
        messages: list of {"role": ..., "content": ...} dicts forming the
            conversation so far (system/user/assistant/tool turns).
        tools: list of Ollama/OpenAI-style tool schemas
            ({"type": "function", "function": {...}}).
        model: Ollama model name; must support tool calling.
        host: base URL of the local Ollama server.
        think: whether to let the model use its extended "thinking" mode
            (supported by reasoning models like Qwen3) before responding.
            Defaults to True: disabling it was measured to save only ~25%
            latency while making the model request the same tool multiple
            times instead of concluding cleanly, so correctness of the
            tool-calling behavior is prioritized over that modest speedup.

    Output:
        The assistant message dict as returned by Ollama
        (response.json()["message"]).
    """
    import requests

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "tools": tools,
            "think": think,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["message"]


# ----------------------------------------------------------------------
# 6. Contribution persistence
# ----------------------------------------------------------------------
def save_contribution(contribution: AgentContribution, output_dir=None, file_name="contribution"):
    """
    Serialize an AgentContribution to a JSON file for later auditing by the
    Sep Agent or a human reviewer.

    Inputs:
        contribution: AgentContribution instance to save.
        output_dir: directory to write into; defaults to "./results" next to
            this module if not provided (matches the ECG agent's original
            default behavior, preserved here for backward compatibility).
        file_name: base file name (without extension) for the JSON file.

    Output:
        Path to the written JSON file (str).
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{file_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(contribution), f, indent=2, ensure_ascii=False)

    return output_path
