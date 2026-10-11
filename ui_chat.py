"""
Conversational assistant over the current case (a VLM chat).

Grounds a free-text conversation in the case's already-produced findings (and the
MRI image when present), with a role-specific framing. Uses the local multimodal
MedGemma backend so it can actually SEE the MRI slice.

Safety: the framing is role-dependent. For the PATIENT role the assistant may
mention the possible conditions the report highlights (aligned with the
privileged patient report, which runs RAG/CoT/KG), but must present them as
possibilities to review with a doctor, never as a confirmed diagnosis, and never
give a prognosis or treatment advice.
"""

_ROLE_SYSTEM = {
    "patient": (
        "You are a friendly, supportive health assistant chatting with a PATIENT. "
        "Talk naturally: greet them back, answer general and everyday questions normally, and keep a warm, "
        "conversational tone. When the topic turns to their test results, use plain, simple language a "
        "non-expert understands. You MAY mention the possible conditions the report highlights (including any "
        "cross-modality convergence), but ALWAYS present them as possibilities to review with a doctor, never "
        "as a confirmed diagnosis. Safety rules for medical topics: never claim a disease is confirmed, never "
        "give a prognosis, and never give treatment or medication advice — instead explain what the findings "
        "describe and tell them to discuss the results with a qualified doctor. This rule does NOT stop you "
        "from normal small talk or answering simple questions."),
    "doctor": (
        "You are a clinical decision-support assistant chatting with a CLINICIAN. Converse naturally — greet "
        "them back and answer general questions normally. When discussing the case, reason about likely "
        "diagnoses and differentials from the findings and their evidence, but present them as decision "
        "support, not a final verdict. Be concise and precise, and flag uncertainty."),
    "researcher": (
        "You are a helpful technical assistant. Converse naturally — greet the user back and answer general "
        "questions normally. When relevant, discuss the findings, the model outputs and the pipeline openly, "
        "including limitations and how each number was produced."),
}


def build_case_context(contributions, sep_decision=None):
    """A compact text summary of the current case, used to ground the chat."""
    lines = []
    if sep_decision is not None and getattr(sep_decision, "summary", None):
        lines.append("Integrated impression: " + sep_decision.summary)
        if sep_decision.flags_for_review:
            lines.append("Review flags (all modalities): " + ", ".join(sep_decision.flags_for_review))
    for agent_id, c in contributions.items():
        flags = ", ".join(c.flags_for_review) or "none"
        lines.append(f"[{agent_id.upper()}] flags: {flags}")
        for f in c.findings[:8]:
            lines.append(f"  - {f.feature} = {f.value} ({f.rationale})")
    return "\n".join(lines) if lines else "No case has been produced yet."


def answer(role, case_context, history, user_message, images=None):
    """Return the assistant's reply. `history` is the prior [{role, content}] turns
    (excluding the new user message). `images` = optional list of base64 strings."""
    from agent_common import generate_report_medgemma

    system = _ROLE_SYSTEM.get(role, _ROLE_SYSTEM["researcher"])
    convo = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history)
    prompt = (
        f"{system}\n\n"
        "The following is background about the current case, for reference IF the user asks about it. "
        "You do not need to mention it for greetings or general questions. When you do cite specific "
        "numbers, use only these — do not invent values beyond them.\n"
        "=== CASE CONTEXT ===\n"
        f"{case_context}\n\n"
        "=== CONVERSATION SO FAR ===\n"
        f"{convo}\nUser: {user_message}\nAssistant:")
    return generate_report_medgemma(prompt, images=images) if images else generate_report_medgemma(prompt)
