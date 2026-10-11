"""
Sep Agent - cross-modality aggregation
========================================
This module is the aggregating component described throughout the ECG/MRI
agent docstrings: it combines the structured "contribution" objects
(agent_common.AgentContribution) produced by each independent modality
agent (ECG, MRI, and eventually structured patient data) into a single
integrated clinical impression, with full traceability back to each source
finding.

Unlike the modality agents, the Sep Agent IS allowed to synthesize a
cross-modality impression - that is its whole purpose. It never invents
clinical facts beyond what the modality agents already reported; its job is
to combine and contextualize, not to re-derive findings from raw signals or
images (those were already interpreted by the modality agents).

Demo/testing note: PTB-XL (ECG) and ACDC (MRI) share no real patients, so
any ECG+MRI case used to exercise this module is necessarily a synthetic
pairing for architecture validation, not a real joint clinical case.

Uses the same shared MedGemma backbone as the modality agents
(agent_common.generate_report_medgemma), called text-only - by the time
findings reach the Sep Agent, images have already been interpreted into
structured text by the MRI agent, so no image input is needed here.

RAG integration (opt-in via generate_sep_agent_decision's use_rag=False
default - existing callers, e.g. report_evaluation.py's ECG-only scoring
pipeline, keep their exact current behavior unless they explicitly opt in;
enabling it by default would have silently changed report_evaluation.py's
already-published n=300 results). When enabled: build_retrieval_query turns
a case's flags into a query, rag_knowledge_base.retrieve() fetches the
top-k most relevant chunks (see that module - 276 chunks across 4 sources:
personal clinical notes, the PTB-XL dataset paper, the ACDC challenge
paper, and this project's ECG-preprocessing methodology source), and those
chunks are injected into the prompt as additional evidence the model may
cite. Retrieved-knowledge citations are validated by the EXACT SAME
mechanism that already validates (agent_id, feature) citations - no new
validation code path: each retrieved chunk is treated as a finding from a
virtual "knowledge" agent_id (feature = "knowledge_<i>"), so
parse_sep_agent_response's existing fabricated-reference rejection (already
tested) covers knowledge citations automatically. This is the concrete
"x Hallucination" check from the RAG architecture sketch this module's
docstring elsewhere refers to.

Clinical-correctness prompt requirement (new, addresses a real, previously
unresolved limitation): sep_agent_demo.ipynb's own demo run once
mischaracterized a 295.5 mL LV end-diastolic volume as "low" instead of
elevated/dilated - traceability confirmed the citation was real but never
checked whether the CHARACTERIZATION was correct, since a raw finding
citation alone doesn't encode what counts as "low" or "high". The prompt
now requires every qualitative claim (low/high/elevated/reduced/suspected)
to be grounded in a cited knowledge chunk that actually defines the
relevant threshold, not just a bare number. This is a prompt-level
requirement plus a retrievable audit trail, not a formal automated proof of
correctness - a model could still misapply a correctly-cited definition;
catching that would need a further, not-yet-built rule-based check (e.g.
cross-referencing a claim against the SAME deterministic flags already
computed in detect_flags_mri/detect_flags, which is a cheaper and
more reliable signal than RAG for the specific claims that already have a
hard threshold in code).

Chain-of-Thought integration (opt-in via generate_sep_agent_decision's
use_cot=False default, same reasoning as use_rag: existing callers keep
their exact current behavior unless they explicitly opt in). When enabled,
the response schema gains a "reasoning" field the model must fill BEFORE
"summary" - an explicit step-by-step pass (what does each modality suggest
independently, do they agree or conflict, what does retrieved knowledge -
if any - confirm) rather than jumping straight to a conclusion. This is
most valuable exactly at the Sep Agent, not the modality agents: this is
the one place in the pipeline that actually combines evidence across
sources, so it's the one place where a reasoning chain has something real
to chain together. The "reasoning" text is stored on SepAgentDecision but
deliberately kept OUT of "summary" (which stays the same 3-5 sentence
length as before) - it is an audit artifact, not user-facing prose.
require_reasoning in parse_sep_agent_response rejects (and triggers a
retry on) a response that's missing the field when use_cot=True, so the
step cannot be silently skipped by the model.

Directional-consistency check (src/cot_reliability_test.py, 5 cases - one
per ACDC pathology group, paired with a fixed real ECG record): CoT was
mechanically reliable (5/5 parsed first try, 5/5 produced a reasoning
field, 5/5 changed the output) but 2/5 cases had a real clinical
characterization error that traceability, by design, cannot catch (a cited
feature can be real while its direction is stated backwards) - exactly the
sep_agent_demo.ipynb "295.5 mL called low" bug, still present after RAG's
correctness_rule. Concretely: patient102 (LV EF 70.9%, the
high_lv_ejection_fraction_suspected flag) had the model call it "decreased"
in both the reasoning and the summary; patient109 (RV EF 33.7%, the
low_rv_ejection_fraction_suspected flag) had a summary that said "mildly
reduced" then, one sentence later, "no signs of reduced ejection fraction"
- a self-contradiction introduced only during the reasoning-to-summary
compression step, since the reasoning text itself was accurate. Both cases
have one thing in common: the correct direction was already sitting in
flags_for_review (computed deterministically in detect_flags_mri/
detect_flags) - the model didn't need to judge the number itself and
got it wrong anyway when it tried to. Fixed the same way agent_common.py
already fixed an analogous MedGemma over-citation problem: not by prompting
harder alone, but by a deterministic post-generation check
(_check_directional_consistency) that scans the summary and reasoning text
for a set of known direction-implying flags (the two EF flags, tachycardia,
bradycardia) and rejects any sentence that mentions the relevant measurement
and ventricle/quantity alongside a word implying the opposite direction.
This is a clause-level keyword heuristic, not NLI - it can miss phrasing it
doesn't recognize and is scoped only to the flags carrying a `directional` block
in the clinical knowledge graph (clinical_kg.json - read by
_load_directional_flags_config below), not a general correctness prover. A
fixed prompt rule was also added (unconditional, not gated behind
use_rag/use_cot) that tells the model flags already encode the correct
direction, to reduce how often the retry even triggers - later
strengthened per-flag by _build_directional_anchor_block (round 4 of
results/cot_rag_narrative_accuracy_limitation.md).

Two real false positives were found and fixed while building this, both via
actually re-running the 5-case test rather than trusting the design on
paper - a direct instance of why the sample-based test mattered, not just a
single validated case: (1) "elevated LV volumes ... and reduced ejection
fraction" in one sentence flagged "elevated" as a wrong-direction claim
about the ejection fraction, when it actually modified "volumes", a
different clause - fixed by splitting sentences into clauses (on
comma/and/with/while/but) and requiring the measurement term and the
forbidden word to be in the SAME clause. (2) A sentence naming both
chambers ("mild LV dilation ... but also reveals a slightly reduced RV
ejection fraction") let a same-sentence LV mention validate the ventricle
check for a clause that was actually about the RV, wrongly blaming a
correct "reduced RV EF" statement on the LV-only flag - fixed by resolving
which chamber a clause refers to via "most recent explicit chamber mention
wins" in reading order, instead of "does this ventricle term appear
anywhere in the sentence".

Known remaining gap, found during the same re-run and left unfixed
deliberately: patient109's original failure was a summary that said
"mildly reduced" RV EF, then one sentence later, "no signs of reduced
ejection fraction" - a self-contradiction via NEGATING the correct word in
a different clause, not asserting the wrong word. The clause-level fix
above (needed to kill false positive #1) means the trigger word ("normal",
in "no signs of reduced ejection fraction with normal cardiac function")
and the measurement-bearing clause can end up in different clauses,
missing this specific phrasing. Catching it would need negation-scope
detection, which risks reintroducing false positives - not implemented, in
favor of precision over recall for a check that gates a retry loop (a false
positive here repeatedly rejects an already-correct response, which is
worse than missing one hard-to-detect phrasing). See
src/cot_directional_fix_verification.py and
results/_archive/cot_reliability_test_after_fix.json (superseded by the
later N=25 rounds in results/cot_rag_narrative_accuracy_limitation.md, kept
for the original before/after re-run this was first validated against).
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

from agent_common import AgentContribution, generate_report_medgemma

__all__ = [
    "TraceabilityEntry",
    "SepAgentDecision",
    "build_retrieval_query",
    "build_sep_agent_prompt",
    "parse_sep_agent_response",
    "generate_sep_agent_decision",
    "save_sep_agent_decision",
]


# ----------------------------------------------------------------------
# 0. Directional-consistency check (see module docstring)
# ----------------------------------------------------------------------
# Per-flag config (measure terms, forbidden words, prompt anchor text) lives in
# the clinical knowledge graph (clinical_kg.json, `directional` block on finding
# nodes) - the single source of the direction facts - not hardcoded here. Adding a
# new directional flag (e.g. for a future patient-data agent, or extending
# pathology_<CLASS>_pattern_suspected coverage) means adding its `directional`
# block in that graph, not editing this module. One entry per flag (not two dicts
# kept in sync by hand) so the check's forbidden_words and the prompt anchor's
# "never use" list can never drift apart - see _load_directional_flags_config.
_VENTRICLE_TERM_SETS = {
    "lv": ["left ventric", " lv "],
    "rv": ["right ventric", " rv "],
}
# Single source of the directional facts: the clinical knowledge graph
# (clinical_kg.json), read here with PLAIN json (no networkx - we only need the
# `directional` block on finding nodes, not graph traversal), so sep_agent stays
# importable without networkx. This replaced the former separate
# directional_flags_config.json (migrated so the direction facts live in ONE place
# alongside the graph relations); the check/anchor LOGIC below is unchanged, only
# the data source moved - verified config-identical before the old file was removed.
_CLINICAL_KG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clinical_kg.json")
_REQUIRED_DIRECTIONAL_FLAG_KEYS = {"ventricle_key", "measure_terms", "forbidden_words", "anchor_description"}


def _load_directional_flags_config(path=_CLINICAL_KG_PATH) -> Dict[str, Dict[str, Any]]:
    """
    Build the per-flag directional config from the clinical KG's finding nodes:
    every node carrying a `directional` block yields one entry (flag name ->
    {ventricle_key, measure_terms, forbidden_words, anchor_description}). Same
    output shape the separate directional_flags_config.json used to produce, so
    _check_directional_consistency / _build_directional_anchor_block are unchanged.
    Validated at load time (a malformed block would otherwise KeyError deep inside
    the check).

    Raises:
        ValueError: if any directional block is missing a required key.
    """
    with open(path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    config = {}
    for node in kg.get("nodes", []):
        spec = node.get("directional")
        if spec is None:
            continue
        missing = _REQUIRED_DIRECTIONAL_FLAG_KEYS - spec.keys()
        if missing:
            raise ValueError(f"clinical_kg.json directional block for {node['id']!r} is missing keys: {missing}")
        config[node["id"]] = spec
    return config


_DIRECTIONAL_FLAGS_CONFIG = _load_directional_flags_config()


def _build_directional_anchor_block(flags_for_review) -> str:
    """
    Builds an explicit, per-case reminder line for each directional flag
    actually present in this case (from the KG's `directional` blocks) -
    unlike the generic "flags already indicate direction" rule, this names
    the SPECIFIC flag and the SPECIFIC forbidden words for it, since the
    generic rule alone was not enough to prevent the
    high_lv_ejection_fraction_suspected characterization bias (see round 3
    of results/cot_rag_narrative_accuracy_limitation.md).

    Inputs:
        flags_for_review: iterable of flag strings for this case.

    Output:
        Prompt-ready string block (empty string if no relevant flag is present).
    """
    flags = set(flags_for_review)
    lines = []
    for flag, spec in _DIRECTIONAL_FLAGS_CONFIG.items():
        if flag not in flags:
            continue
        never_words = ", ".join(f'"{w}"' for w in spec["forbidden_words"])
        lines.append(
            f'- Flag "{flag}" is present: {spec["anchor_description"]}. Describe it accordingly - '
            f"never use these words for this specific measurement: {never_words}."
        )
    if not lines:
        return ""
    return "\n\nIMPORTANT - case-specific direction reminders:\n" + "\n".join(lines)


# clause boundaries within a sentence - keeps "elevated LV volumes ... and
# reduced ejection fraction" from letting "elevated" (which modifies
# volumes, a different clause) get blamed on the ejection fraction. Shared
# by both _check_directional_consistency and _check_ef_citation_coverage.
_CLAUSE_SPLIT_RE = re.compile(r",| and | with | while | but ")
_EF_PERCENT_RE = re.compile(r"\d+(\.\d+)?\s*(%|percent)", re.IGNORECASE)
_EF_FEATURE_BY_VENTRICLE = {"lv": "lv_ejection_fraction_pct", "rv": "rv_ejection_fraction_pct"}
# "dilated left and right ventricles" never contains the substring "left
# ventric" (English ellipsis - "left" borrows "ventricles" from the second
# conjunct), so chamber detection would silently miss the LV mention and
# wrongly let an unrelated last_seen value carry over. Expand it first so
# both terms appear literally before any clause splitting/term matching.
_VENTRICLE_ELLIPSIS_RE = re.compile(r"\b(left|right) and (left|right) ventric", re.IGNORECASE)


def _expand_ventricle_ellipsis(text: str) -> str:
    return _VENTRICLE_ELLIPSIS_RE.sub(lambda m: f"{m.group(1)} ventricle and {m.group(2)} ventric", text)


def _mentions(text_lower, terms):
    return any(t in text_lower for t in terms)


def _split_into_clauses(text: str):
    """(sentence, clause) pairs in reading order, for both directional and citation-coverage checks."""
    text = _expand_ventricle_ellipsis(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [(sentence, clause) for sentence in sentences for clause in _CLAUSE_SPLIT_RE.split(sentence)]


def _resolve_ventricle_referents(text: str):
    """
    For each clause in reading order, resolves which ventricle ("lv"/"rv")
    it should be attributed to, or None if genuinely ambiguous. Shared by
    _check_directional_consistency and _check_ef_citation_coverage so both
    checks agree on chamber attribution instead of each re-deriving it
    (which is exactly how the two false positives documented in this
    module's docstring diverged from each other during testing).

    Resolution order per clause, most to least specific:
      1. The clause itself unambiguously names one chamber - use it.
      2. Else, its own sentence unambiguously names exactly one chamber
         elsewhere in the sentence - use that.
      3. Else, if the sentence names BOTH chambers (e.g. "left and right
         ventricles, with an ejection fraction of X%") and this clause
         doesn't specify - genuinely ambiguous (None), NOT "whichever
         chamber happens to be textually closer".
      4. Else (the sentence names neither chamber - a pure anaphoric
         follow-up like "...no signs of reduced ejection fraction")
         inherit whichever chamber was most recently and unambiguously
         resolved in an earlier clause/sentence.

    Output:
        List of (sentence, clause, referent) tuples, referent in {"lv", "rv", None}.
    """
    all_clauses = _split_into_clauses(text)
    sentence_chamber_info = {}
    for sentence, _clause in all_clauses:
        if sentence not in sentence_chamber_info:
            sl = " " + sentence.lower() + " "
            sentence_chamber_info[sentence] = (
                _mentions(sl, _VENTRICLE_TERM_SETS["lv"]),
                _mentions(sl, _VENTRICLE_TERM_SETS["rv"]),
            )

    resolved = []
    last_seen = None
    for sentence, clause in all_clauses:
        c = " " + clause.lower() + " "
        clause_has_lv = _mentions(c, _VENTRICLE_TERM_SETS["lv"])
        clause_has_rv = _mentions(c, _VENTRICLE_TERM_SETS["rv"])
        if clause_has_lv and not clause_has_rv:
            referent = "lv"
        elif clause_has_rv and not clause_has_lv:
            referent = "rv"
        else:
            sent_has_lv, sent_has_rv = sentence_chamber_info[sentence]
            if sent_has_lv and sent_has_rv:
                referent = None
            elif sent_has_lv or sent_has_rv:
                referent = "lv" if sent_has_lv else "rv"
            else:
                referent = last_seen
        if referent is not None:
            last_seen = referent
        resolved.append((sentence, clause, referent))
    return resolved


def _check_directional_consistency(text: str, flags_for_review) -> List[str]:
    """
    Clause-level keyword heuristic (NOT a formal correctness proof): for
    each flag in flags_for_review that has a known direction (from the KG's
    `directional` blocks), scan text for a clause that mentions the
    relevant measurement, is attributed (via _resolve_ventricle_referents)
    to the flag's chamber, and contains a word implying the OPPOSITE
    direction of what the flag already establishes - e.g.
    "high_lv_ejection_fraction_suspected" + an LV-attributed clause
    containing "ejection fraction" and "reduced" is a contradiction, since
    the flag was only raised because that value is elevated, not reduced.

    Inputs:
        text: reasoning or summary text to check (may be None/empty).
        flags_for_review: iterable of flag strings for this case.

    Output:
        List of human-readable violation strings (empty if none found).
    """
    if not text:
        return []

    violations = []
    flags = set(flags_for_review)
    resolved_clauses = _resolve_ventricle_referents(text)

    for flag, spec in _DIRECTIONAL_FLAGS_CONFIG.items():
        if flag not in flags:
            continue
        vkey = spec["ventricle_key"]
        for sentence, clause, referent in resolved_clauses:
            c = " " + clause.lower() + " "
            if not any(m in c for m in spec["measure_terms"]):
                continue
            if vkey is not None and referent != vkey:
                continue
            hits = [w for w in spec["forbidden_words"] if re.search(rf"\b{w}\b", c)]
            if hits:
                violations.append(
                    f'flag "{flag}" implies the opposite of {hits} in: "{clause.strip()}" (sentence: "{sentence.strip()}")'
                )
    return violations


def _check_ef_citation_coverage(text: str, traceability_pairs) -> List[str]:
    """
    Sibling check to _check_directional_consistency, reusing the same
    clause/chamber-resolution machinery (_split_into_clauses, _mentions,
    _VENTRICLE_TERM_SETS) but for a DIFFERENT failure mode: an ejection-
    fraction claim with a concrete percentage that has NO traceability
    entry backing it at all - which _check_directional_consistency cannot
    catch, since it only fires for a flag that's actually present, and a
    value that never crossed a threshold raises no flag (see patient104's
    "RV ejection fraction is slightly low at 65.6%" - 65.6% crosses no
    threshold, so nothing to check the DIRECTION against, but the claim
    also had zero supporting citation - a distinct gap, documented in
    results/cot_rag_narrative_accuracy_limitation.md).

    Deliberately conservative: only fires on a clause containing an actual
    percentage number (not just the bare phrase "ejection fraction"), and
    only when _resolve_ventricle_referents can attribute it to exactly one
    chamber - a genuinely ambiguous clause is skipped rather than guessed,
    to avoid a false-positive class (see that function's docstring for why
    "most recent mention wins" alone isn't safe here, e.g. "left and right
    ventricles, with an ejection fraction of X%").

    Inputs:
        text: reasoning or summary text to check (may be None/empty).
        traceability_pairs: iterable of (agent_id, feature) tuples already
            cited in this response (e.g. from parse_sep_agent_response's
            returned traceability entries).

    Output:
        List of human-readable violation strings (empty if none found).
    """
    if not text:
        return []

    violations = []
    cited_features = {feature for (_agent_id, feature) in traceability_pairs}
    for sentence, clause, referent in _resolve_ventricle_referents(text):
        c = " " + clause.lower() + " "
        if "ejection fraction" not in c and " ef " not in c:
            continue
        if not _EF_PERCENT_RE.search(clause):
            continue  # only claims with an actual number are checked
        if referent is None:
            continue

        expected_feature = _EF_FEATURE_BY_VENTRICLE[referent]
        if expected_feature not in cited_features:
            violations.append(
                f'clause "{clause.strip()}" states a {referent.upper()} ejection fraction value but no '
                f'traceability entry cites "{expected_feature}" (sentence: "{sentence.strip()}")'
            )
    return violations


# ----------------------------------------------------------------------
# 1. Output schema
# ----------------------------------------------------------------------
@dataclass
class TraceabilityEntry:
    """
    One link between a statement in the Sep Agent's integrated summary and
    the specific modality-agent finding it is grounded in.

    -> guarantee that each statement in the final summary is linked to a specific piece of evidence from a specific agent

    Inputs (fields):
        statement: short phrase or clause from the summary that this entry
            justifies.
        agent_id: which modality agent produced the underlying finding
            (e.g. "ecg", "mri") - or the literal string "knowledge" when
            this entry cites a retrieved RAG chunk instead of an agent
            finding (see module docstring's RAG integration section).
        feature: the finding's "feature" name within that agent's
            contribution (see agent_common.Finding.feature) - or
            "knowledge_<i>" (the retrieved chunk's index for this call)
            when agent_id == "knowledge".
    """

    statement: str
    agent_id: str
    feature: str


@dataclass
class SepAgentDecision:
    """
    Structured output of the Sep Agent: an integrated cross-modality
    summary with traceability back to each contributing agent's findings.

    Inputs (fields):
        record_id: identifier for this case, supplied by the caller, or None.
        contributing_agents: list of agent_id values present in this decision
            (e.g. ["ecg", "mri"]).
        summary: short integrated clinical impression synthesized across all
            contributing agents' findings.
        flags_for_review: union of flags_for_review from all input
            contributions, computed deterministically (not LLM-generated),
            so no agent's review flag can be silently dropped in synthesis.
        traceability: list of TraceabilityEntry linking each part of the
            summary back to a specific agent/finding.
        trace: dict capturing what fed into this decision (model, retry
            attempts, which agent_ids/record_ids were combined), for audit.
        reasoning: the model's step-by-step Chain-of-Thought pass, only
            present when generate_sep_agent_decision was called with
            use_cot=True - see module docstring's CoT integration section.
            None otherwise (not just omitted - explicitly None, so callers
            can tell "CoT was off" apart from "CoT ran but produced empty
            text").
    """

    record_id: Optional[str]
    contributing_agents: List[str]
    summary: str
    flags_for_review: List[str]
    traceability: List[TraceabilityEntry]
    trace: Dict[str, Any]
    reasoning: Optional[str] = None


class SepAgentParseError(Exception):
    """Raised when the Sep Agent's LLM response cannot be parsed/validated."""


# ----------------------------------------------------------------------
# 2. Prompt construction
# ----------------------------------------------------------------------
# Curated mapping from a deterministic flag to a natural clinical phrase, so the
# RAG query reads like a real clinical question instead of raw internal jargon
# ("low_lv_ejection_fraction_suspected" -> "reduced left ventricular ejection
# fraction, systolic dysfunction"). An A/B test (results/rag_ab_test/) showed the
# old flag-name-concatenation query retrieved dataset-documentation boilerplate,
# not clinical criteria - the query, not only the knowledge base, was part of the
# problem. Small and auditable, same discipline as flag_validation.py's ground-
# truth code sets; any flag not listed falls back to a plain underscore->space
# rendering so new flags degrade gracefully rather than breaking retrieval.
_FLAG_QUERY_PHRASES = {
    # ECG rate / rhythm / conduction
    "tachycardia_suspected": "sinus tachycardia, elevated heart rate",
    "bradycardia_suspected": "bradycardia, low heart rate",
    "wide_qrs_suspected": "wide QRS complex, bundle branch block, intraventricular conduction delay",
    "irregular_rhythm_suspected": "irregular rhythm, atrial fibrillation",
    # ECG learned-embedding diagnostic superclasses
    "embedding_CD_pattern_suspected": "cardiac conduction disturbance",
    "embedding_HYP_pattern_suspected": "left ventricular hypertrophy, voltage criteria",
    "embedding_MI_pattern_suspected": "myocardial infarction",
    "embedding_NORM_pattern_suspected": "normal ECG",
    "embedding_STTC_pattern_suspected": "ST-T changes, myocardial ischemia",
    # ECG-IMAGE learned classifier diagnostic superclasses (same 5 superclasses
    # as the signal embedding, from the GenECG image classifier - see
    # ecg_image_agent.compute_pathology_features)
    "pathology_CD_pattern_suspected": "cardiac conduction disturbance",
    "pathology_HYP_pattern_suspected": "left ventricular hypertrophy, voltage criteria",
    "pathology_MI_pattern_suspected": "myocardial infarction",
    "pathology_NORM_pattern_suspected": "normal ECG",
    "pathology_STTC_pattern_suspected": "ST-T changes, myocardial ischemia",
    # MRI ejection-fraction flags
    "low_lv_ejection_fraction_suspected": "reduced left ventricular ejection fraction, systolic dysfunction",
    "low_rv_ejection_fraction_suspected": "reduced right ventricular ejection fraction",
    "high_lv_ejection_fraction_suspected": "elevated left ventricular ejection fraction, hyperdynamic left ventricle, hypertrophic cardiomyopathy",
    # MRI learned pathology classifier
    "pathology_DCM_pattern_suspected": "dilated cardiomyopathy",
    "pathology_HCM_pattern_suspected": "hypertrophic cardiomyopathy",
    "pathology_MINF_pattern_suspected": "myocardial infarction with systolic heart failure",
    "pathology_NOR_pattern_suspected": "normal cardiac function",
    "pathology_RV_pattern_suspected": "abnormal right ventricle, right ventricular dysfunction",
    # Structured patient-data flags
    "high_resting_bp_suspected": "hypertension, elevated resting blood pressure",
    "high_cholesterol_suspected": "hypercholesterolemia, elevated serum cholesterol",
    "reduced_exercise_capacity_suspected": "reduced exercise capacity, low maximum heart rate achieved",
    "significant_st_depression_suspected": "ST segment depression, exercise-induced ischemia",
    "exercise_induced_angina_present": "exercise-induced angina pectoris",
    "multiple_vessels_involved_suspected": "coronary artery disease, multi-vessel involvement",
    "abnormal_thallium_result_suspected": "abnormal thallium stress test, myocardial perfusion defect",
}


def build_retrieval_query(contributions: List[AgentContribution]) -> str:
    """
    Build a RAG retrieval query string from a case's deterministic flags,
    phrased as natural clinical language via _FLAG_QUERY_PHRASES (e.g.
    "low_lv_ejection_fraction_suspected" -> "reduced left ventricular ejection
    fraction, systolic dysfunction") so embedding retrieval matches clinical-
    criteria text rather than internal flag jargon. Flags absent from the map
    fall back to a plain underscore->space rendering.

    Inputs:
        contributions: list of AgentContribution instances for this case.

    Output:
        A single query string (str) - empty if no contribution raised any
        flag.
    """
    phrases = []
    for contribution in contributions:
        for flag in contribution.flags_for_review:
            phrase = _FLAG_QUERY_PHRASES.get(flag, flag.replace("_", " "))
            if phrase not in phrases:
                phrases.append(phrase)
    if not phrases:
        return ""
    return "Clinical significance and diagnostic criteria for " + "; ".join(phrases) + "."


def build_sep_agent_prompt(contributions: List[AgentContribution], retrieved_chunks=None, use_cot=False, integrate_criteria=False, use_kg=False) -> str:
    """
    Build the prompt asking the LLM to synthesize an integrated impression
    across all contributing agents' findings, with mandatory per-statement
    traceability back to a specific (agent_id, feature) pair - or, when
    retrieved_chunks is given, to a retrieved knowledge chunk instead.

    Inputs:
        contributions: list of AgentContribution instances, one per
            modality agent for this case.
        retrieved_chunks: optional list of chunk dicts as returned by
            rag_knowledge_base.retrieve() (each with "text", "source",
            "header", "score"). None (the default) omits the knowledge
            section entirely - existing callers/behavior unaffected.
        use_cot: if True, the schema gains a "reasoning" field the model
            must fill before "summary", with explicit step-by-step
            instructions - see module docstring's CoT integration section.
            False (the default) omits it entirely - existing callers
            unaffected.

    Output:
        Prompt string ready to send to an LLM backend.
    """
    sections = []
    for contribution in contributions:
        lines = [f"=== {contribution.agent_id.upper()} agent (record {contribution.record_id!r}) ==="]
        if not contribution.findings:
            lines.append("(no findings - insufficient signal for this modality)")
        for finding in contribution.findings:
            lines.append(
                f'- feature="{finding.feature}": {finding.value} '
                f"(confidence {finding.confidence:.2f}) - {finding.rationale}"
            )
        if contribution.flags_for_review:
            lines.append(f"Flags raised by this agent: {contribution.flags_for_review}")
        sections.append("\n".join(lines))

    findings_block = "\n\n".join(sections)
    valid_pairs = [f'({c.agent_id}, "{f.feature}")' for c in contributions for f in c.findings]
    all_flags = [flag for c in contributions for flag in c.flags_for_review]
    directional_anchor_block = _build_directional_anchor_block(all_flags)

    # Consumer #3: cross-modality convergence from the clinical knowledge graph
    # (opt-in via use_kg; lazy import so sep_agent stays importable without
    # networkx unless this is actually used). Surfaces diagnoses that >= 2
    # DIFFERENT modality agents' flags converge on - something concatenating
    # findings in the prompt cannot make explicit.
    convergence_block = ""
    if use_kg:
        from knowledge_graph import convergent_diagnoses, load_kg
        conv = convergent_diagnoses(load_kg(), all_flags)
        if conv:
            clines = ["=== Cross-modality convergent evidence (clinical knowledge graph) ==="]
            for _d, e in sorted(conv.items(), key=lambda kv: -len(kv[1]["agents"])):
                agents = ", ".join(sorted(e["agents"]))
                fnds = ", ".join(e["supporting_findings"])
                clines.append(f"- {e['label']}: independently supported by {agents} ({fnds})")
            clines.append("Convergence across independent modalities is STRONGER evidence than any "
                          "single modality - weight it accordingly in the impression.")
            convergence_block = "\n\n" + "\n".join(clines)

    knowledge_block = ""
    correctness_rule = ""
    if retrieved_chunks:
        knowledge_lines = ["=== Retrieved clinical knowledge (cite as agent_id=\"knowledge\") ==="]
        for i, chunk in enumerate(retrieved_chunks):
            knowledge_lines.append(f'- knowledge_{i} (source: {chunk["source"]}): {chunk["text"]}')
            valid_pairs.append(f'(knowledge, "knowledge_{i}")')
        knowledge_block = "\n\n" + "\n".join(knowledge_lines)
        correctness_rule = (
            "\n- Every qualitative claim in the summary (e.g. \"low\", \"high\", \"elevated\", \"reduced\", "
            "\"suspected\") must be grounded in a cited knowledge_<i> chunk that actually defines the "
            "relevant threshold or criterion - not just the raw numeric finding alone. If no retrieved "
            "chunk supports a qualitative claim you would otherwise make, state the number without the "
            "qualitative label rather than asserting it ungrounded."
        )
        # integrate_criteria: opt-in experiment knob (default False, no behavior
        # change). TRIED AND NOT ADOPTED - an A/B on the 3-modality case
        # (results/rag_ab_test/g2_ab_result.json) found this stronger directive
        # did NOT make MedGemma 4B weave the criterion into the prose AND
        # degraded grounding (knowledge citations dropped from 7 to 0): the
        # deep-integration gap is bounded by the small model's capability, not by
        # the prompt. Kept as a documented knob, same convention as the
        # temperature/seed experiment.
        if integrate_criteria:
            correctness_rule += (
                "\n- INTEGRATION: when a cited knowledge_<i> chunk defines a threshold or category that a "
                "numeric finding meets or crosses, write that link EXPLICITLY IN the summary sentence - the "
                "measured value together with the threshold/classification it satisfies "
                "(e.g. \"LVEF 23.7%, below the 40% HFrEF threshold, i.e. reduced ejection fraction\"), "
                "rather than stating the value and the criterion as two separate disconnected statements."
            )

    valid_pairs_str = ", ".join(valid_pairs)

    cot_task_instructions = ""
    reasoning_schema_line = ""
    reasoning_rule = ""
    if use_cot:
        cot_task_instructions = """

Before writing the summary, think step by step and write out that reasoning
in the "reasoning" field:
1. What does each contributing agent's findings suggest INDEPENDENTLY, on their own?
2. Do the agents agree, or does one modality's evidence conflict with or fail to
   confirm another's? State this explicitly, even if they simply agree.
3. If knowledge chunks were retrieved, which ones actually apply to this case's
   findings, and what do they establish?
4. Only after that reasoning, decide what the integrated summary should say."""
        reasoning_schema_line = '\n  "reasoning": "<step-by-step reasoning per the 4 points above>",'
        reasoning_rule = (
            "\n- The \"reasoning\" field is required and must be filled in BEFORE \"summary\" conceptually "
            "(it comes first in the JSON) - it is an audit trail, not shown to the end user, so it can be "
            "longer and more exploratory than the summary itself."
        )

    return f"""You are the Sep Agent, an aggregating component that combines structured
findings from independent modality-specific agents (ECG, MRI, ...) into one
integrated clinical impression. Each agent already analyzed its own modality
in isolation and reported findings with a rationale and confidence - you do
NOT re-derive anything from raw signals or images, you only combine and
contextualize what has already been reported.

Findings from each contributing agent for this case:

{findings_block}{knowledge_block}{convergence_block}{directional_anchor_block}

Task: write a short (3-5 sentence) integrated clinical impression that
synthesizes across all agents' findings. Do not introduce any clinical fact
that is not present in the findings above. If agents disagree or an
individual finding has low confidence, reflect that uncertainty rather than
asserting certainty.{cot_task_instructions}

Respond with ONLY a single JSON object (no markdown fences, no extra text)
with this exact shape:
{{{reasoning_schema_line}
  "summary": "<3-5 sentence integrated clinical impression>",
  "traceability": [
    {{"statement": "<short phrase from the summary>", "agent_id": "<agent_id or 'knowledge'>", "feature": "<exact feature name or knowledge_<i>>"}}
  ]
}}

Rules:
- Every entry in "traceability" must reference one of these exact (agent_id, feature) pairs: {valid_pairs_str}
- Every part of the summary that states a specific fact must have a corresponding traceability entry.
- The flags listed above (e.g. "high_..._suspected" vs "low_..._suspected") already indicate the correct
  direction for that measurement - describe it consistently with the flag, do not independently judge from
  the raw number whether it counts as high or low.{correctness_rule}{reasoning_rule}
- Do not include any field or text outside this JSON object.
"""


# ----------------------------------------------------------------------
# 3. Response parsing and validation
# ----------------------------------------------------------------------
def parse_sep_agent_response(raw_text: str, valid_pairs, require_reasoning: bool = False) -> tuple:
    """
    Parse and validate the Sep Agent LLM response into a summary string, a
    list of TraceabilityEntry, and an optional reasoning string, rejecting
    responses that cite an (agent_id, feature) pair that wasn't actually
    among the input findings.

    Inputs:
        raw_text: raw string returned by the LLM backend.
        valid_pairs: set of (agent_id, feature) tuples that are valid
            traceability targets (built from the actual input contributions).
        require_reasoning: if True (i.e. the prompt was built with
            use_cot=True), a missing/empty "reasoning" field is treated as a
            parse failure - ***forces a retry rather than silently accepting a
            response that skipped the CoT step***. False (the default) never
            requires it.

    Output:
        (summary, traceability, reasoning) tuple: summary is a str,
        traceability is a list of TraceabilityEntry instances, reasoning is
        a str or None (None when require_reasoning=False and the field
        wasn't present).

    Raises:
        SepAgentParseError: if the text is not valid JSON, is missing
        required fields, a traceability entry cites an unknown
        (agent_id, feature) pair, or require_reasoning=True and "reasoning"
        is missing/empty.
    """
    import json

    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SepAgentParseError(f"Response is not valid JSON: {exc}") from exc

    if "summary" not in payload or "traceability" not in payload:
        raise SepAgentParseError("Response JSON is missing 'summary' or 'traceability'")

    if require_reasoning and not payload.get("reasoning"):
        raise SepAgentParseError("Response JSON is missing required 'reasoning' field (use_cot=True)")

    traceability = []
    for item in payload["traceability"]:
        missing = {"statement", "agent_id", "feature"} - item.keys()
        if missing:
            raise SepAgentParseError(f"Traceability entry is missing fields: {missing}")
        pair = (item["agent_id"], item["feature"])
        if pair not in valid_pairs:
            raise SepAgentParseError(
                f"Traceability entry cites unknown (agent_id, feature) pair {pair}; "
                f"expected one of {sorted(valid_pairs)}"
            )
        traceability.append(
            TraceabilityEntry(
                statement=item["statement"], agent_id=item["agent_id"], feature=item["feature"]
            )
        )

    return payload["summary"], traceability, payload.get("reasoning")


# ----------------------------------------------------------------------
# 4. Orchestration
# ----------------------------------------------------------------------
def generate_sep_agent_decision(
    contributions: List[AgentContribution],
    llm_call: Optional[Callable[[str], str]] = None,
    record_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 1,
    use_rag: bool = False,
    top_k_knowledge: int = 5,
    kb_dir: str = "results/rag_knowledge_base",
    use_cot: bool = False,
    integrate_criteria: bool = False,
    use_kg: bool = False,
) -> SepAgentDecision:
    """
    Combine multiple modality agents' AgentContribution objects into a
    single SepAgentDecision: a deterministic union of flags_for_review, plus
    an LLM-synthesized integrated summary with per-statement traceability
    back to specific agent findings. Retries on a parse/validation failure
    by appending the error to the prompt and asking the model to correct it
    - this now also includes a directional-consistency failure (see module
    docstring), not just a fabricated/unknown traceability citation.

    Inputs:
        contributions: list of AgentContribution instances, one per
            modality agent for this case (e.g. one "ecg" and one "mri").
        llm_call: callable taking a single prompt string and returning the
            raw text response. Defaults to agent_common.generate_report_medgemma
            (free, local, no API key).
        record_id: optional identifier for this case, stored in the output.
        model_name: optional string identifying the model/backend used,
            stored in the trace for auditability.
        max_retries: number of extra attempts if the LLM response fails to
            parse/validate.
        use_rag: if True, retrieves relevant chunks from the RAG knowledge
            base (rag_knowledge_base.retrieve, query built by
            build_retrieval_query from this case's flags) and injects them
            into the prompt - see module docstring's RAG integration
            section. Defaults to False: existing callers (e.g.
            report_evaluation.py's ECG-only scoring pipeline) keep their
            exact current behavior unless they explicitly opt in.
        top_k_knowledge: number of chunks to retrieve when use_rag=True.
        kb_dir: directory holding the saved RAG index (chunks.json,
            embeddings.npy), only used when use_rag=True.
        use_cot: if True, requires the model to produce a step-by-step
            "reasoning" field before the summary - see module docstring's
            CoT integration section. Defaults to False: existing callers
            keep their exact current behavior unless they explicitly opt in.

    Output:
        SepAgentDecision instance. When use_rag=True, trace["retrieved_knowledge"]
        holds the retrieved chunks (text, source, score) for auditability.
        When use_cot=True, decision.reasoning holds the model's step-by-step
        pass; otherwise decision.reasoning is None.

    Raises:
        SepAgentParseError: if no valid response is obtained within
        max_retries + 1 attempts.
    """
    if llm_call is None:
        llm_call = generate_report_medgemma
        if model_name is None:
            model_name = "ollama:medgemma:4b"

    deterministic_flags = sorted(
        {flag for c in contributions for flag in c.flags_for_review}
    )
    valid_pairs = {(c.agent_id, f.feature) for c in contributions for f in c.findings}

    retrieved_chunks = None
    if use_rag:
        from rag_knowledge_base import retrieve

        query = build_retrieval_query(contributions)
        retrieved_chunks = retrieve(query, top_k=top_k_knowledge, kb_dir=kb_dir) if query else []
        valid_pairs |= {("knowledge", f"knowledge_{i}") for i in range(len(retrieved_chunks))}

    prompt = build_sep_agent_prompt(contributions, retrieved_chunks=retrieved_chunks, use_cot=use_cot, integrate_criteria=integrate_criteria, use_kg=use_kg)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raw_response = llm_call(prompt)
        try:
            summary, traceability, reasoning = parse_sep_agent_response(
                raw_response, valid_pairs, require_reasoning=use_cot
            )
            traceability_pairs = [(e.agent_id, e.feature) for e in traceability]
            violations = (
                _check_directional_consistency(summary, deterministic_flags)
                + _check_directional_consistency(reasoning or "", deterministic_flags)
                + _check_ef_citation_coverage(summary, traceability_pairs)
                + _check_ef_citation_coverage(reasoning or "", traceability_pairs)
            )
            if violations:
                raise SepAgentParseError(
                    "Directional/citation consistency check failed: " + "; ".join(violations)
                )
            trace = {
                "model": model_name,
                "attempts": attempt + 1,
                "source_record_ids": {c.agent_id: c.record_id for c in contributions},
            }
            if use_rag:
                trace["retrieved_knowledge"] = retrieved_chunks
            if use_kg:
                # Deterministic, LLM-independent cross-modality convergence (same
                # spirit as flags_for_review): auditable regardless of the prose.
                # Both the specific-DIAGNOSIS level and the CATEGORY level (PTB-XL
                # superclass, via the `is_a` hierarchy - catches agreement between a
                # specific diagnosis in one modality and a superclass pattern in
                # another, incl. the deterministic-vs-embedding ECG pathways).
                from knowledge_graph import convergent_categories, convergent_diagnoses, load_kg
                _g = load_kg()
                trace["kg_convergent_diagnoses"] = {
                    d: {"label": e["label"], "agents": sorted(e["agents"]),
                        "supporting_findings": e["supporting_findings"]}
                    for d, e in convergent_diagnoses(_g, deterministic_flags).items()
                }
                trace["kg_convergent_categories"] = {
                    c: {"label": e["label"], "agents": sorted(e["agents"]),
                        "supporting_findings": e["supporting_findings"]}
                    for c, e in convergent_categories(_g, deterministic_flags).items()
                }
            return SepAgentDecision(
                record_id=record_id,
                contributing_agents=[c.agent_id for c in contributions],
                summary=summary,
                flags_for_review=deterministic_flags,
                traceability=traceability,
                trace=trace,
                reasoning=reasoning,
            )
        except SepAgentParseError as exc:
            last_error = exc
            prompt = (
                prompt
                + f"\n\nYour previous response could not be parsed: {exc}. "
                "Return ONLY the corrected JSON object, nothing else."
            )

    raise SepAgentParseError(
        f"Failed to obtain a valid Sep Agent decision after "
        f"{max_retries + 1} attempts: {last_error}"
    )


# ----------------------------------------------------------------------
# 5. Decision persistence
# ----------------------------------------------------------------------
def save_sep_agent_decision(decision: SepAgentDecision, output_dir=None, file_name="sep_agent_decision"):
    """
    Serialize a SepAgentDecision to a JSON file for later auditing/discussion
    (e.g. with a supervisor) - mirrors agent_common.save_contribution, kept
    as its own function so the type/name stays accurate for a different
    object shape (a cross-modality decision, not a single agent's findings).

    Inputs:
        decision: SepAgentDecision instance to save.
        output_dir: directory to write into; defaults to "./results" next to
            this module if not provided.
        file_name: base file name (without extension) for the JSON file.

    Output:
        Path to the written JSON file (str).
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{file_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(decision), f, indent=2, ensure_ascii=False)

    return output_path
