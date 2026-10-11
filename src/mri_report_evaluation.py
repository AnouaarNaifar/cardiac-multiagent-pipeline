"""
MRI report evaluation (BLEU / ROUGE / clinical cosine) - STRUCTURE READY, references PENDING
============================================================================================
Counterpart to report_evaluation.py (ECG) for the MRI agent: it scores the
MRI agent's generated narrative against real cardiologist REFERENCE reports,
using the exact same three metrics as the ECG evaluation - BLEU, ROUGE-1,
and a biomedical whole-report cosine similarity - so the two modalities are
directly comparable.

    generated MRI narrative  vs  reference MRI report   ->  BLEU / ROUGE-1 / clinical cosine

To stay identical to the ECG pipeline and avoid duplicating validated code,
the three scorers and the biomedical embedding model are IMPORTED from
report_evaluation.py (compute_bleu, compute_rouge,
compute_clinical_cosine_similarity_batch, load_clinical_embedding_model) -
nothing is re-implemented here. Like report_evaluation.py, the MRI agent
never emits free text (only Finding objects), so scoreable text is obtained
by routing the MRI contribution through the Sep Agent with ONLY the MRI
contribution present (generate_mri_text_for_scoring below), exactly as the
ECG module routes an ECG-only contribution.

=============================== BLOCKER (read me) ===============================
Unlike PTB-XL, the ACDC dataset ships NO free-text cardiologist report - only a
pathology group label (NOR/DCM/HCM/MINF/RV), segmentation masks, and a few
scalars (ED/ES frames, height/weight). There is therefore currently NO
reference-report corpus to score against, which is why this module is a READY
STRUCTURE, not a finished evaluation: everything is wired EXCEPT the reference
source, which plugs into load_mri_reference_reports() below.

Candidate reference sources to look for / build (fill load_mri_reference_reports
once one is available):
  1. An ACDC-linked radiology/cardiology report corpus, if one is released or
     obtained (the most faithful option - real expert reports for these exact
     patients).
  2. Another cardiac-MRI dataset that DOES pair volumes with free-text reports
     (e.g. a MICCAI/PhysioNet MRI-report set), re-running the agent on those
     patients so image and reference come from the same record.
  3. Templated ``reference'' text synthesised from the ACDC ground-truth labels
     (group + expert EF), used only as a weak, clearly-labelled proxy baseline -
     NOT a substitute for real reports (documented as such if used).

Until one of these is wired, run_mri_evaluation() stops with an explicit message
rather than fabricating references or scores.
================================================================================

Dependencies: reuses report_evaluation.py's (sacrebleu, rouge-score,
sentence-transformers). No new dependency.
"""
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# --- reuse the ECG evaluation's validated scorers + embedding loader ---------
# (import, never re-implement - keeps the two modalities byte-comparable)
from report_evaluation import (
    compute_bleu,
    compute_rouge,
    compute_clinical_cosine_similarity_batch,
    load_clinical_embedding_model,
)
from agent_common import AgentContribution, save_contribution
from mri_preprocessing import preprocess_patient
from mri_agent import generate_mri_contribution
from sep_agent import generate_sep_agent_decision

# ACDC's five pathology groups, used for stratified sampling (mirrors the ECG
# module's diagnostic-superclass stratification so per-group skew doesn't bias
# aggregate scores).
ACDC_GROUPS = ["NOR", "DCM", "HCM", "MINF", "RV"]


# ----------------------------------------------------------------------
# 1. Reference-report source  <-- THE ONLY MISSING PIECE (see BLOCKER above)
# ----------------------------------------------------------------------
def load_mri_reference_reports(reference_source: Optional[str] = None) -> Dict[str, str]:
    """
    Return {patient_id: reference_report_text} for ACDC patients.

    *** NOT YET IMPLEMENTED - this is the plug-in point. ***
    ACDC provides no free-text report (see the module BLOCKER note), so there is
    no corpus to load yet. When a reference source is found/built, implement one
    of the candidate loaders here (real ACDC-linked reports, another
    MRI+report dataset, or a clearly-labelled templated proxy) and return the
    mapping. Until then this returns {} so run_mri_evaluation() reports the gap
    explicitly instead of inventing references.

    Inputs:
        reference_source: path/identifier of the reference corpus once available
            (unused until a loader is implemented).

    Output:
        dict {patient_id -> reference report text}. Empty until wired.
    """
    # TODO: implement once a reference-report corpus is available.
    return {}


# ----------------------------------------------------------------------
# 2. Sampling (group-stratified, mirrors the ECG stratified sampler)
# ----------------------------------------------------------------------
def stratified_sample_patients(
    reference_ids: List[str],
    patient_group: Dict[str, str],
    n_samples: int,
    random_seed: int = 42,
) -> List[str]:
    """
    Sample ~equally across the five ACDC pathology groups from the patients that
    actually have a reference report, so shorter/templated groups don't dominate
    the aggregate (same rationale as report_evaluation.stratified_sample).
    """
    rng = np.random.default_rng(random_seed)
    by_group: Dict[str, List[str]] = {g: [] for g in ACDC_GROUPS}
    for pid in reference_ids:
        g = patient_group.get(pid)
        if g in by_group:
            by_group[g].append(pid)

    per_group = max(1, n_samples // len(ACDC_GROUPS))
    picked: List[str] = []
    for g in ACDC_GROUPS:
        pool = by_group[g]
        if not pool:
            continue
        k = min(per_group, len(pool))
        picked.extend(rng.choice(pool, size=k, replace=False).tolist())
    return picked[:n_samples]


# ----------------------------------------------------------------------
# 3. Contribution + text-for-scoring (mirrors the ECG pipeline exactly)
# ----------------------------------------------------------------------
def get_or_create_mri_contribution(
    patient_dir: str,
    record_id: str,
    contributions_dir: str,
    llm_call: Optional[Callable[..., str]] = None,
    image_mode: str = "ed_only",
) -> AgentContribution:
    """
    Run the MRI agent for one ACDC patient (preprocess_patient ->
    generate_mri_contribution) and cache the AgentContribution JSON, so a
    re-run of the evaluation does not re-call the model. Mirrors
    report_evaluation.get_or_create_ecg_contribution.
    """
    os.makedirs(contributions_dir, exist_ok=True)
    cache_path = os.path.join(contributions_dir, f"{record_id}.json")
    # NB: a load_contribution helper could restore from cache_path here; kept
    # minimal on purpose - re-running regenerates unless a loader is added.
    preprocessing_result = preprocess_patient(patient_dir)
    contribution = generate_mri_contribution(
        preprocessing_result,
        llm_call=llm_call,
        record_id=record_id,
        image_mode=image_mode,
    )
    save_contribution(contribution, output_dir=contributions_dir, file_name=record_id)
    return contribution


def generate_mri_text_for_scoring(mri_contribution: AgentContribution, record_id: str) -> str:
    """
    Turn an MRI AgentContribution into scoreable free text by routing it through
    the Sep Agent with ONLY the MRI contribution present - identical pattern to
    report_evaluation.generate_text_for_scoring (which does this for ECG), so the
    MRI-only summary cannot reference any other modality.
    """
    decision = generate_sep_agent_decision([mri_contribution], record_id=record_id)
    return decision.summary


# ----------------------------------------------------------------------
# 4. End-to-end evaluation (ready; stops cleanly until references exist)
# ----------------------------------------------------------------------
def run_mri_evaluation(
    acdc_dir: str,
    reference_source: Optional[str] = None,
    n_samples: int = 20,
    output_dir: str = "results/mri_report_evaluation",
    image_mode: str = "ed_only",
    random_seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Full MRI report-evaluation run, structurally identical to
    report_evaluation.run_evaluation:
      load reference reports -> group-stratified sample -> for each patient:
      MRI agent contribution -> Sep-Agent MRI-only summary -> score with
      BLEU / ROUGE-1 / clinical cosine -> save evaluation_results.csv
      (+ MEAN/STD) and evaluation_interpretation.md.

    Returns the per-record results DataFrame, or None (with an explanatory
    message) if no reference reports are available yet - the module never
    fabricates references or scores.
    """
    os.makedirs(output_dir, exist_ok=True)
    contributions_dir = os.path.join(output_dir, "contributions")

    references = load_mri_reference_reports(reference_source)
    if not references:
        print(
            "[mri_report_evaluation] No reference reports available yet - ACDC ships "
            "no free-text report. Implement load_mri_reference_reports() with a "
            "reference corpus (see the module BLOCKER note), then re-run. "
            "Structure is ready; nothing was fabricated."
        )
        return None

    # Map patient_id -> pathology group (from each patient's Info.cfg) for the
    # stratified sampler. Only patients that have a reference are considered.
    from mri_preprocessing import load_patient_info
    patient_group: Dict[str, str] = {}
    for pid in references:
        info_cfg = os.path.join(acdc_dir, pid, "Info.cfg")
        if os.path.exists(info_cfg):
            patient_group[pid] = load_patient_info(info_cfg).get("Group", "UNKNOWN")

    sampled = stratified_sample_patients(
        list(references.keys()), patient_group, n_samples, random_seed
    )

    rows: List[Dict[str, Any]] = []
    for pid in sampled:
        try:
            patient_dir = os.path.join(acdc_dir, pid)
            contribution = get_or_create_mri_contribution(
                patient_dir, pid, contributions_dir, image_mode=image_mode
            )
            generated = generate_mri_text_for_scoring(contribution, pid)
            reference = references[pid]
            rouge = compute_rouge(generated, reference)
            rows.append({
                "patient_id": pid,
                "group": patient_group.get(pid, "UNKNOWN"),
                "generated_text": generated,
                "reference_report": reference,
                "bleu": compute_bleu(generated, reference),
                "rouge1": rouge.get("rouge1", np.nan),
            })
        except Exception as exc:  # noqa: BLE001 - per-record failures logged, not fatal
            print(f"[mri_report_evaluation] {pid} failed: {exc}")

    if not rows:
        print("[mri_report_evaluation] No records scored (all failed).")
        return None

    results = pd.DataFrame(rows)

    # Clinical cosine is batched once per run (as in the ECG module).
    model = load_clinical_embedding_model()
    results["clinical_cosine"] = compute_clinical_cosine_similarity_batch(
        results["generated_text"].tolist(), results["reference_report"].tolist(), model
    )

    # Append MEAN/STD rows, mirroring report_evaluation's output layout.
    summary = pd.DataFrame({
        "patient_id": ["MEAN", "STD"],
        "bleu": [results["bleu"].mean(), results["bleu"].std()],
        "rouge1": [results["rouge1"].mean(), results["rouge1"].std()],
        "clinical_cosine": [results["clinical_cosine"].mean(), results["clinical_cosine"].std()],
    })
    out = pd.concat([results, summary], ignore_index=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    out.to_csv(csv_path, index=False)

    _write_mri_interpretation(results, output_dir)
    print(f"[mri_report_evaluation] Wrote {csv_path} ({len(results)} records scored).")
    return out


def _write_mri_interpretation(results: pd.DataFrame, output_dir: str) -> str:
    """Write a short interpretation note mirroring the ECG evaluation's .md."""
    path = os.path.join(output_dir, "evaluation_interpretation.md")
    lines = [
        "# MRI report evaluation",
        "",
        f"Scored {len(results)} ACDC patients, MRI agent narrative vs. reference report.",
        "",
        "| Metric | Mean | Std |",
        "|---|---|---|",
        f"| BLEU | {results['bleu'].mean():.2f} | {results['bleu'].std():.2f} |",
        f"| ROUGE-1 | {results['rouge1'].mean():.3f} | {results['rouge1'].std():.3f} |",
        f"| Clinical cosine | {results['clinical_cosine'].mean():.3f} | {results['clinical_cosine'].std():.3f} |",
        "",
        "Same three metrics as the ECG evaluation (report_evaluation.py), so the "
        "modalities are directly comparable. The same caveats apply: BLEU/ROUGE "
        "measure surface overlap, not clinical correctness; the trustworthy MRI "
        "accuracy evidence is the ejection-fraction flag validation and the "
        "segmentation Dice, not this text table.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    run_mri_evaluation(acdc_dir="ACDC/database/testing")
