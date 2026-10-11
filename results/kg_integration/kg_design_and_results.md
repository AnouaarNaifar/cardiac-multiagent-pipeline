# Clinical knowledge graph (KG) for the Sep Agent — design & results

Opt-in structured-reasoning layer for the cross-modality aggregator, complementing (not
replacing) the text-retrieval RAG layer. Addresses limitations RAG cannot: structured
finding→criterion→diagnosis relations, explicit cross-modality convergence, and principled
directional-consistency checking.

## Approach (why this architecture)

- **Engine: `networkx`** (light, pure-Python, no server) — gives graph traversal / paths /
  reachability for free, instead of re-implementing them over raw dicts.
- **Source: a human-editable JSON** (`src/clinical_kg.json`) loaded into the graph at runtime
  — auditable and version-controllable, same spirit as the RAG index (source → `chunks.json`)
  and `directional_flags_config.json`.
- **Not Neo4j / RDF**: a graph database (Cypher/SPARQL, external service) is the production
  choice at scale but overkill and against this project's local/lightweight ethos for a scoped
  prototype. Noted as a future evolution, not used here.
- **Node ids for findings = the agents' actual flag names**, so the graph connects directly to
  real `AgentContribution` output. Node types: `finding → criterion → diagnosis → severity`;
  edge relations: `meets`, `indicates`, `supports`, `has_severity`.

## Two scoped consumers

1. **Principled directional-consistency check** (`check_directional_consistency_kg`) — replaces
   the brittle keyword heuristic (`sep_agent._check_directional_consistency`) with a check
   driven by each finding node's own `measurement` / `direction` / `incompatible_terms`
   attributes. Clause-scoped to avoid cross-clause false positives.
2. **Cross-modality convergence** (`convergent_diagnoses`) — surfaces the diagnoses supported by
   findings from ≥2 DIFFERENT modality agents, which concatenating findings in the prompt cannot
   make explicit. Fed to the Sep Agent as a structured "convergent evidence" block.

## Foundation — validated (n=1 structured test, no LLM)

`src/knowledge_graph.py` + `src/clinical_kg.json` (18 nodes, 13 edges). On the 3-modality active
findings `{lvh_sokolow_lyon_suspected (ecg), pathology_HCM_pattern_suspected (mri),
high_resting_bp_suspected (patient), low_lv_ejection_fraction_suspected (mri)}`:

- **Cross-modality convergence works**: `lv_hypertrophy` ← ECG (Sokolow) **+** MRI (HCM);
  `hypertensive_heart_disease` ← ECG **+** MRI **+** patient (3 modalities). Non-convergent
  diagnoses (`hcm`, `hfref`, single-agent) correctly excluded.
- **Reasoning path** is interpretable: *"LVH by Sokolow-Lyon → Sokolow-Lyon ≥ 3.5 mV → left
  ventricular hypertrophy"* — an auditable explanation, stronger than the current traceability
  (which only guarantees a cited finding is real).
- **Directional check works**: with an active `low_lv_ejection_fraction_suspected`, the clause
  "the ejection fraction is normal" is flagged as a contradiction; "the ejection fraction is
  markedly reduced" passes. Driven purely by KG node attributes, no hardcoded per-flag rule.

## Consumer #3 — wired + A/B (POSITIVE, both models)

Wired into `sep_agent` (opt-in `use_kg`, lazy import): the KG's convergent diagnoses are injected
as a "cross-modality convergent evidence" block. A/B on a synthetic convergent-hypertrophy case
(ECG Sokolow-LVH + MRI HCM-pattern + patient elevated-BP → the KG computes `lv_hypertrophy` [ecg+mri]
and `hypertensive_heart_disease` [ecg+mri+patient] as convergent), same contributions, fixed seed,
KG off vs on:

| Model | A (use_kg=False) | B (use_kg=True) |
|---|---|---|
| **MedGemma 4B** (local default) | lists the findings; names no convergent diagnosis | adds *"These findings are consistent with **hypertensive heart disease**"* — names the KG's 3-modality convergent node |
| **gpt-oss-120b** (Groq) | lists findings; no convergence language | *"MRI… **further supporting LVH**… **indicating hypertensive heart disease**. **Convergent evidence across ECG, MRI, and blood pressure** strengthens the impression… though MRI raises a possibility of primary HCM"* — full convergence synthesis + clinical nuance |

**Result: the KG convergence block genuinely changes the synthesis** — the model names the
cross-modality convergent diagnosis it otherwise omits.

**Robustness on MedGemma 4B — reproducible, not a lucky seed** (`experiments/kg_robustness.py`, 5 seeds
per condition, counting whether the summary names the convergent "hypertensive heart disease"):

| 4B, 5 seeds | use_kg=False | use_kg=True |
|---|---|---|
| clean case (no EF finding) | 0/5 | **5/5** |
| noisy case (extra EF finding) | 0/5 | **4/5** |

**9/10 with the KG name the convergent diagnosis vs 0/10 without** — a robust, reproducible effect on
the local 4B, in both a clean and a noisier case. (An earlier single WITH-EF run that did *not* name it
was the 1/5 noisy-case exception, not the norm.) This is notably **stronger than the G2 result**, where
the 4B could not use *injected text criteria* at all: here the hard reasoning (convergence) is done by
the graph deterministically, so the small model only has to narrate a pre-computed fact - which it does
reliably. gpt-oss-120b uses it even more richly (explicit "convergent evidence across ECG, MRI, and BP"
+ HCM-vs-hypertensive nuance). The **most reliable layer is model-independent**: the deterministic
`convergent_diagnoses` output (trace + PDF section) holds regardless of what any model writes.
Reproduce: `experiments/kg_ab.py` / `kg_robustness.py` (MedGemma), same with
`generate_report_groq(model="openai/gpt-oss-120b")`.

## Consumer #1 — directional facts: MIGRATED so the KG is the single source (logic unchanged)

Two steps here. First, a standalone KG-native check (`check_directional_consistency_kg`) was compared
against the existing keyword heuristic on the documented failure cases and found **detection-equivalent**
(catches patient102, misses patient109's negation-scope case - both share the clause-level limitation).
No detection gain → not worth replacing the proven logic with new logic.

But the *architectural* win (one source of truth) was worth capturing without the regression risk. So
instead of swapping the logic, the **data source** was migrated: each directional finding node now carries
a `directional` block (`ventricle_key`, `measure_terms`, `forbidden_words`, `anchor_description`), and
`sep_agent._load_directional_flags_config` reads these (plain json, no networkx) instead of a separate
`directional_flags_config.json`, which was **deleted**. The battle-tested check + anchor LOGIC
(`_check_directional_consistency` with its ventricle-referent resolution) is **unchanged**.

**Zero-regression proof:** before deleting the old file, the KG-derived config was verified
**byte-identical** to it (`old == new` True, same 5 flags incl. bradycardia), so the check behaves
identically by construction; functionally re-confirmed on patient102 (flags), the correct case (clean),
and bradycardia (now covered). Net: all clinical knowledge - the diagnosis relations AND the directional
facts - now lives in one auditable graph, with no duplicated/second source.

## Deterministic convergence output (shipped)

Beyond the prompt injection (whose prose effect is model-dependent, though positive even at 4B),
the convergent diagnoses are also stored **deterministically** on the decision's `trace`
(`trace["kg_convergent_diagnoses"]`) when `use_kg=True` — an auditable, LLM-independent structured
output, in the same spirit as `flags_for_review` being a deterministic union rather than
LLM-generated. So the cross-modality convergence is reliable regardless of whether the model weaves
it into the prose.

## Pathology hierarchy (taxonomy) — added

The KG originally did only evidential reasoning (finding→criterion→diagnosis, and evidential
diagnosis→diagnosis `supports` edges). A **taxonomy layer** was then added: `category` nodes for the
PTB-XL diagnostic superclasses (CD/HYP/MI/STTC/NORM) plus a clinical `heart failure` category, linked
by an **`is_a`** relation (e.g. `old_inferior_mi is_a MI`, `lv_hypertrophy / hcm is_a HYP`,
`hfref is_a heart failure`). The learned-embedding superclass flags and the DCM pathology flag point
straight at their category node, so the deterministic and probabilistic pathways meet there.

New queries (`knowledge_graph.py`): `category_of` (a specific diagnosis → its parent category),
`categories_for_findings` (roll findings up to superclasses), `convergent_categories` (cross-modality
convergence at the superclass level). Validated (no LLM):

- **Taxonomy**: `old_inferior_mi → MI`, `lv_hypertrophy/hcm → HYP`, `hfref → heart failure`.
- **Deterministic↔embedding bridge**: the **MI** superclass is reached by both `pathological_q_inferior`
  (deterministic Q-wave flag) AND `embedding_MI_pattern_suspected` (the learned CNN) - the two ECG
  pathways corroborate each other at the category level, which neither the diagnosis level nor the RAG
  could express.
- **Category-level cross-modality convergence**: `HYP` ← ECG (Sokolow + embedding_HYP) + MRI (HCM) - a
  more abstract convergence than the diagnosis level, grouping several specific findings under one
  superclass.
- **No regression**: `convergent_diagnoses` is unchanged (still the 2 diagnosis-level convergences);
  the directional config is still the 5 flags (embedding nodes carry no `directional` block).

Both the deterministic diagnosis-level and category-level convergences are stored in the decision
`trace` (`kg_convergent_diagnoses`, `kg_convergent_categories`) when `use_kg=True`. So the KG now plays
two roles: **cross-modality convergence** (evidential) **and pathology hierarchy** (taxonomic), plus it
is the single source of the directional facts.

## CAD / ischaemia cluster — added (the richest cross-modality convergence)

The convergence coverage was extended with the most flag-dense cluster available: **coronary artery
disease / myocardial ischaemia**. This one needed no new detection - the **patient agent already
produces four CAD/ischaemia flags** (`abnormal_thallium_result_suspected`,
`multiple_vessels_involved_suspected`, `significant_st_depression_suspected`,
`exercise_induced_angina_present`); they were simply wired to a new `coronary_artery_disease` diagnosis
node. The ECG side flows in through the **hierarchy** already built: `pathological_q_inferior →
old_inferior_mi → is_a → MI superclass → supports → CAD` (and `embedding_MI` via the same MI category).

- **Deterministic convergence** (no LLM): CAD is reached by **ECG + patient** (`pathological_q_inferior`,
  `embedding_MI` + the four patient flags) - a genuine cross-modality convergence. The reasoning path is
  fully interpretable and uses the taxonomy: *"pathological Q inferior → old inferior MI → MI superclass →
  coronary artery disease"*.
- **A/B (use_kg off vs on, MedGemma 4B, seeded)**: without the KG the summary lists the findings; with it,
  it concludes *"These findings are consistent with **coronary artery disease**"* - the convergent diagnosis,
  which A/B stored deterministically in the trace (`ecg` + `patient`).
- **No regression**: the hypertrophy/hypertensive cluster still converges; directional config still 5 flags.
- **Note (data honesty)**: the MRI side is NOT in this cluster - scar / prior-MI characterisation needs
  late-gadolinium-enhancement imaging, which ACDC (cine MRI only) does not contain. No vision model can
  recover a signal the imaging never captured; a stronger VLM could at most add an *indirect* functional
  sign (regional akinesia/thinning) from cine, not scar. So CAD converges across ECG + patient (rich), MRI
  stays out by a data limitation, honestly.

KG now: 36 nodes / 28 edges.

## End-to-end demo (deliverable artifact)

`report_pdf.generate_case_report_pdf` now renders a **"Cross-Modality Convergent Evidence (Knowledge
Graph)"** section whenever the decision carries `trace["kg_convergent_diagnoses"]` (i.e. the Sep
Agent ran with `use_kg=True`). A synthetic convergent-hypertrophy case (ECG-LVH + MRI-HCM +
patient-HTN) run through the Sep Agent (`use_kg=True`, MedGemma) produces
`results/kg_integration/kg_demo_case_report.pdf` (+ `kg_demo_sep_decision.json`): the integrated
summary names *"hypertensive heart disease"*, and the PDF shows the deterministic convergence block
("hypertensive heart disease - independently supported by ecg, mri, patient"; "left ventricular
hypertrophy - ecg, mri") with a note that it is computed from the KG, not the LLM. Reproduce:
`experiments/kg_demo_pdf.py`.

## Summary

The KG delivers one clear, validated win — **cross-modality convergence** (consumer #3): a new
structured-reasoning capability RAG cannot provide, improving the synthesis on both the local 4B and
a strong cloud model, plus a deterministic auditable convergence output. Consumer #1 (directional
check) is architecturally cleaner but detection-equivalent, so not shipped as a replacement. The KG
does NOT address G2 (prose integration, model-bound, already closed) - it is a different axis
(structured reasoning / verification), as intended.

Requires `pip install networkx`. Reproduce the foundation test: load `clinical_kg.json` via
`knowledge_graph.load_kg` and call `convergent_diagnoses` / `check_directional_consistency_kg`.
