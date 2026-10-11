## Limitation: traceability validates citation existence, not qualitative
## correctness of the reasoning built on top of it (CoT + RAG)

**Status: fix implemented and empirically validated across N=25 repeated
generations (four rounds below). Round 4 (a flag-specific prompt anchor,
chosen over the round-3 temperature/seed approach for the reasons given
there) cut the round-2 baseline error rate from 12.0% to 4.0% - real
improvement, not full elimination.**

### Round 4 - flag-specific prompt anchor (adopted over temperature/seed)

Round 3 showed temperature alone made things worse and temperature+seed's
0% result was likely partly a small-sample/seed-non-reproducibility
artifact rather than a structural fix (see round 3 below) - so a prompt-
level fix targeting the diagnosed root cause was tried instead:
`_build_directional_anchor_block` (`sep_agent.py`) injects an explicit,
per-case reminder for each directional flag actually present, naming the
flag and its forbidden words directly (config externalized to
`src/directional_flags_config.json` since this note was first written -
one entry per flag with its measure terms, forbidden words, AND anchor
description together, so the check and the prompt anchor can't drift
apart) - e.g. for
`high_lv_ejection_fraction_suspected`: *"the LEFT ventricular ejection
fraction is ELEVATED (hyperdynamic / preserved-high)... never use these
words: 'reduced', 'decreased', 'diminished', 'depressed', 'low',
'normal'."* This is stronger than the pre-existing generic direction rule
(which stayed in place unchanged) because it names the SPECIFIC flag and
SPECIFIC words, not just a general instruction to "be consistent with
flags."

Tested on the identical N=25 sample (5 patients x 5 repeats, CoT+RAG, both
checks active, default temperature/no seed - isolating this one variable
against the round-2 baseline):

| Outcome | Round 2 (baseline) | Round 4 (+ anchor) |
|---|---|---|
| Clean on first attempt | 19/25 | 15/25 |
| Clean after a retry | 3/25 | 9/25 |
| Exhausted retry budget | 3/25 | 1/25 |
| **Final-output error rate** | **12.0%** | **4.0%** |

**Real improvement, not elimination.** All 3 round-2 exhausted failures
were `high_lv_ejection_fraction_suspected` - with the anchor, that flag
had ZERO exhausted failures (patient102: 3/5 repeats still needed a
retry for it, but all 3 now recovered instead of 2/5 exhausting). The one
remaining round-4 failure is a DIFFERENT flag - patient109,
`low_rv_ejection_fraction_suspected`, calling 33.7% RV EF "normal" across
all 3 attempts - despite the SAME anchor mechanism applying to this flag
too (its own explicit "never use... normal" reminder was present in the
prompt). So the anchor substantially reduces, but does not guarantee
against, this failure class, even for the exact flag it was written for.
6 total directional/citation retries fired this round (vs. only 3 in round
2); 5 of 6 recovered via retry (vs. 3 of 3 in round 2, but round 2 also
had 2 additional exhaustions from checks firing 0 times and getting stuck
purely on JSON truncation instead - not a like-for-like comparison of
"check reliability," see the table above for the real total).

**Not yet tried:** combining the anchor with the round-3 temperature
finding, or extending this config-file approach to flags outside the
current 5 (e.g. `pathology_<CLASS>_pattern_suspected`, which has no
directional check at all yet) - adding one is now a JSON edit
(`src/directional_flags_config.json`), not a code change.

### Round 3 - does lowering temperature (and fixing a seed) help?

Tested empirically (src/temperature_seed_reliability_test.py), not adopted
on intuition - same discipline as the embedding-model comparison. Two
things confirmed first, directly (not assumed): (1) medgemma:4b's own
Modelfile sets no `PARAMETER temperature` or `seed` - only `stop` - so
every prior round used Ollama's own runtime default (~0.8); (2) 3 identical
short calls with no options produced 3 different phrasings (non-determinism
confirmed), while 2 identical calls with an explicit seed returned
byte-identical text on that SAME short prompt.

Same N=25 sample (5 patients x 5 repeats, CoT+RAG, both checks active),
three configurations compared:

| Config | Clean 1st try | Clean after retry | Exhausted | **Error rate** |
|---|---|---|---|---|
| Default (~0.8, no seed) - round 2 | 19/25 | 3/25 | 3/25 | **12.0%** |
| Temperature=0.2, no seed | 19/25 | 2/25 | 4/25 | **16.0% - WORSE** |
| Temperature=0.2 + per-patient fixed seed | 21/25 | 4/25 | 0/25 | **0.0%** |

**Temperature alone made it worse, not better - confirmed, not just
theorized.** 3 of the 4 exhausted failures at temperature=0.2 were the
exact same pattern as round 2 (`high_lv_ejection_fraction_suspected` /
`low_lv_ejection_fraction_suspected` characterization errors, on
patient104/103/102). This is the predicted failure mode of sharpening the
distribution when the model's underlying bias already favors the WRONG
word for this flag ("reduced"/"normal" more probable than "elevated" in
its training distribution for this phrasing) - lower temperature makes the
model pick its most likely token MORE reliably, which is actively harmful
when the most likely token is wrong.

**Temperature + seed reached 0/25 exhausted failures - but with an
important caveat found while checking the result, not assumed.** The seed
did NOT produce byte-identical output at the length/complexity of the real
production prompts, despite doing so on the short test prompt above: only
1 of 5 patients (patient103) had fully identical text across all 5
"repeats" with its fixed seed; the other 4 had 3-5 distinct summaries each,
sometimes substantively different (e.g. patient101's reps 3-4 mention
"dilated left AND right ventricles", reps 1/2/5 mention only the LV). The
directional/citation check DID fire 4 times even in this configuration
(confirmed by grepping the raw logs) - it just always succeeded within the
retry budget this time, unlike round 2 where the same check firing 3 times
led to exhaustion. **0% is a genuine, positive result on this sample, not
a guarantee the model became deterministic - the improvement is real, the
"reproducibility via seed" framing is only partially true at this prompt
length**, likely floating-point non-associativity in batched GPU
inference compounding over a long generation (a known llama.cpp/Ollama
characteristic, not unique to this project).

**Not adopted as the production default.** Per the task that requested
this measurement: report only, decide together afterward. Two reasons for
caution before adopting temperature=0.2+seed anyway: (1) N=25 with only 5
distinct seeds is a small sample to generalize a 0% claim from, especially
since the seed's effect on THESE 5 specific prompts might not transfer to
different findings/patients; (2) since temperature alone made things
measurably worse, any partial/future change to the prompt or checks could
shift the model's probability landscape again, and temperature+seed with
these fixed values hasn't been tested against that variation.

### Round 2 - after fixing MedGemma's JSON truncation (num_ctx)

While validating round 1 (below), 9 of 13 retries and both exhausted
failures turned out to share one root cause unrelated to this limitation:
`agent_common.generate_report_medgemma` sent no `options` to Ollama, so it
fell back to its small default context window (~2-4k tokens) instead of
using medgemma:4b's real 131k-token capacity (confirmed via `ollama show`)
- with a long CoT+RAG prompt, generation got cut off mid-JSON
("Unterminated string..."). Fixed by explicitly setting
`options: {"num_ctx": 8192, "num_predict": 2048}` in the Ollama request.
Verified directly: the two prompts that previously exhausted all 3
retries via truncation succeeded 6/6 times when resent standalone after
the fix.

Re-running the full N=25 sample after this fix: **truncation-caused
retries dropped from 9/13 to 2/9** - the fix works for its intended
purpose. But the overall exhausted-failure rate went UP, from 8.0% (2/25)
to **12.0% (3/25)**:

| Outcome | Round 1 (before num_ctx fix) | Round 2 (after) |
|---|---|---|
| Clean on first attempt | 12/25 | 19/25 |
| Clean after a retry | 11/25 | 3/25 |
| Exhausted retry budget | 2/25 | 3/25 |

**Why more failures, not fewer, after fixing truncation.** Truncation was
previously cutting some responses off BEFORE the model finished writing a
qualitative claim - meaning some of what looked like "just a truncation
problem" was actually also masking a directional error that never got the
chance to fully materialize (and get caught) in the truncated text. With
generation now completing reliably, `_check_directional_consistency` sees
the FULL text and catches more real errors than before - this is the check
working correctly, not a regression.

**All 3 round-2 exhausted failures share the exact same flag:
`high_lv_ejection_fraction_suspected`** (patient102, LV EF 70.9%;
patient104, LV EF 84.4% - both across multiple repeats). This is not
random: MedGemma appears to have a systematic, reproducible tendency to
under-characterize an ELEVATED ejection fraction specifically - writing
"normal" or even "reduced" where the flag says elevated - persisting
across all 3 retry attempts in these cases. Plausible reason: "reduced
ejection fraction" is a far more common/familiar clinical phrase in the
model's training distribution than "elevated ejection fraction" (which is
the rarer, HCM-associated finding this flag exists to catch), so the model
defaults toward the familiar phrasing under uncertainty. The check
correctly detects this every time; it does not always correct it within a
3-attempt budget for this specific flag.

**Not yet done - candidate next step, not implemented:** a stronger,
flag-specific prompt anchor (e.g. explicitly pairing
`high_lv_ejection_fraction_suspected` with "describe as elevated/
hyperdynamic, never as reduced or normal" inline near the flag, rather
than the current generic direction rule) to reduce how often this specific
flag needs a retry at all. Untested - would need its own before/after
comparison, not assumed to work.

### Round 1 (original N=25 result, before the num_ctx fix)

**Why a single small-sample run isn't enough.** The original test below (5
cases, one per ACDC pathology group) found 2 real errors and 3 clean
cases - but a single run cannot give a reliable error rate, because
MedGemma's generation is non-deterministic. This was confirmed directly
while validating the fix: patient104, one of the 3 CLEAN cases in the
original run, produced a genuinely new characterization error
("...RV ejection fraction is slightly low at 65.6%" - not low, and with
zero supporting traceability citation) on a later re-run of the exact same
fixed input. **Errors are not tied to specific "hard" patients - they can
occur on any generation of any case.** The methodology was corrected
accordingly: error rate is now measured across N=25 repeated generations
(5 patients x 5 repeats each, fixed MRI contribution per patient so the
MRI agent's own non-determinism isn't a confound), not a single 5-case
pass/fail. See src/cot_repeated_reliability_sample.py.

**The fix, implemented in `sep_agent.py`:**
1. `_check_directional_consistency` - rejects a claim stating the OPPOSITE
   direction of what an already-raised flag establishes (e.g. calling an
   elevated EF "reduced"). Catches Case 1 below.
2. `_check_ef_citation_coverage` - rejects an ejection-fraction claim with
   a concrete percentage that has NO traceability entry backing it at all.
   Added specifically because the first version of check #1 could not
   catch a claim about a value that never crossed any flag threshold (see
   the patient104 re-run finding above) - both checks share the same
   clause/chamber-resolution logic (`_resolve_ventricle_referents`) so
   they agree on which measurement a clause refers to.

Both are deterministic, clause-level keyword heuristics (not NLI) that
re-enter the SAME retry loop as a fabricated citation - not a new
mechanism. Two real false positives were found and fixed while building
these (a same-sentence-different-clause misattribution, and an English
ellipsis pattern - "left and right ventricles" - that broke chamber
attribution); see `sep_agent.py`'s module docstring for the full account.
One deliberate gap remains unfixed: a self-contradiction expressed via
negating the correct word in a different clause (patient109's original
phrasing) is not caught, in favor of precision over recall for a check
that gates a retry loop.

**Result across N=25 generations (5 patients x 5 repeats, CoT+RAG - the
production config - both checks active):**

| Outcome | Count |
|---|---|
| Clean on first attempt | 12/25 |
| Clean after a retry fired and corrected it | 11/25 |
| Exhausted retry budget (3 attempts) without a clean response | 2/25 |
| **Final-output error rate** | **2/25 = 8.0%** |

Per patient:

| Patient | Group | Clean | Needed a retry | Exhausted |
|---|---|---|---|---|
| patient101 | DCM | 5/5 | 2/5 | 0/5 |
| patient104 | HCM | 5/5 | 3/5 | 0/5 |
| patient103 | MINF | 5/5 | 2/5 | 0/5 |
| patient102 | NOR | 3/5 | 1/5 | 2/5 |
| patient109 | RV | 5/5 | 3/5 | 0/5 |

**What actually triggered each retry** (checked directly against the raw
prompt/response logs, preserved at
`results/cot_reliability_history/01_default_temp_before_numctx_fix_raw/`
- not inferred):
of the 13 generations that needed at least one retry, 9 were malformed or
truncated JSON from MedGemma (a pre-existing, unrelated output-formatting
issue), 1 was the PRE-EXISTING fabricated-citation check (an
agent_id="knowledge" citation using a feature name that actually belongs
to "mri" - unrelated to this fix), and only 3 were the two NEW checks
specifically (1 from `_check_directional_consistency`, 2 from
`_check_ef_citation_coverage`). **Both checks self-corrected 3/3 times they
fired; neither ever exhausted the retry budget in this sample.** Both of
the 2 EXHAUSTED failures (patient102, reps 2 and 3) were caused purely by
persistent JSON malformation across all 3 attempts - a separate,
pre-existing reliability issue, not the qualitative-characterization
problem this fix targets.

**Caveat.** N=25 is a real improvement in rigor over a single 5-case run,
but still modest - a firing rate of 3/25 (12%) for the new checks means
the confidence interval on the true underlying error rate is wide. This is
evidence the fix works as intended within a repeated, diagnostically
diverse sample, not a claim of statistical certainty at scale.

---

## Original 5-case test (superseded by the N=25 result above; kept for the
## detailed case studies, which remain accurate as illustrations)

**Summary.** Adding Chain-of-Thought (CoT) reasoning and RAG-grounded
context to the Sep Agent did not eliminate a pre-existing narrative-accuracy
limitation - it produced two additional, distinct concrete examples of it.
On a small but diagnostically diverse 5-case test (one case per diagnostic
group: NOR, DCM, MINF, HCM, RV), the pipeline was structurally reliable
5/5 (no crashes, valid JSON, reasoning field always populated,
traceability always resolved to a real feature) but clinically correct in
only 3/5 cases (60%) once the actual narrative content was read, not just
its structural validity.

**Why the existing hallucination check cannot catch this.** Traceability
verification (`parse_findings_response` / the equivalent check for
free-text reasoning) confirms that a cited feature reference
(`evidence_ref`) genuinely exists in the computed feature set - it says
nothing about whether the qualitative word attached to that feature
("reduced", "elevated", "mild") correctly characterizes its actual value.
Both failure cases below cite a real, correctly-valued feature; the
citation-existence check therefore passes in both, by design, and this is
expected to remain the case for any prompt wording that keeps qualitative
characterization as free text rather than a structurally-validated field.

**Case 1 - directional misreading during reasoning (patient102, true NOR).**

- Ground truth: LV ejection fraction = 70.9% (elevated - this is exactly
  what triggered `high_lv_ejection_fraction_suspected`).
- CoT reasoning text: "a decreased ejection fraction (70.9%)".
- CoT summary: "ejection fraction reduced, raising suspicion for dilated
  cardiomyopathy" - both the direction (reduced vs. elevated) and the
  resulting suspected pathology (DCM, associated with *low* EF) are wrong
  for this value.
- The no-CoT summary for the same case avoided this entirely, by staying
  purely numeric ("ejection fraction of 70.9%", no qualitative adjective).
  This suggests CoT's added reasoning step introduced this specific error
  rather than preventing it - the opposite of the intended effect.

**Case 2 - contradiction introduced during reasoning-to-summary
compression (patient109, true RV).**

- Ground truth: RV ejection fraction = 33.7% (~12 points below the review
  threshold).
- CoT reasoning text: accurate - "the right ventricle has a low ejection
  fraction (33.7%)".
- CoT summary: drifts within the same paragraph - "mildly reduced right
  ventricular ejection fraction at 33.7%" followed, one sentence later, by
  "no signs of reduced ejection fraction" - internally self-contradictory,
  and understates the severity of a clinically significant value.
- Unlike Case 1, the reasoning step itself was correct here; the error was
  introduced specifically during the reasoning-to-final-summary
  compression step.

**Cases held up on inspection.** The remaining 3 of 5 cases (DCM, MINF,
HCM) showed correct direction, no internal contradiction, and valid
traceability throughout.

**Interpretation.** These are two distinct failure modes, not one:

| | Case 1 (patient102) | Case 2 (patient109) |
|---|---|---|
| Where the error appears | During reasoning itself | During summary compression, after correct reasoning |
| Effect of CoT | Introduced the error (no-CoT version was correct) | Reasoning was right; only the final summary drifted |
| Caught by traceability check? | No (citation is real) | No (citation is real) |

This extends the pre-existing narrative-accuracy limitation already noted
in `sep_agent.py` (the original DCM ejection-fraction mischaracterization
case) with two further, independently observed instances - confirming it
is a systematic class of failure for this LLM/prompt configuration, not
an isolated incident, and that adding CoT does not resolve it and can,
in at least one observed case, actively introduce it.

**Practical takeaway for interpreting any generated report from this
system.** A valid traceability entry (`evidence_ref` resolves to a real
feature) confirms the underlying data point is not fabricated. It does
NOT confirm that the qualitative characterization built on top of that
data point ("reduced", "elevated", "mildly...") is directionally or
internally consistent. Any narrative summary produced by this pipeline
should be read with this distinction in mind, and - per the existing
project-wide "review, not diagnosis" framing - is not a substitute for a
clinician's own reading of the underlying numeric values.

**Status (superseded - see top of document): implemented and validated.**
Instead of the numeric-threshold prompt mitigation originally proposed
here, a deterministic post-generation check
(`_check_directional_consistency`) was built and wired into the existing
retry loop, later joined by a second check (`_check_ef_citation_coverage`)
for a distinct gap it couldn't cover. Case 1 (patient102's direction
inversion) is caught reliably. Case 2 (patient109's original exact
self-contradiction phrasing) turned out NOT to reproduce on the fix's
validation re-runs, and the specific negation pattern is a known,
documented gap in the current checks (see top of document) - the concern
about a reasoning-only rule not covering summary-compression errors was
valid caution at the time, but the fix that was actually built works on
both the reasoning AND summary text, not just the reasoning step alone.
Measured error rate after the fix, across N=25 repeated generations: 8%
(2/25), and both those failures were unrelated to this limitation (see
top of document).
