# ECG-Image Branch — Design Logic and Comparative VLM Evaluation

*Prepared to justify the design choices for the ECG-image modality. All tests below were
run locally on real data; every number is reproducible with the scripts named in each section.*

---

## 1. Why an image branch at all

The system accepts an ECG in **two** forms:

- **Signal** — a raw WFDB record (`.hea`/`.dat`) or a CSV of samples. This is a *research-dataset*
  artifact.
- **Image** — a photo, scan, or printout of a 12-lead ECG. **This is the realistic real-world input**:
  in practice a clinician usually has the *printout*, not the digital signal file.

A single entry point, `ecg_router.route_ecg_upload`, decides which branch applies from the upload
itself and returns the same `AgentContribution` object either way, so the rest of the multi-agent
system does not need to know which branch produced it.

**Key architectural fact (already implemented):** the image branch goes **directly through a
Vision-Language Model (VLM)** and touches **none** of the signal code.
`ecg_image_agent.generate_image_ecg_contribution` imports only `adaptive_ecg_pipeline` (image
preprocessing) and `agent_common` (prompt / parse / LLM). It does **not** import
`ecg_preprocessing`, `ecg_embedding`, `ecg_multilead`, or `ecg_territorial`. So the design already
matches the "if the user brings an image, go straight to the VLM without those codes" principle.

---

## 2. The core question this branch must answer

The signal branch grounds **every** finding in a **measured value compared to a validated
threshold**:

| Signal finding | Measured value | Threshold | Validated in |
|---|---|---|---|
| tachycardia | heart rate (bpm) | 120 bpm | `flag_validation.py` |
| wide QRS | QRS duration (ms) | 120 ms | `flag_validation.py` |
| LV hypertrophy | Sokolow-Lyon index (mV) | 3.5 mV | `ecg_multilead_validation.py` |
| low LV EF (MRI) | ejection fraction (%) | 45 % | `mri_flag_validation.py` |

Those findings are **deterministic, falsifiable, and validated** against ground truth.

The **image branch has no such numeric clinical values.** Its only numeric keys are
*image-quality* metrics (blur, contrast, brightness). **All clinical content comes from the VLM
reading pixels.** There is no measured mV/ms/bpm, no threshold, no deterministic flag.

> **The question:** how can an image-derived ECG finding be *justified* if there are no values and
> no thresholds to compare against?

This report answers that question with evidence, and settles the secondary practical problem
("the VLM does not return well-formed JSON").

---

## 3. Experimental setup

- **Record:** PTB-XL / GenECG `ecg_id = 1` (record `00001`). GenECG renders PTB-XL signals as ECG
  *images*, so the **ground truth is known**: **NORM, sinus rhythm, low QRS voltage (LVOLT)**.
- **Two image variants:** **A** = clean render, **B** = with imperfections (blur/noise/skew).
- **Identical pipeline for every model:** `adaptive_ecg_pipeline.process_ecg_image` (adaptive
  quality correction) → `build_image_findings_prompt` (the agent's real prompt, with per-lead
  evidence keys) → VLM with the image attached → `json_repair` → strict
  `parse_findings_response` (schema + evidence-key validation).
- **Four models**, from smallest/most-specialized to frontier.

---

## 4. Comparative results

Ground truth: **NORM · sinus rhythm · low voltage.**

| Model | Size | Valid JSON (A / B) | Hallucination | Clinical reading vs. ground truth | Measured values + thresholds |
|---|---|---|---|---|---|
| `medgemma-ecg-gguf` (ECG fine-tune) | 4B, 4-bit | ✗ / ✗ (broken fragments, missing fields) | **Severe** — invented *tachycardia, ST-elevation, LBBB, prolonged PR* on a NORM tracing | **Wrong** | None |
| `medgemma:4b` (base) | 4B | ✓ / ✗ (truncated) | None | **Shallow** — correct "sinus rhythm" but padded with image-quality restatements; confidence = 1.0 for everything (uncalibrated) | None |
| `qwen2.5vl:7b` | 7B | **✓ / ✓** | None | **Correct but monotone** — "sinus rhythm" repeated per lead; followed the per-lead schema exactly | None |
| Frontier VLM (Claude, direct image read) | — | ✓ / ✓ | None | **Correct + richer** — sinus rhythm, narrow QRS, **low voltage** | None (qualitative gestalt only; cannot produce calibrated mV/ms) |

Raw evidence: the ECG fine-tune returned e.g. `{"feature":"sinus_rhythm",}` (no value/rationale/
confidence/evidence_ref, trailing comma) and, on the *clean* image, the contradictory flag set
`["tachycardia","incomplete_left_bundle_branch_block","sinus rhythm","prolonged pr interval", ...]`.
`qwen2.5vl:7b` returned 12 well-formed findings on both variants, one `sinus_rhythm` per lead,
confidence 0.9.

---

## 5. Two conclusions that hold across **every** model

**(1) A better model fixes the FORM, never the SUBSTANCE.**
Moving up in model quality removes the JSON breakage and the hallucination, and recovers the
correct *gestalt* reading. But **not one model** produces a measured value or a threshold
comparison. A frontier VLM (my own direct read) can say *"low voltage"* but **cannot** say
*"S(V1)+R(V5) = 1.8 mV < 3.5 mV."* The **validity gap is model-independent** — it is a property of
reading pixels instead of a calibrated signal.

**(2) The ECG-specialized fine-tune is paradoxically the LEAST reliable for this task.**
`medgemma-ecg-gguf` was fine-tuned to *describe* ECG images in free text, **not** to emit our rigid
JSON schema — so it produces the worst structured output and even hallucinates. Reliability
ranking for our structured task: **`qwen2.5vl:7b` > `medgemma:4b` base > ECG fine-tune.**

---

## 6. Architectural decision (what is defended)

1. **The image branch stays a direct VLM path** (no signal/embedding code). This matches the
   realistic input scenario and is already how the router works. **Digitization** (image → signal →
   reuse the signal pipeline) was considered and **set aside**: it would re-couple the image path to
   PTB-XL-signal-specific code and add heavy machinery (grid detection, trace extraction,
   calibration), against treating the image as a first-class native input.

2. **Image findings are honestly scoped as QUALITATIVE**, not measurements. This is carried
   explicitly by the schema: `Finding.evidence_type = "image_region"` (vs `"numeric_feature"` on the
   signal side). The Sep Agent then combines these qualitative image findings alongside the
   quantitatively-grounded signal / MRI / patient findings — the multi-agent design already supports
   heterogeneous evidence classes.

3. **Default model for the branch = `qwen2.5vl:7b`** (free, local, no data leaves the machine): the
   most reliable JSON, no hallucination, correct reading. It should replace the ECG fine-tune as the
   default. A frontier cloud VLM is an **optional** upgrade for richer reading (paid, and the image
   leaves the machine — acceptable for public GenECG data, a concern for real patient images).

---

## 7. How the image branch IS validated — the honest justification

The signal branch is justified *per finding vs. a threshold*. The image branch cannot be (no
measurements). Its justification is **different but equally rigorous: agreement with ground truth
over a test set.**

Because **GenECG = PTB-XL rendered as images**, every image has a known diagnostic superclass label
(NORM / MI / STTC / CD / HYP). The validation — the **image-branch analogue of
`flag_validation.py`** — runs the VLM over *N* GenECG images in a zero-shot 5-class presence query
and computes **precision / recall / F1** against the true PTB-XL labels. **No classifier and no
signal code are involved — 100 % VLM.**

**Result — `qwen2.5vl:7b`, 50 class-stratified images (variant A, clean), PTB-XL test fold 10:**

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| CD | 0.469 | 0.682 | 0.556 | 22 |
| HYP | 0.600 | 0.200 | 0.300 | 15 |
| **MI** | **0.000** | **0.000** | **0.000** | 18 |
| NORM | 0.167 | 0.273 | 0.207 | 11 |
| STTC | 0.406 | 0.684 | 0.510 | 19 |
| **macro** | **0.328** | **0.368** | **0.314** | |

Exact-match (all 5 classes correct): **0.10** (5/50).

**Interpretation — this is the defensible, honest number for the branch:**

- **The VLM-only image branch is weakly reliable at best (macro F1 ≈ 0.31).** For context, the
  *signal* branch's CNN reaches macro F1 ≈ 0.67 on the same 5 superclasses — roughly **2× better**.
- **It never detects MI (F1 = 0.00, 0/18).** This is the clinically most dangerous failure: a
  VLM-only image reading would miss every myocardial infarction in this sample.
- **It over-calls "normal" (NORM precision 0.17):** it labels many abnormal ECGs as normal — false
  reassurance.
- It is least-bad on CD (0.556) and STTC (0.510), both high-recall / low-precision (over-flagging).

**This is exactly why the image branch must be scoped as a qualitative aid, not a stand-alone
diagnostic** — and it is now backed by a measured number, not an assertion. It also quantifies the
upside of the classifier option (§10): a discriminative image classifier trained on GenECG labels
would be expected to approach the signal branch's ≈0.67 macro F1 and, crucially, actually detect MI.

Reproduce: `results/ecg_image_pipeline/vlm_zeroshot_validation.md` (+ per-record `.csv`).

---

## 8. Immediate reliability fixes (independent of the model)

Two are prompt/design issues, not model issues, surfaced by the tests above:

- **Remove image-quality metrics from the citable `evidence_keys`.** They are *context*, not
  findings; exposing them made the base 4B turn every metric into a "finding" (noise) and blow the
  token budget (truncation on variant B).
- **Force confidence calibration in the prompt.** A flat 1.0 (base) or 0.9 (7B) on every finding is
  meaningless.
- **Keep per-lead grounding** (`ecg_image_<lead>`): `qwen2.5vl:7b` followed it well, and it is the
  only lightweight traceability the branch can offer.

---

## 9. Real probabilities via a GenECG image classifier — BUILT AND VALIDATED

A VLM cannot output genuine per-class probabilities — its self-reported `confidence` is
uncalibrated (flat 0.9/1.0 in §4). Real probabilities require a **trained classification head**,
which the signal branch already has (`ECGEmbeddingNet` → sigmoid → 5 calibrated probabilities).

The image-equivalent was built: a **discriminative image classifier trained on GenECG images**
(image → PTB-XL 5-superclass label), reusing `ecg_embedding.py`'s exact label logic. **No signal
code, no digitization** — pure image → ResNet18 (ImageNet-pretrained) → 5 sigmoid probabilities.
Trained on 887 GenECG images (folds 1–9, 15 % carved for early-stopping validation), tested on
**217 held-out fold-10 images**. Early stopping was essential: the model over-fit fast (best
validation at **epoch 2**; by epoch 7 train loss 0.13 but val loss 0.80), and the best epoch was
restored before testing.

**Result — held-out fold 10, threshold 0.5:**

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| CD | 0.508 | 0.769 | 0.612 | 78 |
| HYP | 0.580 | 0.691 | 0.631 | 68 |
| **MI** | **0.632** | **0.608** | **0.619** | 79 |
| NORM | 0.525 | 0.857 | 0.651 | 49 |
| STTC | 0.825 | 0.612 | 0.703 | 85 |
| **macro** | | | **0.643** | |

Exact-match: **0.267**.

**Classifier vs. 100 % VLM zero-shot (same 5 classes):**

| metric | VLM zero-shot | **Image classifier** |
|---|---:|---:|
| macro F1 | 0.314 | **0.643** (≈ 2×) |
| **MI F1** | **0.000** | **0.619** |
| exact-match | 0.100 | **0.267** |
| real probabilities | no (fake confidence) | **yes (sigmoid)** |

The classifier **doubles** the branch's macro F1, **closes the MI blind spot** (0.00 → 0.62 — the
VLM's most dangerous failure), and approaches the *signal* branch's ≈0.67 macro F1 — all from the
image alone. Two complementary components now form the branch: the **VLM = qualitative narrative**,
the **classifier = measured probabilities + validation**. Reproduce:
`results/ecg_image_pipeline/genecg_classifier_validation.md`; training in
`src/genecg_classifier_train.py`, data in `src/genecg_classifier_download.py`.

## 10. Bottom line for the defense

- The image branch is **architecturally correct** as a direct VLM path — matching the realistic
  "user brings an image" scenario, with **no dependence on the signal pipeline**.
- Its findings are **honestly labeled as qualitative visual reads**, not measurements — a property
  of the modality, carried by the `AgentContribution` schema and combined by the Sep Agent.
- The JSON-reliability problem is **real but solved by model choice**, settled empirically:
  **adopt `qwen2.5vl:7b`** (free, local) over the ECG fine-tune.
- The branch's reliability is now **measured, not asserted**: zero-shot macro **F1 ≈ 0.31**, with
  **MI undetected (F1 0.00)** — ~2× weaker than the signal branch. This **justifies scoping the
  image branch as a qualitative aid** and is the honest number to present.
- For quantitative image diagnosis, the defensible route is **not a bigger VLM** but a
  **GenECG-trained image classifier**, now built (§9): **macro F1 0.643 vs the VLM's 0.314**, and
  **MI F1 0.619 vs 0.000** — with real per-class probabilities. Final design = **VLM (qualitative
  narrative) + image classifier (measured probabilities), both from the image, no signal code.**

---

### Reproducibility

- Preprocessing + agent: `src/adaptive_ecg_pipeline.py`, `src/ecg_image_agent.py`, `src/ecg_router.py`.
- Single-call raw inspection: `src/debug_llm_raw_response.py --ecg_id 1 --variant A|B`.
- Multi-model comparison harness used here: a small script calling the agent's own
  `build_image_findings_prompt` + `_repair_raw_response` + `parse_findings_response`, looped over
  models `hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M`, `medgemma:4b`, `qwen2.5vl:7b`.
- Test images: GenECG `ecg_id=1`, variants A/B (fetched via `src/genecg_fixtures.py`).
