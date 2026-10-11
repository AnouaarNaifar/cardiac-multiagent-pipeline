# Territorial MI features — validation report

Territorial ECG features for myocardial infarction (`src/ecg_territorial.py`), built on the
anatomical logic that each cardiac wall is watched by specific leads with reciprocal leads
opposite. Two complementary MI signals in one framework, validated against PTB-XL `scp_codes`
(no LLM, `src/ecg_territorial_validation.py`, n=4000).

| Wall | Leads | Reciprocal leads |
|---|---|---|
| inferior | II, III, aVF | I, aVL |
| anterior | V1–V4 | (posterior V7–V9, not in 12-lead) |
| lateral | I, aVL, V5, V6 | II, III, aVF |

## (1) ST elevation + reciprocal depression (acute STEMI) — NOT viable in PTB-XL

Validated against the acute injury/ST-elevation codes (`INJIN/INJIL`, `INJAS/INJAL`, `INJLA`,
`STE_`). **Best F1 ≈ 0.00–0.02.** Not a code bug — a **data** limitation: PTB-XL is a
resting-ECG dataset of a general population, so acute STEMI is almost absent (inferior injury
3 cases, lateral 10, anterior 42 in 4000). Reciprocal-depression gating changed nothing (too
few positives). The clinical logic (territorial elevation + reciprocal depression) is correct;
PTB-XL simply doesn't contain the acute presentations to validate it. **Kept as descriptive
features** (`st_elev_<wall>_mv`, `st_recip_<wall>_mv`); no flag.

## (2) Territorial pathological Q waves (old/chronic MI) — real signal, moderate

This is where PTB-XL's MI actually lives (chronic MI diagnosed by Q waves / loss of R:
IMI 2676, ASMI 2357, …). Per-lead pathological Q = Q/R ≥ 0.25; per wall, the COUNT of
territorial leads meeting it (`qcount_<wall>`). **Territorial consensus (≥ k contiguous leads)
is the specificity lever** — a single lead over-fires (a lone septal q is physiological):

| Wall | MAX-Q/R (1 lead) | **Consensus** (best) | vs base rate |
|---|---|---|---|
| **inferior** | F1 0.261 | **F1 0.366** @≥3 leads (P 0.267 / R 0.584) | 2.9× |
| anterior | F1 0.192 | F1 0.371 @≥4 leads (P 0.281 / R 0.546) | — |
| lateral | F1 0.131 | F1 0.212 @≥2 leads (P 0.127 / R 0.644) | 4.3× |

At the clinical standard **≥2 contiguous leads**: inferior F1 0.328 (P 0.200 / **R 0.915**,
2.2× the 9.2% base rate), anterior 0.209 (only 1.1× — no real enrichment at ≥2), lateral 0.212.

## What was shipped

- **Flag: `pathological_q_inferior_suspected`** = pathological Q in ≥2 inferior leads (the
  clinical consensus rule, recall-favoring). Real, recall-favoring (R 0.915) MI alert, 2.2×
  the base rate — and the **only deterministic MI signal in the agent** (all other MI coverage
  is the embedding's probability). Far better than several already-shipped flags (tachycardia
  F1 0.086). Wired in `ecg_agent.detect_flags` (param `inferior_q_min_leads=2`);
  `build_clinical_summary` reports the per-wall Q counts. Verified end-to-end on a real IMI
  (flag raised) and a NORM (an expected false positive — precision is 0.20, this is a screening
  alert not a diagnosis).
- **Descriptive-only** (no flag): anterior Q (enriched only at ≥3–4, threshold-sensitive),
  lateral Q (MI rare), and all ST features (acute STEMI absent from PTB-XL). Real measurements,
  kept as interpretable context; not reliable enough to auto-flag.

**Net:** the territorial framework confirmed the data-driven split — acute ST elevation is
undetectable *here* for lack of acute cases, while chronic inferior-Q MI is a real, moderate,
recall-favoring signal now flagged. Stronger MI detection (anterior/lateral, and acute ST on a
dataset that has it) would need proper Q-wave delineation (width, not just depth) and an acute
dataset — future work.

Reproduce: `python src/ecg_territorial_validation.py [n_records]`. Cache:
`results/ecg_territorial_validation/territorial_cache_*.csv`.
