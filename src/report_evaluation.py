"""
Report evaluation - generated ECG narrative vs. PTB-XL reference reports
==========================================================================
Standalone evaluation module. Does NOT modify ecg_agent.py, agent_common.py,
sep_agent.py, or ecg_embedding.py - it only consumes their existing,
already-built outputs (AgentContribution, SepAgentDecision) to produce free
text, then scores that text against real PTB-XL cardiologist reports with
BLEU, ROUGE, and BERTScore.

Scope (ECG only): PTB-XL's reference reports were written by a cardiologist
looking at the ECG alone. The generated text scored here must therefore be
ECG-only -> generate_text_for_scoring() below calls 
sep_agent.generate_sep_agent_decision() with a list containing ONLY the ECG
AgentContribution. Why go through the Sep Agent at all, instead of scoring
ecg_agent.py's output directly: ecg_agent.py produces only structured Finding
JSON objects. BLEU/ROUGE/BERTScore compare natural-language text; scoring raw
JSON against a reference report would be invalid. generate_sep_agent_decision()
is the existing component that already turns structured findings into a
synthesized 3-5 sentence narrative - reusing it here (with an ECG-only input
list) avoids adding a second, redundant text-generation path.

Known limitation, preserved rather than hidden: PTB-XL reference reports are
short, telegraphic clinical shorthand (e.g. "sinusrhythmus normales ekg" /
its English equivalent), structurally very different in style and length
from this project's generated 3-5 sentence narrative summaries. BLEU/ROUGE
are sensitive to exact n-gram/length overlap and will be depressed by this
style mismatch even when the generated text is clinically consistent with
the reference, and weight BERTScore (a semantic
similarity measure, less sensitive to surface length/style) more heavily
than BLEU/ROUGE when interpreting these results.

Additional dependencies beyond the rest of the project (flag before running):
    pip install langdetect sacrebleu rouge-score bert-score
bert-score downloads a HuggingFace model on first use (see compute_bertscore_batch's
docstring for the exact model and its size).

Manual step required (mandatory - the reference pool is now built entirely
from this source, see build_reference_subset):
    git clone https://github.com/danikiyasseh/RTLP
This is the RTLP repository accompanying Kiyasseh et al., "Let Your Heart
Speak in its Mother Tongue" - its translations/en_df_round4.csv file
(columns: ecg_id, report) covers 16247 of PTB-XL's 21799 records (74.5%,
NOT the full dataset despite the paper's description - verified directly).

Reference pool source, and why: detect_report_language (langdetect) was
found unreliable on this dataset - a full cross-check against RTLP's
ground-truth English text found a 13.3% false-positive rate among records
langdetect called English (807 of 6050 were actually non-English text
misclassified, e.g. ecg_id 5186's unambiguously German report). Language
detection is therefore no longer used to build the reference pool at all;
build_reference_subset instead uses RTLP's translation for every ecg_id it
covers. See cross_check_native_vs_rtlp for the full numbers and
detect_report_language's docstring - it is kept only for that cross-check,
not for reference-pool construction anymore. Translation-quality caveat:
since most references are now machine-translated from German (or another
source language), that translation is an additional source of noise in the
scores, alongside the length/style and descriptive-vs-diagnostic mismatches
already documented - see write_interpretation_note.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agent_common import AgentContribution, Finding, generate_report_medgemma, save_contribution
from ecg_agent import generate_ecg_contribution
from ecg_embedding import DIAGNOSTIC_SUPERCLASSES, load_scp_superclass_map, scp_codes_to_multihot
from ecg_preprocessing import ecg_preprocessing_pipeline
from sep_agent import generate_sep_agent_decision

# ----------------------------------------------------------------------
# Clinical / biomedical embedding cosine similarity
# ----------------------------------------------------------------------

CLINICAL_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def load_clinical_embedding_model(model_name: str = CLINICAL_EMBEDDING_MODEL):
    """
    Load the biomedical sentence-embedding model used for
    whole-report cosine similarity.

    The model is loaded once per evaluation run, not once per ECG.
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading biomedical embedding model: {model_name}")
    return SentenceTransformer(model_name)


def compute_clinical_cosine_similarity_batch(
    generated_texts: List[str],
    reference_texts: List[str],
    model,
) -> List[float]:
    """
    Compute whole-report cosine similarity between generated ECG
    narratives and PTB-XL reference reports.

    Each generated report and reference report is encoded as one
    embedding vector. Cosine similarity is then computed between
    the two corresponding vectors.

    Inputs:
        generated_texts: generated ECG reports.
        reference_texts: PTB-XL reference reports.
        model: SentenceTransformer biomedical embedding model.

    Output:
        List of cosine similarity scores, one per report pair.
        Values are theoretically in [-1, 1], although sentence
        embedding models typically produce positive similarities
        for semantically related clinical text.
    """
    if len(generated_texts) != len(reference_texts):
        raise ValueError(
            "generated_texts and reference_texts must have the same length."
        )

    if not generated_texts:
        return []

    generated_embeddings = model.encode(
        generated_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    reference_embeddings = model.encode(
        reference_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Because embeddings are L2-normalized, their dot product
    # is equivalent to cosine similarity.
    similarities = np.sum(
        generated_embeddings * reference_embeddings,
        axis=1,
    )

    return similarities.astype(float).tolist()
# ----------------------------------------------------------------------
# 1. Reference-report selection: native-English, validated_by_human subset
# ----------------------------------------------------------------------
def detect_report_language(text: str) -> Optional[str]:
    """
    Detect the language of a single report string.

    Inputs:
        text: raw string from PTB-XL's "report" column (may be NaN/empty).

    Output:
        ISO 639-1 language code (e.g. "en", "de") as detected by langdetect,
        or None if the text is empty/NaN or langdetect cannot decide
        (langdetect raises LangDetectException on inputs with no reliable
        signal, e.g. very short or purely numeric strings - treated as
        "undetectable" here rather than propagating the exception).
    """
    from langdetect import DetectorFactory, LangDetectException, detect

    # langdetect's underlying algorithm is stochastic unless seeded - without
    # this, the same text can be classified differently across runs, making
    # the reference subset irreproducible.
    DetectorFactory.seed = 0

    if not isinstance(text, str) or not text.strip():
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


# Exact-match administrative placeholder strings found in PTB-XL's "report"
# field: no cardiologist interpretation was ever written for these records -
# just an administrative note, with no diagnostic content at all.
# Strings - "trace only requested"/ "trace only" was the ONLY placeholder
# pattern present Matched case-insensitively after stripping whitespace :
# 20 records also start with "trace only requested." but have a real
# diagnostic code appended after an "Edit:" marker (e.g."trace only requested.
# Edit: CLBBB") - those carry clinical content despite the shared
# prefix and are deliberately NOT excluded here, since excluding them would
# discard real signal over a formatting quirk.

_ADMINISTRATIVE_PLACEHOLDER_REPORTS = {"trace only requested.", "trace only."}


def is_administrative_placeholder_report(report_text) -> bool:
    """
    Check whether a PTB-XL "report" field value is a pure administrative
    placeholder (e.g. "trace only requested.") rather than an actual
    clinical description - see _ADMINISTRATIVE_PLACEHOLDER_REPORTS for how
    this list is derived and why it is an exact-match, not substring,
    check.

    Inputs:
        report_text: raw string value from the "report" column (may be
            NaN/non-string).

    Output:
        True if report_text is a known administrative placeholder, False
        otherwise (including for non-string/NaN input).
    """
    if not isinstance(report_text, str):
        return False
    return report_text.strip().lower() in _ADMINISTRATIVE_PLACEHOLDER_REPORTS


def build_native_english_reference_subset(database_csv, validated_only: bool = True) -> pd.DataFrame:
    """
    NOT used by build_reference_subset anymore - kept only as input to
    cross_check_native_vs_rtlp, since detect_report_language was found
    unreliable on this dataset (13.3% false-positive rate, see module
    docstring). Filters PTB-XL's ptbxl_database.csv to records whose own
    "report" field is DETECTED as English (not verified as English),
    optionally restricted to validated_by_human == True, excluding
    administrative placeholders.

    Inputs:
        database_csv: path to PTB-XL's ptbxl_database.csv.
        validated_only: if True (default), also require
            validated_by_human == True.

    Output:
        DataFrame subset of the original database (same columns, plus a
        "detected_language" column), index preserved from the source file.
    """
    df = pd.read_csv(database_csv)
    df["detected_language"] = df["report"].apply(detect_report_language)

    subset = df[df["detected_language"] == "en"]
    if validated_only:
        subset = subset[subset["validated_by_human"] == True]
    subset = subset[~subset["report"].apply(is_administrative_placeholder_report)]
    # '~' is the NOT operator, used here to invert the boolean Series returned by is_administrative_placeholder_report
    return subset.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. RTLP English translations - now the PRIMARY reference source
# ----------------------------------------------------------------------
# Language-detection reliability finding that motivated abandoning
# detect_report_language for reference-pool construction: a full-dataset
# check (all 16247 PTB-XL records covered by RTLP) found detect_report_language
# flagged 6050 records as English, but only 5243 (86.7%) of those actually
# matched their own RTLP English translation - the other 807 (13.3%) were
# German (or other non-English) text misclassified as English. This is a real,
# systemic error rate on this dataset's very short specialized-vocabulary
# reports, not an isolated edge case - see cross_check_native_vs_rtlp for the full numbers.
def load_rtlp_translations(rtlp_repo_dir) -> Dict[int, str]:
    """
    Load RTLP's English translation of PTB-XL reports.

    Requires the RTLP repository cloned locally first:
        git clone https://github.com/danikiyasseh/RTLP

    Inputs:
        rtlp_repo_dir: path to the cloned RTLP repository root (the
            directory containing "translations/en_df_round4.csv").

    Output:
        dict mapping ecg_id (int) -> English report text (str). Verified
        file format: translations/en_df_round4.csv has exactly two columns,
        "ecg_id" and "report", with ecg_id using the same numbering as
        PTB-XL's own ptbxl_database.csv "ecg_id" column. NOT a full-dataset
        translation despite the source paper's description: verified
        coverage is 16247 of PTB-XL's 21799 records (74.5%) - the remaining
        5552 records have no RTLP translation at all and are therefore
        absent from any reference pool built from this source.

    Raises:
        FileNotFoundError: if the expected CSV is not found under
        rtlp_repo_dir - most likely meaning the repo has not been cloned
        yet, or a different subfolder layout is present.
    """
    csv_path = Path(rtlp_repo_dir) / "translations" / "en_df_round4.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected RTLP translations file not found at {csv_path}. "
            "Clone the repository first: git clone https://github.com/danikiyasseh/RTLP"
        )
    rtlp_df = pd.read_csv(csv_path)
    return dict(zip(rtlp_df["ecg_id"], rtlp_df["report"]))


def build_reference_subset(
    database_csv,
    rtlp_repo_dir: str,
    validated_only: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build the final reference-report table used for evaluation, from RTLP's
    English translations for every ecg_id RTLP covers - detect_report_language
    is NOT used here at all;
    Optionally restricted to validated_by_human == True, and always excludes
    administrative-placeholder reports.

    Inputs:
        database_csv: path to PTB-XL's ptbxl_database.csv.
        rtlp_repo_dir: path to a locally cloned RTLP repository (see
            load_rtlp_translations) - now REQUIRED, not an optional
            fallback, since RTLP is the sole reference source.
        validated_only: if True (default), require validated_by_human == True.

    Output:
        (reference_df, summary) tuple:
        - reference_df: DataFrame with columns ["ecg_id", "record_id",
          "reference_report", "reference_source", "filename_lr",
          "patient_id", "strat_fold", "validated_by_human"].
          "record_id" is the basename of filename_lr (e.g. "00001_lr"),
          matching the record_id convention used elsewhere in this project
          (see ecg_agent.py / sep_agent_demo.ipynb). "reference_source" is
          always "rtlp_translation" now.
        - summary: dict with "rtlp_coverage_size" (records RTLP has a
          translation for at all), "validated_size" (after the
          validated_by_human filter), "placeholder_excluded_count", and
          "total_reference_pool_size" (final size) - meant to be
          logged/saved verbatim for the report (see write_interpretation_note).
    """
    full_df = pd.read_csv(database_csv)
    rtlp_map = load_rtlp_translations(rtlp_repo_dir)

    candidates = full_df[full_df["ecg_id"].isin(rtlp_map.keys())].copy()
    rtlp_coverage_size = len(candidates)

    if validated_only:
        candidates = candidates[candidates["validated_by_human"] == True]
    validated_size = len(candidates)
    
    candidates["reference_report"] = candidates["ecg_id"].map(rtlp_map)
    is_placeholder = candidates["reference_report"].apply(is_administrative_placeholder_report)
    placeholder_excluded_count = int(is_placeholder.sum())
    candidates = candidates[~is_placeholder]

    reference_df = pd.DataFrame(
        {
            "ecg_id": candidates["ecg_id"],
            "record_id": candidates["filename_lr"].apply(lambda p: Path(p).name),
            "reference_report": candidates["reference_report"],
            "reference_source": "rtlp_translation",
            "filename_lr": candidates["filename_lr"],
            "patient_id": candidates["patient_id"],
            "strat_fold": candidates["strat_fold"],
            "validated_by_human": candidates["validated_by_human"],
        }
    ).reset_index(drop=True)

    summary = {
        "rtlp_coverage_size": rtlp_coverage_size,
        "validated_size": validated_size,
        "placeholder_excluded_count": placeholder_excluded_count,
        "total_reference_pool_size": len(reference_df),
    }
    return reference_df, summary


def cross_check_native_vs_rtlp(database_csv, rtlp_repo_dir, validated_only: bool = True) -> Dict[str, Any]:
    """
    This exists purely to quantify and document the 
    language-detection reliability finding that motivated that change.

    Method: for each record, check if detect_report_language says "English",
    then compare the native report text to its RTLP translation. If the
    native text was really English, the RTLP "translation" should look almost
    the same (nothing to translate). If they're very different, the record
    was probably misdetected as English by mistake.

    Inputs:
        database_csv: path to PTB-XL's ptbxl_database.csv.
        rtlp_repo_dir: path to a locally cloned RTLP repository.
        validated_only: if True (default), restrict to validated_by_human == True.

    Output:
        dict with "rtlp_coverage_size", "detected_as_english_count",
        "exact_match_count" (of those detected as English, how many exactly
        match their RTLP translation - i.e. were genuinely English),
        "mismatch_count" (likely language-detection false positives), and
        "mismatch_rate", plus "example_mismatches" (up to 5 (ecg_id, native,
        rtlp) tuples for inspection).
    """
    full_df = pd.read_csv(database_csv)
    rtlp_map = load_rtlp_translations(rtlp_repo_dir)

    covered = full_df[full_df["ecg_id"].isin(rtlp_map.keys())].copy()
    if validated_only:
        covered = covered[covered["validated_by_human"] == True]

    covered["detected_lang"] = covered["report"].apply(detect_report_language)
    covered["rtlp_report"] = covered["ecg_id"].map(rtlp_map)
    covered["exact_match"] = covered.apply(
        lambda r: str(r["report"]).strip().lower() == str(r["rtlp_report"]).strip().lower(), axis=1
    )

    detected_en = covered[covered["detected_lang"] == "en"]
    mismatches = detected_en[~detected_en["exact_match"]]

    return {
        "rtlp_coverage_size": len(covered),
        "detected_as_english_count": len(detected_en),
        "exact_match_count": int(detected_en["exact_match"].sum()),
        "mismatch_count": len(mismatches),
        "mismatch_rate": len(mismatches) / len(detected_en) if len(detected_en) else float("nan"),
        "example_mismatches": [
            (int(r["ecg_id"]), r["report"], r["rtlp_report"]) for _, r in mismatches.head(5).iterrows()
        ],
    }


def assign_primary_diagnostic_class(reference_df: pd.DataFrame, database_csv, scp_statements_csv) -> pd.Series:
    """
    Assign each row in a reference_df (see build_reference_subset) a single
    "primary" PTB-XL diagnostic superclass, for use as a stratification key.

    Reuses ecg_embedding.py's existing scp_codes -> multi-hot label mapping
    (load_scp_superclass_map, scp_codes_to_multihot) rather than duplicating
    it - read-only reuse, ecg_embedding.py itself is not modified.


    Inputs:
        reference_df: DataFrame as returned by build_reference_subset (must
            have an "ecg_id" column).
        database_csv: path to PTB-XL's ptbxl_database.csv (source of
            "scp_codes" for each ecg_id).
        scp_statements_csv: path to PTB-XL's scp_statements.csv.

    Output:
        pandas Series (same index as reference_df) of primary class strings,
        one of DIAGNOSTIC_SUPERCLASSES, or "NONE" for the rare record with no
        positive diagnostic superclass at all (~1% of the reference pool).
    """
    code_map = load_scp_superclass_map(scp_statements_csv)
    scp_by_ecg_id = pd.read_csv(database_csv, index_col="ecg_id")["scp_codes"]

    def primary_class(ecg_id):
        label = scp_codes_to_multihot(scp_by_ecg_id.loc[ecg_id], code_map)
        positive = np.nonzero(label)[0]
        return DIAGNOSTIC_SUPERCLASSES[positive[0]] if len(positive) else "NONE"

    return reference_df["ecg_id"].apply(primary_class)


def stratified_sample(reference_df: pd.DataFrame, primary_classes: pd.Series, n_samples: int, random_seed: int = 42) -> pd.DataFrame:
    """
    Sample records with (approximately) equal representation across each
    diagnostic superclass, instead of plain random sampling.

    Rationale: plain random sampling at a small N over 5 overlapping classes
    can (and, in an earlier run of this pipeline, did) produce a skewed
    sample - e.g. 53% NORM / 10% MI in a 30-record random draw, versus 46%
    NORM / 21% MI in the full reference pool. That skew inflates or deflates
    aggregate scores for reasons unrelated to report quality.

    Inputs:
        reference_df: DataFrame as returned by build_reference_subset.
        primary_classes: Series as returned by assign_primary_diagnostic_class,
            same index as reference_df.
        n_samples: total number of records to sample; divided as evenly as
            possible across DIAGNOSTIC_SUPERCLASSES. Records with primary_class
            "NONE" are excluded from the sampling pool
        random_seed: seed for reproducible sampling.

    Output:
        DataFrame subset of reference_df (reset index), with an added
        "primary_class" column, size n_samples (or fewer if a class has
        fewer available records than its quota).
    """
    rng = np.random.default_rng(random_seed)
    #rng a generator that produces random numbers, seeded for reproducibility
    classes = list(DIAGNOSTIC_SUPERCLASSES)
    base_quota, remainder = divmod(n_samples, len(classes))
    #divmod returns the quotient and the rest when dividing n_samples by the number of classes
    quotas = {cls: base_quota + (1 if i < remainder else 0) for i, cls in enumerate(classes)}
    #gives each class its base quota, and gives an extra record to the first remainder classes in the list, so that the total comes out exactly right.
    sampled_parts = []
    for cls in classes:
        pool = reference_df[primary_classes == cls]
        take = min(quotas[cls], len(pool))
        if take > 0:
            idx = rng.choice(pool.index, size=take, replace=False)
            part = reference_df.loc[idx].copy()
            part["primary_class"] = cls
            sampled_parts.append(part)

    return pd.concat(sampled_parts, ignore_index=True) if sampled_parts else reference_df.iloc[0:0].copy()


# ----------------------------------------------------------------------
# 3. Generated text: existing agents' outputs, unmodified, ECG-only
# ----------------------------------------------------------------------
def load_ecg_signal(dataset_dir, filename_lr) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Load a single PTB-XL record's signal - both the single lead used by the
    rest of this project's ECG pipeline (see ecg_preprocessing.py's
    documented single-lead limitation) and the full multi-lead array needed
    for the optional learned-embedding integration (see
    get_or_create_ecg_contribution / ecg_agent.compute_embedding_features).

    Inputs:
        dataset_dir: PTB-XL root directory (containing "records100/", etc.).
        filename_lr: value from the database's "filename_lr" column.

    Output:
        (signal, fs, multilead_signal) tuple: signal is a 1D numpy array
        (lead 0 only), fs is the sampling frequency in Hz, multilead_signal
        is the full (n_samples, n_leads) array (wfdb.rdsamp's raw shape).
    """
    import wfdb

    sig, meta = wfdb.rdsamp(str(Path(dataset_dir) / filename_lr))
    return sig[:, 0], meta["fs"], sig


def get_or_create_ecg_contribution(
    record_id: str,
    dataset_dir,
    filename_lr,
    contributions_dir,
    embedding_model=None,
) -> AgentContribution:
    """
    Obtain the ECG AgentContribution for one record: reuse a previously
    saved contribution JSON if one already exists at the expected path
    (<output_dir>/<file_name>.json),
    otherwise run the full ECG pipeline + generate_ecg_contribution() fresh
    and save the result there for next time.


    Inputs:
        record_id: identifier used as both the AgentContribution.record_id
            and the JSON file's containing subfolder name (e.g. "00001_lr").
        dataset_dir: PTB-XL root directory.
        filename_lr: value from the database's "filename_lr" column, used to
            load the signal if regeneration is needed.
        contributions_dir: base directory under which each record's
            contribution is cached, one subfolder per record_id.
        embedding_model: optional trained ECGEmbeddingNet instance;
            if given, the full multi-lead signal is loaded and passed through to
            generate_ecg_contribution so its findings include the learned
            diagnostic-superclass probabilities. Omitted by default (no
            behavior change from before this parameter existed).

    Output:
        AgentContribution instance (agent_id="ecg").
    """
    record_dir = Path(contributions_dir) / record_id
    cached_path = record_dir / "ecg_contribution.json"

    if cached_path.exists():
        with open(cached_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return AgentContribution(
            agent_id=payload["agent_id"],
            record_id=payload["record_id"],
            findings=[Finding(**item) for item in payload["findings"]],
            flags_for_review=payload["flags_for_review"],
            trace=payload["trace"],
        )

    signal, fs, multilead_signal = load_ecg_signal(dataset_dir, filename_lr)
    results = ecg_preprocessing_pipeline(signal, fs)
    contribution = generate_ecg_contribution(
        results,
        fs,
        record_id=record_id,
        multilead_signal=multilead_signal if embedding_model is not None else None,
        embedding_model=embedding_model,
    )
    save_contribution(contribution, output_dir=str(record_dir), file_name="ecg_contribution")
    return contribution


def generate_text_for_scoring(ecg_contribution: AgentContribution, record_id: str) -> str:
    """
    Produce ECG-only free-text narrative for scoring, by calling
    sep_agent.generate_sep_agent_decision() with a list containing ONLY the
    given ECG contribution.

    Inputs:
        ecg_contribution: AgentContribution instance (agent_id="ecg").
        record_id: identifier stored on the resulting SepAgentDecision.

    Output:
        The SepAgentDecision.summary string (the 3-5 sentence integrated
        narrative) - used as "generated_text" for BLEU/ROUGE/BERTScore.
    """
    decision = generate_sep_agent_decision([ecg_contribution], record_id=record_id)
    return decision.summary


def build_brief_summary_prompt(ecg_contribution: AgentContribution) -> str:
    """
    Build a prompt for a SHORT, telegraphic, number-free clinical note, in
    the style of real cardiologist shorthand (e.g. "sinus rhythm normal
    ecg"), as an alternative style to generate_text_for_scoring's 3-5
    sentence narrative.

    This is a deliberately SEPARATE, dedicated prompt/function - it does
    NOT go through sep_agent.py or modify build_sep_agent_prompt in any way.
    sep_agent.py stays exactly as-is and keeps being used for its own
    purposes (the Multi-Agent Case Report, sep_agent_demo.ipynb, etc.) -
    this function exists only to test, in this evaluation module, whether a
    terser/number-free STYLE (as opposed to the descriptive-vs-diagnostic
    CONTENT gap, tested separately via the embedding integration) narrows
    the gap to PTB-XL's telegraphic reference reports. If this style turns
    out to score meaningfully better, that would be a separate decision to
    make about sep_agent.py's default prompt - not something this function
    does on its own.

    Inputs:
        ecg_contribution: AgentContribution instance (agent_id="ecg").

    Output:
        Prompt string ready to send to an LLM backend.
    """
    if not ecg_contribution.findings:
        findings_block = "(no findings - insufficient signal for this modality)"
    else:
        findings_block = "\n".join(
            f'- feature="{f.feature}": {f.value} (confidence {f.confidence:.2f}) - {f.rationale}'
            for f in ecg_contribution.findings
        )

    return f"""You are summarizing ECG findings into a SHORT clinical note, in the
style of real cardiologist shorthand (e.g. "sinus rhythm normal ecg",
"sinus tachycardia otherwise normal ecg", "sinusrhythmus rechtstyp
linksposteriorer hemiblock"). Do not introduce any clinical fact not
present in the findings below.

Findings:
{findings_block}

Task: write a SHORT telegraphic clinical note (3-10 words, comma/period
separated phrases, not full sentences) summarizing these findings.

Rules:
- Use categorical/qualitative language only (e.g. "normal heart rate",
  "tachycardia", "possible conduction disturbance pattern") - do NOT
  include any raw numeric value (no bpm, ms, %, or digits) anywhere in
  your output.
- Do not use full grammatical sentences - use short telegraphic phrases,
  matching real ECG report shorthand style.
- Respond with ONLY the short note text - no JSON, no extra commentary,
  no markdown.
"""


def generate_brief_ecg_summary(
    ecg_contribution: AgentContribution,
    llm_call: Optional[Any] = None,
) -> str:
    """
    Generate a short, telegraphic, number-free ECG summary via
    build_brief_summary_prompt - the "brief" alternative to
    generate_text_for_scoring, for testing whether a terser style (as
    opposed to content) narrows the gap to PTB-XL's telegraphic reference
    reports. Calls agent_common.generate_report_medgemma directly - does
    NOT go through sep_agent.py, so that module's default behavior is
    completely unaffected by this experiment.

    Inputs:
        ecg_contribution: AgentContribution instance (agent_id="ecg").
        llm_call: callable taking a single prompt string and returning the
            raw text response. Defaults to generate_report_medgemma (free,
            local, no API key).

    Output:
        Raw text response (str) - the short clinical note, used as
        "generated_text" for BLEU/ROUGE/BERTScore in place of
        generate_text_for_scoring's output.
    """
    if llm_call is None:
        llm_call = generate_report_medgemma

    prompt = build_brief_summary_prompt(ecg_contribution)
    return llm_call(prompt).strip()


# ----------------------------------------------------------------------
# 4. Scoring: BLEU (sacrebleu), ROUGE (rouge-score), BERTScore (bert-score)
# ----------------------------------------------------------------------
def compute_bleu(generated_text: str, reference_text: str) -> float:
    """
    Sentence-level BLEU score for one (generated, reference) pair.

    Library/variant: sacrebleu's sentence_bleu (BLEU with the standard
    4-gram weighting and exponential smoothing method "exp", sacrebleu's
    default) - chosen over the older nltk implementation because sacrebleu
    applies a standardized, reproducible tokenization, which matters for
    reports that are only one short clause long (this project's reference
    reports are extremely short - so smoothing matters more here than on
    long-document BLEU use cases).

    Inputs:
        generated_text: model-generated narrative text.
        reference_text: reference report text (single reference; PTB-XL/ RTLP).

    Output:
        BLEU score as a float in [0, 100] (sacrebleu's native scale).
    """
    import sacrebleu

    return sacrebleu.sentence_bleu(generated_text, [reference_text]).score


def compute_rouge(generated_text: str, reference_text: str) -> Dict[str, float]:
    """
    ROUGE-1, ROUGE-2, and ROUGE-L F-measure for one (generated, reference)
    pair.

    Library/variant: the "rouge-score" package (Google's reference
    implementation), stemmer enabled (use_stemmer=True) to reduce words
    to their root form - appropriate here given the reference reports'
    telegraphic, non-fluent style.

    Inputs:
        generated_text: model-generated narrative text.
        reference_text: reference report text.

    Output:
        dict with keys "rouge1", "rouge2", "rougeL", each an F-measure float
        in [0, 1].
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return {key: scores[key].fmeasure for key in ("rouge1", "rouge2", "rougeL")}


def compute_bertscore_batch(
    generated_texts: List[str], reference_texts: List[str], model_type: str = "distilbert-base-uncased"
) -> Tuple[List[float], List[float], List[float]]:
    """
    BERTScore precision/recall/F1 for a whole batch of (generated,
    reference) pairs at once.

    Library/variant: the "bert-score" package's bert_score.score(). Batched
    here (rather than one call per pair) because each call loads a
    transformer model into memory, doing that once for N pairs instead of N
    times is both much faster and avoids redundant memory churn.

    model_type defaults to "distilbert-base-uncased" rather than
    bert-score's own default ("roberta-large"): roberta-large gives
    marginally better correlation with human judgment in the original
    BERTScore paper, but is a much larger download/inference cost. Given
    this project's local, resource-constrained hardware, the lighter model is used
    by default; pass model_type="roberta-large" to use the paper's default
    if bandwidth/compute allows.

    Inputs:
        generated_texts: list of model-generated narrative texts.
        reference_texts: list of reference report texts, same length and
            order as generated_texts.
        model_type: HuggingFace model name used as BERTScore's backbone.

    Output:
        (precision, recall, f1) tuple, each a list of floats, one per pair,
        same order as the inputs.
    """
    import bert_score

    precision, recall, f1 = bert_score.score(
        generated_texts, reference_texts, model_type=model_type, lang="en", verbose=False
    )
    return precision.tolist(), recall.tolist(), f1.tolist()


def compute_bertscore_contrastive_baseline(
    generated_texts: List[str], reference_texts: List[str], model_type: str = "distilbert-base-uncased"
) -> Tuple[float, float, List[float]]:
    """
    Compute a BERTScore F1 baseline for MISMATCHED (generated, reference)
    pairs, as a sanity check on whether the real pairs' BERTScore is
    actually meaningful.

    Rationale: raw BERTScore is known to compress into a narrow high band -
    even semantically unrelated sentence pairs commonly score 0.6-0.75 with
    this kind of model, simply because
    the embedding space isn't uniformly spread out. A real-pairs mean of,
    say, 0.70 is only meaningful evidence of semantic alignment if
    genuinely mismatched pairs score noticeably lower - otherwise 0.70 may
    just be this model's baseline "floor" for any two English clinical
    sentences, real match or not.

    Mismatching method: each reference is paired with the generated text of
    a DIFFERENT record via a cyclic shift by one position - deterministic
    and reproducible, and guarantees no reference is paired with its own
    generated text as long as there are at least 2 pairs and all reference
    texts are not identical.

    Inputs:
        generated_texts: list of model-generated narrative texts (same list
            used for the real scoring).
        reference_texts: list of reference report texts, same order as
            generated_texts.
        model_type: passed through to compute_bertscore_batch.

    Output:
        (mean_f1, std_f1, per_pair_f1) tuple over the mismatched pairs.
        per_pair_f1 is a list, same length/order as the inputs, so the
        caller can persist individual mismatched-pair scores (see
        run_evaluation, which saves these to evaluation_results.csv rather
        than keeping only the aggregate mean/std). Compare mean_f1 directly
        to the real pairs' mean BERTScore F1 (see write_interpretation_note):
        a small or negative gap means the real score is not distinguishable
        from this model's baseline noise floor.
    """
    if len(generated_texts) < 2:
        return float("nan"), float("nan"), []

    shifted_references = reference_texts[1:] + reference_texts[:1]
    _, _, f1 = compute_bertscore_batch(generated_texts, shifted_references, model_type=model_type)
    return float(np.mean(f1)), float(np.std(f1)), f1


# ----------------------------------------------------------------------
# 5. End-to-end evaluation pipeline
# ----------------------------------------------------------------------
def run_evaluation(
    database_csv,
    dataset_dir,
    rtlp_repo_dir: str,
    output_dir="results/evaluation",
    scp_statements_csv: Optional[str] = None,
    validated_only: bool = True,
    n_samples: Optional[int] = 30,
    stratify_by_class: bool = True,
    random_seed: int = 42,
    bertscore_model_type: str = "distilbert-base-uncased",
    embedding_model=None,
    brief_mode: bool = False,
    clinical_embedding_model=None,
    clinical_embedding_model_name: str = CLINICAL_EMBEDDING_MODEL,
) -> pd.DataFrame:
    """
    Run the full ECG-only report evaluation: build the reference subset,
    sample n_samples records from it, generate ECG-only text for each
    (reusing/caching AgentContributions), score every pair with BLEU/ROUGE/
    BERTScore plus a BERTScore contrastive (mismatched-pair) baseline, and
    save a results CSV plus an interpretation note.

    n_samples defaults to 30, NOT the full reference pool (thousands of
    records): each record requires two real local LLM calls (one for
    generate_ecg_contribution, one for generate_sep_agent_decision), and
    this project's own measurements (see CLAUDE.md) show local Ollama
    inference costs several seconds to tens of seconds per call plus a
    ~10-15s cold-load penalty - running the full pool would take hours for
    a demonstration evaluation. Pass n_samples=None to run on the entire
    reference pool if that time cost is acceptable.

    stratify_by_class (default True): sample (approximately) equally across
    PTB-XL's 5 diagnostic superclasses (see stratified_sample) instead of
    plain random sampling. Plain random sampling at small N was found in
    practice to skew heavily toward NORM and underrepresent MI (a 30-record
    random draw came out 53% NORM / 10% MI, versus 46% / 21% in the full
    pool) - since NORM reports are shorter/more templated, that skew can
    make aggregate scores look better or worse for reasons unrelated to
    report quality. Set to False to restore plain random sampling.

    Records where the ECG pipeline finds fewer than 2 QRS complexes (see
    ecg_agent.detect_flags's skip_llm_call) produce an ECG contribution
    with no findings; these are still scored (an empty-findings summary is
    a valid, if uninformative, generated text) but flagged with
    status="skipped_insufficient_ecg_signal" so they can be filtered out of
    the aggregate if desired. Any other per-record failure (e.g. a
    non-retriable parse error) is caught, logged with
    status="failed:<error>", and does not stop the rest of the run.

    Inputs:
        database_csv: path to PTB-XL's ptbxl_database.csv.
        dataset_dir: PTB-XL root directory (containing "records100/", etc.).
        output_dir: directory to write evaluation_results.csv and the
            interpretation note into; also holds a "contributions/"
            subfolder used as this evaluation's AgentContribution cache
            (see get_or_create_ecg_contribution).
        scp_statements_csv: path to PTB-XL's scp_statements.csv, needed for
            stratify_by_class's diagnostic-class labels; defaults to
            "scp_statements.csv" next to database_csv (PTB-XL's standard
            layout) if not given. Unused if stratify_by_class is False.
        rtlp_repo_dir: path to a locally cloned RTLP repository - now the
            sole reference source (see build_reference_subset), not optional.
        validated_only: if True (default), require validated_by_human ==
            True when selecting reference reports.
        n_samples: number of records to sample from the reference pool, or
            None to use the entire pool (see caveat above).
        stratify_by_class: see rationale above.
        random_seed: seed for the sampling, for reproducibility.
        bertscore_model_type: passed through to compute_bertscore_batch.
        embedding_model: optional trained ECGEmbeddingNet instance (see
            ecg_agent.compute_embedding_features); if given, every record's
            ECG contribution includes the learned diagnostic-superclass
            probabilities as findings (see get_or_create_ecg_contribution).
            Use a fresh output_dir when toggling this on/off between runs -
            a contributions/ cache built without it will NOT be
            regenerated just because this is now given.
        brief_mode: if True, use generate_brief_ecg_summary (short,
            telegraphic, number-free style, via its own dedicated prompt -
            does NOT touch sep_agent.py) instead of generate_text_for_scoring
            (the default 3-5 sentence narrative via sep_agent.py) as the
            generated_text for scoring. An experiment testing whether STYLE
            (as opposed to the descriptive-vs-diagnostic CONTENT gap tested
            by embedding_model) narrows the gap to PTB-XL's telegraphic
            references. sep_agent.py's own default behavior is unaffected
            either way.

    Output:
        DataFrame with one row per evaluated record (ecg_id, record_id,
        primary_class, reference_source, status, generated_text,
        reference_report, bleu, rouge1, rouge2, rougeL, bertscore_precision,
        bertscore_recall, bertscore_f1) plus a final row with ecg_id "MEAN"
        and one with "STD" holding the aggregate across successfully-scored
        records. Also written to <output_dir>/evaluation_results.csv. The
        BERTScore contrastive baseline (see compute_bertscore_contrastive_baseline)
        is written into evaluation_interpretation.md, not this DataFrame.
    """
    output_dir = Path(output_dir)
    contributions_dir = output_dir / "contributions"
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_df, subset_summary = build_reference_subset(
        database_csv, rtlp_repo_dir=rtlp_repo_dir, validated_only=validated_only
    )
    print(f"Reference pool: {subset_summary}")

    if stratify_by_class and n_samples is not None and n_samples < len(reference_df):
        if scp_statements_csv is None:
            scp_statements_csv = str(Path(database_csv).parent / "scp_statements.csv")
        primary_classes = assign_primary_diagnostic_class(reference_df, database_csv, scp_statements_csv)
        sample_df = stratified_sample(reference_df, primary_classes, n_samples, random_seed=random_seed)
    elif n_samples is not None and n_samples < len(reference_df):
        sample_df = reference_df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
        sample_df["primary_class"] = None
    else:
        sample_df = reference_df.reset_index(drop=True)
        sample_df["primary_class"] = None
    print(f"Evaluating {len(sample_df)} records.")

    rows = []
    for _, row in sample_df.iterrows():
        record_id = row["record_id"]
        try:
            contribution = get_or_create_ecg_contribution(
                record_id, dataset_dir, row["filename_lr"], contributions_dir, embedding_model=embedding_model
            )
            if brief_mode:
                generated_text = generate_brief_ecg_summary(contribution)
            else:
                generated_text = generate_text_for_scoring(contribution, record_id)
            status = "ok" if contribution.findings else "skipped_insufficient_ecg_signal"
        except Exception as exc:  # noqa: BLE001 - any per-record failure must not abort the batch
            rows.append(
                {
                    "ecg_id": row["ecg_id"],
                    "record_id": record_id,
                    "primary_class": row["primary_class"],
                    "reference_source": row["reference_source"],
                    "status": f"failed:{exc}",
                    "generated_text": None,
                    "reference_report": row["reference_report"],
                }
            )
            continue

        rows.append(
            {
                "ecg_id": row["ecg_id"],
                "record_id": record_id,
                "primary_class": row["primary_class"],
                "reference_source": row["reference_source"],
                "status": status,
                "generated_text": generated_text,
                "reference_report": row["reference_report"],
            }
        )

    results_df = pd.DataFrame(rows)
    scorable = results_df["generated_text"].notna()

    # BLEU/ROUGE are cheap (no model loading) - computed per row directly.
    results_df.loc[scorable, "bleu"] = results_df[scorable].apply(
        lambda r: compute_bleu(r["generated_text"], r["reference_report"]), axis=1
    )
    rouge_scores = results_df[scorable].apply(
        lambda r: compute_rouge(r["generated_text"], r["reference_report"]), axis=1
    )
    for key in ("rouge1", "rouge2", "rougeL"):
        results_df.loc[scorable, key] = rouge_scores.apply(lambda d: d[key])

    # BERTScore is batched once across every scorable pair (see
    # compute_bertscore_batch's docstring for why this matters).
    contrastive_baseline = (float("nan"), float("nan"))
    if scorable.any():
        generated_list = results_df.loc[scorable, "generated_text"].tolist()
        reference_list = results_df.loc[scorable, "reference_report"].tolist()
        record_id_list = results_df.loc[scorable, "record_id"].tolist()
        precision, recall, f1 = compute_bertscore_batch(
            generated_list, reference_list, model_type=bertscore_model_type
        )
        results_df.loc[scorable, "bertscore_precision"] = precision
        results_df.loc[scorable, "bertscore_recall"] = recall
        results_df.loc[scorable, "bertscore_f1"] = f1
        # --------------------------------------------------------------
        # Biomedical / clinical embedding cosine similarity
        # --------------------------------------------------------------
        if clinical_embedding_model is None:
            clinical_embedding_model = load_clinical_embedding_model(
                clinical_embedding_model_name
            )

        clinical_cosine_scores = compute_clinical_cosine_similarity_batch(
            generated_list,
            reference_list,
            clinical_embedding_model,
        )

        results_df.loc[
            scorable, "clinical_cosine_similarity"
        ] = clinical_cosine_scores
        
        mean_f1, std_f1, per_pair_f1 = compute_bertscore_contrastive_baseline(
            generated_list, reference_list, model_type=bertscore_model_type
        )
        contrastive_baseline = (mean_f1, std_f1)
        # Persist the individual mismatched-pair scores, not just the
        # aggregate mean/std - lets the contrastive check be audited or
        # re-analyzed later instead of only trusting the summary numbers.
        results_df.loc[scorable, "bertscore_f1_mismatched"] = per_pair_f1
        results_df.loc[scorable, "mismatched_against_record_id"] = record_id_list[1:] + record_id_list[:1]

    metric_cols = [
    "bleu", "rouge1", "rouge2", "rougeL",
    "bertscore_precision", "bertscore_recall", "bertscore_f1",  "bertscore_f1_mismatched",
    "clinical_cosine_similarity",
    ]

    aggregate_rows = []
    for label, fn in (("MEAN", np.mean), ("STD", np.std)):
        agg = {"ecg_id": label, "record_id": "", "primary_class": "", "reference_source": "", "status": ""}
        for col in metric_cols:
            agg[col] = fn(results_df.loc[scorable, col]) if scorable.any() else None
        aggregate_rows.append(agg)

    full_results_df = pd.concat([results_df, pd.DataFrame(aggregate_rows)], ignore_index=True)

    csv_path = output_dir / "evaluation_results.csv"
    full_results_df.to_csv(csv_path, index=False)
    print(f"Saved results table to {csv_path}")

    write_interpretation_note(full_results_df, subset_summary, output_dir, contrastive_baseline=contrastive_baseline)

    return full_results_df


# ----------------------------------------------------------------------
# 6. Interpretation note (for direct use in the project report)
# ----------------------------------------------------------------------
def write_interpretation_note(
    results_df: pd.DataFrame,
    subset_summary: Dict[str, Any],
    output_dir,
    contrastive_baseline: Tuple[float, float] = (float("nan"), float("nan")),
) -> str:
    """
    Write a short markdown interpretation note summarizing one evaluation
    run, meant to be dropped directly into the project report rather than
    read only as code comments.

    Inputs:
        results_df: DataFrame as returned by run_evaluation (including its
            "MEAN"/"STD" aggregate rows and, if stratify_by_class was used,
            a "primary_class" column).
        subset_summary: summary dict as returned by build_reference_subset.
        output_dir: directory to write "evaluation_interpretation.md" into.
        contrastive_baseline: (mean, std) of BERTScore F1 on mismatched pairs,
            as returned by compute_bertscore_contrastive_baseline - used to
            judge whether the real pairs' BERTScore is actually meaningful
            or within this model's baseline noise floor. Pass (nan, nan)
            (the default) to omit this section.

    Output:
        Path to the written markdown file (str).
    """
    mean_row = results_df[results_df["ecg_id"] == "MEAN"].iloc[0]
    std_row = results_df[results_df["ecg_id"] == "STD"].iloc[0]
    n_scored = results_df["status"].eq("ok").sum() + results_df["status"].eq(
        "skipped_insufficient_ecg_signal"
    ).sum()
    n_failed = results_df["status"].astype(str).str.startswith("failed:").sum()

    lines = [
        "# ECG report evaluation - results interpretation",
        "",
        "## Reference subset (RTLP English translations - language detection no longer used)",
        f"- PTB-XL records with an RTLP translation available: {subset_summary['rtlp_coverage_size']}",
        f"- After validated_by_human filter: {subset_summary['validated_size']}",
        f"- Administrative placeholders excluded: {subset_summary['placeholder_excluded_count']}",
        f"- Total reference pool available: {subset_summary['total_reference_pool_size']}",
        f"- Records actually evaluated in this run: {len(results_df) - 2} "
        f"({n_scored} scored, {n_failed} failed)",
        "",
        "## Aggregate scores (mean +/- std, over scored records)",
        f"- BLEU: {mean_row['bleu']:.2f} +/- {std_row['bleu']:.2f}",
        f"- ROUGE-1 (F1): {mean_row['rouge1']:.3f} +/- {std_row['rouge1']:.3f}",
        f"- ROUGE-2 (F1): {mean_row['rouge2']:.3f} +/- {std_row['rouge2']:.3f}",
        f"- ROUGE-L (F1): {mean_row['rougeL']:.3f} +/- {std_row['rougeL']:.3f}",
        f"- BERTScore F1: {mean_row['bertscore_f1']:.3f} +/- {std_row['bertscore_f1']:.3f}",
        f"- Biomedical embedding cosine similarity: "
        f"{mean_row['clinical_cosine_similarity']:.3f} +/- "
        f"{std_row['clinical_cosine_similarity']:.3f}",
    ]

    if "primary_class" in results_df.columns and results_df["primary_class"].notna().any():
        lines += ["", "## Per-class breakdown (stratified sample)"]
        per_class = results_df[
            results_df["status"].isin(["ok", "skipped_insufficient_ecg_signal"])
        ].groupby("primary_class")
        for cls, group in per_class:
            lines.append(
                f"- {cls} (n={len(group)}): BLEU {group['bleu'].mean():.2f}, "
                f"ROUGE-1 {group['rouge1'].mean():.3f}, BERTScore F1 {group['bertscore_f1'].mean():.3f}"
            )

    if not np.isnan(contrastive_baseline[0]):
        gap = mean_row["bertscore_f1"] - contrastive_baseline[0]
        lines += [
    "",
    "## BERTScore contrastive baseline (mismatched pairs)",
    f"- Real pairs: {mean_row['bertscore_f1']:.3f} +/- {std_row['bertscore_f1']:.3f}",
    f"- Mismatched pairs (each generated text scored against a DIFFERENT record's "
    f"reference): {contrastive_baseline[0]:.3f} +/- {contrastive_baseline[1]:.3f}",
    f"- Gap (real - mismatched): {gap:+.3f}",
    (
    "Raw BERTScore (without baseline rescaling) can occupy a relatively "
    "narrow and high numerical range, making the absolute score difficult "
    "to interpret in isolation. The contrastive baseline provides a useful "
    "reference point: a meaningfully positive gap indicates that the "
    "generated text is more similar to its true reference than to a "
    "mismatched reference. A small or negative gap means that the observed "
    "BERTScore may not provide strong evidence that the generated report "
    "matches its corresponding reference."
    ),
    "",
    "## Biomedical embedding cosine similarity",
    (
        "This metric compares the generated ECG narrative and the PTB-XL "
        "reference report as whole-report biomedical embeddings using cosine "
        "similarity. BERTScore instead compares contextual token embeddings "
        "through token-level matching and aggregates the matches into "
        "precision, recall, and F1. The embedding cosine score therefore "
        "provides a document-level semantic similarity measure. It should be "
        "interpreted as semantic similarity rather than direct clinical "
        "correctness."
    ),    
]

    lines += [
        "",
        "## Interpretation caveat",
        (
            "These scores measure textual/semantic overlap with a stylistically "
            "different, abbreviated reference - not clinical correctness. PTB-XL's "
            "reference reports are short, telegraphic clinical shorthand (e.g. "
            "\"sinusrhythmus normales ekg\"), while the generated text is a "
            "synthesized 3-5 sentence narrative. This length/style mismatch "
            "structurally depresses BLEU and ROUGE (both sensitive to exact "
            "n-gram/length overlap) even when the generated report is clinically "
            "consistent with the reference. A low BLEU/ROUGE score does not, by "
            "itself, mean the generated report is wrong. BERTScore is a semantic "
            "similarity measure and is comparatively less sensitive to this "
            "surface-form mismatch - weight it more heavily than BLEU/ROUGE when "
            "judging report quality from these numbers, but see the contrastive "
            "baseline above before treating the absolute BERTScore number as "
            "meaningful on its own. The evaluation sample is stratified to give "
            "roughly equal representation to each diagnostic class (see "
            "stratified_sample) rather than mirroring the reference pool's natural "
            "class distribution - the aggregate above is a stratified estimate of "
            "per-class performance, not a naturalistic average."
        ),
        "",
        (
            "Additional caveat - machine-translated references: reference reports now come "
            "from RTLP's machine translation of PTB-XL's mostly-German original reports "
            "(language detection was found unreliable on this dataset - 13.3% false-positive "
            "rate - and is no longer used to build the reference pool; see "
            "cross_check_native_vs_rtlp and the module docstring). This introduces "
            "translation-quality noise as a further source of mismatch alongside the "
            "length/style and descriptive-vs-diagnostic gaps above - a generated text scored "
            "against an imperfectly translated reference may be penalized (or credited) for "
            "reasons unrelated to its own quality."
        ),
    ]

    note_path = Path(output_dir) / "evaluation_interpretation.md"
    with open(note_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return str(note_path)


# ----------------------------------------------------------------------
# 7. One-page PDF summary (companion to report_pdf.py's pattern)
# ----------------------------------------------------------------------
def generate_evaluation_summary_pdf(results_df: pd.DataFrame, subset_summary: Dict[str, Any], output_path) -> str:
    """
    Render a one-page PDF summarizing one evaluation run: the reference
    subset used, aggregate scores, and the 5 worst/best-scoring records by
    BERTScore F1 (the metric this module's interpretation note recommends
    weighting most heavily - see write_interpretation_note). Each of those
    records shows its generated text next to its reference report, so the
    two can be compared directly rather than reading the generated text
    alone.

    Reuses report_pdf.py's table styling (_table_style) rather than
    duplicating it, without modifying report_pdf.py itself.

    Inputs:
        results_df: DataFrame as returned by run_evaluation (including its
            "MEAN"/"STD" aggregate rows).
        subset_summary: summary dict as returned by build_reference_subset.
        output_path: path to write the PDF to.

    Output:
        output_path as a string.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    from report_pdf import _table_style

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10))

    mean_row = results_df[results_df["ecg_id"] == "MEAN"].iloc[0]
    scored = results_df[results_df["status"] == "ok"].copy()
    scored = scored.sort_values("bertscore_f1")

    doc = SimpleDocTemplate(str(output_path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                             topMargin=2 * cm, bottomMargin=2 * cm)
    story = [
        Paragraph("ECG Report Evaluation Summary", styles["Title"]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"Reference pool (RTLP English translations, validated_by_human, "
            f"placeholders excluded): {subset_summary['total_reference_pool_size']} records "
            f"(of {subset_summary['rtlp_coverage_size']} RTLP-covered, "
            f"{subset_summary['placeholder_excluded_count']} excluded as placeholders). "
            f"Records evaluated in this run: {len(results_df) - 2}.",
            styles["Normal"],
        ),
        Spacer(1, 0.3 * cm),
        Paragraph("Aggregate scores (mean)", styles["Heading2"]),
        Table(
            [
                ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore F1"],
                [
                    f"{mean_row['bleu']:.2f}",
                    f"{mean_row['rouge1']:.3f}",
                    f"{mean_row['rouge2']:.3f}",
                    f"{mean_row['rougeL']:.3f}",
                    f"{mean_row['bertscore_f1']:.3f}",
                ],
            ],
            style=_table_style(),
        ),
        Spacer(1, 0.4 * cm),
        Paragraph(
            "Note: BLEU/ROUGE compare exact surface-form overlap with a much "
            "shorter, telegraphic reference report; BERTScore (semantic "
            "similarity) should be weighted more heavily - see "
            "evaluation_interpretation.md for the full caveat.",
            styles["Small"],
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Worst-scoring records (by BERTScore F1)", styles["Heading2"]),
        Table(
            [["record_id", "BERTScore F1", "Generated text", "Reference report"]]
            + [
                [
                    r["record_id"],
                    f"{r['bertscore_f1']:.3f}",
                    Paragraph(str(r["generated_text"])[:300], styles["Small"]),
                    Paragraph(str(r["reference_report"])[:300], styles["Small"]),
                ]
                for _, r in scored.head(5).iterrows()
            ],
            colWidths=[2 * cm, 2 * cm, 8 * cm, 5 * cm],
            style=_table_style(),
        ),
        Spacer(1, 0.4 * cm),
        Paragraph("Best-scoring records (by BERTScore F1)", styles["Heading2"]),
        Table(
            [["record_id", "BERTScore F1", "Generated text", "Reference report"]]
            + [
                [
                    r["record_id"],
                    f"{r['bertscore_f1']:.3f}",
                    Paragraph(str(r["generated_text"])[:300], styles["Small"]),
                    Paragraph(str(r["reference_report"])[:300], styles["Small"]),
                ]
                for _, r in scored.tail(5).iloc[::-1].iterrows()
            ],
            colWidths=[2 * cm, 2 * cm, 8 * cm, 5 * cm],
            style=_table_style(),
        ),
    ]

    doc.build(story)
    return str(output_path)