"""
Per-agent report evaluation (ECG signal vs ECG image), on the SAME 20 records.

For each of 20 held-out (PTB-XL fold 10) records we compare, against the SAME
PTB-XL/RTLP cardiologist reference report:
  - the ECG SIGNAL branch's generated narrative (signal -> ecg_agent -> Sep summary)
  - the ECG IMAGE branch's generated narrative (GenECG image -> ecg_image_agent
    + classifier -> Sep summary)
scored with BLEU, ROUGE-1, and clinical (biomedical) cosine similarity. Because
GenECG images ARE PTB-XL records rendered as images, both branches share the same
reference report and the same records, so the two 20-row tables are directly
comparable.

Only ECG has a cardiologist reference report (PTB-XL/RTLP), so this text
evaluation is ECG-only (signal + image); MRI and patient-data "accuracy" is their
flag validation (label-based), not a text-similarity table.

Honest caveat (see report_evaluation.py's findings): BLEU/ROUGE are low by design
here - the reference is a diagnostic interpretation, the generated text is a
quantitative/qualitative description (the prompt forbids diagnosis). The clinical
cosine is the more meaningful semantic metric.

Model thrashing note (16GB RAM -> Ollama keeps ONE model resident): work is phased
(all signal[medgemma] -> all image contributions[qwen] -> all image Sep[medgemma])
so models load ~3 times total, not per record. Resumable: generated texts are
cached to disk per record.

Run:  python src/agent_report_evaluation.py
Out:  results/agent_evaluation/
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("GENECG_DOWNLOAD_DIR", os.path.join(os.getcwd(), "genecg_downloaded"))

import report_evaluation as RE
from report_evaluation import (
    build_reference_subset, assign_primary_diagnostic_class, stratified_sample,
    load_ecg_signal, generate_text_for_scoring,
    compute_bleu, compute_rouge, compute_clinical_cosine_similarity_batch,
    load_clinical_embedding_model,
)
from agent_common import AgentContribution, Finding, save_contribution, generate_report_medgemma
from ecg_agent import generate_ecg_contribution
from ecg_preprocessing import ecg_preprocessing_pipeline
from sep_agent import generate_sep_agent_decision
import ecg_image_agent
from genecg_fixtures import fetch_test_image

ROOT = "kaggle/archive/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
DATABASE_CSV = f"{ROOT}/ptbxl_database.csv"
SCP_CSV = f"{ROOT}/scp_statements.csv"
RTLP_DIR = "RTLP"
EMBEDDING_CKPT = "results/ecg_embedding/embedding_stratfold.best.pt"
CLASSIFIER_CKPT = "results/ecg_image_pipeline/genecg_classifier.pt"
IMAGE_VLM_TAG = "qwen2.5vl:7b"
OUT = "results/agent_evaluation"
CONTRIB_DIR = f"{OUT}/contributions"
IMG_CONTRIB_DIR = f"{OUT}/image_contributions"
CACHE = f"{OUT}/generated_texts.json"
N = 20
SEED = 42
os.makedirs(OUT, exist_ok=True)


def _load_cache():
    return json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def _save_cache(cache):
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def _load_contribution(path):
    p = json.load(open(path, encoding="utf-8"))
    return AgentContribution(agent_id=p["agent_id"], record_id=p["record_id"],
                             findings=[Finding(**f) for f in p["findings"]],
                             flags_for_review=p["flags_for_review"], trace=p["trace"])


def main():
    # --- sample 20 held-out (fold 10) records with a reference report ----------
    ref_df, summary = build_reference_subset(DATABASE_CSV, RTLP_DIR, validated_only=True)
    ref_df = ref_df[ref_df["strat_fold"] == 10].reset_index(drop=True)
    print(f"reference pool (fold 10): {len(ref_df)} records")
    prim = assign_primary_diagnostic_class(ref_df, DATABASE_CSV, SCP_CSV)
    sample = stratified_sample(ref_df, prim, n_samples=N, random_seed=SEED).reset_index(drop=True)
    records = [{"ecg_id": int(r.ecg_id), "record_id": r.record_id,
                "filename_lr": r.filename_lr, "reference": r.reference_report}
               for r in sample.itertuples()]
    print(f"sampled {len(records)} records")

    cache = _load_cache()

    # --- Phase 1: SIGNAL branch (MedGemma resident) ----------------------------
    print("\n=== Phase 1: signal branch (single-lead) ===")
    for i, r in enumerate(records):
        rid = r["record_id"]
        entry = cache.setdefault(rid, {"ecg_id": r["ecg_id"], "reference": r["reference"]})
        if entry.get("signal_text"):
            continue
        try:
            # Single-lead form for tractability (the multilead+embedding form emits
            # ~35 findings, needs num_predict=4096, and is ~80s/record - too slow to
            # run reliably here). The generated NARRATIVE (rate/rhythm/QRS description)
            # is what's scored, and is single-lead-derived; noted in the report.
            signal, fs, _ = load_ecg_signal(ROOT, r["filename_lr"])
            results = ecg_preprocessing_pipeline(signal, fs)
            contrib = generate_ecg_contribution(results, fs, record_id=rid, max_retries=2)
            save_contribution(contrib, output_dir=f"{CONTRIB_DIR}/{rid}", file_name="ecg_contribution")
            entry["signal_text"] = generate_text_for_scoring(contrib, rid)
            entry.pop("signal_error", None)
        except Exception as e:  # noqa: BLE001
            entry["signal_text"] = ""
            entry["signal_error"] = f"{type(e).__name__}: {e}"
        _save_cache(cache)
        print(f"  [{i+1}/{N}] {rid} signal done")

    # --- Phase 2: IMAGE contributions (qwen resident) --------------------------
    print("\n=== Phase 2: image contributions (VLM + classifier) ===")
    os.makedirs(IMG_CONTRIB_DIR, exist_ok=True)
    from genecg_classifier_train import load_genecg_classifier
    clf = load_genecg_classifier(CLASSIFIER_CKPT, device="cpu")
    ecg_image_agent.DEFAULT_IMAGE_MODEL_TAG = IMAGE_VLM_TAG
    for i, r in enumerate(records):
        rid = r["record_id"]
        cpath = f"{IMG_CONTRIB_DIR}/{rid}.json"
        if cache.get(rid, {}).get("image_text") or os.path.exists(cpath):
            continue
        try:
            img = fetch_test_image(r["ecg_id"], "A")
            c = ecg_image_agent.generate_image_ecg_contribution(
                img, record_id=rid, classifier_model=clf, max_retries=2)
            save_contribution(c, output_dir=IMG_CONTRIB_DIR, file_name=rid)
        except Exception as e:  # noqa: BLE001
            cache.setdefault(rid, {})["image_text"] = ""
            cache[rid]["image_error"] = f"{type(e).__name__}: {e}"
            _save_cache(cache)
        print(f"  [{i+1}/{N}] {rid} image contribution done")

    # --- Phase 3: IMAGE Sep summaries (MedGemma resident) ----------------------
    print("\n=== Phase 3: image Sep summaries ===")
    for i, r in enumerate(records):
        rid = r["record_id"]
        entry = cache.setdefault(rid, {"ecg_id": r["ecg_id"], "reference": r["reference"]})
        if entry.get("image_text"):
            continue
        cpath = f"{IMG_CONTRIB_DIR}/{rid}.json"
        if not os.path.exists(cpath):
            entry["image_text"] = ""
            continue
        try:
            entry["image_text"] = generate_text_for_scoring(_load_contribution(cpath), rid)
        except Exception as e:  # noqa: BLE001
            entry["image_text"] = ""
            entry["image_error"] = f"{type(e).__name__}: {e}"
        _save_cache(cache)
        print(f"  [{i+1}/{N}] {rid} image Sep done")

    # --- Phase 4: scoring ------------------------------------------------------
    print("\n=== Phase 4: scoring ===")
    # NeuML/pubmedbert-base-embeddings (safetensors) rather than
    # report_evaluation's default pritamdeka/S-PubMedBert-MS-MARCO: the latter
    # ships only a torch .bin checkpoint, which the installed transformers refuses
    # to load under torch < 2.6 (CVE-2025-32434). Both are biomedical embeddings.
    CLIN_MODEL = "NeuML/pubmedbert-base-embeddings"
    clin = load_clinical_embedding_model(CLIN_MODEL)
    rows = [cache[r["record_id"]] | {"record_id": r["record_id"]} for r in records]
    refs = [x["reference"] for x in rows]
    sig_txt = [x.get("signal_text", "") for x in rows]
    img_txt = [x.get("image_text", "") for x in rows]
    sig_cos = compute_clinical_cosine_similarity_batch(sig_txt, refs, clin)
    img_cos = compute_clinical_cosine_similarity_batch(img_txt, refs, clin)

    def build_table(texts, cos):
        out = []
        for r, gen, c in zip(rows, texts, cos):
            rouge = compute_rouge(gen, r["reference"]) if gen else {"rouge1": 0.0}
            out.append({
                "record_id": r["record_id"], "ecg_id": r["ecg_id"],
                "reference_report": r["reference"], "generated_text": gen,
                "bleu": compute_bleu(gen, r["reference"]) if gen else 0.0,
                "rouge1": rouge["rouge1"], "clinical_cosine": c,
            })
        return pd.DataFrame(out)

    sig_df = build_table(sig_txt, sig_cos)
    img_df = build_table(img_txt, img_cos)
    sig_df.to_csv(f"{OUT}/ecg_signal_evaluation.csv", index=False)
    img_df.to_csv(f"{OUT}/ecg_image_evaluation.csv", index=False)

    def md_table(df, title):
        lines = [f"\n## {title} (n={len(df)}, PTB-XL fold 10)\n",
                 "| ecg_id | reference (PTB-XL/RTLP) | generated | BLEU | ROUGE-1 | clin.cosine |",
                 "|---|---|---|---:|---:|---:|"]
        for x in df.itertuples():
            ref = (x.reference_report or "")[:60].replace("|", "/").replace("\n", " ")
            gen = (x.generated_text or "")[:60].replace("|", "/").replace("\n", " ")
            lines.append(f"| {x.ecg_id} | {ref} | {gen} | {x.bleu:.2f} | {x.rouge1:.3f} | {x.clinical_cosine:.3f} |")
        lines.append(f"| **MEAN** | | | **{df.bleu.mean():.2f}** | **{df.rouge1.mean():.3f}** | "
                     f"**{df.clinical_cosine.mean():.3f}** |")
        return "\n".join(lines)

    report = (
        "# Per-agent report evaluation — ECG signal vs ECG image\n\n"
        f"Same {N} held-out PTB-XL fold-10 records (held out for BOTH the signal embedding and the "
        "image classifier). GenECG images = the same PTB-XL records rendered as images, so both "
        "branches share the same cardiologist reference report.\n\n"
        "**Interpretation caveat:** BLEU/ROUGE are low BY DESIGN — the reference is a diagnostic "
        "interpretation, the generated text is a quantitative/qualitative description (diagnosis is "
        "forbidden in the agent prompts). The **clinical cosine** (biomedical embedding, "
        "NeuML/pubmedbert-base-embeddings) is the meaningful semantic metric. Only ECG has a "
        "reference report, so MRI/patient accuracy is their flag validation, not this text table. "
        "The ECG SIGNAL branch is run in its single-lead form here for tractability (the "
        "multilead+embedding form is much slower); the generated narrative is what's scored.\n"
        + md_table(sig_df, "ECG SIGNAL branch") + "\n"
        + md_table(img_df, "ECG IMAGE branch") + "\n\n"
        "## Signal vs image (means)\n\n"
        "| branch | BLEU | ROUGE-1 | clinical cosine |\n|---|---:|---:|---:|\n"
        f"| ECG signal | {sig_df.bleu.mean():.2f} | {sig_df.rouge1.mean():.3f} | {sig_df.clinical_cosine.mean():.3f} |\n"
        f"| ECG image | {img_df.bleu.mean():.2f} | {img_df.rouge1.mean():.3f} | {img_df.clinical_cosine.mean():.3f} |\n"
    )
    open(f"{OUT}/agent_evaluation_report.md", "w", encoding="utf-8").write(report)
    print(report)
    print(f"saved -> {OUT}/agent_evaluation_report.md (+ two CSVs)")


if __name__ == "__main__":
    main()
