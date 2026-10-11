"""
Pearson-correlation evaluation of the two RAG embedding models, adapted
from NeuML/pubmedbert-base-embeddings' own model-card methodology
(https://huggingface.co/NeuML/pubmedbert-base-embeddings): correlate each
model's similarity score against a KNOWN relevant/irrelevant label for a
set of (query, chunk) pairs.

Expanded (round 2, 25 queries instead of 10) after the first 10-query test
gave a closer call than expected: the clinical model won on Pearson r but
failed twice on top-1 accuracy specifically on plausible-but-wrong
candidates. This version makes the Pearson test harder and more realistic
to actually probe that same failure mode, instead of only measuring
relevant-vs-obviously-irrelevant separation:
  - 3 fixed EASY negatives per query (bibliography/data-format text, zero
    topical overlap - same as round 1).
  - 2 HARD negatives per query, sampled from two OTHER queries' correct
    answers (fixed seed) - clinically plausible, topically related, but
    wrong for THIS query. This is exactly the kind of confusion the top-1
    test exposed (e.g. mistaking "3.1 The Waves" for the Atrial
    Fibrillation section) that easy negatives cannot detect.
150 total labeled pairs (25 queries x 6 pairs each: 1 positive + 2 hard
negatives + 3 easy negatives).
"""
import json
import random

# numpy/scipy/sentence_transformers are deliberately imported lazily, inside
# run_pearson_evaluation() below - not at module level. QUERIES_AND_POSITIVE_HEADERS
# is also consumed by lightweight, non-ML code (e.g. the PDF report generator),
# which would otherwise be forced to pay for loading the full embedding stack
# just to read a constant list.

QUERIES_AND_POSITIVE_HEADERS = [
    ("low left ventricular ejection fraction dilated cardiomyopathy", "4.2 Structural abnormalities"),
    ("wide QRS complex bundle branch block", "5.3 Bundle Branch Blocks"),
    ("Sokolow-Lyon criterion hypertrophy voltage", "5.4 Left Ventricular Hypertrophy"),
    ("PTB-XL diagnostic superclasses NORM MI STTC CD HYP", "(document start)"),
    ("ACDC dilated cardiomyopathy hypertrophic cardiomyopathy classification", "4.2 Structural abnormalities"),
    ("ECG preprocessing baseline wander wavelet denoising", "2. The few multimodal approaches"),
    ("reduced pumping function of the heart chamber", "2.1 Cine MRI"),
    ("irregular heartbeat coming from upper chambers", "5.1 Atrial Fibrillation"),
    ("heart attack scar tissue detection", "4.3 Tissue/scar abnormalities"),
    ("thickened heart muscle wall abnormal", "4.2 Structural abnormalities"),
    ("P wave QRS complex T wave cardiac cycle sequence", "3.1 The Waves"),
    ("RR interval PR interval QT interval definitions", "3.2 The Intervals"),
    ("ST segment elevation myocardial infarction STEMI leads", "5.2 ST-Elevation Myocardial Infarction"),
    ("regional heart wall motion hypokinetic akinetic dyskinetic", "4.1 Functional abnormalities"),
    ("myocarditis inflammation T2 signal elevated", "4.4 Inflammatory / infiltrative disease"),
    ("T1 T2 mapping tissue characterisation", "2.3 T1 / T2 mapping"),
    ("cardiac MRI sequences overview Cine LGE mapping", "2. Cardiac MRI sequence"),
    ("why is MRI used for the heart", "1.2 Why MRI for the heart specifically"),
    ("how electrical signals recorded skin electrodes leads", "1.1 How Are Signals Recorded"),
    ("twelve lead ECG sampling rate data shape", "1.3 Data Dimensions (Standard ECG)"),
    ("short axis stack slices base to apex 3D reconstruction", "2.4 Short axis stack"),
    ("end diastolic volume blood filling ventricle at rest", "2. End-diastolic volume (EDV)"),
    ("ejection fraction percentage blood pumped per beat", "1. Ejection Fraction (EF)"),
    ("feature extraction morphology temporal spatial levels ECG", "4. Feature Extraction, Three Levels"),
    ("what is an electrocardiogram basic definition", "1. What is an ECG?"),
]

NEGATIVE_HEADERS = [
    ("alqudah_alfraihat2026_multimodal.pdf", "2. Ye C, Kumar BV"),
    ("alqudah_alfraihat2026_multimodal.pdf", "21. Kumar AS, Rekha R"),
    ("ptbxl_wagner2020.pdf", "3. Each record was annotated as follows"),
]


def _find_chunk(chunks, header_substr, source=None):
    for c in chunks:
        if header_substr in c["header"] and (source is None or c["source"] == source):
            return c
    raise ValueError(f"No chunk found for header substring {header_substr!r} (source={source})")


def run_pearson_evaluation(kb_dirs_and_models, output_json="results/rag_pearson_evaluation.json", seed=7):
    import numpy as np
    from scipy.stats import pearsonr
    from sentence_transformers import SentenceTransformer

    with open("results/rag_knowledge_base/chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    negatives = [_find_chunk(chunks, h, s) for s, h in NEGATIVE_HEADERS]
    positives = [_find_chunk(chunks, h) for _, h in QUERIES_AND_POSITIVE_HEADERS]

    rng = random.Random(seed)
    pairs = []  # (query, chunk_text, label, pair_type)
    for i, (query, _) in enumerate(QUERIES_AND_POSITIVE_HEADERS):
        pairs.append((query, positives[i]["text"], 1, "positive"))

        other_indices = [j for j in range(len(positives)) if j != i]
        hard_neg_indices = rng.sample(other_indices, 2)
        for j in hard_neg_indices:
            pairs.append((query, positives[j]["text"], 0, "hard_negative"))

        for neg in negatives:
            pairs.append((query, neg["text"], 0, "easy_negative"))

    results = {}
    for label, (kb_dir, model_name) in kb_dirs_and_models.items():
        model = SentenceTransformer(model_name)
        queries_emb = model.encode([p[0] for p in pairs], normalize_embeddings=True)
        chunks_emb = model.encode([p[1] for p in pairs], normalize_embeddings=True)
        sims = np.sum(queries_emb * chunks_emb, axis=1)
        rel_labels = np.array([p[2] for p in pairs])
        pair_types = np.array([p[3] for p in pairs])

        r, p_value = pearsonr(sims, rel_labels)

        hard_mask = (pair_types == "hard_negative") | (pair_types == "positive")
        r_hard, p_hard = pearsonr(sims[hard_mask], rel_labels[hard_mask])

        results[label] = {
            "pearson_r_all": float(r),
            "p_value_all": float(p_value),
            "pearson_r_positive_vs_hard_negative_only": float(r_hard),
            "p_value_hard": float(p_hard),
            "n_pairs": len(pairs),
            "mean_sim_positive": float(sims[rel_labels == 1].mean()),
            "mean_sim_hard_negative": float(sims[pair_types == "hard_negative"].mean()),
            "mean_sim_easy_negative": float(sims[pair_types == "easy_negative"].mean()),
        }
        print(f"{label}:")
        print(f"  Pearson r (all 150 pairs) = {r:.4f} (p={p_value:.2e})")
        print(f"  Pearson r (positive vs HARD negatives only, 75 pairs) = {r_hard:.4f} (p={p_hard:.2e})")
        print(f"  mean_sim: positive={results[label]['mean_sim_positive']:.3f}, "
              f"hard_neg={results[label]['mean_sim_hard_negative']:.3f}, "
              f"easy_neg={results[label]['mean_sim_easy_negative']:.3f}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    run_pearson_evaluation({
        "general (all-MiniLM-L6-v2)": ("results/rag_knowledge_base", "all-MiniLM-L6-v2"),
        "clinical (NeuML/pubmedbert-base-embeddings)": ("results/rag_knowledge_base_clinical", "NeuML/pubmedbert-base-embeddings"),
    })
