"""
ECG image gate - "is this actually an ECG?" content check (CLIP zero-shot)
=========================================================================
A lightweight guard for the ECG IMAGE branch: before the pipeline spends a
VLM call + the learned classifier + digitization on an uploaded image, verify
the image actually CONTAINS an ECG. Without this, the branch would confidently
"read" a selfie, a landscape, or a scanned letter and emit ECG findings from
noise (the failure mode documented as limitation #4 in
adaptive_ecg_pipeline.py, now addressed here).

Why CLIP zero-shot (and not a trained classifier): this is an open-set problem
- "not an ECG" is everything else in the world - so a supervised binary
classifier would need an impossible-to-curate negative set. CLIP already aligns
images with natural-language descriptions from 400M web pairs, so we can just
ask it, in words, whether the image looks more like "a 12-lead ECG printout" or
like "a photo of something else", with NO training. Validated on real images
(two GenECG ECGs, an ACDC MRI slice, a block-diagram figure): P(ECG) ~0.89 on
the ECGs vs. <0.1 on the non-ECGs - a clean, well-separated margin around the
0.5 threshold (see the prompt-tuning note on _ECG_PROMPTS below).

Mechanism: encode the image and a small set of positive ("this IS an ECG") and
negative ("this is NOT an ECG") text prompts, take cosine similarities, softmax
over ALL prompts, and sum the probability mass on the positive prompts ->
P(ECG). Compare against a threshold.

Self-contained and OPT-IN: nothing imports this unless the image branch is run
with ecg_gate=True (see ecg_image_agent.generate_image_ecg_contribution /
ecg_router.route_ecg_upload). sentence-transformers is lazily imported so the
rest of the project stays importable without it, and the CLIP model is cached
module-level so it loads once, not per call (same pattern as
rag_knowledge_base._MODEL_CACHE).

Dependencies: sentence-transformers, torch, Pillow (torch is already an image-
branch dependency for the classifier). Model: "clip-ViT-B-32" (~0.6 GB,
downloaded once from HuggingFace, then cached locally).
"""
from typing import Any, Dict, Optional, Tuple

# CLIP model name (sentence-transformers' packaging of OpenAI CLIP ViT-B/32).
DEFAULT_CLIP_MODEL = "clip-ViT-B-32"

# Natural-language descriptions CLIP scores the image against. P(ECG) is the
# softmax mass on the POSITIVE prompts. The two HARD negatives - "a scientific
# chart/diagram that is not an ECG" and "a medical scan such as an MRI or X-ray"
# - matter: without them a matplotlib figure or an MRI slice (both plots/medical
# images) drift toward the ECG side. This exact set was tuned on real images
# (two GenECG ECGs, an ACDC MRI slice, a block-diagram figure): ECGs ~0.89,
# non-ECGs <0.1 - a clean, well-separated margin around the 0.5 threshold.
_ECG_PROMPTS = [
    "a 12-lead electrocardiogram (ECG) printout on pink grid paper",
    "an ECG recording showing repeating heart waveform tracings",
    "a printed ECG rhythm strip with QRS complexes",
]
_NON_ECG_PROMPTS = [
    "a photograph of a person or a face",
    "an outdoor landscape or a building",
    "a page of text or a scanned document",
    "a photo of an everyday object",
    "a scientific chart, bar chart, or block diagram that is not an ECG",
    "a medical scan such as an MRI or X-ray",
]

# Module-level cache so the CLIP model (and the fixed text embeddings) load once.
_MODEL_CACHE: Dict[str, Any] = {}
_TEXT_EMB_CACHE: Dict[str, Any] = {}


def load_ecg_gate_model(model_name: str = DEFAULT_CLIP_MODEL):
    """
    Load (and cache) the CLIP model. Lazily imports sentence-transformers so
    this module can be imported without it when the gate isn't used. Returns a
    SentenceTransformer instance.
    """
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _text_embeddings(model, model_name: str):
    """Encode + cache the fixed positive/negative prompt embeddings once."""
    if model_name not in _TEXT_EMB_CACHE:
        prompts = _ECG_PROMPTS + _NON_ECG_PROMPTS
        emb = model.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
        _TEXT_EMB_CACHE[model_name] = emb
    return _TEXT_EMB_CACHE[model_name]


def ecg_probability(image_path: str, model=None,
                    model_name: str = DEFAULT_CLIP_MODEL) -> float:
    """
    Return P(ECG) in [0, 1] for the image: the CLIP-softmax probability mass on
    the ECG prompts vs. the non-ECG prompts. Higher = more ECG-like.

    Inputs:
        image_path: path to the image to check.
        model: optional preloaded CLIP model (from load_ecg_gate_model); loaded
            and cached automatically if None.
        model_name: CLIP model name (only used when model is None).
    """
    import numpy as np
    from PIL import Image

    if model is None:
        model = load_ecg_gate_model(model_name)
    text_emb = _text_embeddings(model, model_name)

    img = Image.open(image_path).convert("RGB")
    img_emb = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]

    sims = text_emb @ img_emb                      # cosine (both normalized)
    logits = sims * 100.0                          # CLIP's standard temperature
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    n_pos = len(_ECG_PROMPTS)
    return float(probs[:n_pos].sum())


def is_ecg_image(image_path: str, threshold: float = 0.5, model=None,
                 model_name: str = DEFAULT_CLIP_MODEL) -> Tuple[bool, Dict[str, Any]]:
    """
    Decide whether an image is an ECG. Returns (is_ecg, info) where info carries
    the P(ECG) score and the threshold used, for logging/traceability.

    Inputs:
        image_path: path to the image to check.
        threshold: minimum P(ECG) to accept the image as an ECG (default 0.5;
            the separation is large so the exact value is not sensitive).
        model / model_name: see ecg_probability.

    Output:
        (is_ecg: bool, info: dict with "ecg_probability", "threshold").
    """
    p = ecg_probability(image_path, model=model, model_name=model_name)
    return (p >= threshold, {"ecg_probability": round(p, 4), "threshold": threshold})
