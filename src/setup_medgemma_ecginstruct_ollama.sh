#!/usr/bin/env bash
#
# One-time setup for the ECG-specialized MedGemma vision fine-tune used by
# ecg_image_agent.py (DEFAULT_IMAGE_MODEL_TAG).
#
# Community checkpoint convaiinnovations/medgemma-4b-ecginstruct, fine-tuned on
# ~1.15M ECG image-text pairs (PULSE-ECG/ECGInstruct) to read 12-lead ECG
# IMAGES - unlike base medgemma:4b, which is used by the ecg_agent.py text/
# numeric branch. Only the IMAGE branch needs this model.
#
# Ollama can pull a GGUF straight from the Hugging Face Hub via the hf.co/
# tag, so no manual `ollama create`/Modelfile is required.
#
# Usage:
#   bash setup_medgemma_ecginstruct_ollama.sh
#
# Requires: Ollama installed and running (https://ollama.com), plus network
# access to huggingface.co. ~3 GB download (Q4_K_M quantization).

set -euo pipefail

MODEL_TAG="hf.co/convaiinnovations/medgemma-ecg-gguf-only:Q4_K_M"
FALLBACK_TAG="medgemma:4b"   # base multimodal model, ecg_image_agent's fallback

echo ">> Pulling ECG-specialized model: ${MODEL_TAG}"
ollama pull "${MODEL_TAG}"

echo ">> Ensuring base fallback model is present: ${FALLBACK_TAG}"
ollama pull "${FALLBACK_TAG}"

echo ">> Installed ECG/MedGemma models:"
ollama list | grep -Ei 'medgemma|ecg' || {
    echo "!! Neither model appears in 'ollama list' - check Ollama is running." >&2
    exit 1
}

echo
echo ">> Done. ecg_image_agent.py will now use '${MODEL_TAG}' by default,"
echo "   falling back to '${FALLBACK_TAG}' if the primary tag is unavailable."
echo "   Sanity-check that the model actually SEES the image with:"
echo "     python sanity_check_image_ingestion.py"
