FROM python:3.11-slim

# Image metadata — travels WITH the image (visible via `docker inspect`). Tells a
# puller what else they need beyond the image itself.
LABEL org.opencontainers.image.title="cardiac-multiagent-pipeline" \
      org.opencontainers.image.description="Multi-agent cardiac decision-support (ECG/MRI/tabular) Streamlit app. NOT included, provide separately: datasets (PTB-XL/ACDC/UCI/RTLP, mounted as volumes), the MedGemma model (via a native Ollama: 'ollama pull medgemma:4b'), and docker-compose.yml. Run: app container + host Ollama (OLLAMA_HOST=http://host.docker.internal:11434). Research/demo on public data, not a medical device." \
      org.opencontainers.image.documentation="See DOCKERHUB_README.md / DEPLOY.md in the project repo"

WORKDIR /app

# Dependencies first (cache layer: only reinstalled if requirements.txt changes).
# NOTE 1: we deliberately do NOT use apt. ultralytics pulls `opencv-python`, which
# needs the system libs libGL/libglib (an apt install) — but on restricted networks
# apt's HTTP mirrors (port 80) are often blocked. `opencv-python-headless` removes
# that system-lib need, so the whole build uses only pip over HTTPS (port 443).
# NOTE 2: the --mount=type=cache keeps pip's downloaded wheels ACROSS build attempts,
# so a retry after a daemon hiccup (torch is a 500+ MB download) does not re-download
# everything. Requires BuildKit (default in modern Docker).
COPY requirements.txt .
# NOTE 3: install the CPU-only torch AND torchvision FIRST, TOGETHER, from PyTorch's
# CPU index, as a matched, pinned pair. In Option A the container's torch runs on CPU
# (the GPU work is done by the host's native Ollama), so the default CUDA build from
# PyPI would pull ~2-3 GB of unused NVIDIA CUDA wheels and bloat the image to ~8 GB.
# CRITICAL: torch and torchvision MUST be the same CPU build and a compatible version
# pair, installed from the SAME index. If only torch is installed here, the later
# `-r requirements.txt` pulls torchvision from PyPI (a CUDA build compiled against a
# different torch), which breaks at import ("operator torchvision::nms does not exist").
# Installing both here first means `-r requirements.txt` sees them satisfied and keeps
# them. 2.5.1 / 0.20.1 is the validated compatible pair.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless || true
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install opencv-python-headless

# The code + checkpoints (small); the large datasets are mounted as volumes
COPY . .

EXPOSE 8501

# Streamlit must listen on 0.0.0.0 to be reachable from outside the container
CMD ["streamlit", "run", "Home.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
