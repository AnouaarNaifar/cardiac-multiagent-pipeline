# cardiac-multiagent-pipeline — Docker image

> Paste this into the **Overview / README** of the Docker Hub repository
> (hub.docker.com → your repo → *Manage repository* → Description / README).
> It tells anyone who `docker pull`s the image what else they must provide.

A local, multi-agent clinical **decision-support** demo for cardiac data (ECG / MRI / tabular), built with
Streamlit + a shared MedGemma backbone. **Research/demo on public data — not a medical device.**

---

## ⚠️ What is IN vs NOT in this image

**Inside the image** (ready to run): the app code, all Python dependencies (CPU PyTorch), and the trained model
**checkpoints** (ECG embedding, YOLO, U-Net, pathology classifier, RAG index).

**NOT in the image — you must provide these separately:**

| What | Why it's not in the image | How to get it |
|---|---|---|
| **Datasets** (PTB-XL, ACDC, UCI, RTLP, RAG PDFs) | too large (~8.5 GB); mounted as volumes | download the public datasets (links below) and mount them |
| **MedGemma model** (~5 GB) | served by **Ollama**, not the app | install [Ollama](https://ollama.com), then `ollama pull medgemma:4b` |
| **`docker-compose.yml`** | run configuration, not app code | from the project repo (defines the volume mounts + `OLLAMA_HOST`) |
| **`.env`** (optional Groq key) | a secret | create it yourself if you want the Groq cloud backend |

---

## How to run (Option A — app container + native host Ollama)

**Prerequisites:** Docker, a native **Ollama** running with `medgemma:4b`, and the datasets on disk.

```bash
# 1) get the image
docker pull anou21/cardiac-pipeline:latest

# 2) use a docker-compose.yml whose app service points at the image and mounts the datasets:
#    services:
#      app:
#        image: anou21/cardiac-pipeline:latest
#        ports: ["8501:8501"]
#        environment: ["OLLAMA_HOST=http://host.docker.internal:11434"]
#        extra_hosts: ["host.docker.internal:host-gateway"]
#        volumes:
#          - ./kaggle:/app/kaggle:ro
#          - ./ACDC:/app/ACDC:ro
#          - ./rag_sources:/app/rag_sources:ro
#          - ./RTLP:/app/RTLP:ro

# 3) run it
docker compose up -d

# 4) open the app
#    http://localhost:8501     (health: curl http://localhost:8501/_stcore/health -> 200)
```

The container reaches your **host** Ollama via `host.docker.internal:11434` — so the LLM stays GPU-fast on the
host while the app runs in the container. No NVIDIA Container Toolkit needed.

---

## Where to get the datasets (all public)

- **PTB-XL** (ECG) — PhysioNet: https://physionet.org/content/ptb-xl/ → place under `kaggle/archive/ptb-xl-.../`
- **ACDC** (cardiac MRI) — https://www.creatis.insa-lyon.fr/Challenge/acdc/ → place under `ACDC/database/`
- **UCI Heart Disease / Cleveland** — https://archive.ics.uci.edu/dataset/45/heart+disease → `uci_heart_disease/`
- **RTLP** (PTB-XL English reports) — https://github.com/danikiyasseh/RTLP → `RTLP/`
- **RAG guideline PDFs** — the curated PDFs go under `rag_sources/` (see the project repo)

(The exact folder layout is described in the project's `README.md` / `CLAUDE.md`.)

---

## Notes

- Data stays local: the default LLM (MedGemma via Ollama) runs on your machine. The optional Groq backend is
  cloud (text-only) and only used if you set `GROQ_API_KEY`.
- The role login (patient / doctor / researcher) uses **demo passwords** — a UX switch, **not** real security.
- Tags: `latest` = most recent build.
