# DEPLOY — build → run local → push to Docker Hub (Option A)

Runbook to do everything yourself from a terminal (PowerShell **or** Git Bash — the `docker` commands are the
same in both). Username `anou21` is already filled in. All commands are run **from the project root**:
`C:\Users\neifa\Documents\ENETCOM\3IDSD\stage_25-26\cardiac-multiagent-pipeline`.

> **Option A** = only the app is containerized; it uses your **native host Ollama** (GPU). So keep Ollama
> running on your machine — there is **no `ollama pull` inside a container**.

---

## 0. Prerequisites (check once, each time before you start)

1. **Docker Desktop** open, icon **green / "Engine running"**, and **signed in as `anou21`** (top-right).
2. **Native Ollama running** with the model:
   ```bash
   ollama list          # must show medgemma:4b  (if not: ollama pull medgemma:4b)
   ```
3. Sanity check the daemon answers:
   ```bash
   docker version       # both Client and Server versions must print (no 500 error)
   ```
   If you get `500 Internal Server Error` or the Server version is empty → see **Troubleshooting T1** below.

---

## 1. Build the image

```bash
docker compose build
```
- Takes a few minutes (downloads torch CPU + deps over HTTPS). Expected result: image `cardiac-pipeline:local`
  ~**3.5 GB**.
- Verify:
  ```bash
  docker images cardiac-pipeline:local
  ```

---

## 2. Run locally & verify it works (smoke test)

```bash
docker compose up -d          # start the app container (Ollama stays native on the host)
```
- Check it serves:
  ```bash
  curl http://localhost:8501/_stcore/health      # -> 200
  ```
- Open **http://localhost:8501** in your browser → the role login page.
- Test the LLM path (proves the container reaches your host Ollama): log in as **researcher** → ECG page →
  enable **LLM structured findings** → a result comes back.
- Follow logs if needed:
  ```bash
  docker compose logs -f app
  ```
- Stop when done:
  ```bash
  docker compose down
  ```

---

## 3. Tag the image for Docker Hub

```bash
docker tag cardiac-pipeline:local anou21/cardiac-pipeline:latest
docker images anou21/cardiac-pipeline:latest        # confirm the tag exists
```

---

## 4. Log in to Docker Hub (skip if already signed in via Docker Desktop)

```bash
docker login                 # enter username: anou21  + password or access token
```
> Tip: Docker Hub prefers a **Personal Access Token** over your password
> (hub.docker.com → Account settings → Security → New Access Token). Use the token as the password.

---

## 5. Push to Docker Hub

```bash
docker push anou21/cardiac-pipeline:latest
```
- ~**3.5 GB upload** → several minutes on the school Wi-Fi. It shows a per-layer progress bar.
- If it drops mid-push, just run the same command again — already-uploaded layers are skipped.

---

## 6. Verify the push

- Browser: **https://hub.docker.com/r/anou21/cardiac-pipeline/tags** → the `latest` tag appears.
- Or from a terminal:
  ```bash
  docker manifest inspect anou21/cardiac-pipeline:latest > /dev/null && echo "OK: image is on Docker Hub"
  ```

> ⚠️ On the free plan the repo is **public** (code + model checkpoints are visible; datasets, `.env`/Groq key,
> and physionet are excluded by `.dockerignore`). For a private repo, create it as **private** on Docker Hub
> first (1 free private repo).

---

## 7. (Optional) Run from the PUBLISHED image on any machine

On another machine that has Docker + native Ollama + the datasets, the image can be pulled and run:
```bash
docker pull anou21/cardiac-pipeline:latest
# then a docker-compose.yml whose app service uses:  image: anou21/cardiac-pipeline:latest  (instead of build: .)
docker compose up -d
```
(That machine still needs the datasets mounted as volumes and `medgemma:4b` in its Ollama — they are never in
the image, by design.)

---

## Troubleshooting

**T1 — `500 Internal Server Error` / empty Server version / daemon stuck.**
Docker Desktop's engine (WSL2) hiccupped — common on 16 GB RAM, often right after signing in.
```bash
# 1) Quit Docker Desktop completely (right-click tray icon -> Quit)
# 2) reset WSL2 networking/VM:
wsl --shutdown
# 3) reopen Docker Desktop, wait for the GREEN "Engine running" icon
# 4) re-run your command (the image is still on disk — no rebuild needed)
```
Prevention: Docker Desktop → **Settings → Resources** → give it **~8 GB memory** and enough disk.

**T2 — build fails at `apt-get ... Unable to connect to deb.debian.org`.**
Your network (ENETCOM) blocks HTTP/port 80. Already handled: the `Dockerfile` uses **no apt** (headless
opencv), pip over HTTPS only. If you see this, you have an old Dockerfile — pull the current one.

**T3 — image is ~8 GB / build downloads many `nvidia_*` / `cuda_*` wheels.**
That's the CUDA torch. Already handled: the `Dockerfile` installs **CPU torch first** from
`download.pytorch.org/whl/cpu`. Option A runs torch on CPU anyway.

**T4 — the app runs but LLM findings fail.**
Your **native Ollama isn't running** (or lacks `medgemma:4b`). Start it: `ollama serve` (or just open the
Ollama app) and `ollama pull medgemma:4b`. The container reaches it via `host.docker.internal:11434`.

**T5 — `denied: requested access to the resource is denied` on push.**
Not logged in, or the tag's username ≠ your Docker Hub user. Re-run `docker login` and check the tag is
`anou21/cardiac-pipeline:latest`.

---

## One-shot copy-paste (once Docker Desktop is green + Ollama running)

```bash
docker compose build
docker compose up -d && curl http://localhost:8501/_stcore/health && echo " <- health"
docker tag cardiac-pipeline:local anou21/cardiac-pipeline:latest
docker login
docker push anou21/cardiac-pipeline:latest
```
