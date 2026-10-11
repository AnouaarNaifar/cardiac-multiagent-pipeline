# Groq backend — why, and how to configure it

## Why Groq (the choice)

The deep-integration gap (G2: the Sep Agent cites a criterion but does not weave
it into the prose) was shown to be **bounded by MedGemma 4B's capability**, not by
the prompt (`integrate_criteria` directive tried and failed - it dropped grounding
from 7 citations to 0). Testing whether a **much stronger model** unblocks G2
needs a model far larger than the ~7-9B ceiling that fits in 16 GB RAM.

- **Claude / GPT** would work but are **paid**.
- **Groq** offers a **free tier** serving large open models — notably
  **Llama-3.3-70B** — over an **OpenAI-compatible REST API**, fast, no local RAM
  cost, no new Python dependency (`requests` only).

So Groq is the best **free** way to answer: *"does a genuinely stronger model
integrate criteria into the prose, where the local 4B could not?"*

Trade-offs (accepted here): the prompt is sent to the **cloud** (fine - the test
case is **synthetic**, not a real patient), it needs a **free API key**, and the
free tier has **rate limits** (requests/tokens per minute). Groq also rotates
which models it hosts, so a model id can be deprecated over time.

## Model choice

- **`llama-3.3-70b-versatile`** (default in the adapter) — strongest free general
  model; the one to test for G2.
- `llama-3.1-8b-instant` — smaller/faster, if rate-limited.
- Note: all Groq models are **general**, not medical - this tests *capability*,
  trading away MedGemma's medical fine-tuning. "Stronger" is the hypothesis under
  test, not a guarantee.

## Configuration steps

1. **Create a free key:** sign up at `https://console.groq.com`, then create an
   API key (free tier, no card required).
2. **Set the key as an environment variable** (do NOT hardcode it in code):
   - PowerShell (this session only): `$env:GROQ_API_KEY = "gsk_..."`
   - PowerShell (persistent, new shells): `setx GROQ_API_KEY "gsk_..."`
   - Git Bash (this session): `export GROQ_API_KEY="gsk_..."`
3. **No install needed** — the adapter uses `requests` (already a dependency).
4. **The adapter** is `agent_common.generate_report_groq(prompt, model=..., seed=...)`
   — same `llm_call(prompt)` shape as the other backends, so it drops into
   `generate_sep_agent_decision(..., llm_call=...)` unchanged.

## How it is used in the test

```python
from functools import partial
from agent_common import generate_report_groq
llm = partial(generate_report_groq, model="llama-3.3-70b-versatile", seed=42)
# then: generate_sep_agent_decision(contribs, llm_call=llm, use_rag=True, ...)
```

The same clean guideline index and fixed seed are reused, so the only variable vs
the MedGemma runs is the **generator model** — isolating the model-capability
question for both the RAG-utility A/B and the G2 (integration) A/B.

## Security note

The key lives only in an environment variable, never in a file. The synthetic
3-modality case contains no real patient data. Do not point this backend at real
patient records without a proper data-processing review — the project is
otherwise deliberately local / no-API-key.
