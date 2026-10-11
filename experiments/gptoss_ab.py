"""
Second strong-model retest: openai/gpt-oss-120b via Groq, same setup as groq_ab.py.
Adds inter-call spacing + 429 backoff (this model has a lower tokens-per-minute
free-tier cap than llama-3.3-70b). Only variable vs the MedGemma/Llama runs is the
generator model. Run from project root.
"""
import json, os, re, sys, time
from functools import partial
from pathlib import Path

import requests

sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_groq
from sep_agent import build_retrieval_query, generate_sep_agent_decision

if not os.environ.get("GROQ_API_KEY"):
    sys.exit("GROQ_API_KEY not visible to this process.")

ROOT = Path(".").resolve()
SRC = ROOT / "results" / "sep_agent_3modality_test"
KB = "results/rag_knowledge_base"
OUT = ROOT / "results" / "rag_ab_test" / "gptoss_ab_result.json"
MODEL = "openai/gpt-oss-120b"

INTEG = re.compile(r"HFrEF|HFmrEF|HFpEF|NYHA|threshold|below the|above the|meets the|"
                   r"consistent with|classif|criteri|≤\s?\d|<\s?\d+\s?%", re.I)


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return AgentContribution(d["agent_id"], d["record_id"],
                             [Finding(**f) for f in d["findings"]],
                             d["flags_for_review"], d["trace"])


def llm(prompt):
    """gpt-oss-120b with 429 backoff."""
    for attempt in range(4):
        try:
            # bigger budget + low reasoning_effort: gpt-oss spends max_tokens on
            # hidden reasoning, which truncated the JSON at the default 2048.
            return generate_report_groq(prompt, model=MODEL, seed=42,
                                        max_tokens=3500, reasoning_effort="low")
        except requests.exceptions.HTTPError as e:
            if getattr(e.response, "status_code", None) == 429 and attempt < 3:
                wait = 30 * (attempt + 1)
                print(f"  [429] backing off {wait}s (attempt {attempt+1})", flush=True)
                time.sleep(wait)
                continue
            raise


contribs = [load(SRC / f"{n}_contribution.json") for n in ("ecg", "mri", "patient")]
print("=== SETUP ===\nmodel:", MODEL, flush=True)
runs = {}


def do(label, use_rag, integrate):
    print(f"\n{'='*60}\n=== {label} (use_rag={use_rag}, integrate={integrate}) ===", flush=True)
    dec = generate_sep_agent_decision(contribs, llm_call=llm, record_id=label,
                                      use_rag=use_rag, top_k_knowledge=5, kb_dir=KB,
                                      integrate_criteria=integrate)
    kc = [(e.feature, e.statement) for e in dec.traceability if e.agent_id == "knowledge"]
    woven = bool(INTEG.search(dec.summary))
    print("attempts:", dec.trace["attempts"])
    print("SUMMARY:\n", dec.summary)
    print("knowledge citations:", len(kc))
    for f, s in kc:
        print(f"  KNOWLEDGE [{f}]: {s}")
    print("PROSE weaves a criterion/threshold term?:", woven)
    runs[label] = {"use_rag": use_rag, "integrate": integrate,
                   "attempts": dec.trace["attempts"], "summary": dec.summary,
                   "prose_weaves_criterion": woven, "n_knowledge_cites": len(kc),
                   "knowledge_citations": [{"feature": f, "statement": s} for f, s in kc]}


for i, (label, ur, integ) in enumerate([("A_norag", False, False),
                                         ("B_rag", True, False),
                                         ("C_rag_integrate", True, True)]):
    if i:
        time.sleep(45)  # stay under gpt-oss-120b TPM cap
    do(label, ur, integ)

print(f"\n{'='*60}\n=== VERDICT (Groq {MODEL}) ===")
print("Test1 RAG utility  - B cites real criteria?:", runs["B_rag"]["n_knowledge_cites"], "citations,",
      "weaves in prose:", runs["B_rag"]["prose_weaves_criterion"])
print("Test2 G2 integrate - C weaves criterion in prose?:", runs["C_rag_integrate"]["prose_weaves_criterion"],
      "(", runs["C_rag_integrate"]["n_knowledge_cites"], "citations)")
json.dump(runs, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Saved:", OUT)
