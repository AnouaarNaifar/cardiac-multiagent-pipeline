"""
Groq (Llama-3.3-70B) A/B: does a much STRONGER free model change the two
conclusions we reached with MedGemma 4B?

  Test 1 (RAG utility): use_rag=False vs True  -> does Groq cite real criteria?
  Test 2 (G2 integration): integrate_criteria=False vs True (both use_rag=True)
          -> does a strong model finally WEAVE the criterion into the prose
             (e.g. "LVEF 23.7% -> HFrEF, LVEF <=40%"), where the 4B could not?

Same contributions, SAME clean guideline index, SAME seed=42 for every run ->
the only variable vs the MedGemma runs is the generator model. Run from root.
"""
import json, os, re, sys
from functools import partial
from pathlib import Path

sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_groq
from sep_agent import build_retrieval_query, generate_sep_agent_decision

if not os.environ.get("GROQ_API_KEY"):
    sys.exit("GROQ_API_KEY not visible to this process. Open a NEW shell after setx, "
             "or run:  $env:GROQ_API_KEY='gsk_...'  in the launching PowerShell.")

ROOT = Path(".").resolve()
SRC = ROOT / "results" / "sep_agent_3modality_test"
KB = "results/rag_knowledge_base"          # production index: cardiac_ai + ESC + AHA
OUT = ROOT / "results" / "rag_ab_test" / "groq_ab_result.json"
MODEL = "llama-3.3-70b-versatile"

# does the PROSE weave value<->criterion (not just cite a chunk)?
INTEG = re.compile(r"HFrEF|HFmrEF|HFpEF|NYHA|threshold|below the|above the|meets the|"
                   r"consistent with|classif|criteri|≤\s?\d|<\s?\d+\s?%", re.I)


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return AgentContribution(d["agent_id"], d["record_id"],
                             [Finding(**f) for f in d["findings"]],
                             d["flags_for_review"], d["trace"])


contribs = [load(SRC / f"{n}_contribution.json") for n in ("ecg", "mri", "patient")]
llm = partial(generate_report_groq, model=MODEL, seed=42)
print("=== SETUP ===", flush=True)
print("model:", MODEL)
print("RAG query:", repr(build_retrieval_query(contribs)))

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


# Test 1: RAG utility
do("A_norag", use_rag=False, integrate=False)
do("B_rag", use_rag=True, integrate=False)
# Test 2: G2 integration (strong-model retest)
do("C_rag_integrate", use_rag=True, integrate=True)

print(f"\n{'='*60}\n=== VERDICT (Groq {MODEL}) ===")
print("Test1 RAG utility  - B cites real criteria?:", runs["B_rag"]["n_knowledge_cites"], "citations,",
      "weaves in prose:", runs["B_rag"]["prose_weaves_criterion"])
print("Test2 G2 integrate - C weaves criterion in prose?:", runs["C_rag_integrate"]["prose_weaves_criterion"],
      "(", runs["C_rag_integrate"]["n_knowledge_cites"], "citations)")
json.dump(runs, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Saved:", OUT)
