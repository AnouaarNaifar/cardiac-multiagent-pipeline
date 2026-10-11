"""
A/B test: does use_rag=True improve the Sep Agent's final synthesis vs use_rag=False?

Same contributions, same fixed seed for both runs -> the ONLY difference is the
retrieved-knowledge block injected into the prompt. Isolates the RAG effect from
LLM sampling noise. Run from the project root.
"""
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_medgemma
from sep_agent import build_retrieval_query, generate_sep_agent_decision

ROOT = Path(".").resolve()
SRC_DIR = ROOT / "results" / "sep_agent_3modality_test"
OUT_DIR = ROOT / "results" / "rag_ab_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_contribution(path):
    d = json.load(open(path, encoding="utf-8"))
    findings = [Finding(**f) for f in d["findings"]]
    return AgentContribution(
        agent_id=d["agent_id"],
        record_id=d["record_id"],
        findings=findings,
        flags_for_review=d["flags_for_review"],
        trace=d["trace"],
    )


contributions = [
    load_contribution(SRC_DIR / "ecg_contribution.json"),
    load_contribution(SRC_DIR / "mri_contribution.json"),
    load_contribution(SRC_DIR / "patient_contribution.json"),
]

flags = sorted({fl for c in contributions for fl in c.flags_for_review})
query = build_retrieval_query(contributions)
print("=== SETUP ===", flush=True)
print("Union flags:", flags)
print("RAG query built from flags:", repr(query))

# Fixed seed for BOTH -> only the prompt content (with/without knowledge) differs.
llm = partial(generate_report_medgemma, seed=42)

print("\n=== RUN A: use_rag=False ===", flush=True)
dec_a = generate_sep_agent_decision(
    contributions, llm_call=llm, record_id="ab_norag", use_rag=False
)
print("attempts:", dec_a.trace["attempts"])
print("SUMMARY A:\n", dec_a.summary)

print("\n=== RUN B: use_rag=True (top_k=5) ===", flush=True)
dec_b = generate_sep_agent_decision(
    contributions, llm_call=llm, record_id="ab_rag",
    use_rag=True, top_k_knowledge=5, kb_dir="results/rag_knowledge_base",
)
print("attempts:", dec_b.trace["attempts"])
print("SUMMARY B:\n", dec_b.summary)

# --- Analysis ---
chunks = dec_b.trace.get("retrieved_knowledge", []) or []
print("\n=== RETRIEVED CHUNKS (B) ===", flush=True)
for i, ch in enumerate(chunks):
    txt = ch.get("text", "").replace("\n", " ")
    print(f"[{i}] score={ch.get('score'):.3f} src={ch.get('source')} :: {txt[:160]}")

knowledge_cites = [(e.feature, e.statement) for e in dec_b.traceability if e.agent_id == "knowledge"]
print("\n=== ANALYSIS ===")
print("B cites knowledge in traceability?:", bool(knowledge_cites),
      "(agents cited:", sorted({e.agent_id for e in dec_b.traceability}), ")")
for feat, stmt in knowledge_cites:
    print(f"  KNOWLEDGE CITED [{feat}]: {stmt}")
print("Summary A == Summary B ?:", dec_a.summary.strip() == dec_b.summary.strip())

json.dump(
    {
        "flags": flags,
        "rag_query": query,
        "summary_A_norag": dec_a.summary,
        "summary_B_rag": dec_b.summary,
        "summaries_identical": dec_a.summary.strip() == dec_b.summary.strip(),
        "B_cites_knowledge": bool(knowledge_cites),
        "knowledge_citations": [{"feature": f, "statement": s} for f, s in knowledge_cites],
        "B_traceability_agents": sorted({e.agent_id for e in dec_b.traceability}),
        "retrieved_chunks": [
            {"score": ch.get("score"), "source": ch.get("source"),
             "header": ch.get("header"), "text": ch.get("text")}
            for ch in chunks
        ],
    },
    open(OUT_DIR / "rag_ab_result.json", "w", encoding="utf-8"),
    indent=2, ensure_ascii=False,
)
print("\nSaved:", OUT_DIR / "rag_ab_result.json")
