"""
A/B experiment #4: KB = ONLY cardiac_ai_internship.pdf (author's own clinical
notes), NO specialist/dataset-description papers. Isolation test: is the RAG
downstream failure caused by the KB CONTENT (dataset-doc papers) rather than the
mechanism? Same fixed seed for both runs -> only the injected knowledge differs.
Run from the project root.
"""
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_medgemma
from rag_knowledge_base import build_knowledge_base
from sep_agent import build_retrieval_query, generate_sep_agent_decision

ROOT = Path(".").resolve()
SRC_DIR = ROOT / "results" / "sep_agent_3modality_test"
KB_DIR = ROOT / "results" / "rag_knowledge_base_cardiaconly"
OUT_DIR = ROOT / "results" / "rag_ab_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Build the cardiac-only KB ---
info = build_knowledge_base(["rag_sources/cardiac_ai_internship.pdf"], output_dir=str(KB_DIR))
print("=== CARDIAC-ONLY KB BUILT ===")
print(info, flush=True)


def load_contribution(path):
    d = json.load(open(path, encoding="utf-8"))
    return AgentContribution(
        agent_id=d["agent_id"], record_id=d["record_id"],
        findings=[Finding(**f) for f in d["findings"]],
        flags_for_review=d["flags_for_review"], trace=d["trace"],
    )


contributions = [
    load_contribution(SRC_DIR / "ecg_contribution.json"),
    load_contribution(SRC_DIR / "mri_contribution.json"),
    load_contribution(SRC_DIR / "patient_contribution.json"),
]
query = build_retrieval_query(contributions)
print("\nRAG query:", query, flush=True)

llm = partial(generate_report_medgemma, seed=42)

print("\n=== RUN A: use_rag=False ===", flush=True)
dec_a = generate_sep_agent_decision(contributions, llm_call=llm, record_id="ab4_norag", use_rag=False)
print("SUMMARY A:\n", dec_a.summary)

print("\n=== RUN B: use_rag=True on CARDIAC-ONLY KB ===", flush=True)
dec_b = generate_sep_agent_decision(
    contributions, llm_call=llm, record_id="ab4_rag",
    use_rag=True, top_k_knowledge=5, kb_dir=str(KB_DIR),
)
print("SUMMARY B:\n", dec_b.summary)

chunks = dec_b.trace.get("retrieved_knowledge", []) or []
print("\n=== RETRIEVED CHUNKS (B) ===")
for i, ch in enumerate(chunks):
    print(f"[{i}] {ch.get('score'):.3f} {ch.get('source')} :: {ch.get('text','')[:150].replace(chr(10),' ')}")

# Which knowledge entries were actually cited, and for what statements?
knowledge_cites = [(e.agent_id, e.feature, e.statement) for e in dec_b.traceability if e.agent_id == "knowledge"]
print("\n=== ANALYSIS ===")
print("B cites knowledge?:", bool(knowledge_cites), "| agents:", sorted({e.agent_id for e in dec_b.traceability}))
for aid, feat, stmt in knowledge_cites:
    print(f"  KNOWLEDGE CITED [{feat}]: {stmt}")
print("A == B ?:", dec_a.summary.strip() == dec_b.summary.strip())

json.dump({
    "experiment": "cardiac_ai_only_KB",
    "kb_info": info,
    "rag_query": query,
    "summary_A_norag": dec_a.summary,
    "summary_B_rag": dec_b.summary,
    "B_cites_knowledge": bool(knowledge_cites),
    "knowledge_citations": [{"feature": f, "statement": s} for _, f, s in knowledge_cites],
    "retrieved_chunks": [{"score": c.get("score"), "source": c.get("source"),
                          "header": c.get("header"), "text": c.get("text")} for c in chunks],
}, open(OUT_DIR / "rag_ab_result_cardiaconly.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\nSaved:", OUT_DIR / "rag_ab_result_cardiaconly.json")
