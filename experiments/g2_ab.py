"""
G2 experiment: does an explicit "integrate the criterion into the prose"
directive (integrate_criteria=True) deepen the Sep Agent's synthesis vs the
current RAG-on behaviour? Both runs: use_rag=True, SAME clean guideline index,
SAME seed -> only the integration directive differs. Run from project root.
"""
import json, re, sys
from functools import partial
from pathlib import Path

sys.path.insert(0, "src")
from agent_common import AgentContribution, Finding, generate_report_medgemma
from sep_agent import generate_sep_agent_decision

ROOT = Path(".").resolve()
SRC = ROOT / "results" / "sep_agent_3modality_test"
KB = "results/rag_knowledge_base"  # current production index (cardiac_ai + ESC + AHA)
OUT = ROOT / "results" / "rag_ab_test" / "g2_ab_result.json"


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return AgentContribution(d["agent_id"], d["record_id"],
                             [Finding(**f) for f in d["findings"]], d["flags_for_review"], d["trace"])


contribs = [load(SRC / f"{n}_contribution.json") for n in ("ecg", "mri", "patient")]
llm = partial(generate_report_medgemma, seed=42)

# integration heuristic: does the PROSE weave value<->criterion?
INTEG = re.compile(r"HFrEF|HFmrEF|HFpEF|NYHA|threshold|below the|above the|meets the|"
                   r"criteri|classif|≤\s?\d|<\s?\d+\s?%", re.I)

runs = {}
for label, integrate in [("B1_no_directive", False), ("B2_integrate", True)]:
    print(f"\n{'='*60}\n=== {label} (use_rag=True, integrate_criteria={integrate}) ===", flush=True)
    dec = generate_sep_agent_decision(contribs, llm_call=llm, record_id=label,
                                      use_rag=True, top_k_knowledge=5, kb_dir=KB,
                                      integrate_criteria=integrate)
    kc = [(e.feature, e.statement) for e in dec.traceability if e.agent_id == "knowledge"]
    woven = bool(INTEG.search(dec.summary))
    print("attempts:", dec.trace["attempts"])
    print("SUMMARY:\n", dec.summary)
    print("PROSE weaves a criterion/threshold term?:", woven)
    print("knowledge citations:", len(kc))
    runs[label] = {"attempts": dec.trace["attempts"], "summary": dec.summary,
                   "prose_weaves_criterion": woven, "n_knowledge_cites": len(kc),
                   "knowledge_citations": [{"feature": f, "statement": s} for f, s in kc]}

print("\n=== VERDICT ===")
print("B1 weaves criterion in prose:", runs["B1_no_directive"]["prose_weaves_criterion"])
print("B2 weaves criterion in prose:", runs["B2_integrate"]["prose_weaves_criterion"])
json.dump(runs, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Saved:", OUT)
