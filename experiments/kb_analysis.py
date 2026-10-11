"""Analyze the RAG knowledge base content quality. Local, no LLM."""
import json, re, statistics as st
from collections import Counter, defaultdict

chunks = json.load(open("results/rag_knowledge_base/chunks.json", encoding="utf-8"))
N = len(chunks)
print(f"TOTAL chunks: {N}\n")

# --- Per source + type ---
by_src = Counter(c["source"] for c in chunks)
by_type = Counter(c["chunk_type"] for c in chunks)
print("=== Chunks per source ===")
for s, n in by_src.most_common():
    lens = [len(c["text"]) for c in chunks if c["source"] == s]
    print(f"  {n:3d}  {s:42s}  mean {int(st.mean(lens)):4d} chars, median {int(st.median(lens)):4d}")
print("chunk_type:", dict(by_type), "\n")

# --- Junk / boilerplate heuristics ---
def is_short(c):        return len(c["text"]) < 150
def has_email(c):       return "@" in c["text"] or "e-mail" in c["text"].lower()
def ref_list(c):        # reference-list / citation boilerplate
    t = c["text"]; return len(re.findall(r"\b(19|20)\d\d\b", t)) >= 4 or t.lower().count("et al") >= 2 or "doi" in t.lower()
def hi_nonalpha(c):
    t = c["text"]; alpha = sum(ch.isalpha() or ch.isspace() for ch in t)
    return len(t) > 0 and alpha / len(t) < 0.75

junk = defaultdict(list)
for i, c in enumerate(chunks):
    if is_short(c):     junk["short(<150)"].append(i)
    if has_email(c):    junk["email/contact"].append(i)
    if ref_list(c):     junk["reference-list"].append(i)
    if hi_nonalpha(c):  junk["hi-nonalpha"].append(i)
print("=== Junk/boilerplate heuristics (chunk counts) ===")
flagged = set()
for k, idxs in junk.items():
    print(f"  {k:16s}: {len(idxs)}")
    flagged |= set(idxs)
print(f"  UNION flagged as low-value: {len(flagged)} / {N} ({100*len(flagged)/N:.0f}%)\n")

# --- Clinical-criteria signal heuristic ---
CRIT = re.compile(r"\b(criteri|threshold|cut-?off|defined as|diagnos|normal range|"
                  r"\d+\s?%|\d+\s?mm|\d+\s?ms|\bEF\b|ejection fraction|guideline|"
                  r"\bmmHg\b|voltage|Sokolow|\bmg/dl\b|classif)", re.I)
DATASET = re.compile(r"\b(dataset|annotat|record was|records? were|Table \d|column|"
                     r"\.nii|\.csv|repositor|benchmark|challenge|training set|test set)", re.I)
clin = [i for i, c in enumerate(chunks) if CRIT.search(c["text"]) and i not in flagged]
dsdoc = [i for i, c in enumerate(chunks) if DATASET.search(c["text"])]
print("=== Content signal ===")
print(f"  chunks with clinical-criteria markers (and not junk): {len(clin)} ({100*len(clin)/N:.0f}%)")
print(f"  chunks with dataset-documentation markers:            {len(dsdoc)} ({100*len(dsdoc)/N:.0f}%)")
print("  clinical-criteria chunks BY SOURCE:")
for s, n in Counter(chunks[i]["source"] for i in clin).most_common():
    print(f"     {n:3d}  {s}")

print("\n=== Sample: 3 clinical-criteria chunks ===")
for i in clin[:3]:
    print(f"  [{chunks[i]['source']}] {chunks[i]['text'][:200].strip().replace(chr(10),' ')}...")
print("\n=== Sample: 3 flagged-junk chunks ===")
for i in list(flagged)[:3]:
    print(f"  [{chunks[i]['source']}] {chunks[i]['text'][:160].strip().replace(chr(10),' ')}...")
