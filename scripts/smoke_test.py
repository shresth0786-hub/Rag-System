import json
import sys
import time
from pathlib import Path

# Robust to cwd: resolve project root (parent of scripts/) instead of relying on "."
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from verirag.index import InvertedIndex
from verirag.query import BooleanTrace, run_boolean
from verirag.retrieve import search

t0 = time.time()
idx = InvertedIndex()
corpus_path = PROJECT_ROOT / "data" / "scifact" / "corpus.jsonl"
for line in open(corpus_path, encoding="utf-8"):
    d = json.loads(line)
    idx.add_document(d["_id"], d["title"], d["text"])
idx.finalize(champion_size=50)
print(f"built {idx.N} docs, {len(idx.postings)} terms in {time.time()-t0:.1f}s")

q = "smoking cessation reduces cardiovascular risk"
t0 = time.time()
ranked, trace = search(idx, q, k=5, explain=True, static_weight=0.1, proximity_weight=0.15)
print(f"search in {time.time()-t0:.3f}s")
for r in ranked:
    doc = idx.docs[r.doc_id]
    print(f"  {r.score:.4f} [{r.doc_id}] {doc.title[:80]}")
print("term stats:", json.dumps(trace.term_stats, indent=1, default=str)[:800])

for fq in [
    '"smoking cessation" AND risk',
    'title:smoking AND NOT cancer',
    '(smoking OR tobacco) AND risk',
]:
    tr = BooleanTrace()
    res = run_boolean(idx, fq, tr)
    print(f"\nBOOL {fq!r} -> {len(res)} docs")
    for s in tr.steps:
        print("   ", s)
    for d in res[:3]:
        print("    ", d, idx.docs[d].title[:70])
