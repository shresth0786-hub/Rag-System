import sys
import time

sys.path.insert(0, ".")

from verirag.evaluate import load_qrels, load_queries, run_config, split_qids
from verirag.index import InvertedIndex

queries = load_queries("data/scifact/queries.jsonl")
qrels = load_qrels("data/scifact/qrels/test.tsv")
_, test_qids = split_qids(qrels, 100)

CONFIGS = [
    ("whole: 1 chunk per document", "chunk_whole", dict(strategy="whole", chunk_params={})),
    ("fixed: 120-token windows, 30 overlap", "chunk_fixed",
     dict(strategy="fixed", chunk_params={"size": 120, "overlap": 30})),
    ("sentence: 3 sentences, step 2", "chunk_sent",
     dict(strategy="sentence", chunk_params={"sentences": 3, "step": 2})),
    ("sentence: 5 sentences, step 3", "chunk_sent5",
     dict(strategy="sentence", chunk_params={"sentences": 5, "step": 3})),
]

from verirag.pipeline import build_from_corpus

rows = []
for name, tag, cfg in CONFIGS:
    path = f"output/index_{tag}.pkl"
    try:
        index = InvertedIndex.load(path)
        print(f"loaded [{tag}]")
    except FileNotFoundError:
        t0 = time.time()
        index, stats = build_from_corpus(
            "data/scifact/corpus.jsonl",
            strategy=cfg["strategy"],
            chunk_params=cfg["chunk_params"],
            stemming=True,
            zone_weight=2.0,
            champion_size=0,
            out_path=path,
        )
        print(f"built [{tag}] {stats['n_chunks']} chunks in {stats['build_seconds']}s")
    s = run_config(
        index, queries, qrels, qids=test_qids, k=100,
        proximity=0.5, proximity_window=30, static=0.0,
    )
    rows.append((name, index.N, s))
    print(f"  {name:40s} nDCG@10={s['nDCG@10']:.4f} P@10={s['P@10']:.4f} R@100={s['R@100']:.4f} MRR={s['MRR@10']:.4f}")

print("\nchunking strategy                      chunks  nDCG@10  P@10   R@100  MRR@10")
for name, n, s in rows:
    print(f"{name:40s} {n:6d}  {s['nDCG@10']:.4f}  {s['P@10']:.4f} {s['R@100']:.4f} {s['MRR@10']:.4f}")
