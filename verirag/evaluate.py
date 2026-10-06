import csv
import json
import os
import time

from .chunker import split_sentences
from .metrics import aggregate, average_precision, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .pipeline import build_from_corpus
from .retrieve import search
from .scoring import analyze_query, cosine, query_vector
from .text import analyze
from .verifier import doc_cosine, novelty_penalty

MAIN_BUILD = dict(strategy="sentence", chunk_params={"sentences": 3, "step": 2}, champion_size=50)


def load_queries(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d
    return out


def load_qrels(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            out.setdefault(row[0], {})[row[1]] = int(float(row[2]))
    return out


def get_index(tag, corpus, args):
    path = f"output/index_{tag}.pkl"
    if os.path.exists(path):
        from .index import InvertedIndex

        return InvertedIndex.load(path)
    stemming = tag != "plain"
    zone = 2.0 if tag == "main" else 1.0
    _, stats = build_from_corpus(
        corpus,
        strategy=MAIN_BUILD["strategy"],
        chunk_params=MAIN_BUILD["chunk_params"],
        stemming=stemming,
        zone_weight=zone,
        champion_size=MAIN_BUILD["champion_size"] if tag == "main" else 0,
        out_path=path,
    )
    print(f"built index [{tag}]: {stats['n_chunks']} chunks in {stats['build_seconds']}s")
    from .index import InvertedIndex

    return InvertedIndex.load(path)


def parent_ranking(index, ranked):
    seen = set()
    docs = []
    for r in ranked:
        parent = index.docs[r.doc_id].meta.get("doc_id", r.doc_id)
        if parent not in seen:
            seen.add(parent)
            docs.append(parent)
    return docs


def eval_ranked_docs(docs, gold, k_list=(5, 10, 100)):
    rels = [1 if d in gold else 0 for d in docs]
    total = len(gold)
    row = {
        "nDCG@10": ndcg_at_k(rels, 10),
        "P@5": precision_at_k(rels, 5),
        "P@10": precision_at_k(rels, 10),
        "R@100": recall_at_k(rels, 100, total),
        "MRR@10": mrr(rels),
        "MAP": average_precision(rels, total),
    }
    return row


def run_config(index, queries, qrels, method="tfidf", proximity=0.0, static=0.0,
               champions=False, eliminate=1.0, k=100, qids=None,
               bm25_k1=1.5, bm25_b=0.75, proximity_window=10):
    rows = []
    latencies = []
    for qid in qids:
        qtext = queries[qid]["text"]
        t0 = time.perf_counter()
        ranked = search(
            index,
            qtext,
            k=k,
            method=method,
            use_champions=champions,
            eliminate_threshold=eliminate,
            proximity_weight=proximity,
            proximity_window=proximity_window,
            static_weight=static,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        docs = parent_ranking(index, ranked)
        rows.append(eval_ranked_docs(docs, qrels[qid]))
    scores = aggregate(rows)
    scores["ms/query"] = sum(latencies) / max(len(latencies), 1)
    return scores


def split_qids(qrels, n_dev=100):
    qids = sorted(qrels, key=lambda x: int(x))
    return qids[:n_dev], qids[n_dev:]


def format_table(results, keys):
    head = f"{'configuration':42s} " + " ".join(f"{k:>9s}" for k in keys)
    lines = [head, "-" * len(head)]
    for name, r in results:
        cells = []
        for k in keys:
            v = r.get(k, 0)
            cells.append(f"{v:9d}" if k == "n" else f"{v:9.4f}")
        lines.append(f"{name:42s} " + " ".join(cells))
    return "\n".join(lines)


def run_retrieval_eval(args):
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    dev_qids, test_qids = split_qids(qrels, n_dev=100)
    test_qids = [q for q in test_qids if q in queries]

    print(f"queries: {len(qrels)} judged (dev={len(dev_qids)}, test={len(test_qids)})")
    plain = get_index("plain", args.corpus, args)
    stem = get_index("stem", args.corpus, args)
    main = get_index("main", args.corpus, args)

    print("tuning proximity weight/window and static weight eta on dev queries...")
    best_cfg = {"proximity": 0.0, "window": 10, "static": 0.0, "ndcg": -1.0}
    for pw in (0.0, 0.5, 1.0):
        for win in (10, 30):
            if pw == 0.0 and win != 10:
                continue
            for eta in (0.0, 0.1):
                s = run_config(
                    main, queries, qrels,
                    proximity=pw, static=eta, proximity_window=win,
                    qids=dev_qids, k=20,
                )
                print(
                    f"  prox={pw:<4} win={win:<3} eta={eta:<3} "
                    f"dev nDCG@10={s['nDCG@10']:.4f}"
                )
                if s["nDCG@10"] > best_cfg["ndcg"]:
                    best_cfg = {"proximity": pw, "window": win, "static": eta, "ndcg": s["nDCG@10"]}
    pw, win, eta = best_cfg["proximity"], best_cfg["window"], best_cfg["static"]
    print(f"chosen: proximity_weight={pw} window={win} eta={eta}")

    configs = [
        ("A0 plain tf-idf (no stem, no zone)", dict(index=plain)),
        ("A1 + Porter stemming", dict(index=stem)),
        ("A2 + title zone weight (lambda=2)", dict(index=main)),
        (
            f"A3 + query term proximity (w={pw}, win={win})",
            dict(index=main, proximity=pw, proximity_window=win),
        ),
        (
            f"A4 + static quality g(d) [eta={eta}]",
            dict(index=main, proximity=pw, proximity_window=win, static=eta),
        ),
        (
            "A4 + champion lists (top-50)",
            dict(index=main, proximity=pw, proximity_window=win, static=eta, champions=True),
        ),
        ("BM25 (k1=1.5, b=0.75)", dict(index=stem, method="bm25")),
        (
            "BM25 (k1=0.9, b=0.4)",
            dict(index=stem, method="bm25", bm25_k1=0.9, bm25_b=0.4),
        ),
    ]

    keys = ["nDCG@10", "P@5", "P@10", "R@100", "MRR@10", "ms/query"]
    results = []
    for name, cfg in configs:
        cfg = dict(cfg)
        index = cfg.pop("index")
        method = cfg.pop("method", "tfidf")
        t0 = time.time()
        scores = run_config(
            index, queries, qrels, method=method, qids=test_qids, **cfg
        )
        results.append((name, scores))
        print(f"  {name:42s} nDCG@10={scores['nDCG@10']:.4f}  ({time.time()-t0:.0f}s)")

    print("\n" + format_table(results, keys))

    os.makedirs("output", exist_ok=True)
    with open("output/eval_retrieval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration"] + keys)
        for name, r in results:
            w.writerow([name] + [round(r.get(k, 0), 5) for k in keys])
    md = ["| configuration | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, r in results:
        md.append(f"| {name} | " + " | ".join(f"{r.get(k, 0):.4f}" for k in keys) + " |")
    with open("output/eval_retrieval.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\nwrote output/eval_retrieval.csv and output/eval_retrieval.md")

    if args.charts:
        _chart_retrieval(results, keys)
    return results


def _chart_retrieval(results, keys):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_keys = ["nDCG@10", "P@10", "R@100"]
    labels = [name for name, _ in results]
    x = range(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for i, key in enumerate(plot_keys):
        vals = [r[key] for _, r in results]
        bars = ax.bar([v + (i - 1) * width for v in x], vals, width, label=key, color=colors[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Retrieval quality on SciFact test queries (n=200, judged)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("output/eval_retrieval.png", dpi=160)
    print("wrote output/eval_retrieval.png")


def _claim_terms(text, index):
    return analyze(
        text,
        stemming=index.stemming,
        remove_stopwords=index.remove_stopwords,
        keep_negators=index.keep_negators,
    )


def score_claim(index, terms, qvec, doc_id, novelty_weight):
    cos = doc_cosine(index, terms, doc_id)
    novel, pen = novelty_penalty(index, terms, doc_id)
    support = min(1.0, max(0.0, cos - novelty_weight * pen))
    return support, cos, pen, novel


def apply_penalty(cos, pen, weight, form):
    if form == "mul":
        return min(1.0, max(0.0, cos * (1.0 - min(1.0, weight * pen))))
    return min(1.0, max(0.0, cos - weight * pen))


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def collect_verifier_items(queries, qrels):
    items = []
    for qid in qrels:
        q = queries.get(qid)
        if q is None:
            continue
        if q["metadata"]:
            for doc_id, entries in q["metadata"].items():
                labels = [e["label"] for e in entries]
                label = "SUPPORT" if "SUPPORT" in labels else "CONTRADICT"
                items.append({"qid": qid, "claim": q["text"], "doc": doc_id, "label": label})
        else:
            items.append({"qid": qid, "claim": q["text"], "doc": None, "label": "NEI"})
    return items


def tune_threshold(scores_by_item, scorer_key, candidates=None):
    candidates = candidates or [i / 100 for i in range(5, 71, 5)]
    best_t, best_f1 = 0.3, -1.0
    for t in candidates:
        tp = fp = fn = 0
        for it in scores_by_item:
            pred = it[scorer_key] >= t
            gold = it["label"] == "SUPPORT"
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif gold and not pred:
                fn += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def binary_metrics(items, key, threshold):
    tp = fp = fn = tn = 0
    for it in items:
        pred = it[key] >= threshold
        gold = it["label"] == "SUPPORT"
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif gold and not pred:
            fn += 1
        else:
            tn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return {"P(SUPPORT)": p, "R(SUPPORT)": r, "F1(SUPPORT)": f1, "accuracy": acc, "n": tp + tn + fp + fn}


def parent_chunks(index, doc_id):
    return [d for d, entry in index.docs.items() if entry.meta.get("doc_id") == doc_id]


def tune_penalty(dev_items):
    best = {"form": "sub", "weight": 0.6, "threshold": 0.3, "f1": -1.0}
    for form in ("sub", "mul"):
        for weight in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
            for it in dev_items:
                it["_tmp"] = apply_penalty(it["cos_score"], it["novel_score"], weight, form)
            thr, f1 = tune_threshold(dev_items, "_tmp")
            if f1 > best["f1"]:
                best = {"form": form, "weight": weight, "threshold": thr, "f1": f1}
    for it in dev_items:
        it.pop("_tmp", None)
    return best


def run_verifier_eval(args):
    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    from .index import InvertedIndex

    if not os.path.exists(args.index):
        raise SystemExit(f"index not found at {args.index}; run `python -m verirag build` first")
    index = InvertedIndex.load(args.index)

    items = collect_verifier_items(queries, qrels)
    if args.limit:
        items = items[: args.limit]
    dev_qids, test_qids = split_qids(qrels, n_dev=100)
    dev_set, test_set = set(dev_qids), set(test_qids)
    dev_items = [it for it in items if it["qid"] in dev_set]
    test_items = [it for it in items if it["qid"] in test_set]

    print(f"verifier items: {len(items)} (dev={len(dev_items)}, test={len(test_items)})")

    gold_chunks_missing = 0
    for it in items:
        terms = _claim_terms(it["claim"], index)
        qvec = query_vector(index, terms)
        it["novel_score"] = 0.0
        if it["doc"] is not None:
            chunks = parent_chunks(index, it["doc"])
            if not chunks:
                gold_chunks_missing += 1
                it["cos_score"] = 0.0
                it["jaccard_score"] = 0.0
                continue
            best = None
            for c in chunks:
                support, cos, pen, _ = score_claim(index, terms, qvec, c, 0.0)
                if best is None or cos > best[1]:
                    best = (support, cos, pen)
            it["cos_score"] = best[1]
            it["novel_score"] = best[2]
            doc_terms = set()
            for c in chunks:
                doc_terms |= set(_claim_terms(index.docs[c].text, index))
            it["jaccard_score"] = jaccard(set(terms), doc_terms)
        else:
            ranked = search(index, it["claim"], k=5)
            cos = 0.0
            pen = 0.0
            jac = 0.0
            for r in ranked:
                _, c, p, _ = score_claim(index, terms, qvec, r.doc_id, 0.0)
                if c > cos:
                    cos, pen = c, p
                jac = max(jac, jaccard(set(terms), set(_claim_terms(index.docs[r.doc_id].text, index))))
            it["cos_score"], it["novel_score"], it["jaccard_score"] = cos, pen, jac

    pen_cfg = tune_penalty(dev_items)
    form, weight = pen_cfg["form"], pen_cfg["weight"]
    for it in items:
        it["ours_score"] = apply_penalty(it["cos_score"], it["novel_score"], weight, form)
    t_ours_thr, f1_ours_dev = tune_threshold(dev_items, "ours_score")
    t_cos, f1_cos_dev = tune_threshold(dev_items, "cos_score")
    t_jac, f1_jac_dev = tune_threshold(dev_items, "jaccard_score")
    print(
        f"dev-tuned verifier: form={form} weight={weight} threshold={t_ours_thr:.2f} "
        f"(F1={f1_ours_dev:.3f})"
    )
    print(
        f"dev-tuned baselines: cosine threshold={t_cos:.2f} (F1={f1_cos_dev:.3f}), "
        f"jaccard threshold={t_jac:.2f} (F1={f1_jac_dev:.3f})"
    )

    results = [
        ("baseline: Jaccard overlap", binary_metrics(test_items, "jaccard_score", t_jac)),
        ("baseline: cosine only", binary_metrics(test_items, "cos_score", t_cos)),
        ("VeriRAG: cosine - novelty penalty", binary_metrics(test_items, "ours_score", t_ours_thr)),
    ]

    keys = ["P(SUPPORT)", "R(SUPPORT)", "F1(SUPPORT)", "accuracy", "n"]
    print("\n" + format_table(results, keys))

    retrieval_rows = []
    hit5 = hit10 = mrr_sum = 0.0
    n_rel = 0
    for it in test_items:
        if it["doc"] is None:
            continue
        n_rel += 1
        ranked = search(index, it["claim"], k=10)
        docs = parent_ranking(index, ranked)
        rels = [1 if d == it["doc"] else 0 for d in docs]
        mrr_sum += mrr(rels)
        if rels[:5].count(1):
            hit5 += 1
        if rels[:10].count(1):
            hit10 += 1
    if n_rel:
        retrieval_rows.append(
            {
                "metric": "claim -> gold-document retrieval",
                "hit@5": hit5 / n_rel,
                "hit@10": hit10 / n_rel,
                "MRR": mrr_sum / n_rel,
                "n": n_rel,
            }
        )
        print("\n-- retrieval stage (test, claim -> gold evidence doc) --")
        print(
            f"  hit@5={hit5/n_rel:.4f}  hit@10={hit10/n_rel:.4f}  "
            f"MRR={mrr_sum/n_rel:.4f}  (n={n_rel})"
        )

    ans_rows = _answer_evidence_eval(index, queries, test_items)
    if ans_rows:
        print("\n-- answer stage: does the extractive answer include a gold evidence sentence? --")
        for row in ans_rows:
            print("  " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))

    os.makedirs("output", exist_ok=True)
    with open("output/eval_verifier.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + keys)
        for name, r in results:
            w.writerow([name] + [round(r.get(k, 0), 5) if k != "n" else r["n"] for k in keys])
    with open("output/eval_verifier.md", "w", encoding="utf-8") as f:
        md = ["| method | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        for name, r in results:
            md.append(
                f"| {name} | " + " | ".join(str(round(r[k], 4)) if k != "n" else str(r[k]) for k in keys) + " |"
            )
        f.write("\n".join(md) + "\n")
    print("\nwrote output/eval_verifier.csv / .md")

    if args.charts:
        _chart_verifier(test_items, t_ours_thr, t_cos, t_jac)
    return results


def _answer_evidence_eval(index, queries, test_items):
    from .generator import extractive_answer

    support_items = [it for it in test_items if it["label"] == "SUPPORT" and it["doc"]]
    if not support_items:
        return []
    hit_full = 0
    hit_base = 0
    n = 0
    for it in support_items:
        gold = it["doc"]
        chunks = parent_chunks(index, gold)
        if not chunks:
            continue
        meta = queries[it["qid"]]["metadata"].get(gold, [])
        idxs = set()
        for entry in meta:
            idxs.update(entry.get("sentences", []))
        if not idxs:
            continue
        gold_texts = set()
        for cid in chunks:
            entry = index.docs[cid]
            sents = split_sentences(entry.text)
            start = entry.meta.get("sentence_start", 0) or 0
            for i in idxs:
                rel = i - start
                if 0 <= rel < len(sents):
                    gold_texts.add(sents[rel].strip())
        if not gold_texts:
            continue
        n += 1
        ranked = search(index, it["claim"], k=5)
        answer = extractive_answer(index, ranked, it["claim"], max_sentences=3)
        ans_texts = {s["text"].strip() for s in answer}
        if ans_texts & gold_texts:
            hit_full += 1
        top = ranked[0]
        top_doc = index.docs[top.doc_id].meta.get("doc_id")
        if top_doc == gold:
            top_sents = split_sentences(index.docs[top.doc_id].text)
            if top_sents and top_sents[0].strip() in gold_texts:
                hit_base += 1
    if not n:
        return []
    return [
        {"metric": "evidence-sentence recall", "n": n},
        {"baseline: first sentence of top-1 chunk": hit_base / n},
        {"VeriRAG extractive answer (3 sentences)": hit_full / n},
    ]


def _chart_verifier(test_items, t_ours, t_cos, t_jac):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"SUPPORT": "#55A868", "CONTRADICT": "#C44E52", "NEI": "#8172B3"}
    for key, ax, title, thr in [
        ("cos_score", axes[0], "baseline: raw cosine", t_cos),
        ("ours_score", axes[1], "VeriRAG: cosine - novelty penalty", t_ours),
    ]:
        for label in ("SUPPORT", "CONTRADICT", "NEI"):
            vals = [it[key] for it in test_items if it["label"] == label]
            if vals:
                ax.hist(vals, bins=25, range=(0, 1), alpha=0.55, label=label, color=colors[label])
        ax.axvline(thr, color="black", linestyle="--", linewidth=1.2, label=f"threshold={thr:.2f}")
        ax.set_title(title)
        ax.set_xlabel("support score")
        ax.set_ylabel("claims")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Claim-level support scores by gold label (test split)")
    fig.tight_layout()
    fig.savefig("output/eval_verifier.png", dpi=160)
    print("wrote output/eval_verifier.png")
