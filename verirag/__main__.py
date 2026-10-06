import argparse
import json
import sys

from .index import InvertedIndex
from .pipeline import Pipeline, build_from_corpus
from .query import BooleanTrace, run_boolean
from .retrieve import search
from .verifier import verdict_counts, verify


def _load(args):
    return InvertedIndex.load(args.index)


def cmd_build(args):
    params = {}
    if args.strategy == "fixed":
        params = {"size": args.chunk_size, "overlap": args.chunk_overlap}
    elif args.strategy == "sentence":
        params = {"sentences": args.chunk_sentences, "step": args.chunk_step}
    index, stats = build_from_corpus(
        args.corpus,
        strategy=args.strategy,
        chunk_params=params,
        stemming=not args.no_stemming,
        zone_weight=args.zone_weight,
        champion_size=args.champions,
        out_path=args.out,
    )
    print(json.dumps(stats, indent=2))
    print(f"saved index -> {args.out}")


def cmd_search(args):
    index = _load(args)
    ranked, trace = search(
        index,
        args.query,
        k=args.k,
        method=args.method,
        use_champions=args.champions,
        eliminate_threshold=args.eliminate,
        proximity_weight=args.proximity,
        proximity_window=args.prox_window,
        static_weight=args.static,
        explain=True,
    )
    print(f"query: {args.query}")
    print(
        f"terms: {trace.query_terms}  candidates={trace.candidates}  "
        f"eliminated={trace.eliminated}"
    )
    if args.explain:
        print("\n-- term statistics --")
        for term, st in trace.term_stats.items():
            print(
                f"  {term:16s} df={st['df']:<5d} idf={st['idf']:.3f} "
                f"qw={st['query_weight']:.4f} postings={st['postings_processed']}"
            )
    print("\n-- ranked results --")
    for i, r in enumerate(ranked, 1):
        doc = index.docs[r.doc_id]
        print(
            f"  {i:2d}. {r.score:.4f}  [{r.doc_id}]  {doc.title[:78]}  "
            f"(base={r.base:.4f} prox={r.proximity:.3f} g={r.g:.3f})"
        )
        if args.explain:
            for term, w in sorted(r.contributions.items(), key=lambda x: -x[1]):
                print(f"        {term:16s} -> {w:.4f}")


def cmd_term(args):
    index = _load(args)
    info = index.describe_term(args.term, limit=args.limit)
    print(f"term: {info['term']}")
    print(f"  df={info['df']}  idf={info['idf']:.4f}  postings={info['postings_len']}")
    print("  postings (doc, tf, tf_title, positions):")
    for p in info["postings_sample"]:
        print(f"    {p}")
    if info["skips"]:
        print(f"  skip pointers: {info['skips']}")
    if info["champions"]:
        print(f"  champion list: {info['champions']}")


def cmd_bool(args):
    index = _load(args)
    trace = BooleanTrace()
    doc_ids = run_boolean(index, args.query, trace)
    print(f"query: {args.query}")
    print(f"matches: {len(doc_ids)}")
    for step in trace.steps:
        print(f"  [{step['label']}] size={step['size']}  {step['detail']}")
    print("-- results --")
    for doc_id in doc_ids[: args.k]:
        doc = index.docs[doc_id]
        print(f"  [{doc_id}] {doc.title[:80]}")


def _print_answer(result, show_text=True):
    print(f"\n-- retrieved chunks (k={len(result['retrieved'])}) --")
    for r in result["retrieved"]:
        print(
            f"  [{r['rank']}] {r['score']:.4f}  {r['chunk_id']:<12} "
            f"doc={r['parent_doc']:<9} {r['title'][:60]}"
        )
        if show_text:
            print(f"      {r['text'][:160].replace(chr(10), ' ')}...")
    print("\n-- generated answer --")
    for i, s in enumerate(result["answer"], 1):
        cites = "".join(f"[{c}]" for c in s["citations"])
        print(f"  {i}. {s['text']} {cites}   (rel={s['relevance']})")
    if not result["answer"]:
        print("  (no answer sentences produced)")
    print("\n-- citation verification --")
    print(
        f"  {'#':>2} {'support':>7} {'cos':>6} {'novel':>6} {'verdict':<26} claim"
    )
    for i, row in enumerate(result["verification"], 1):
        print(
            f"  {i:2d} {row['support']:7.4f} {row['cosine']:6.4f} "
            f"{row['novelty_penalty']:6.3f} {row['verdict']:<26} {row['claim'][:60]}"
        )
        if row["novel_terms"]:
            print(f"      novel terms not in cited chunk: {row['novel_terms'][:8]}")
        if row["cite_mismatch"]:
            print(
                f"      ! better support in rank {row['best_rank']} "
                f"({row['best_support']:.4f}) than cited rank {row['citations'][0]}"
            )
    print(f"\nverdicts: {result['verdicts']}  |  total {result['elapsed_ms']} ms")


def cmd_ask(args):
    pipe = Pipeline.load(args.index)
    result = pipe.ask(
        args.query,
        k=args.k,
        generator="llm" if args.llm else "extractive",
        max_sentences=args.sentences,
        method=args.method,
        use_champions=args.champions,
        proximity_weight=args.proximity,
        proximity_window=args.prox_window,
        static_weight=args.static,
        eliminate_threshold=args.eliminate,
        model=args.model,
        explain=args.explain,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print(f"query: {args.query}")
    if args.explain and "trace" in result:
        print("\n-- query analysis --")
        print(f"  terms: {result['trace']['query_terms']}")
        print(f"  eliminated: {result['trace']['eliminated']}")
        for term, st in result["trace"]["term_stats"].items():
            print(
                f"  {term:16s} df={st['df']:<5d} idf={st['idf']:.3f} "
                f"qw={st['query_weight']:.4f} postings={st['postings_processed']}"
            )
    _print_answer(result)
    print("\n-- full verification JSON --")
    print(json.dumps(result["verification"], indent=2, ensure_ascii=False))


def cmd_verify(args):
    from .scoring import analyze_query
    from .verifier import doc_cosine, novelty_penalty, support_score

    index = _load(args)
    terms = analyze_query(
        args.claim,
        stemming=index.stemming,
        remove_stopwords=index.remove_stopwords,
        keep_negators=index.keep_negators,
    )
    ranked = search(index, args.claim, k=args.k, explain=False)
    print(f"claim: {args.claim}")
    print(f"terms: {terms}")
    print(f"\n-- support of this claim against each retrieved chunk --")
    print(f"  {'rank':>4} {'chunk':<12} {'cos':>6} {'novel':>6} {'support':>7}  title")
    rows = []
    for rank, r in enumerate(ranked, 1):
        cos = doc_cosine(index, terms, r.doc_id)
        novel, pen = novelty_penalty(index, terms, r.doc_id)
        support = support_score(cos, pen, args.novelty_weight)
        rows.append(support)
        doc = index.docs[r.doc_id]
        print(
            f"  {rank:4d} {r.doc_id:<12} {cos:6.4f} {pen:6.3f} {support:7.4f}  "
            f"{doc.title[:52]}"
        )
        if novel:
            print(f"        novel terms: {novel[:10]}")
    if rows:
        best = max(rows)
        verdict = "SUPPORTED" if best >= args.high else ("WEAK" if best >= args.low else "UNSUPPORTED")
        print(f"\nbest support={best:.4f} -> verdict: {verdict}")
        print(f"thresholds: high={args.high} low={args.low} novelty_weight={args.novelty_weight}")


def cmd_eval(args):
    from .evaluate import run_retrieval_eval

    run_retrieval_eval(args)


def cmd_web(args):
    from .web import serve

    serve(
        args.index,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        method=args.method,
        champions=args.champions,
        high=args.high,
        low=args.low,
        novelty_weight=args.novelty_weight,
    )


REPL_VERBS = {
    "search": "search", "s": "search",
    "bool": "bool", "b": "bool",
    "term": "term", "t": "term",
    "verify": "verify", "v": "verify",
    "ask": "ask", "a": "ask",
}


def cmd_repl(args):
    import sys as _sys

    _sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    from .retrieve import search as _search
    from .scoring import analyze_query as _analyze
    from .verifier import doc_cosine as _doc_cos
    from .verifier import novelty_penalty as _novel
    from .verifier import support_score as _support

    index = _load(args)
    pipe = Pipeline.load(args.index)
    k, explain, sents = args.k, False, args.sentences
    print(f"\nVeriRAG interactive  (index: {args.index}, {index.N} chunks)")
    print("Bare text      -> ask (RAG answer + verdict)")
    print("s: / search:   -> ranked search      b: / bool:   -> boolean trace")
    print("t: / term:     -> postings           v: / verify: -> support score")
    print("options: k N   sents N   explain on/off   help   quit\n")
    while True:
        try:
            line = input("verirag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "q"):
            return
        if low == "help":
            print("s: TEXT | b: QUERY | t: WORD | v: CLAIM | a: TEXT | k N | sents N | explain on/off | quit")
            continue
        parts = line.split(None, 1)
        head = parts[0].lower().rstrip(":")
        rest = parts[1].strip() if len(parts) > 1 else ""
        if head == "k":
            try:
                k = int(rest)
            except ValueError:
                print("k needs an integer")
            continue
        if head in ("sents", "sentences"):
            try:
                sents = int(rest)
            except ValueError:
                print("sents needs an integer")
            continue
        if head == "explain":
            explain = rest.startswith("on") or rest in ("1", "true")
            print("explain =", explain)
            continue
        mode = REPL_VERBS.get(head)
        if mode is None:
            mode, text = "ask", line
        elif rest:
            text = rest
        else:
            print("missing text after %r" % head)
            continue

        if mode == "ask":
            result = pipe.ask(
                text, k=k, generator="extractive", max_sentences=sents,
                method=args.method, use_champions=args.champions,
                proximity_weight=args.proximity, proximity_window=args.prox_window,
                static_weight=args.static, eliminate_threshold=args.eliminate,
                explain=explain,
            )
            print(f"\nquery: {text}")
            if explain and "trace" in result:
                print("terms: %s  eliminated=%s" % (result["trace"]["query_terms"], result["trace"]["eliminated"]))
            _print_answer(result, show_text=False)
        elif mode == "verify":
            terms = _analyze(text, stemming=index.stemming,
                             remove_stopwords=index.remove_stopwords,
                             keep_negators=index.keep_negators)
            ranked = _search(index, text, k=k, explain=False)
            print(f"\nclaim: {text}")
            print("terms: %s" % terms)
            print("  rank  chunk       cos     novel  support  title")
            best = 0.0
            for rank, r in enumerate(ranked, 1):
                cos = _doc_cos(index, terms, r.doc_id)
                novel, pen = _novel(index, terms, r.doc_id)
                support = _support(cos, pen, 0.6)
                best = max(best, support)
                doc = index.docs[r.doc_id]
                print("  %4d %-12s %.4f  %.3f  %.4f  %s" % (rank, r.doc_id, cos, pen, support, doc.title[:40]))
                if novel:
                    print("       novel terms: %s" % novel[:8])
            verdict = "SUPPORTED" if best >= 0.20 else ("WEAK" if best >= 0.12 else "UNSUPPORTED")
            print("best support=%.4f -> verdict: %s" % (best, verdict))
        elif mode == "search":
            result = _search(index, text, k=k, method=args.method,
                             use_champions=args.champions,
                             eliminate_threshold=args.eliminate,
                             proximity_weight=args.proximity,
                             proximity_window=args.prox_window,
                             static_weight=args.static, explain=explain)
            ranked, trace = (result if explain else (result, None))
            print(f"\nquery: {text}")
            if trace is not None:
                print("terms: %s  candidates=%d  eliminated=%s" % (trace.query_terms, trace.candidates, trace.eliminated))
            for i, r in enumerate(ranked, 1):
                doc = index.docs[r.doc_id]
                print("  %2d. %.4f  [%s]  %s  (base=%.4f prox=%.3f g=%.3f)"
                      % (i, r.score, r.doc_id, doc.title[:45], r.base, r.proximity, r.g))
        elif mode == "bool":
            tr = BooleanTrace()
            doc_ids = run_boolean(index, text, tr)
            print(f"\nquery: {text}\nmatches: {len(doc_ids)}")
            for step in tr.steps:
                print("  [%s] size=%d  %s" % (step["label"], step["size"], step["detail"]))
            for doc_id in doc_ids[: k]:
                print("  [%s] %s" % (doc_id, index.docs[doc_id].title[:50]))
        elif mode == "term":
            info = index.describe_term(text, limit=6)
            print("term: %s  df=%s  idf=%.4f" % (info["term"], info["df"], info["idf"]))
            for p in info["postings_sample"]:
                print("   %s" % p)
            if info["skips"]:
                print("   skip pointers: %s" % info["skips"])


def cmd_eval_verify(args):
    from .evaluate import run_verifier_eval

    run_verifier_eval(args)


def build_parser():
    p = argparse.ArgumentParser(
        prog="verirag",
        description="VeriRAG: an inspectable tf-idf RAG pipeline with claim-level citation verification",
    )
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="chunk a corpus and build the inverted index")
    b.add_argument("--corpus", default="data/scifact/corpus.jsonl")
    b.add_argument("--out", default="output/index.pkl")
    b.add_argument("--strategy", choices=["whole", "fixed", "sentence"], default="sentence")
    b.add_argument("--chunk-size", type=int, default=120)
    b.add_argument("--chunk-overlap", type=int, default=30)
    b.add_argument("--chunk-sentences", type=int, default=3)
    b.add_argument("--chunk-step", type=int, default=2)
    b.add_argument("--zone-weight", type=float, default=2.0)
    b.add_argument("--champions", type=int, default=0)
    b.add_argument("--no-stemming", action="store_true")
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("search", help="ranked (vector-space) retrieval")
    s.add_argument("query")
    s.add_argument("--index", default="output/index.pkl")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--method", choices=["tfidf", "bm25"], default="tfidf")
    s.add_argument("--champions", action="store_true")
    s.add_argument("--eliminate", type=float, default=1.0)
    s.add_argument("--proximity", type=float, default=0.5)
    s.add_argument("--prox-window", type=int, default=30)
    s.add_argument("--static", type=float, default=0.0)
    s.add_argument("--explain", action="store_true")
    s.set_defaults(func=cmd_search)

    t = sub.add_parser("term", help="show dictionary entry / postings for a term")
    t.add_argument("term")
    t.add_argument("--index", default="output/index.pkl")
    t.add_argument("--limit", type=int, default=8)
    t.set_defaults(func=cmd_term)

    bo = sub.add_parser("bool", help="boolean/phrase/zone query with query-processor trace")
    bo.add_argument("query")
    bo.add_argument("--index", default="output/index.pkl")
    bo.add_argument("-k", type=int, default=10)
    bo.set_defaults(func=cmd_bool)

    a = sub.add_parser("ask", help="end-to-end RAG: retrieve, answer, verify citations")
    a.add_argument("query")
    a.add_argument("--index", default="output/index.pkl")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--llm", action="store_true", help="use an OpenAI-compatible LLM as generator")
    a.add_argument("--model", default=None)
    a.add_argument("--sentences", type=int, default=3)
    a.add_argument("--method", choices=["tfidf", "bm25"], default="tfidf")
    a.add_argument("--champions", action="store_true")
    a.add_argument("--eliminate", type=float, default=1.0)
    a.add_argument("--proximity", type=float, default=0.5)
    a.add_argument("--prox-window", type=int, default=30)
    a.add_argument("--static", type=float, default=0.0)
    a.add_argument("--explain", action="store_true")
    a.add_argument("--json", action="store_true")
    a.set_defaults(func=cmd_ask)

    v = sub.add_parser("verify", help="score one claim against retrieved chunks")
    v.add_argument("claim")
    v.add_argument("--index", default="output/index.pkl")
    v.add_argument("-k", type=int, default=5)
    v.add_argument("--high", type=float, default=0.20)
    v.add_argument("--low", type=float, default=0.12)
    v.add_argument("--novelty-weight", type=float, default=0.6)
    v.set_defaults(func=cmd_verify)

    e = sub.add_parser("eval", help="retrieval evaluation (nDCG/P/R/MRR) vs baselines")
    e.add_argument("--index", default="output/index.pkl")
    e.add_argument("--corpus", default="data/scifact/corpus.jsonl")
    e.add_argument("--queries", default="data/scifact/queries.jsonl")
    e.add_argument("--qrels", default="data/scifact/qrels/test.tsv")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--charts", action="store_true")
    e.set_defaults(func=cmd_eval)

    ev = sub.add_parser("eval-verify", help="citation-verifier evaluation against SciFact labels")
    ev.add_argument("--index", default="output/index.pkl")
    ev.add_argument("--corpus", default="data/scifact/corpus.jsonl")
    ev.add_argument("--queries", default="data/scifact/queries.jsonl")
    ev.add_argument("--qrels", default="data/scifact/qrels/test.tsv")
    ev.add_argument("--limit", type=int, default=0)
    ev.add_argument("--charts", action="store_true")
    ev.set_defaults(func=cmd_eval_verify)

    rp = sub.add_parser("repl", help="interactive prompt: type queries, claims or commands")
    rp.add_argument("--index", default="output/index.pkl")
    rp.add_argument("-k", type=int, default=5)
    rp.add_argument("--sentences", type=int, default=3)
    rp.add_argument("--method", choices=["tfidf", "bm25"], default="tfidf")
    rp.add_argument("--champions", action="store_true")
    rp.add_argument("--eliminate", type=float, default=1.0)
    rp.add_argument("--proximity", type=float, default=0.5)
    rp.add_argument("--prox-window", type=int, default=30)
    rp.add_argument("--static", type=float, default=0.0)
    rp.set_defaults(func=cmd_repl)

    wb = sub.add_parser("web", help="browser UI: local web app for queries, claims, index inspection")
    wb.add_argument("--index", default="output/index.pkl")
    wb.add_argument("--host", default="127.0.0.1")
    wb.add_argument("--port", type=int, default=8123, help="start here, auto-increments if blocked")
    wb.add_argument("--method", choices=["tfidf", "bm25"], default="tfidf")
    wb.add_argument("--champions", action="store_true")
    wb.add_argument("--high", type=float, default=0.20)
    wb.add_argument("--low", type=float, default=0.12)
    wb.add_argument("--novelty-weight", type=float, default=0.6)
    wb.add_argument("--no-open", action="store_true", help="do not auto-open the browser")
    wb.set_defaults(func=cmd_web)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
