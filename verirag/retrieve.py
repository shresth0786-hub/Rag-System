import heapq
from collections import Counter, defaultdict

from .scoring import (
    analyze_query,
    bm25_score,
    doc_weight,
    prefetch_positions,
    proximity_bonus,
    query_vector,
)


class RankedDoc:
    __slots__ = ("doc_id", "score", "base", "proximity", "g", "contributions")

    def __init__(self, doc_id, score, base, proximity, g, contributions):
        self.doc_id = doc_id
        self.score = score
        self.base = base
        self.proximity = proximity
        self.g = g
        self.contributions = contributions


class RetrievalTrace:
    def __init__(self):
        self.query_terms = []
        self.eliminated = []
        self.term_stats = {}
        self.candidates = 0
        self.method = "tfidf"


def search(
    index,
    raw_query,
    k=10,
    method="tfidf",
    use_champions=False,
    eliminate_threshold=1.0,
    proximity_weight=0.5,
    proximity_window=30,
    static_weight=0.0,
    bm25_k1=1.5,
    bm25_b=0.75,
    explain=False,
):
    terms = analyze_query(
        raw_query,
        stemming=index.stemming,
        remove_stopwords=index.remove_stopwords,
        keep_negators=index.keep_negators,
    )
    trace = RetrievalTrace()
    trace.method = method
    trace.query_terms = terms

    unique_terms = list(dict.fromkeys(terms))
    used = []
    for term in unique_terms:
        df = index.df.get(term, 0)
        if df == 0:
            trace.eliminated.append((term, "not in dictionary"))
            continue
        if df / index.N > eliminate_threshold:
            trace.eliminated.append((term, f"df={df} exceeds threshold"))
            continue
        used.append(term)

    qvec = query_vector(index, [t for t in terms if t in set(used)])

    base_scores = defaultdict(float)
    contributions = defaultdict(dict)
    docs_with_term = defaultdict(int)

    for term in used:
        plist = index.postings[term]
        postings_seen = 0
        if use_champions and term in index.champions:
            champion_set = set(index.champions[term])
            candidates = [p for p in plist if p.doc_id in champion_set]
        else:
            candidates = plist
        qw = qvec.get(term, 0.0)
        for p in candidates:
            postings_seen += 1
            if method == "bm25":
                w = qw * bm25_score(index, p, term, k1=bm25_k1, b=bm25_b)
            else:
                w = qw * doc_weight(index, p)
            base_scores[p.doc_id] += w
            contributions[p.doc_id][term] = w
            docs_with_term[p.doc_id] += 1
        trace.term_stats[term] = {
            "df": index.df.get(term, 0),
            "idf": index.idf(term),
            "query_weight": qw,
            "postings_processed": postings_seen,
            "champion_list": bool(use_champions and term in index.champions),
        }

    trace.candidates = len(base_scores)

    pos_index = None
    if proximity_weight > 0 and len(used) >= 2:
        pos_index = prefetch_positions(index, used)

    results = []
    for doc_id, base in base_scores.items():
        prox = 0.0
        if pos_index is not None and docs_with_term[doc_id] >= 2:
            prox, _, _ = proximity_bonus(pos_index, doc_id, used, window=proximity_window)
        g = index.docs[doc_id].g if static_weight > 0 else 0.0
        score = base * (1.0 + proximity_weight * prox) + static_weight * g
        results.append(RankedDoc(doc_id, score, base, prox, g, dict(contributions[doc_id])))

    top = heap_top_k(results, k)
    return (top, trace) if explain else top


def heap_top_k(items, k):
    heap = []
    for item in items:
        if len(heap) < k:
            heapq.heappush(heap, (item.score, item.doc_id, item))
        elif item.score > heap[0][0]:
            heapq.heapreplace(heap, (item.score, item.doc_id, item))
    out = [entry[2] for entry in heap]
    out.sort(key=lambda r: (-r.score, r.doc_id))
    return out


def match_count(index, doc_id, terms):
    matched = 0
    for term in set(terms):
        plist = index.postings.get(term, [])
        for p in plist:
            if p.doc_id == doc_id:
                matched += 1
                break
            if p.doc_id > doc_id:
                break
    return matched
