import bisect

from .scoring import doc_weight, query_vector
from .text import analyze


def _terms(text, index):
    return analyze(
        text,
        stemming=index.stemming,
        remove_stopwords=index.remove_stopwords,
        keep_negators=index.keep_negators,
    )


def find_posting(plist, doc_id):
    i = bisect.bisect_left(plist, doc_id, key=lambda p: p.doc_id)
    if i < len(plist) and plist[i].doc_id == doc_id:
        return plist[i]
    return None


def doc_cosine(index, terms, doc_id):
    qvec = query_vector(index, terms)
    dot = 0.0
    for t, qw in qvec.items():
        p = find_posting(index.postings.get(t, ()), doc_id)
        if p is not None:
            dot += qw * doc_weight(index, p)
    return min(1.0, max(0.0, dot))


def novelty_penalty(index, terms, doc_id):
    idf_total = 0.0
    novel_idf = 0.0
    novel = []
    for t in set(terms):
        idf = index.idf(t)
        if idf <= 0:
            continue
        idf_total += idf
        p = find_posting(index.postings.get(t, ()), doc_id)
        if p is None:
            novel.append(t)
            novel_idf += idf
    if idf_total <= 0:
        return [], 0.0
    return sorted(novel), novel_idf / idf_total


def support_score(cos, pen, novelty_weight, form="mul"):
    if form == "mul":
        return min(1.0, max(0.0, cos * (1.0 - min(1.0, novelty_weight * pen))))
    return min(1.0, max(0.0, cos - novelty_weight * pen))


def verify(
    index,
    sentences,
    retrieved,
    high=0.20,
    low=0.12,
    novelty_weight=0.6,
):
    rows = []
    for s in sentences:
        terms = _terms(s["text"], index)
        ranks = s.get("citations") or [s.get("rank", 1)]
        primary_rank = ranks[0]
        primary_doc = retrieved[primary_rank - 1].doc_id
        cos = doc_cosine(index, terms, primary_doc)
        novel, pen = novelty_penalty(index, terms, primary_doc)
        support = support_score(cos, pen, novelty_weight)

        per_chunk = []
        for rank, r in enumerate(retrieved, 1):
            c = doc_cosine(index, terms, r.doc_id)
            n, p = novelty_penalty(index, terms, r.doc_id)
            per_chunk.append((rank, support_score(c, p, novelty_weight)))
        best_rank, best_support = max(per_chunk, key=lambda x: x[1])
        mismatch = best_rank not in ranks and best_support > support + 0.10

        if support >= high:
            verdict = "SUPPORTED"
        elif support >= low:
            verdict = "WEAK"
        else:
            verdict = "UNSUPPORTED"
        if mismatch and verdict == "SUPPORTED":
            verdict = "SUPPORTED (cite mismatch)"

        rows.append(
            {
                "claim": s["text"],
                "citations": ranks,
                "cited_chunk": primary_doc,
                "cosine": round(cos, 4),
                "novel_terms": novel,
                "novelty_penalty": round(pen, 4),
                "support": round(support, 4),
                "verdict": verdict,
                "best_rank": best_rank,
                "best_support": round(best_support, 4),
                "cite_mismatch": mismatch,
            }
        )
    return rows


def verdict_counts(rows):
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    return counts
