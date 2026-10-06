import math
from collections import Counter

from .text import analyze

PROXIMITY_WINDOW = 30


def analyze_query(raw_query, stemming=True, remove_stopwords=True, keep_negators=True):
    return analyze(
        raw_query, stemming=stemming, remove_stopwords=remove_stopwords, keep_negators=keep_negators
    )


def query_vector(index, terms):
    counts = Counter(terms)
    vec = {}
    for term, freq in counts.items():
        idf = index.idf(term)
        if idf <= 0:
            continue
        vec[term] = (1.0 + math.log(freq)) * idf
    norm = math.sqrt(sum(w * w for w in vec.values()))
    if norm > 0:
        for term in vec:
            vec[term] /= norm
    return vec


def doc_weight(index, posting):
    doc = index.docs[posting.doc_id]
    tf = index._effective_tf(posting)
    return (1.0 + math.log(tf)) / doc.norm


def bm25_score(index, posting, term, k1=1.5, b=0.75):
    doc = index.docs[posting.doc_id]
    tf = index._effective_tf(posting)
    len_norm = 1.0 - b + b * (doc.length / max(index.avg_length, 1e-9))
    return index.bm25_idf(term) * tf * (k1 + 1.0) / (tf + k1 * len_norm)


def prefetch_positions(index, terms):
    pos_index = {}
    for term in set(terms):
        plist = index.postings.get(term)
        if plist:
            pos_index[term] = {p.doc_id: p.positions for p in plist}
    return pos_index


def proximity_bonus(pos_index, doc_id, query_terms, window=PROXIMITY_WINDOW):
    unique = [t for t in set(query_terms) if t in pos_index]
    if len(unique) < 2:
        return 0.0, 0, 0
    doc_positions = {}
    for term in unique:
        positions = pos_index[term].get(doc_id)
        if positions:
            doc_positions[term] = positions
    terms = list(unique)
    total_pairs = len(terms) * (len(terms) - 1) // 2
    close_pairs = 0
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a = doc_positions.get(terms[i])
            b = doc_positions.get(terms[j])
            if a and b and any(abs(x - y) <= window for x in a for y in b):
                close_pairs += 1
    if total_pairs == 0:
        return 0.0, 0, 0
    return close_pairs / total_pairs, close_pairs, total_pairs


def build_answer_vector(terms):
    counts = Counter(terms)
    vec = {t: 1.0 + math.log(f) for t, f in counts.items()}
    norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
    return {t: w / norm for t, w in vec.items()}, counts


def cosine(a, b):
    if not a or not b:
        return 0.0
    small, other = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(w * other[t] for t, w in small.items() if t in other)
    return max(0.0, min(1.0, dot))
