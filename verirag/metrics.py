import math


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(rels, k):
    gains = [1.0 if r else 0.0 for r in rels[:k]]
    ideal = sorted(gains, reverse=True)
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def precision_at_k(rels, k):
    if k <= 0:
        return 0.0
    head = rels[:k]
    return sum(1 for r in head if r) / k


def recall_at_k(rels, k, total_relevant):
    if total_relevant == 0:
        return 0.0
    return sum(1 for r in rels[:k] if r) / total_relevant


def mrr(rels):
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def average_precision(rels, total_relevant):
    if total_relevant == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(rels, 1):
        if r:
            hits += 1
            score += hits / i
    return score / total_relevant


def aggregate(per_query_metrics):
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {k: sum(m[k] for m in per_query_metrics) / len(per_query_metrics) for k in keys}
