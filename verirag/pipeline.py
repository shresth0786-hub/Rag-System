import json
import time

from .chunker import build_chunks
from .generator import extractive_answer, llm_answer
from .index import InvertedIndex
from .query import BooleanTrace, run_boolean
from .retrieve import RetrievalTrace, search
from .verifier import verdict_counts, verify


def build_index_from_chunks(chunks, stemming=True, zone_weight=2.0, champion_size=0,
                            remove_stopwords=True, keep_negators=True):
    index = InvertedIndex(
        stemming=stemming,
        remove_stopwords=remove_stopwords,
        zone_weight=zone_weight,
        keep_negators=keep_negators,
    )
    for c in chunks:
        index.add_document(
            c.chunk_id,
            c.title,
            c.text,
            meta={
                "doc_id": c.doc_id,
                "sentence_start": c.sentence_start,
                "sentence_end": c.sentence_end,
                "strategy": c.strategy,
                "n_tokens": c.tokens,
            },
        )
    index.finalize(champion_size=champion_size)
    return index


def build_from_corpus(corpus_path, strategy="sentence", chunk_params=None,
                      stemming=True, zone_weight=2.0, champion_size=0, out_path=None):
    chunk_params = chunk_params or {}
    t0 = time.time()
    chunks = build_chunks(corpus_path, strategy, **chunk_params)
    index = build_index_from_chunks(
        chunks,
        stemming=stemming,
        zone_weight=zone_weight,
        champion_size=champion_size,
    )
    stats = {
        "corpus": corpus_path,
        "strategy": strategy,
        "chunk_params": chunk_params,
        "n_docs": index.N,
        "n_chunks": len(chunks),
        "n_terms": len(index.postings),
        "build_seconds": round(time.time() - t0, 2),
        "avg_chunk_tokens": round(sum(c.tokens for c in chunks) / max(len(chunks), 1), 1),
    }
    index.meta_build = stats
    if out_path:
        index.save(out_path)
    return index, stats


class Pipeline:
    def __init__(self, index):
        self.index = index

    @classmethod
    def load(cls, path):
        return cls(InvertedIndex.load(path))

    def ask(
        self,
        query,
        k=5,
        generator="extractive",
        max_sentences=3,
        method="tfidf",
        use_champions=False,
        proximity_weight=0.5,
        proximity_window=30,
        static_weight=0.0,
        eliminate_threshold=1.0,
        model=None,
        high=0.35,
        low=0.18,
        novelty_weight=0.6,
        explain=False,
    ):
        t0 = time.time()
        out = search(
            self.index,
            query,
            k=k,
            method=method,
            use_champions=use_champions,
            eliminate_threshold=eliminate_threshold,
            proximity_weight=proximity_weight,
            proximity_window=proximity_window,
            static_weight=static_weight,
            explain=True,
        )
        retrieved, trace = out
        if generator == "llm":
            sentences = llm_answer(self.index, retrieved, query, model=model)
        else:
            sentences = extractive_answer(
                self.index, retrieved, query, max_sentences=max_sentences
            )
        rows = verify(
            self.index,
            sentences,
            retrieved,
            high=high,
            low=low,
            novelty_weight=novelty_weight,
        )
        result = {
            "query": query,
            "retrieved": [
                {
                    "rank": i,
                    "chunk_id": r.doc_id,
                    "parent_doc": self.index.docs[r.doc_id].meta.get("doc_id"),
                    "score": round(r.score, 4),
                    "base": round(r.base, 4),
                    "proximity": round(r.proximity, 4),
                    "g": round(r.g, 4),
                    "title": self.index.docs[r.doc_id].title,
                    "text": self.index.docs[r.doc_id].text,
                }
                for i, r in enumerate(retrieved, 1)
            ],
            "answer": sentences,
            "verification": rows,
            "verdicts": verdict_counts(rows),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "generator": generator,
        }
        if explain:
            result["trace"] = {
                "query_terms": trace.query_terms,
                "eliminated": trace.eliminated,
                "term_stats": trace.term_stats,
                "candidates": trace.candidates,
            }
        return result

    def boolean(self, query, limit=20):
        trace = BooleanTrace()
        doc_ids = run_boolean(self.index, query, trace)
        hits = []
        for doc_id in doc_ids[:limit]:
            doc = self.index.docs[doc_id]
            hits.append({"chunk_id": doc_id, "parent_doc": doc.meta.get("doc_id"), "title": doc.title})
        return {"query": query, "n_results": len(doc_ids), "hits": hits, "trace": trace.steps}
