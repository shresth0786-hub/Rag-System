import os
import re

from .chunker import split_sentences
from .scoring import cosine, query_vector
from .text import analyze


def _terms(text, index):
    return analyze(
        text,
        stemming=index.stemming,
        remove_stopwords=index.remove_stopwords,
        keep_negators=index.keep_negators,
    )


def sentence_candidates(index, retrieved, query, min_rel=0.02):
    q_terms = _terms(query, index)
    qvec = query_vector(index, q_terms)
    candidates = []
    for rank, r in enumerate(retrieved, 1):
        doc = index.docs[r.doc_id]
        for sent in split_sentences(doc.text):
            terms = _terms(sent, index)
            if len(terms) < 4:
                continue
            vec = query_vector(index, terms)
            rel = cosine(qvec, vec)
            if rel < min_rel:
                continue
            candidates.append(
                {
                    "text": sent,
                    "citations": [rank],
                    "rank": rank,
                    "chunk_id": r.doc_id,
                    "parent_doc": doc.meta.get("doc_id", r.doc_id),
                    "title": doc.title,
                    "relevance": round(rel, 4),
                    "generator": "extractive",
                    "_vec": vec,
                }
            )
    return candidates


def extractive_answer(index, retrieved, query, max_sentences=3, mmr_lambda=0.7):
    candidates = sentence_candidates(index, retrieved, query)
    selected = []
    while candidates and len(selected) < max_sentences:
        best_i = max(
            range(len(candidates)),
            key=lambda i: mmr_lambda * candidates[i]["relevance"]
            - (1.0 - mmr_lambda)
            * max(
                (cosine(candidates[i]["_vec"], s["_vec"]) for s in selected),
                default=0.0,
            ),
        )
        selected.append(candidates.pop(best_i))
    for s in selected:
        s.pop("_vec", None)
    return selected


def build_prompt(query, retrieved, index):
    sources = []
    for rank, r in enumerate(retrieved, 1):
        doc = index.docs[r.doc_id]
        sources.append(f"[{rank}] {doc.title}\n{doc.text}")
    return (
        "You are a retrieval-augmented assistant. Answer the query using ONLY the "
        "numbered sources below. After every sentence, add citations like [1] or "
        "[2][3] pointing to the sources that support that sentence. If the sources "
        "do not contain the answer, say that the sources do not support an answer."
        "\n\nSources:\n" + "\n\n".join(sources) + f"\n\nQuery: {query}\n\nAnswer:"
    )


def llm_answer(index, retrieved, query, model=None, temperature=0.0, max_tokens=400):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("VERIRAG_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("VERIRAG_BASE_URL")
    model = model or os.environ.get("VERIRAG_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError(
            "No API key found. Set OPENAI_API_KEY (or VERIRAG_API_KEY), optionally "
            "OPENAI_BASE_URL for any OpenAI-compatible endpoint."
        )
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    prompt = build_prompt(query, retrieved, index)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return parse_cited_text(text, retrieved, index)


def parse_cited_text(text, retrieved, index):
    sentences = split_sentences(text.replace("\n", " "))
    out = []
    for sent in sentences:
        ranks = [int(m) for m in re.findall(r"\[(\d+)\]", sent)]
        clean = re.sub(r"\s*\[\d+\]", "", sent).strip()
        if not clean:
            continue
        ranks = [r for r in ranks if 1 <= r <= len(retrieved)] or [1]
        rank = ranks[0]
        r = retrieved[rank - 1]
        doc = index.docs[r.doc_id]
        out.append(
            {
                "text": clean,
                "citations": ranks,
                "rank": rank,
                "chunk_id": r.doc_id,
                "parent_doc": doc.meta.get("doc_id", r.doc_id),
                "title": doc.title,
                "relevance": None,
                "generator": "llm",
            }
        )
    return out
