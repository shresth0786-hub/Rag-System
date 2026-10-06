import re

from .text import analyze

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text):
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts


class Chunk:
    def __init__(self, chunk_id, doc_id, title, text, sentence_start, sentence_end, strategy, tokens):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end
        self.strategy = strategy
        self.tokens = tokens

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "sentence_start": self.sentence_start,
            "sentence_end": self.sentence_end,
            "strategy": self.strategy,
            "tokens": self.tokens,
        }


def chunk_document(doc_id, title, text, strategy="sentence", **params):
    raw_tokens = analyze(text, stemming=False, remove_stopwords=False)
    if strategy == "whole":
        return [
            Chunk(
                f"{doc_id}#0", doc_id, title, text, 0, 0, "whole", len(raw_tokens)
            )
        ]
    if strategy == "fixed":
        size = params.get("size", 120)
        overlap = params.get("overlap", 30)
        sentences = split_sentences(text)
        chunks = []
        current = []
        cur_tokens = 0
        start_idx = 0
        i = 0
        idx = 0

        def count(tok_text):
            return len(analyze(tok_text, stemming=False, remove_stopwords=False))

        while i < len(sentences):
            s = sentences[i]
            current.append(s)
            cur_tokens += count(s)
            i += 1
            if cur_tokens >= size:
                piece = " ".join(current)
                chunks.append(
                    Chunk(f"{doc_id}#{idx}", doc_id, title, piece, start_idx, i, "fixed", cur_tokens)
                )
                idx += 1
                keep = []
                ov = 0
                k = len(current) - 1
                while k >= 0 and ov < overlap:
                    ov += count(current[k])
                    keep.insert(0, current[k])
                    k -= 1
                if len(keep) >= len(current):
                    keep = []
                    ov = 0
                start_idx = i - len(keep)
                current = keep
                cur_tokens = ov
        if current:
            piece = " ".join(current)
            chunks.append(
                Chunk(f"{doc_id}#{idx}", doc_id, title, piece, start_idx, i, "fixed", cur_tokens)
            )
        return chunks or [
            Chunk(f"{doc_id}#0", doc_id, title, text, 0, 0, "fixed", len(raw_tokens))
        ]
    sentences = split_sentences(text)
    n = params.get("sentences", 3)
    step = params.get("step", 2)
    if not sentences:
        return [Chunk(f"{doc_id}#0", doc_id, title, text, 0, 0, "sentence", len(raw_tokens))]
    chunks = []
    i = 0
    idx = 0
    while i < len(sentences):
        j = min(len(sentences), i + n)
        piece = " ".join(sentences[i:j])
        chunks.append(
            Chunk(
                f"{doc_id}#{idx}",
                doc_id,
                title,
                piece,
                i,
                j,
                "sentence",
                len(analyze(piece, stemming=False, remove_stopwords=False)),
            )
        )
        idx += 1
        if j >= len(sentences):
            break
        i += step
    return chunks


def build_chunks(corpus_path, strategy="sentence", **params):
    import json

    chunks = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.extend(chunk_document(d["_id"], d["title"], d["text"], strategy, **params))
    return chunks
