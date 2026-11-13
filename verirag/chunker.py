import re

from .text import analyze

SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def split_sentences(text):
    parts = [p.strip() for p in SENT_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts


# Finance: detect table-like blocks (| or multiple aligned numbers) to avoid splitting mid-row
_TABLE_LINE_RE = re.compile(r"(\|.*\|)|(\$?\s*\d[\d,\.]*\s*(B|M|%|million|billion))", re.IGNORECASE)
_ABBREV_PROTECT = ["Corp.", "Inc.", "Ltd.", "U.S.", "U.K.", "e.g.", "i.e.", "vs."]


def _protect_abbrev(text: str) -> tuple[str, dict[str, str]]:
    """Replace abbreviations with placeholders to avoid false sentence splits."""
    placeholders: dict[str, str] = {}
    for idx, ab in enumerate(_ABBREV_PROTECT):
        ph = f"__ABBR{idx}__"
        if ab in text:
            placeholders[ph] = ab
            text = text.replace(ab, ph)
    return text, placeholders


def _restore_abbrev(text: str, placeholders: dict[str, str]) -> str:
    for ph, ab in placeholders.items():
        text = text.replace(ph, ab)
    return text


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
    if strategy == "finance":
        # Finance-aware: protect abbreviations, keep table blocks together
        protected, ph_map = _protect_abbrev(text)
        lines = protected.split("\n")
        sentences: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # table rows: keep as single sentence-like unit
            if "|" in line or (_TABLE_LINE_RE.search(line) and line.count("$") >= 1):
                # restore abbrev inside
                sentences.append(_restore_abbrev(line, ph_map))
                continue
            # normal sentence split per line
            parts = [p.strip() for p in SENT_SPLIT_RE.split(line) if p.strip()]
            for p in parts:
                sentences.append(_restore_abbrev(p, ph_map))
        if not sentences:
            sentences = split_sentences(_restore_abbrev(protected, ph_map))
        n = params.get("sentences", 3)
        step = params.get("step", 2)
        chunks = []
        i = 0
        idx = 0
        while i < len(sentences):
            j = min(len(sentences), i + n)
            piece = " ".join(sentences[i:j])
            # avoid splitting a table in middle: if next sentence is a table continuation and we cut inside, extend
            if j < len(sentences) and "|" in sentences[j]:
                # keep table row with previous chunk
                if j - i < n:
                    j = min(len(sentences), j + 1)
                    piece = " ".join(sentences[i:j])
            chunks.append(
                Chunk(
                    f"{doc_id}#{idx}",
                    doc_id,
                    title,
                    piece,
                    i,
                    j,
                    "finance",
                    len(analyze(piece, stemming=False, remove_stopwords=False, finance_mode=True)),
                )
            )
            idx += 1
            if j >= len(sentences):
                break
            i += step
        return chunks or [Chunk(f"{doc_id}#0", doc_id, title, text, 0, 0, "finance", len(raw_tokens))]
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
            # pass finance_mode flag for chunking but keep generic api; strategy handles it
            chunks.extend(chunk_document(d["_id"], d["title"], d["text"], strategy, **params))
    return chunks


def build_chunks_from_docs(docs: list[dict], strategy: str = "sentence", **params):
    """Build chunks from in-memory doc dicts (for ingest pipeline)."""
    chunks = []
    for d in docs:
        doc_id = d.get("_id") or d.get("doc_id") or d.get("id")
        title = d.get("title", "")
        text = d.get("text", "")
        if not doc_id or not text:
            continue
        chunks.extend(chunk_document(str(doc_id), title, text, strategy, **params))
    return chunks
