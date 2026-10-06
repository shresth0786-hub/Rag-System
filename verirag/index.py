import math
import pickle

from .text import analyze

TITLE_GAP = 1000


class Posting:
    __slots__ = ("doc_id", "tf", "tf_title", "positions")

    def __init__(self, doc_id, tf, tf_title, positions):
        self.doc_id = doc_id
        self.tf = tf
        self.tf_title = tf_title
        self.positions = positions

    def __repr__(self):
        return f"Posting({self.doc_id}, tf={self.tf}, tf_title={self.tf_title}, pos={self.positions[:6]})"


class DocEntry:
    __slots__ = ("doc_id", "title", "text", "length", "title_length", "meta", "norm", "g")

    def __init__(self, doc_id, title, text, length, title_length, meta):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.length = length
        self.title_length = title_length
        self.meta = meta
        self.norm = 1.0
        self.g = 0.0


class InvertedIndex:
    def __init__(self, stemming=True, remove_stopwords=True, zone_weight=2.0, keep_negators=True):
        self.stemming = stemming
        self.remove_stopwords = remove_stopwords
        self.keep_negators = keep_negators
        self.zone_weight = zone_weight
        self.postings = {}
        self.docs = {}
        self.df = {}
        self.champions = {}
        self.N = 0
        self.avg_length = 0.0
        self.idf_max = 1.0
        self._built = False

    def _analyze(self, text):
        return analyze(
            text,
            stemming=self.stemming,
            remove_stopwords=self.remove_stopwords,
            keep_negators=self.keep_negators,
        )

    def add_document(self, doc_id, title, text, meta=None):
        title_terms = self._analyze(title)
        body_terms = self._analyze(text)
        term_seq = title_terms + body_terms
        pos_seq = list(range(len(title_terms))) + [
            TITLE_GAP + i for i in range(len(body_terms))
        ]
        if not term_seq:
            return
        per_term = {}
        for pos, term in zip(pos_seq, term_seq):
            per_term.setdefault(term, []).append(pos)
        n_title = len(title_terms)
        for term, poss in per_term.items():
            tf_title = sum(1 for p in poss if p < TITLE_GAP)
            self.postings.setdefault(term, []).append(
                Posting(doc_id, len(poss), tf_title, poss)
            )
        self.docs[doc_id] = DocEntry(
            doc_id, title, text, len(term_seq), n_title, meta or {}
        )
        self.N += 1

    def finalize(self, champion_size=0):
        for term, plist in self.postings.items():
            plist.sort(key=lambda p: p.doc_id)
            self.df[term] = len(plist)
        self.idf_max = math.log(max(self.N, 2))
        total_len = 0
        for doc in self.docs.values():
            total_len += doc.length
        self.avg_length = total_len / max(self.N, 1)
        self._compute_doc_vectors()
        self._compute_static_quality()
        if champion_size > 0:
            self._build_champions(champion_size)
        self._built = True
        return self

    def _effective_tf(self, posting):
        return posting.tf + (self.zone_weight - 1.0) * posting.tf_title

    def idf(self, term):
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(self.N / df)

    def bm25_idf(self, term):
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def _compute_doc_vectors(self):
        norms = {d: 0.0 for d in self.docs}
        for term, plist in self.postings.items():
            for p in plist:
                w = 1.0 + math.log(self._effective_tf(p))
                norms[p.doc_id] += w * w
        for doc_id, n in norms.items():
            doc = self.docs[doc_id]
            doc.norm = math.sqrt(n) if n > 0 else 1.0

    def _compute_static_quality(self):
        for doc in self.docs.values():
            body_terms = set(self._analyze(doc.text))
            title_terms = set(self._analyze(doc.title))
            body_rich = sum(self.idf(t) for t in body_terms) / (len(body_terms) * self.idf_max) if body_terms else 0.0
            title_rich = sum(self.idf(t) for t in title_terms) / (len(title_terms) * self.idf_max) if title_terms else 0.0
            doc.g = 0.5 * body_rich + 0.5 * title_rich

    def _build_champions(self, size):
        for term, plist in self.postings.items():
            ranked = sorted(
                plist, key=lambda p: -(1.0 + math.log(self._effective_tf(p))) * self.idf(term)
            )
            self.champions[term] = [p.doc_id for p in ranked[:size]]

    def skip_plan(self, term):
        plist = self.postings.get(term, [])
        n = len(plist)
        if n < 32:
            return []
        step = max(1, int(math.sqrt(n)))
        return [(i, plist[i].doc_id) for i in range(step, n, step)]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def describe_term(self, term, limit=8):
        plist = self.postings.get(term, [])
        out = {
            "term": term,
            "df": self.df.get(term, 0),
            "idf": self.idf(term),
            "postings_len": len(plist),
            "postings_sample": plist[:limit],
            "skips": self.skip_plan(term)[:limit],
            "champions": self.champions.get(term, [])[:limit],
        }
        return out
