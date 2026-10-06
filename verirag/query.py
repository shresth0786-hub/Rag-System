import math
import re

from .index import TITLE_GAP
from .text import stem as stem_term
from .text import tokenize

TOKEN_RE = re.compile(r'"[^"]*"|\(|\)|\S+')
FIELD_RE = re.compile(r"^(title|body|text):(.*)$", re.IGNORECASE)


class QueryNode:
    def __init__(self, kind, value=None, children=None):
        self.kind = kind
        self.value = value
        self.children = children or []


def lex(query):
    query = query.replace("(", " ( ").replace(")", " ) ")
    tokens = []
    for raw in TOKEN_RE.findall(query):
        if raw in ("(", ")", "AND", "OR", "NOT"):
            tokens.append(raw)
            continue
        m = FIELD_RE.match(raw)
        if m:
            zone = m.group(1).lower()
            inner = m.group(2)
            if inner.startswith('"') and inner.endswith('"') and len(inner) >= 2:
                tokens.append(("ZONE", zone, inner[1:-1]))
            else:
                tokens.append(("ZONE", zone, inner))
            continue
        if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
            tokens.append(("PHRASE", raw[1:-1]))
        else:
            tokens.append(("TERM", raw))
    return tokens


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self):
        tok = self.peek()
        self.pos += 1
        return tok

    def parse(self):
        node = self.parse_or()
        return node

    def parse_or(self):
        nodes = [self.parse_and()]
        while self.peek() == "OR":
            self.next()
            nodes.append(self.parse_and())
        if len(nodes) == 1:
            return nodes[0]
        return QueryNode("or", children=nodes)

    def parse_and(self):
        nodes = [self.parse_not()]
        while True:
            nxt = self.peek()
            if nxt == "AND":
                self.next()
                nodes.append(self.parse_not())
            elif nxt in ("OR", ")", None) or nxt is None:
                break
            elif isinstance(nxt, tuple) or nxt == "(":
                nodes.append(self.parse_not())
            else:
                break
        if len(nodes) == 1:
            return nodes[0]
        return QueryNode("and", children=nodes)

    def parse_not(self):
        if self.peek() == "NOT":
            self.next()
            child = self.parse_not()
            return QueryNode("not", children=[child])
        return self.parse_atom()

    def parse_atom(self):
        tok = self.next()
        if tok == "(":
            node = self.parse_or()
            if self.peek() == ")":
                self.next()
            return node
        if tok is None:
            return QueryNode("or", children=[])
        if isinstance(tok, tuple):
            if tok[0] == "TERM":
                return QueryNode("term", value=tok[1])
            if tok[0] == "PHRASE":
                return QueryNode("phrase", value=tok[1])
            if tok[0] == "ZONE":
                return QueryNode("zone", value=tok[1], children=[self._zone_atom(tok[2])])
        if tok == "(":
            return self.parse_atom()
        return QueryNode("term", value=str(tok))

    def _zone_atom(self, text):
        text = text.strip()
        if text.startswith('"') and text.endswith('"') and len(text) >= 2:
            return QueryNode("phrase", value=text[1:-1])
        return QueryNode("term", value=text)


def parse(query):
    return Parser(lex(query)).parse()


class BooleanTrace:
    def __init__(self):
        self.steps = []

    def add(self, label, size, detail=""):
        self.steps.append({"label": label, "size": size, "detail": detail})


def _analyze_terms(text, index):
    toks = tokenize(
        text,
        remove_stopwords=index.remove_stopwords,
        keep_negators=index.keep_negators,
    )
    if index.stemming:
        toks = [stem_term(t) for t in toks]
    return toks


def intersect(a, b, trace=None):
    i = j = 0
    comps = 0
    out = []
    while i < len(a) and j < len(b):
        comps += 1
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    if trace is not None:
        trace.add("intersect", len(out), f"{comps} comparisons over |{len(a)}| x |{len(b)}|")
    return out


def intersect_postings_skip(plist_a, plist_b, trace=None):
    i = j = 0
    comps = 0
    skip_hits = 0
    out = []
    n_a, n_b = len(plist_a), len(plist_b)
    step_a = max(1, int(math.sqrt(n_a)))
    step_b = max(1, int(math.sqrt(n_b)))
    while i < n_a and j < n_b:
        comps += 1
        da, db = plist_a[i].doc_id, plist_b[j].doc_id
        if da == db:
            out.append(da)
            i += 1
            j += 1
        elif da < db:
            if i % step_a == 0 and i + step_a < n_a and plist_a[i + step_a].doc_id < db:
                i += step_a
                skip_hits += 1
            else:
                i += 1
        else:
            if j % step_b == 0 and j + step_b < n_b and plist_b[j + step_b].doc_id < da:
                j += step_b
                skip_hits += 1
            else:
                j += 1
    if trace is not None:
        trace.add(
            "intersect (skip pointers)",
            len(out),
            f"{comps} comparisons, {skip_hits} skip jumps",
        )
    return out


def zone_docids(index, term, zone):
    plist = index.postings.get(term, [])
    if zone == "title":
        return [p.doc_id for p in plist if p.tf_title > 0]
    if zone == "body":
        return [p.doc_id for p in plist if p.tf_title < p.tf]
    return [p.doc_id for p in plist]


def phrase_docids(index, terms, zone=None, use_skips=True):
    if not terms:
        return []
    if zone == "title":
        lists = [
            [p for p in index.postings.get(t, []) if p.tf_title > 0] for t in terms
        ]
    elif zone == "body":
        lists = [
            [p for p in index.postings.get(t, []) if p.tf_title < p.tf] for t in terms
        ]
    else:
        lists = [index.postings.get(t, []) for t in terms]
    lists = [l for l in lists if l]
    if not lists:
        return []
    if len(lists) == 1:
        return [p.doc_id for p in lists[0]]
    lists.sort(key=len)
    if use_skips and len(lists[0]) >= 32 and len(lists[1]) >= 32:
        candidates = set(intersect_postings_skip(lists[0], lists[1]))
    else:
        candidates = set(p.doc_id for p in lists[0]) & set(
            p.doc_id for p in lists[1]
        )
    for other in lists[2:]:
        if not candidates:
            break
        candidates &= set(p.doc_id for p in other)
    out = []
    width = len(terms)
    pos_maps = {}
    for t in terms:
        pos_maps[t] = {p.doc_id: p.positions for p in index.postings.get(t, [])}
    for doc_id in sorted(candidates):
        spans = pos_maps[terms[0]].get(doc_id, [])
        ok = False
        for start in spans:
            if start >= TITLE_GAP and zone == "title":
                continue
            if start < TITLE_GAP and zone == "body":
                continue
            good = True
            for offset, t in enumerate(terms[1:], start=1):
                if start + offset not in pos_maps[t].get(doc_id, ()):
                    good = False
                    break
            if good:
                ok = True
                break
        if ok:
            out.append(doc_id)
    return out


def evaluate(node, index, trace=None, universe=None):
    if universe is None:
        universe = sorted(index.docs)
    if node.kind == "term":
        terms = _analyze_terms(node.value, index)
        if not terms:
            return list(universe)
        result = zone_docids(index, terms[0], None)
        if trace is not None:
            trace.add(f"postings[{terms[0]}]", len(result), f"df={index.df.get(terms[0], 0)}")
        return result
    if node.kind == "phrase":
        terms = _analyze_terms(node.value, index)
        result = phrase_docids(index, terms, zone=None)
        if trace is not None:
            trace.add(f'phrase "{node.value}"', len(result), f"terms={terms}")
        return result
    if node.kind == "zone":
        zone = node.value
        child = node.children[0]
        if child.kind == "term":
            terms = _analyze_terms(child.value, index)
            if not terms:
                return list(universe)
            result = zone_docids(index, terms[0], zone)
            if trace is not None:
                trace.add(f"postings[{terms[0]}] zone={zone}", len(result), "")
            return result
        if child.kind == "phrase":
            terms = _analyze_terms(child.value, index)
            result = phrase_docids(index, terms, zone=zone)
            if trace is not None:
                trace.add(f'phrase "{child.value}" zone={zone}', len(result), "")
            return result
        return evaluate(child, index, trace, universe)
    if node.kind == "not":
        child = evaluate(node.children[0], index, trace, universe)
        child_set = set(child)
        result = [d for d in universe if d not in child_set]
        if trace is not None:
            trace.add("NOT", len(result), "")
        return result
    if node.kind == "and":
        children = node.children
        sizes = []
        for child in children:
            if child.kind in ("term", "phrase"):
                terms = (
                    _analyze_terms(child.value, index)
                    if child.kind == "term"
                    else _analyze_terms(child.value, index)
                )
                df = sum(index.df.get(t, 0) for t in terms)
                sizes.append(df)
            else:
                sizes.append(len(universe))
        order = sorted(range(len(children)), key=lambda i: sizes[i])
        if trace is not None and len(order) > 1:
            labels = []
            for i in order:
                c = children[i]
                labels.append(str(c.value) if c.value else c.kind)
            trace.add(
                "query optimisation",
                len(order),
                "processing in increasing df order: " + " -> ".join(labels),
            )
        result = evaluate(children[order[0]], index, trace, universe)
        for idx in order[1:]:
            other = evaluate(children[idx], index, trace, universe)
            if not result:
                break
            result = intersect(result, other, trace)
        return result
    if node.kind == "or":
        merged = set()
        for child in node.children:
            merged.update(evaluate(child, index, trace, universe))
        result = sorted(merged)
        if trace is not None:
            trace.add("OR", len(result), "")
        return result
    return list(universe)


def run_boolean(index, query, trace=None):
    node = parse(query)
    universe = sorted(index.docs)
    result = evaluate(node, index, trace, universe)
    return result
