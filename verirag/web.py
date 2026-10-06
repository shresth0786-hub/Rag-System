"""Zero-dependency browser frontend for VeriRAG.

Serves a small JSON API (python standard library only) and a static
single-page UI. Run with:  python -m verirag web
"""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .pipeline import Pipeline
from .query import BooleanTrace, run_boolean
from .retrieve import search
from .scoring import analyze_query
from .verifier import doc_cosine, novelty_penalty, support_score

WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

SUPPORTED = "SUPPORTED"
WEAK = "WEAK"
UNSUPPORTED = "UNSUPPORTED"


class VeriRAGApp:
    """Loads the index once and answers JSON requests from worker threads."""

    def __init__(
        self,
        index_path,
        method="tfidf",
        champions=False,
        high=0.20,
        low=0.12,
        novelty_weight=0.6,
    ):
        self.pipe = Pipeline.load(index_path)
        self.index = self.pipe.index
        self.method = method
        self.champions = champions
        self.high = high
        self.low = low
        self.novelty_weight = novelty_weight

    # -- endpoints ------------------------------------------------------

    def stats(self, payload):
        meta = getattr(self.index, "meta_build", None)
        idx = self.index
        return {
            "n_docs": meta.get("n_docs") if meta else "?",
            "n_chunks": idx.N,
            "n_terms": len(idx.postings),
            "avg_chunk_tokens": meta.get("avg_chunk_tokens") if meta else "?",
            "strategy": meta.get("strategy") if meta else "?",
            "stemming": idx.stemming,
            "index_path": payload.get("index", "output/index.pkl"),
        }

    def ask(self, payload):
        result = self.pipe.ask(
            payload["query"],
            k=int(payload.get("k", 5)),
            generator="extractive",
            max_sentences=int(payload.get("sentences", 3)),
            method=payload.get("method", self.method),
            use_champions=self.champions or bool(payload.get("champions")),
            proximity_weight=float(payload.get("proximity", 0.5)),
            proximity_window=int(payload.get("prox_window", 30)),
            static_weight=float(payload.get("static", 0.0)),
            eliminate_threshold=float(payload.get("eliminate", 1.0)),
            high=self.high,
            low=self.low,
            novelty_weight=self.novelty_weight,
            explain=bool(payload.get("explain")),
        )
        return result

    def search(self, payload):
        ranked, trace = search(
            self.index,
            payload["query"],
            k=int(payload.get("k", 10)),
            method=payload.get("method", self.method),
            use_champions=self.champions or bool(payload.get("champions")),
            eliminate_threshold=float(payload.get("eliminate", 1.0)),
            proximity_weight=float(payload.get("proximity", 0.5)),
            proximity_window=int(payload.get("prox_window", 30)),
            static_weight=float(payload.get("static", 0.0)),
            explain=True,
        )
        rows = []
        for i, r in enumerate(ranked, 1):
            doc = self.index.docs[r.doc_id]
            rows.append(
                {
                    "rank": i,
                    "chunk_id": r.doc_id,
                    "parent_doc": doc.meta.get("doc_id"),
                    "score": round(r.score, 4),
                    "base": round(r.base, 4),
                    "proximity": round(r.proximity, 4),
                    "g": round(r.g, 4),
                    "title": doc.title,
                    "text": doc.text,
                }
            )
        return {
            "query": payload["query"],
            "hits": rows,
            "trace": {
                "query_terms": trace.query_terms,
                "eliminated": trace.eliminated,
                "candidates": trace.candidates,
                "term_stats": trace.term_stats,
            },
        }

    def term(self, payload):
        info = self.index.describe_term(payload["term"], limit=8)
        sample = []
        for p in info["postings_sample"]:
            sample.append(
                {
                    "chunk_id": p.doc_id,
                    "tf": p.tf,
                    "tf_title": p.tf_title,
                    "positions": p.positions[:8],
                }
            )
        return {
            "term": info["term"],
            "df": info["df"],
            "idf": round(info["idf"], 4),
            "postings_len": info["postings_len"],
            "postings_sample": sample,
            "skips": [[i, d] for i, d in info["skips"]],
            "champions": info["champions"],
        }

    def boolean(self, payload):
        trace = BooleanTrace()
        doc_ids = run_boolean(self.index, payload["query"], trace)
        hits = []
        for doc_id in doc_ids[: int(payload.get("limit", 20))]:
            doc = self.index.docs[doc_id]
            hits.append(
                {
                    "chunk_id": doc_id,
                    "parent_doc": doc.meta.get("doc_id"),
                    "title": doc.title,
                    "text": doc.text,
                }
            )
        return {
            "query": payload["query"],
            "n_results": len(doc_ids),
            "hits": hits,
            "trace": trace.steps,
        }

    def verify(self, payload):
        raw = payload["claim"]
        terms = analyze_query(
            raw,
            stemming=self.index.stemming,
            remove_stopwords=self.index.remove_stopwords,
            keep_negators=self.index.keep_negators,
        )
        ranked = search(
            self.index, raw, k=int(payload.get("k", 5)), explain=False
        )
        rows = []
        best = 0.0
        for i, r in enumerate(ranked, 1):
            cos = doc_cosine(self.index, terms, r.doc_id)
            novel, pen = novelty_penalty(self.index, terms, r.doc_id)
            support = support_score(cos, pen, self.novelty_weight)
            best = max(best, support)
            doc = self.index.docs[r.doc_id]
            rows.append(
                {
                    "rank": i,
                    "chunk_id": r.doc_id,
                    "parent_doc": doc.meta.get("doc_id"),
                    "title": doc.title,
                    "text": doc.text[:400],
                    "cosine": round(cos, 4),
                    "novelty_penalty": round(pen, 4),
                    "support": round(support, 4),
                    "novel_terms": novel[:10],
                }
            )
        if best >= self.high:
            verdict = SUPPORTED
        elif best >= self.low:
            verdict = WEAK
        else:
            verdict = UNSUPPORTED
        return {
            "claim": raw,
            "terms": terms,
            "best_support": round(best, 4),
            "verdict": verdict,
            "high": self.high,
            "low": self.low,
            "rows": rows,
        }


# -- HTTP layer ---------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    app = None
    server_version = "VeriRAG/1.0"

    # static ------------------------------------------------------------

    def _send_bytes(self, data, ctype, status=200):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _serve_static(self, name, ctype):
        path = os.path.join(WEB_DIR, name)
        try:
            with open(path, "rb") as f:
                self._send_bytes(f.read(), ctype)
        except OSError:
            self._send_json({"error": f"missing static file: {name}"}, 404)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._serve_static("index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self._serve_static("style.css", "text/css; charset=utf-8")
        elif path == "/app.js":
            self._serve_static("app.js", "application/javascript; charset=utf-8")
        else:
            self._send_json({"error": "not found", "path": path}, 404)

    # API ---------------------------------------------------------------

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_POST(self):
        path = urlparse(self.path).path
        route = {
            "/api/stats": self.app.stats,
            "/api/ask": self.app.ask,
            "/api/search": self.app.search,
            "/api/term": self.app.term,
            "/api/bool": self.app.boolean,
            "/api/verify": self.app.verify,
        }.get(path)
        if route is None:
            self._send_json({"error": f"unknown endpoint: {path}"}, 404)
            return
        try:
            payload = self._read_json()
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"bad json body: {exc}"}, 400)
            return
        try:
            result = route(payload)
        except Exception as exc:  # noqa: BLE001 - surface to the UI
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, 500)
            return
        self._send_json(result)

    def log_message(self, fmt, *args):
        pass


def serve(index_path, host="127.0.0.1", port=8123, open_browser=True, **opts):
    """Start the browser UI. Blocks until Ctrl+C."""
    import webbrowser

    app = VeriRAGApp(index_path, **opts)
    Handler.app = app

    httpd = None
    last_err = None
    for candidate in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            last_err = exc
    if httpd is None:
        raise SystemExit(f"could not bind any port from {port} to {port + 19}: {last_err}")

    url = f"http://{host}:{port}/"
    print(f"VeriRAG UI: {url}  (index: {index_path})")
    print("Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()