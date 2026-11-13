# VeriRAG — an inspectable tf-idf RAG pipeline with claim-level citation verification
*Now finance-ready — same TF-IDF/BM25 core, now filing-aware for any corporate corpus.*

CSD358 IR Hackathon 2026 — Track T1 (RAG + trustworthy answers).

**Repository:** https://github.com/shresth0786-hub/Rag-System
**Demo video:** unlisted — link to be added here after upload (record per `VIDEO.md`)

VeriRAG is a retrieval-augmented generation system built from scratch on top of a
classical IR engine: an inverted index with tf-idf/BM25 scoring, a Boolean/phrase
query processor with skip-list optimization, and a citation verifier that checks
every generated sentence against the chunk it cites. No FAISS, no embeddings, no
vector database — every score in the pipeline is inspectable from the CLI.

Finance mode keeps the engine identical — it adds a finance-aware tokenizer
(`$ % 10-K Q3` preserved), table-aware chunking, ticker/filing/period Boolean filters,
and a heavier numeric novelty penalty, plus a corporate disclaimer and filing metadata
in every citation. Default behaviour (SciFact) is unchanged.

## Pipeline

```
corpus.jsonl (BEIR SciFact 5,183 docs  •  Finance filings 20 docs synthetic)
      │
      ▼
 tokenizer ── Porter stemmer (from scratch) ── chunker (sentence / fixed / finance)
             finance_mode keeps $ % - / and 10-K, Q3, FY24; no-stem for $12.4B  [text.py]
      │
      ▼
 inverted index: tf-idf (lnc/ltc), title zone (λ=2), BM25, static quality g(d),
                 skip pointers, champion lists, finance meta (ticker/filing/period)  [index.py, scoring.py]
      │
      ▼
 ranked retrieval: term-at-a-time scoring, index elimination, proximity re-rank
                   (multiplicative, window=30)                                    [retrieve.py]
 ranked + Boolean: ticker:AAPL, filing:10-K, period:2024-Q3, phrase, NOT, AND/OR  [query.py]
      │
      ▼
 answer generation: extractive (tf-idf + MMR sentence selection)
                    or OpenAI-compatible LLM (--llm) with finance prompt            [generator.py]
      │
      ▼
 citation verification: support = cosine(claim, cited chunk) × (1 − 0.6·novelty)
                        finance: numeric tokens weighted 1.5× in novelty
                        SUPPORTED / WEAK / UNSUPPORTED + citation-mismatch check
                        + corporate disclaimer when finance_mode active             [verifier.py, pipeline.py]
```

## Install & quickstart

```powershell
pip install -r requirements.txt          # optional: only for charts/stemmer checks; pypdf optional for ingest
python -m verirag build                  # chunk corpus + build output/index.pkl (~30 s SciFact; ~0.01 s finance)
```

### SciFact (default)

```powershell
python -m verirag search "antibiotic resistance genes in soil bacteria" --explain
python -m verirag ask "metformin reduces mortality in type 2 diabetes patients" --explain
python -m verirag bool '"antibiotic resistance" AND bacteria'
python -m verirag eval --charts          # retrieval benchmark vs BM25 (writes output/)
```

### Finance — any corporate (synthetic sample included)

```powershell
# 1) Use the bundled finance sample (20 filings, tickers AAPL/MSFT/TSLA/JPM/NVDA/BRK/…)
python -m verirag build --corpus data/finance/corpus.jsonl --out output/index_finance.pkl --strategy finance

# 2) Inspect & ask with finance filters
python -m verirag search "Apple revenue Q3 2024" --index output/index_finance.pkl --explain
python -m verirag bool "ticker:AAPL AND revenue" --index output/index_finance.pkl
python -m verirag bool "ticker:MSFT AND filing:10-K" --index output/index_finance.pkl
python -m verirag ask "What was Apple revenue in Q3 FY2024 and guidance for Q4?" --index output/index_finance.pkl --json
python -m verirag verify "Apple Q3 FY2024 revenue was $81.8 billion" --index output/index_finance.pkl
python -m verirag verify "Apple paid $50B dividend in Q3 2024" --index output/index_finance.pkl   # -> WEAK (numeric mismatch)
python -m verirag eval --corpus data/finance/corpus.jsonl --queries data/finance/queries.jsonl --qrels data/finance/qrels/test.tsv

# 3) Ingest your own filings (PDFs/CSVs/TXTs)
python -m verirag ingest --input data/raw --out data/finance/corpus.jsonl --ticker AAPL --filing-type 10-K --period 2024-Q3
# or directly:
python scripts/ingest_finance.py --input report.pdf --ticker TSLA --filing-type 10-Q --period 2024-Q2 --out data/finance/corpus.jsonl --append
pip install pypdf   # only if ingesting PDFs

# 4) Browser UI (auto-detects finance vs SciFact index)
python -m verirag web --index output/index_finance.pkl
python -m verirag web --index output/index.pkl
```

Auto-detection: `build` switches to `finance` chunking + tokenizer if corpus docs contain `ticker`/`filing_type`/`period`. Force via `--strategy finance` / `--finance-mode` / `--no-finance-mode`.

Everything is inspectable:

```powershell
python -m verirag repl                          # interactive: type your own queries
python -m verirag web                           # browser UI: chat-style RAG assistant; also opens the source in VS Code (--no-open-code to skip)
python -m verirag term resist            # dictionary entry: df, idf, postings, skip pointers
python -m verirag bool '"antibiotic resistance" AND bacteria'   # query-processor trace (PowerShell escaping)
python -m verirag bool 'title:CRISPR AND NOT review'              # zone + NOT, 54 matches
python -m verirag search "antibiotic resistance genes in soil bacteria" --explain
python -m verirag ask "metformin reduces mortality in type 2 diabetes patients" --explain
python -m verirag verify "probiotics prevent antibiotic-associated diarrhea"
python -m verirag eval --charts          # retrieval benchmark vs BM25 (writes output/)
python -m verirag eval-verify --charts   # verifier benchmark vs Jaccard/cosine baselines
```

Optional LLM mode (OpenAI-compatible endpoint):

```powershell
$env:OPENAI_API_KEY="sk-..."; $env:OPENAI_BASE_URL="https://api.openai.com/v1"
python -m verirag ask "your question" --llm
# finance LLM prompt adds: "Never extrapolate numbers… not financial advice" (see generator.py:FINANCE_SYSTEM_PROMPT_EXTRA)
```

## Dataset & protocol

**SciFact** — BEIR SciFact (5,183 abstracts, 300 test claims with gold evidence documents
and SUPPORT/CONTRADICT evidence labels). The 300 judged claims are split
deterministically: **first 100 qids = dev** (all hyperparameter tuning:
proximity weight/window, static-quality η, verifier penalty weight/form and
threshold), **remaining 200 = test** (reporting only). No tuning on test.

**Finance** — `data/finance/` synthetic sample (20 filings, 15 claims, 13 qrels).
* `corpus.jsonl` — fields: `_id, ticker, company, title, text, filing_type, period, source_url`
* `queries.jsonl` — claims + `metadata` evidence labels (SUPPORT); empty metadata = NEI/unsupported
* `qrels/test.tsv` — `query-id  corpus-id  score`
Finance `eval` auto-uses 50/50 dev/test split when n<100 (see `evaluate.py:run_retrieval_eval`). Numbers below are toy-corpus sanity checks, not SciFact-scale benchmarks.

## Results

### SciFact (test split, n=200 / n=214) — unchanged

Retrieval (doc-level, chunk→parent dedup):

| configuration | nDCG@10 | P@10 | R@100 | MRR@10 | ms/query |
|---|---|---|---|---|---|
| A0 plain tf-idf (no stem, no zone) | 0.6010 | 0.0810 | 0.8515 | 0.5558 | 16.6 |
| A1 + Porter stemming | 0.6522 | 0.0865 | 0.8865 | 0.6044 | 35.7 |
| A2 + title zone (λ=2) | **0.6561** | 0.0870 | 0.8815 | 0.6083 | 26.9 |
| A3 + proximity (w=0.5, win=30) | 0.6553 | 0.0860 | 0.8815 | **0.6125** | 64.9 |
| A3 + champion lists (top-50) | 0.6031 | 0.0775 | 0.8049 | 0.5653 | **5.7** |
| BM25 (k1=0.9, b=0.4) | 0.6533 | 0.0875 | 0.9085 | 0.6097 | 32.0 |

Citation verifier (SUPPORT detection, dev-tuned threshold 0.20):

| method | precision | recall | F1 | accuracy |
|---|---|---|---|---|
| baseline: Jaccard overlap | 0.396 | 0.724 | 0.512 | 0.439 |
| baseline: cosine only | 0.414 | 0.966 | 0.579 | 0.430 |
| **VeriRAG: cosine × (1 − 0.6·novelty)** | **0.565** | **0.805** | **0.664** | **0.668** |

Answer stage (evidence-sentence recall, n=87): first sentence of top-1 chunk
= 0.103 → VeriRAG extractive answer (3 sentences) = **0.575**.

### Finance sample (toy, n=7 test, for smoke only — do not compare to SciFact)

```
python -m verirag eval --corpus data/finance/corpus.jsonl --queries data/finance/queries.jsonl --qrels data/finance/qrels/test.tsv
# A2 + title zone: nDCG@10 0.9473  R@100 1.0  MRR 0.9286
```
Verifies filing-aware retrieval works; finance verifier demo:
`verify "Apple Q3 FY2024 revenue was $81.8 billion"` → SUPPORTED (0.45), NEI `Apple paid $50B dividend` → WEAK (1.5× numeric novelty penalty).

## Repository layout

```
verirag/            the package (all standard library; pypdf optional)
  text.py           tokenizer, stopwords, negators, Porter stemmer + finance_mode ($ % 10-K, Q3 preserved, FINANCE_KEEP)
  chunker.py        whole / fixed / sentence / finance chunking ("what is a document?" + table-aware)
  index.py          inverted index, zones, skips, champions, static quality + finance_mode flag + finance meta
  scoring.py        ltc/lnc tf-idf, BM25, query analysis (finance_mode), proximity
  query.py          Boolean/phrase/zone parser + finance meta filters ticker/filing/period + query-processor trace
  retrieve.py       term-at-a-time ranking, index elimination, heap top-k (finance-aware analyze)
  generator.py      extractive + optional LLM answers with citations + finance prompt/disclaimer
  verifier.py       support scoring, numeric-weighted novelty penalty in finance_mode, verdicts
  evaluate.py       benchmark harness, dev sweeps, charts (finance auto split n<100)
  metrics.py        nDCG, P@k, R@k, MRR, MAP
  pipeline.py       build_from_corpus (auto finance detection), Pipeline.ask (adds finance_mode + disclaimer + meta)
  web.py            ThreadingHTTPServer, /api/*, finance meta in hits, finance_mode in stats
  web/              index.html + app.js (finance chips, ticker pills, disclaimer banner, mode badge)
scripts/            stemmer validation, smoke test, chunking ablation, ingest_finance.py
data/scifact/       BEIR SciFact (corpus, queries, qrels)
data/finance/       synthetic finance corpus (corpus.jsonl 20 docs, queries.jsonl 15 claims, qrels/test.tsv)
output/             index + evaluation tables/charts (index.pkl SciFact, index_finance.pkl finance)
```

## Corporate finance usage (for any finance corporate)

1. **Prepare corpus** — convert PDFs/Excels/CSVs to `corpus.jsonl` via `scripts/ingest_finance.py` (keep `ticker, filing_type, period, source_url` per doc).
2. **Build** — `python -m verirag build --corpus data/finance/corpus.jsonl --out output/index_finance.pkl --strategy finance`.
3. **Serve** — `python -m verirag web --index output/index_finance.pkl` — analysts ask natural-language questions, filter by `ticker:`, `filing:`, `period:` in Boolean tab, and every answer sentence shows green/amber/red verification with numeric hallucination penalty.
4. **Audit** — every `ask` JSON contains `retrieved[].meta` + `disclaimer` + `verification[].support`; cite-mismatch flagged. No embeddings: every score traceable to postings.
5. **Extend** — add new filings with `ingest --append` and rebuild; evaluate with your own `qrels`.

Finance-specific inspectability:
* `search --explain` shows `q3`, `10-k`, `$` tokens preserved.
* `bool 'ticker:AAPL AND revenue'` filters on chunk meta, not postings.
* `verify` highlights novel numeric terms with 1.5× penalty.

Disclaimer **always** surfaced: *"Generated from retrieved filings only; not financial advice. Verify with source documents and a qualified professional."* — in `Pipeline.ask` JSON (`disclaimer`), CLI (`_print_answer`), and Web UI evidence fold.

## Work division

Single-team-member entry (team size 1): **SHRESTH** — full ownership of indexing,
query processing, ranking, chunking, generation, verification, evaluation and
reporting. Finance extension (corpus + ingest + finance mode + UI) by same author.

## AI-use declaration

Permitted by the competition rules. AI assistants (opencode/ChatGPT) were used
for scaffolding, debugging, stemmer validation against NLTK, and report editing;
all core algorithmic decisions (zone weighting, proximity formulation, novelty
penalty, dev/test protocol, finance tokenizer/mode) were designed and validated by the author, with
every reported number reproduced by `python -m verirag eval --charts` and
`python -m verirag eval-verify --charts` from a cold start. Finance numbers are toy-corpus sanity checks.

## Known limitations

* Synthetic finance corpus (20 docs) is smoke-only; real corpora need domain tuning of thresholds.
* Finance numeric preservation is heuristic (keeps `10-K`/`Q3`/`$`/`%`, decimal dots may split e.g. `$81.8B` → tokens `8`/`billion`); upgrade `text.py:normalize(finance_mode=True)` for locale-specific decimals if needed.
* Table chunking is line-based, not cell-aware; complex SEC tables may need HTML-aware chunking.
* No auth/multi-tenancy; add reverse proxy for corporate deployment.
* LLM mode optional and carries hallucination risk — finance mode adds prompt guardrail but extractive is recommended for compliance.
