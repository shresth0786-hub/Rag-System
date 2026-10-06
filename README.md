# VeriRAG — an inspectable tf-idf RAG pipeline with claim-level citation verification

CSD358 IR Hackathon 2026 — Track T1 (RAG + trustworthy answers).

VeriRAG is a retrieval-augmented generation system built from scratch on top of a
classical IR engine: an inverted index with tf-idf/BM25 scoring, a Boolean/phrase
query processor with skip-list optimization, and a citation verifier that checks
every generated sentence against the chunk it cites. No FAISS, no embeddings, no
vector database — every score in the pipeline is inspectable from the CLI.

## Pipeline

```
corpus.jsonl (BEIR SciFact, 5,183 docs)
      │
      ▼
 tokenizer ── Porter stemmer (from scratch) ── sentence chunker (3 sentences / step 2)
      │
      ▼
 inverted index: tf-idf (lnc/ltc), title zone (λ=2), BM25, static quality g(d),
                 skip pointers, champion lists            [verirag/index.py, scoring.py]
      │
      ▼
 ranked retrieval: term-at-a-time scoring, index elimination, proximity re-rank
                   (multiplicative, window=30)            [verirag/retrieve.py]
      │
      ▼
 answer generation: extractive (tf-idf + MMR sentence selection)
                    or OpenAI-compatible LLM (--llm)      [verirag/generator.py]
      │
      ▼
 citation verification: support = cosine(claim, cited chunk) × (1 − 0.6·novelty)
                        SUPPORTED / WEAK / UNSUPPORTED + citation-mismatch check
                                                       [verirag/verifier.py]
```

## Install & quickstart

```powershell
pip install -r requirements.txt          # optional: only for charts/stemmer checks
python -m verirag build                  # chunk corpus + build output/index.pkl (~30 s)
```

Everything is inspectable:

```powershell
python -m verirag term resist            # dictionary entry: df, idf, postings, skip pointers
python -m verirag bool '\"antibiotic resistance\" AND bacteria'   # query-processor trace (PowerShell escaping)
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
```

## Dataset & protocol

BEIR **SciFact** (5,183 abstracts, 300 test claims with gold evidence documents
and SUPPORT/CONTRADICT evidence labels). The 300 judged claims are split
deterministically: **first 100 qids = dev** (all hyperparameter tuning:
proximity weight/window, static-quality η, verifier penalty weight/form and
threshold), **remaining 200 = test** (reporting only). No tuning on test.

## Results (test split, n=200 / n=214)

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

## Repository layout

```
verirag/            the package (all standard library)
  text.py           tokenizer, stopwords, negators, from-scratch Porter stemmer
  chunker.py        whole / fixed / sentence chunking ("what is a document?")
  index.py          inverted index, zones, skips, champions, static quality
  scoring.py        ltc/lnc tf-idf, BM25, query analysis, proximity
  query.py          Boolean/phrase/zone parser + query-processor trace
  retrieve.py       term-at-a-time ranking, index elimination, heap top-k
  generator.py      extractive + optional LLM answers with citations
  verifier.py       support scoring, novelty penalty, verdicts
  evaluate.py       benchmark harness, dev sweeps, charts
  metrics.py        nDCG, P@k, R@k, MRR, MAP
scripts/            stemmer validation, smoke test, chunking ablation
data/scifact/       BEIR SciFact (corpus, queries, qrels)
output/             index + evaluation tables/charts
```

## Work division

Single-team-member entry (team size 1): **SHRESTH** — full ownership of indexing,
query processing, ranking, chunking, generation, verification, evaluation and
reporting.

## AI-use declaration

Permitted by the competition rules. AI assistants (opencode/ChatGPT) were used
for scaffolding, debugging, stemmer validation against NLTK, and report editing;
all core algorithmic decisions (zone weighting, proximity formulation, novelty
penalty, dev/test protocol) were designed and validated by the author, with
every reported number reproduced by `python -m verirag eval --charts` and
`python -m verirag eval-verify --charts` from a cold start.
