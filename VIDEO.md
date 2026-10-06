# VeriRAG — demo video shot list (5–8 min, screen recording only, no slides)

Presenter: SHRESTH (solo team — narrates every component; state at the start that you own
all components). Record the terminal full-screen (1080p, font ≥16pt). Everything below runs
from the repo root; expected outputs are noted so you can cut/retake if a run differs.

Tip: run `python -m verirag` through a prompt so the command itself is visible on camera.

---

## 0:00–0:35 | Hook + what this is
- No title card — start with the terminal, say: "This is VeriRAG, a RAG system whose entire
  retrieval engine is written from scratch in Python stdlib — inverted index, Porter stemmer,
  Boolean parser, tf-idf ranking — plus a citation verifier. Dataset: BEIR SciFact."
- `Get-ChildItem verirag\*.py` (one line) to show the package is small: "13 modules, ~2,100 lines."

## 0:35–1:45 | Build: chunking + indexing (owned: text.py, chunker.py, index.py)
- Live: `python -m verirag build`
  - Narrate while it runs (~30 s): tokenizer keeps negators; Porter stemmer written from
    scratch; 3-sentence chunks with step 2 → "what is a document?" choice.
  - Point at stats JSON: `n_docs=5183`, `n_chunks=21233`, `n_terms=26319`, build seconds.
- Live: `python scripts\check_stemmer_full.py`
  - Show "18 / 29,699 mismatches, all non-ASCII ligatures" → from-scratch stemmer validated
    against NLTK's original algorithm.

## 1:45–2:45 | Inside the index (owned: index.py, scoring.py)
- Live: `python -m verirag term resist`
  - Point at: `df`, `idf`, a few `(doc, tf, tf_title, positions)` postings, skip pointers.
  - Say: "title terms carry zone weight λ=2 via a +1000 position gap; skips built for lists
    >128 entries; champion list top-50 exists as a fast mode."

## 2:45–3:40 | Query processing (owned: query.py)
- Live: `python -m verirag bool '\"antibiotic resistance\" AND bacteria'`
  - (PowerShell needs the backslashes: `\"...\"`; cmd.exe can use plain quotes.)
  - Point at the trace line(s): "df-ordered AND — rarest term first, skip pointers jump
    over long postings; phrase check is positional, not bag-of-words" (expect 7 matches).
- Live: `python -m verirag bool 'title:CRISPR AND NOT review'` (expect 54 matches)
  - Point at `[postings[crispr] zone=title]` — "field query + NOT in one parse; 54 title-zone
    matches vs 62 for plain CRISPR."

## 3:40–4:45 | Ranked retrieval with explain (owned: retrieve.py, scoring.py)
- Live: `python -m verirag search "antibiotic resistance genes in soil bacteria" --explain`
  - Point at term stats (idf, query weight ltc), then per-result breakdown:
    `score = base × (1 + w·prox) + η·g` with actual numbers from the `(base=… prox=… g=…)` line.
  - Say: "w=0.5 and window=30 were chosen on the 100-claim dev split; η tuned to 0.0 —
    a null result we report in the paper."

## 4:45–6:10 | RAG answer + citation verification (owned: generator.py, verifier.py)
- Live: `python -m verirag ask "metformin reduces mortality in type 2 diabetes patients" --explain`
  - Point at: retrieved chunks with scores → extractive answer sentences each ending `[1] [3]`
    → verification table: `support, cos, novel, verdict` (expect SUPPORTED rows here).
  - Say: "support = cosine(claim, cited chunk) × (1 − 0.6·novelty); novelty = idf-weighted
    share of claim terms absent from the chunk it cites."
- Live: `python -m verirag verify "probiotics prevent antibiotic-associated diarrhea"`
  - Expect a WEAK/UNSUPPORTED outcome with `novel terms: [...]` printed — this is the
    money shot: the system distrusts its own plausible-looking citation.
  - Optional adversarial twist: cite-tamper by asking about a claim whose top chunk is off-topic
    and show the mismatch warning in `ask` output.

## 6:10–7:15 | Evaluation, live (owned: evaluate.py, metrics.py)
- Live: `python -m verirag eval-verify --charts` (~16 s)
  - Read the table off the terminal: Jaccard F1 0.512 / cosine 0.579 / **VeriRAG 0.664**,
    accuracy 0.43 → 0.67; then the retrieval stage and evidence-recall lines.
- Live (or narrate while it runs, ~95 s): `python -u -m verirag eval --charts`
  - Point at the ladder: A0 0.601 → A1 0.652 (stemming!) → A2 0.656 (zones) → A3 0.655
    (proximity: neutral nDCG, +MRR) vs BM25 0.653; champions row = 5.7 ms/query.
  - Open `output\eval_retrieval.png` and `output\eval_verifier.png` for 5 s each.

## 7:15–7:50 | Close (honest limitations + where things are)
- Say: "Sentence chunks cost us 1.8 nDCG vs whole-doc indexing — we pay that for span-level
  citations. g(d) didn't help on SciFact. LLM mode is wired (`--llm`) but the numbers above
  are the deterministic extractive path."
- Show `Get-ChildItem report` → `report.pdf` (4 pages) and README: "Everything here reproduces
  with python -m verirag eval --charts."
- End on `python -m verirag ask "<claim>"` output. No end card needed.

**Total ≈ 7:50.** Upload as **unlisted**; paste the link into README.md (replace the placeholder)
and into the report's header line.
