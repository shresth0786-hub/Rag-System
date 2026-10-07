# VeriRAG — demo video shot list (5–8 min, screen recording only, no slides)

Presenter: SHRESTH (solo team — narrates every component; state at the start that you own
all components). Record the terminal full-screen (1080p, font ≥16pt). Everything below runs
from the repo root; expected outputs are noted so you can cut/retake if a run differs.

**Story of the video:** VeriRAG is a *RAG system*, so the video is one question flowing through
the loop — **ask → retrieve → generate → verify → answer** — and then we peel back each layer
of the retrieval and verification internals.

Tip: run `python -m verirag` through a prompt so the command itself is visible on camera.

---

## 0:00–0:35 | Hook: what "RAG" means here
- Open the terminal and print the loop as a one-shot diagram:
  `python -c "print('QUESTION -> RETRIEVE (inverted index) -> GENERATE (cited answer) -> VERIFY (each citation) -> ANSWER')"`
- `Get-ChildItem verirag\*.py` — "13 modules, ~2,100 lines, all owned by me, pure stdlib."
- Say: "This is VeriRAG — a retrieval-augmented generation system. A question goes in, evidence
  chunks are retrieved, an answer is generated with citations, and **every citation is verified
  before it is shown**. Corpus: BEIR SciFact, 5,183 papers."

## 0:35–1:50 | The whole RAG loop in ONE command (ask)
- Live: `python -m verirag ask "metformin reduces mortality in type 2 diabetes patients"`
- Point at the four layers bottom-to-top as they appear:
  (1) `query:` — the question,
  (2) `-- retrieved chunks (k=5) --` — the retrieve stage, scores on the left,
  (3) `-- generated answer --` — sentences each ending in `[1] [3]` (the generate stage + citations),
  (4) `-- citation verification --` + `verdicts:` — the verify stage (expect SUPPORTED rows).
- Say: "One command: retrieve, generate, verify. Every bracket is a real chunk you can open."

## 1:50–2:50 | Same question in the browser UI (cleanest way to SEE the RAG loop)
- Live: `python -m verirag web` (auto-opens the browser)
- Ask tab → paste the same metformin question → Run.
- Point at: question box at top → evidence chunks (yellow = your words) → cited answer with
  badges → green SUPPORTED verdict strip at the end.
- Say: "Same pipeline, one page — you can literally watch question → evidence → cited answer → verdict."
  (This is the strongest RAG shot; keep the browser open for later.)

## 2:50–3:45 | Peel back the retrieval layer (index)
- Live: `python -m verirag term resist` — df, idf, postings, skip pointers.
- Say: "Behind 'retrieve' is an inverted index I wrote from scratch — term frequencies, zone
  weight, skip pointers. The retrieval stage is inspectable all the way down."

## 3:45–4:30 | Peel back query processing + scoring
- Live: `python -m verirag search "soil bacteria genes"` — highlight shows your words in red
  inside the matched chunk text.
- Live: `python -m verirag bool '"antibiotic resistance" AND bacteria'` — df-ordered AND,
  skip-pointer intersection, positional phrase check (expect 7 matches).
- Say: "Ranked and boolean retrieval share one index; every query leaves an audit trail."

## 4:30–5:40 | The stage that makes this RAG trustworthy: verification
- Live: `python -m verirag verify "probiotics prevent antibiotic-associated diarrhea"`
  - Expect WEAK, with `novel terms:` printed. Say: "A plausible claim, but novelty-penalized
    support says weak — the words it needs never appear in the cited chunk."
- Live: `python -m verirag ask "metformin ..."` again? No — instead run:
  `python -m verirag ask "probiotics prevent antibiotic-associated diarrhea"`
  - Show the generated answer still cites [2]/[4] but the verification table flags the weak
    sentence. "Generation trusts nothing: it can cite, but the verifier can veto."
- Say: "support = cosine × (1 − 0.6·novelty). That veto is force-on for every answer."

## 5:40–6:50 | Evaluate the loop (retrieval + verifier, live)
- Live: `python -m verirag eval-verify --charts` — F1 0.512 Jaccard → 0.579 cosine → **0.664
  VeriRAG**; accuracy 0.43 → 0.67; evidence recall 0.10 → 0.57.
- Live: `python -u -m verirag eval --charts` — ladder A0 0.601 → A1 0.652 (stemming) → A2 0.656
  (zones) → A3 0.655 (prox,+MRR) vs BM25 0.653; champions 5.7 ms/query.
- Open `output\eval_retrieval.png` and `output\eval_verifier.png` for 5 s each.

## 6:50–7:50 | Honest limitations + close
- Say: "Sentence chunks cost ~2 nDCG vs whole-doc, but buy span-level citations. g(d) didn't
  help on SciFact — we report that null. LLM generation is wired (`--llm`) but every number
  here is the deterministic extractive path."
- `Get-ChildItem report` → report.pdf (4 pages): "Reproduces with `python -m verirag eval --charts`."
- End on one more `python -m verirag ask "<claim>"` run so the LAST image is the RAG loop again.

**Total ≈ 7:50.** Upload as **unlisted**; paste the link into README.md (replace the placeholder)
and into the report's header line.