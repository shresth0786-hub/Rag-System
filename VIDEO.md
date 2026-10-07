# VeriRAG — demo video shot list (~7:50, screen recording only, no slides)

Track **T1** (RAG + trustworthy answers). This shot list maps 1:1 to the assignment's
"What the video must show": (1) problem + track relevance ≤1 min, (2) end-to-end on real
inputs incl. a limitation, (3) pipeline then evaluation with actual code/intermediate output,
(4) evaluation vs a baseline, (5) each member explaining the component they own.

Presenter: SHRESTH (solo team — every component explained by its builder). Record terminal
full-screen (1080p, font ≥16pt). Everything runs live against the real index — **nothing
hard-coded or pre-recorded**. Run from the repo root. Expected outputs noted so you can
cut/retake if a run differs.

---

## 0:00–0:50 | 1. The problem + why it belongs to T1 (≤1 min)
- `python -c "print('QUESTION -> RETRIEVE (inverted index) -> GENERATE (cited answer) -> VERIFY (each citation) -> ANSWER')"`
- `Get-ChildItem verirag\*.py` — "13 modules, ~2,100 lines, all mine, pure stdlib."
- Say: problem = LLM only as good as what gets retrieved; uncited answers = hallucinations.
  Track T1 because the retriever is a real inspectable IR component and every claim is
  traced to a ranked source. Solo team. Corpus: BEIR SciFact, 5,183 papers.

## 0:50–2:05 | 2. System running end to end on a real query
- Live: `python -m verirag ask "metformin reduces mortality in type 2 diabetes patients"`
- Point at 4 layers as they print: query → `-- retrieved chunks (k=5) --` (scores left)
  → `-- generated answer --` (`[1] [3]` citations) → `-- citation verification --` verdicts.
- Say: "One command: retrieve, generate, verify. Every bracket is a real chunk."

## 2:05–2:50 | Same loop in the browser UI (visible end to end)
- Live: `python -m verirag web` → Ask tab → same metformin question → Run.
- Point at: question → evidence chunks (yellow = your words) → cited answer badges →
  green SUPPORTED strip. "Same pipeline, one page — the full RAG sequence on screen."

## 2:50–3:55 | A limitation, live + the verify stage's weights & scores (REQ 2+3)
- Live: `python -m verirag verify "probiotics prevent antibiotic-associated diarrhea"`
  - Point at intermediate output: cited chunk → printed novel terms → support score with
    novelty penalty visible → verdict **WEAK**.
  - Say: "cosine × (1 − 0.6·novelty); the words it needs aren't in the chunk — the system
    finds its own limitation, live."
- Live: `python -m verirag ask "probiotics prevent antibiotic-associated diarrhea"`
  - Answer still generated but the weak sentence is flagged in the verification table.
  - Say: "Generation can cite; the verifier can veto. The veto is force-on for every answer."

## 3:55–4:50 | The engine: index + the actual code (REQ 3)
- Live: `python -m verirag term resist` — df → idf → postings → skip pointers.
- `Get-Content verirag\index.py -TotalCount 28` — dictionary+postings build, zone weights.
- Say: "Chunking is the 'what is a document?' decision — sentence-level chunks scored and
  cited individually, which is what makes span-level citations possible."

## 4:50–5:40 | Query processing + scoring (ranked, Boolean, phrase) (REQ 3)
- Live: `python -m verirag search "soil bacteria genes"` — red highlights, cosine scores.
- Live: `python -m verirag bool '"antibiotic resistance" AND bacteria'` — expect 7 matches;
  traced df-ordered AND, skip-pointer intersection, positional phrase check.
- Say: "Ranked and Boolean retrieval share one index; every query leaves an audit trail of
  postings and scores — not just a top-k screen."

## 5:40–6:45 | 4. Evaluation: P@k and recall against a baseline
- Live: `python -m verirag eval-verify --charts` — Jaccard 0.512 → cosine 0.579 → VeriRAG
  **0.664 F1**; accuracy 0.43 → 0.67; evidence recall 0.10 → 0.57; claim→doc hit@5/hit@10/MRR.
- Live: `python -u -m verirag eval --charts` — every row shows **P@5, P@10, R@100, nDCG@10,
  MRR@10** vs the **BM25 baseline**: ladder 0.601 → 0.652 (stemming) → 0.656 (zones) → 0.655
  (proximity, +MRR) vs BM25 0.653; champions 5.7 ms/query.
- Open `output\eval_retrieval.png` / `output\eval_verifier.png` for 5 s. "The numbers in the
  report, reproduced live."

## 6:45–7:50 | 5. Component ownership + honest close
- Say, per component: chunker/tokenizer — mine; inverted index/zones/scoring/champions —
  mine; retrieval pipeline + extractive cited generator — mine; verifier + novelty penalty —
  mine; evaluation harness — mine. "Solo team: every component explained by its builder."
- Honest limitations: sentence chunks ≈ −2 nDCG vs whole-doc but buy span-level citations;
  static-quality g(d) null reported; LLM wired but all shown numbers are the deterministic
  extractive path.
- `Get-ChildItem report` → report.pdf (4 pages): "README reproduces every number with one
  command." End on one more `python -m verirag ask "<claim>"` so the LAST image is the loop.

**Total ≈ 7:50.** Upload **unlisted** (YouTube/Drive); paste the link into README.md and the
report header. Report also carries work division + AI-use declaration.