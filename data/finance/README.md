# Finance sample corpus — VeriRAG

Synthetic, BEIR-compatible. Replace with your filings via `scripts/ingest_finance.py`.

## Files
- `corpus.jsonl` — 20 docs: `{_id, ticker, company, title, text, filing_type, period, source_url}` covering earnings calls, 10-K/10-Q/8-K, risk disclosures, debt notes.
- `queries.jsonl` — 15 claims (10 SUPPORT with evidence sentence refs, 2 NEI unsupported, 3 smoke).
- `qrels/test.tsv` — `query-id<TAB>corpus-id<TAB>1` (13 judged).

## Build
```powershell
python -m verirag build --corpus data/finance/corpus.jsonl --out output/index_finance.pkl --strategy finance
python -m verirag search "Apple revenue Q3 2024" --index output/index_finance.pkl --explain
python -m verirag bool "ticker:AAPL AND revenue" --index output/index_finance.pkl
python -m verirag ask "What was Apple revenue in Q3 FY2024 and guidance for Q4?" --index output/index_finance.pkl --json
```

## Ingest your own
```powershell
pip install pypdf
python scripts/ingest_finance.py --input data/raw --out data/finance/corpus.jsonl
python scripts/ingest_finance.py --input report.pdf --ticker AAPL --filing-type 10-K --period 2024-Q3 --out data/finance/corpus.jsonl --append
```

## Eval (toy)
```powershell
python -m verirag eval --corpus data/finance/corpus.jsonl --queries data/finance/queries.jsonl --qrels data/finance/qrels/test.tsv
```
