"""Finance ingestion: convert PDFs/CSVs/plain text into finance corpus.jsonl.

Usage:
  python scripts/ingest_finance.py --input data/raw --out data/finance/corpus.jsonl
  python scripts/ingest_finance.py --input report.pdf --ticker AAPL --filing-type 10-K --period 2024-Q3

If --input is a directory, every .pdf/.txt/.csv inside is ingested.
PDF support requires `pip install pypdf` (optional - otherwise skipped with hint).
Output is BEIR-compatible jsonl with finance meta keys: ticker, filing_type, period, source_url, company.
"""
import argparse, csv, hashlib, json, os, sys, glob

def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        print(f"[skip] {path}: pypdf not installed. Run: pip install pypdf", file=sys.stderr)
        return ""
    try:
        reader = PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as exc:
        print(f"[error] {path}: {exc}", file=sys.stderr)
        return ""

def _id_for(path: str, text: str) -> str:
    h = hashlib.sha1((path+text[:200]).encode()).hexdigest()[:8]
    return f"FIN_{h.upper()}"

def main():
    ap = argparse.ArgumentParser(description="Finance ingestion for VeriRAG")
    ap.add_argument("--input", required=True, help="file or directory containing PDFs/TXTs/CSVs")
    ap.add_argument("--out", default="data/finance/corpus.jsonl", help="output jsonl path")
    ap.add_argument("--ticker", default=None, help="default ticker for files without metadata")
    ap.add_argument("--filing-type", default=None, dest="filing_type", help="default filing_type e.g. 10-K, 10-Q, 8-K")
    ap.add_argument("--period", default=None, help="default period e.g. 2024-Q3")
    ap.add_argument("--company", default=None)
    ap.add_argument("--source-url", default=None, dest="source_url")
    ap.add_argument("--append", action="store_true", help="append instead of overwrite")
    args = ap.parse_args()

    inputs = []
    if os.path.isdir(args.input):
        for ext in ("*.pdf","*.txt","*.csv","*.md"):
            inputs.extend(glob.glob(os.path.join(args.input, ext)))
            inputs.extend(glob.glob(os.path.join(args.input, "**", ext), recursive=True))
        inputs = sorted(set(inputs))
    elif os.path.isfile(args.input):
        inputs = [args.input]
    else:
        print(f"input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not inputs:
        print(f"no ingestable files in {args.input}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    n = 0
    with open(args.out, mode, encoding="utf-8") as out:
        for path in inputs:
            if path.endswith(".pdf"):
                text = _extract_pdf(path)
            elif path.endswith(".csv"):
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        reader = csv.DictReader(f)
                        if reader.fieldnames and "text" in reader.fieldnames:
                            rows = list(reader)
                            for r in rows:
                                doc = {
                                    "_id": r.get("_id") or _id_for(path, r.get("text","")),
                                    "ticker": r.get("ticker") or args.ticker or "UNKNOWN",
                                    "company": r.get("company") or args.company or "",
                                    "title": r.get("title") or os.path.basename(path),
                                    "text": r.get("text") or "",
                                    "filing_type": r.get("filing_type") or args.filing_type or "UNKNOWN",
                                    "period": r.get("period") or args.period or "UNKNOWN",
                                    "source_url": r.get("source_url") or args.source_url or path,
                                }
                                if doc["text"]:
                                    out.write(json.dumps(doc, ensure_ascii=False)+"\n")
                                    n+=1
                            continue
                        else:
                            text = open(path, encoding="utf-8", errors="replace").read()
                except Exception as exc:
                    print(f"[error] csv {path}: {exc}", file=sys.stderr)
                    continue
            else:
                text = open(path, encoding="utf-8", errors="replace").read()
            if not text or not text.strip():
                continue
            title = os.path.splitext(os.path.basename(path))[0].replace("_"," ").title()
            doc = {
                "_id": _id_for(path, text),
                "ticker": args.ticker or "UNKNOWN",
                "company": args.company or "",
                "title": title,
                "text": text[:50000],
                "filing_type": args.filing_type or "UNKNOWN",
                "period": args.period or "UNKNOWN",
                "source_url": args.source_url or path,
            }
            out.write(json.dumps(doc, ensure_ascii=False)+"\n")
            n+=1
    print(f"ingested {n} docs -> {args.out}")

if __name__ == "__main__":
    main()
