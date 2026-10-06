import json
import sys

sys.path.insert(0, ".")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

from nltk.stem.porter import PorterStemmer as N

from verirag.text import PorterStemmer

mine = PorterStemmer()
ref = N(mode=N.ORIGINAL_ALGORITHM)
words = set()
for path in ["data/scifact/corpus.jsonl", "data/scifact/queries.jsonl"]:
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        text = d.get("title", "") + " " + d["text"]
        for tok in text.replace("/", " ").replace("-", " ").split():
            tok = tok.strip(".,;:()[]{}\"'").lower()
            if tok.isalpha():
                words.add(tok)
mm = [(w, mine.stem(w), ref.stem(w)) for w in sorted(words) if mine.stem(w) != ref.stem(w)]
print("total", len(words), "mismatch", len(mm))
print(mm[:40])
