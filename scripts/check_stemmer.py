import json
import random
import sys

sys.path.insert(0, ".")

from nltk.stem.porter import PorterStemmer as NLTKPorter

from verirag.text import PorterStemmer

mine = PorterStemmer()
ref = NLTKPorter(mode=NLTKPorter.ORIGINAL_ALGORITHM)

paper_vocab = """
consign consigned consigning consignment consist existed existing exist
generalization generalized generalize generalized R R's RB R's depend
duplicated argue argued arguing ponies caress ties feed agreed plastered
bled motoring sing conflated troubled sized hopping tanned falling hissing
fizzed failing filing happy sky spy by my enjoy play cry dry try fry shy
probate rate cease controllable roll roll rolls rolling rolls courage
handler analogy operator airportaru
""".split()

words = set(w.strip("'\u2019,").lower() for w in paper_vocab if w.strip())

corpus_words = set()
with open("data/scifact/corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        for tok in (d["title"] + " " + d["text"]).split():
            tok = tok.strip(".,;:()[]{}/\\\"'-").lower()
            if tok.isalpha() and len(tok) > 2:
                corpus_words.add(tok)

random.seed(0)
sample = sorted(corpus_words)
random.shuffle(sample)
words.update(sample[:3000])

mismatch = []
for w in sorted(words):
    a, b = mine.stem(w), ref.stem(w)
    if a != b:
        mismatch.append((w, a, b))

print(f"compared {len(words)} words, mismatches: {len(mismatch)}")
for row in mismatch[:60]:
    print(row)
