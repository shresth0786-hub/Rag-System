import re
import unicodedata

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself just me more most mustn't my myself
    of off on once only or other ought our ours ourselves out over own same
    shan't she she'd she'll she's should shouldn't so some such than that
    that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up
    very was wasn't we we'd we'll we're we've were weren't what what's when
    when's where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves
    also may might must shall will can could amongst upon within without
    """.split()
)

NEGATORS = frozenset({"not", "no", "nor", "neither", "never", "none", "without", "against"})

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text, remove_stopwords=True, keep_negators=True):
    norm = normalize(text)
    if not norm:
        return []
    terms = []
    for tok in norm.split(" "):
        if not tok:
            continue
        if remove_stopwords and tok in STOPWORDS:
            if not (keep_negators and tok in NEGATORS):
                continue
        terms.append(tok)
    return terms


class PorterStemmer:
    _VOWELS = frozenset("aeiou")

    def is_consonant(self, word, i):
        ch = word[i]
        if ch in self._VOWELS:
            return False
        if ch == "y":
            return i == 0 or not self.is_consonant(word, i - 1)
        return True

    def measure(self, stem):
        n = len(stem)
        i = 0
        m = 0
        while i < n and self.is_consonant(stem, i):
            i += 1
        while i < n:
            while i < n and not self.is_consonant(stem, i):
                i += 1
            if i >= n:
                break
            j = i
            while j < n and self.is_consonant(stem, j):
                j += 1
            if j == i:
                break
            m += 1
            i = j
        return m

    def vowel_in_stem(self, stem):
        return any(not self.is_consonant(stem, i) for i in range(len(stem)))

    def double_consonant(self, stem):
        return len(stem) >= 2 and stem[-1] == stem[-2] and self.is_consonant(stem, len(stem) - 1)

    def ends_cvc(self, stem):
        n = len(stem)
        if n < 3:
            return False
        if not self.is_consonant(stem, n - 1):
            return False
        if self.is_consonant(stem, n - 2):
            return False
        if not self.is_consonant(stem, n - 3):
            return False
        return stem[n - 1] not in "wxy"

    def stem(self, word):
        word = word.casefold()
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5(word)
        return word

    def _step1a(self, word):
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word):
        if word.endswith("eed"):
            stem = word[:-3]
            if self.measure(stem) > 0:
                return stem + "ee"
            return word
        if word.endswith("ed"):
            stem = word[:-2]
            if self.vowel_in_stem(stem):
                return self._step1b_cleanup(stem)
            return word
        if word.endswith("ing"):
            stem = word[:-3]
            if self.vowel_in_stem(stem):
                return self._step1b_cleanup(stem)
            return word
        return word

    def _step1b_cleanup(self, stem):
        if stem.endswith(("at", "bl", "iz")):
            return stem + "e"
        if self.double_consonant(stem) and stem[-1] not in "lsz":
            return stem[:-1]
        if self.measure(stem) == 1 and self.ends_cvc(stem):
            return stem + "e"
        return stem

    def _step1c(self, word):
        if word.endswith("y") and self.vowel_in_stem(word[:-1]):
            return word[:-1] + "i"
        return word

    _STEP2 = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ]

    def _step2(self, word):
        for suffix, repl in self._STEP2:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self.measure(stem) > 0:
                    return stem + repl
                return word
        return word

    _STEP3 = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]

    def _step3(self, word):
        for suffix, repl in self._STEP3:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self.measure(stem) > 0:
                    return stem + repl
                return word
        return word

    _STEP4 = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ]

    def _step4(self, word):
        if word.endswith("ion"):
            stem = word[:-3]
            if self.measure(stem) > 1 and stem.endswith(("s", "t")):
                return stem
            return word
        for suffix in self._STEP4:
            if word.endswith(suffix):
                stem = word[: -len(suffix)]
                if self.measure(stem) > 1:
                    return stem
                return word
        return word

    def _step5(self, word):
        if word.endswith("e"):
            stem = word[:-1]
            m = self.measure(stem)
            if m > 1 or (m == 1 and not self.ends_cvc(stem)):
                return stem
        if word.endswith("ll"):
            stem = word[:-1]
            if self.measure(stem) > 1:
                return stem
        return word


_STEMMER = PorterStemmer()


def stem(term):
    return _STEMMER.stem(term)


def analyze(text, stemming=True, remove_stopwords=True, keep_negators=True):
    terms = tokenize(text, remove_stopwords=remove_stopwords, keep_negators=keep_negators)
    if stemming:
        terms = [stem(t) for t in terms]
    return terms
