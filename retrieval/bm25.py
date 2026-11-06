"""BM25 keyword search: score every chunk by the question's words it contains.

Dense embeddings blur exact terms ("total current liabilities", "EBITDA", "1,577"); keyword matching keeps them.
BM25 adds up, for each query word in a chunk:
  idf(word)  rare words count more (a word in 5 chunks says more than one in 15,000)
  tf part    repeating a word helps with diminishing returns (K1 caps it)
  length     long chunks are discounted, so they don't win just by containing more words (B sets how much)
Written by hand (about 40 lines) instead of a library, to keep every step visible.
"""

import math
import re
from collections import Counter, defaultdict

import numpy as np

K1 = 1.5  # term-frequency saturation (standard range 1.2-2.0)
B = 0.75  # length normalization: 0 = none, 1 = full (standard 0.75)
TOKEN = re.compile(r"[a-z0-9]+")
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")  # "1,577" -> "1577", so numbers in questions and tables match
STOPWORDS = frozenset(
    "a an and are as at be based between by did do does for from has have how in is it its of on or than that "
    "the this to was were what which with".split())


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, thousands separators removed, stopwords dropped."""
    return [t for t in TOKEN.findall(THOUSANDS.sub("", text.lower())) if t not in STOPWORDS]


class BM25:
    """An inverted index (word -> the chunks containing it, and how often) over a list of texts."""

    def __init__(self, texts: list[str]):
        docs = [tokenize(t) for t in texts]
        self.n = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=float)
        self.length_norm = K1 * (1 - B + B * lengths / lengths.mean())  # per-chunk part of the tf denominator
        postings = defaultdict(lambda: ([], []))
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                postings[term][0].append(i)
                postings[term][1].append(tf)
        self.postings = {term: (np.array(ids), np.array(tfs, dtype=float)) for term, (ids, tfs) in postings.items()}

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every text for the query (0 for texts sharing no word with it)."""
        scores = np.zeros(self.n)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            ids, tfs = self.postings[term]
            idf = math.log(1 + (self.n - len(ids) + 0.5) / (len(ids) + 0.5))
            scores[ids] += idf * tfs * (K1 + 1) / (tfs + self.length_norm[ids])
        return scores
