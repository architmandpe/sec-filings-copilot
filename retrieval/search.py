"""Search the chunk index: find the chunks most relevant to a question.

Stages, each a switch so every combination can be measured (defaults = the Week 2 baseline):
  1. candidates  dense (cosine similarity of bge embeddings), or hybrid (dense and BM25 lists merged by
                 reciprocal rank fusion); optionally restricted to some filings (doc_names)
  2. rerank      optionally re-order the top `candidates` with a cross-encoder (retrieval/rerank.py)
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.bm25 import BM25
from retrieval.index import EMBED_MODEL, INDEX_DIR

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention, queries only
SEARCH_MODES = ("dense", "hybrid")
RRF_K = 60  # reciprocal rank fusion constant: a chunk scores 1 / (RRF_K + rank) in each list (60 is the standard)
FUSION_DEPTH = 100  # how far down each list (dense, BM25) is merged
RERANK_CANDIDATES = 50  # how many first-stage chunks the reranker re-orders


def rrf(rankings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Merge ranked lists of chunk indices: sum of 1 / (RRF_K + rank) over the lists. Returns (indices, scores)."""
    fused = {}
    for ranking in rankings:
        for rank, i in enumerate(ranking[:FUSION_DEPTH], 1):
            fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank)
    order = sorted(fused, key=lambda i: (-fused[i], i))  # ties broken by index, so runs are reproducible
    return np.array(order, dtype=int), np.array([fused[i] for i in order])


class Retriever:
    """Holds the embedding model, the chunk index and the optional BM25 index and reranker in memory."""

    def __init__(self, mode: str = "dense", rerank: bool = False, candidates: int = RERANK_CANDIDATES):
        if mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}")
        self.model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        with open(INDEX_DIR / "chunks.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]
        self.doc_names = np.array([c["doc_name"] for c in self.chunks])  # for filtering by filing
        self.mode, self.candidates = mode, candidates
        self.bm25 = BM25([f"{c['title']}\n{c['text']}" for c in self.chunks]) if mode == "hybrid" else None
        self.reranker = None
        if rerank:
            from retrieval.rerank import Reranker  # imported only when used
            self.reranker = Reranker()

    def config(self) -> dict:
        """The search settings, for the results log."""
        from retrieval.rerank import RERANK_MODEL
        cfg = {"search": self.mode, "rerank_model": RERANK_MODEL if self.reranker else None}
        if self.mode == "hybrid":
            cfg |= {"rrf_k": RRF_K, "fusion_depth": FUSION_DEPTH}
        if self.reranker:
            cfg["rerank_candidates"] = self.candidates
        return cfg

    def search(self, query: str, k: int = 5, doc_names: set[str] | None = None) -> list[dict]:
        """Return the k best chunks for the query, best first, each with the score of the last stage that ranked it.

        doc_names: if given, search only the chunks of these filings (None = all chunks, the baseline)."""
        if doc_names is None:
            allowed = np.arange(len(self.chunks))
        else:
            allowed = np.flatnonzero(np.isin(self.doc_names, list(doc_names)))
        query_vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        dense = self.embeddings @ query_vec
        ranked = allowed[np.argsort(-dense[allowed])]
        scores = dense[ranked]
        if self.mode == "hybrid":
            keyword = self.bm25.scores(query)
            ranked, scores = rrf([ranked, allowed[np.argsort(-keyword[allowed], kind="stable")]])
        if self.reranker:
            pool = ranked[:max(self.candidates, k)]
            rerank_scores = np.array(self.reranker.scores(query, [self.chunks[i] for i in pool]))
            order = np.argsort(-rerank_scores, kind="stable")
            ranked, scores = pool[order], rerank_scores[order]
        return [{**self.chunks[i], "score": float(s)} for i, s in zip(ranked[:k], scores[:k])]


if __name__ == "__main__":
    retriever = Retriever()
    for question in [
        "What are Apple's main supply-chain risk factors?",
        "How much did Apple spend on research and development in fiscal 2024?",
        "How much was Apple's CEO paid in fiscal 2024?",
    ]:
        print(f"\n{question}")
        for hit in retriever.search(question):
            print(f"  {hit['score']:.3f}  {hit['id']:<28} {hit['text'][:80]!r}")
