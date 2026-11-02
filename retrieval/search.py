"""Dense retrieval: find the chunks most similar to a question."""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.index import EMBED_MODEL, INDEX_DIR

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention, queries only


class Retriever:
    """Holds the embedding model and the chunk index in memory for repeated searches."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")
        with open(INDEX_DIR / "chunks.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the k chunks most similar to the query, best first, each with its similarity score."""
        query_vec = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        scores = self.embeddings @ query_vec
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


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
