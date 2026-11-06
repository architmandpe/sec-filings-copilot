"""Cross-encoder reranker: re-score a short list of candidate chunks by reading each one together with the question.

The first stage (dense, or dense + BM25) embeds questions and chunks separately, which is fast but can't connect
"capital expenditure" to a table row "Purchases of property, plant and equipment". A cross-encoder reads the pair
as one input and outputs a relevance score, so it can; it costs one model call per (question, chunk) pair, so it
only re-orders the first stage's top candidates.
"""

from sentence_transformers import CrossEncoder

RERANK_MODEL = "BAAI/bge-reranker-base"
MAX_LENGTH = 512  # tokens of question + chunk the model reads (chunks are 400 tokens, so they fit)


def passage(chunk: dict) -> str:
    """What the reranker reads for a chunk: its section title, then its text."""
    return f"{chunk['title']}\n{chunk['text']}"


class Reranker:
    """Holds the cross-encoder in memory for repeated scoring."""

    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name, max_length=MAX_LENGTH)

    def scores(self, query: str, chunks: list[dict]) -> list[float]:
        """One relevance score per chunk (higher = more relevant; only the order matters)."""
        return self.model.predict([(query, passage(c)) for c in chunks], batch_size=16).tolist()
