"""RAG: retrieve chunks, then have the local LLM answer from them with chunk-ID citations.

Retrieval options are switches; the defaults reproduce the Week 2 baseline (dense search over all chunks).
mlx_lm (Apple silicon only) is imported only when the generator is loaded, so retrieval also runs on Linux
(the hosted demo)."""

from retrieval.filters import FilingFilter
from retrieval.search import Retriever

GEN_MODEL = "mlx-community/Qwen3-8B-4bit"
TOP_K = 5
MAX_TOKENS = 512
REFUSAL = "I cannot answer this from the provided filings."

SYSTEM_PROMPT = f"""You answer questions about companies using only excerpts from their SEC filings.

Rules:
1. Use only the information in the excerpts. Do not use outside knowledge.
2. After every claim, cite the excerpt it comes from by its ID in square brackets, e.g. [APPLE_2024_10K-1A-003].
3. Copy numbers exactly as they appear in the excerpts, with their units (e.g. "in millions").
4. If the excerpts do not contain the answer, reply with exactly: "{REFUSAL}" followed by one sentence saying what information is missing."""


def format_context(chunks: list[dict]) -> str:
    """Label each chunk with its ID and source so the model can cite it."""
    return "\n\n".join(
        f"[{c['id']}] (calculator result)\n{c['text']}" if c["form"] == "calculation" else
        f"[{c['id']}] ({c['company']} {c['form']} FY{c['fiscal_year']} {c['fiscal_period']}, Item {c['item']}: {c['title']})\n{c['text']}"
        for c in chunks
    )


class RAG:
    """Holds the retriever and the generator model in memory for repeated questions."""

    def __init__(self, filter_filings: bool = False, search: str = "dense", rerank: bool = False,
                 load_generator: bool = True):
        """filter_filings: search only the filings of the companies and years the question names (retrieval/filters.py).
        search: "dense" or "hybrid" (dense + BM25). rerank: re-order the top candidates with a cross-encoder.
        load_generator: False = retrieval only (no LLM; for machines without MLX)."""
        self.retriever = Retriever(mode=search, rerank=rerank)
        self.filing_filter = FilingFilter(self.retriever.chunks) if filter_filings else None
        self.model = self.tokenizer = None
        if load_generator:
            from mlx_lm import load
            self.model, self.tokenizer = load(GEN_MODEL)

    def retrieve(self, question: str) -> list[dict]:
        """The TOP_K chunks given to the LLM for this question."""
        doc_names = self.filing_filter.infer(question) if self.filing_filter else None
        return self.retriever.search(question, k=TOP_K, doc_names=doc_names)

    def prompt(self, question: str, chunks: list[dict]) -> str:
        """The chat-formatted prompt: rules, then the excerpts and the question (thinking mode off)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"},
        ]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)

    def answer(self, question: str, verbose: bool = True) -> tuple[str, list[dict]]:
        """Retrieve evidence for the question and generate a cited answer; return the answer and the chunks used.

        verbose streams the answer and prints speed stats (on for the demo, off for evals)."""
        from mlx_lm import generate
        chunks = self.retrieve(question)
        text = generate(self.model, self.tokenizer, prompt=self.prompt(question, chunks), max_tokens=MAX_TOKENS,
                        verbose=verbose)
        return text, chunks

    def stream(self, question: str, chunks: list[dict]):
        """Yield the answer as it grows, piece by piece (for the web demo); same prompt and decoding as answer()."""
        from mlx_lm import stream_generate
        text = ""
        for response in stream_generate(self.model, self.tokenizer, self.prompt(question, chunks), max_tokens=MAX_TOKENS):
            text += response.text
            yield text


if __name__ == "__main__":
    rag = RAG()
    for question in [
        "What are Apple's main supply-chain risk factors?",
        "How much did Apple spend on research and development in fiscal 2024?",
        "How much was Apple's CEO paid in fiscal 2024?",
        "What will Apple's revenue be in 2030?",
    ]:
        print(f"\n\n### {question}")
        _, chunks = rag.answer(question)
        print("Retrieved:", ", ".join(c["id"] for c in chunks))
