"""Baseline RAG: retrieve chunks, then have the local LLM answer from them with chunk-ID citations."""

from mlx_lm import generate, load

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
        f"[{c['id']}] ({c['company']} {c['form']} FY{c['fiscal_year']} {c['fiscal_period']}, Item {c['item']}: {c['title']})\n{c['text']}"
        for c in chunks
    )


class RAG:
    """Holds the retriever and the generator model in memory for repeated questions."""

    def __init__(self):
        self.retriever = Retriever()
        self.model, self.tokenizer = load(GEN_MODEL)

    def answer(self, question: str) -> tuple[str, list[dict]]:
        """Retrieve evidence for the question and generate a cited answer; return the answer and the chunks used."""
        chunks = self.retriever.search(question, k=TOP_K)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        text = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, verbose=True)
        return text, chunks


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
