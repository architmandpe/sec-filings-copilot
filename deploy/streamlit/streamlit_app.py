"""Hosted demo (Streamlit Community Cloud): ask a question, see the passages the system retrieves from SEC filings.

Retrieval only: the answer model (Qwen3-8B on Apple MLX) runs on a Mac, not on a free Linux CPU server; the full
demo with streamed, cited answers is app.py, run locally. Same retrieval code and pipeline as the evaluated system
(company/fiscal-year filter, hybrid dense + BM25 search, cross-encoder reranker).

The chunk index (73 MB) is downloaded at startup from a Hugging Face dataset repo, pinned to one revision.
Local test: streamlit run deploy/streamlit/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]  # repository root: agent/ and retrieval/ live there
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # the retriever reads data/index relative to the root

from agent.answer import TOP_K, RAG  # noqa: E402  (needs the path set above)
from retrieval.index import INDEX_DIR  # noqa: E402

INDEX_REPO = "architmandpe02/sec-filings-copilot-index"
INDEX_REVISION = "a3807abc579ff20fb951bcec41fb9bbc06e82f04"  # index built from repo commit 1826cbb (20,494 chunks)
GITHUB = "https://github.com/architmandpe/sec-filings-copilot"
EXAMPLES = [
    "What was Apple's gross margin percentage in fiscal 2024?",
    "How much did Apple's total net sales grow from fiscal 2023 to fiscal 2025, in percent?",
    "Where are Apple's outsourcing partners that manufacture its hardware products primarily located, "
    "according to its fiscal 2024 10-K?",
    "What segments did Microsoft use to report its financial performance in fiscal 2023?",
    "Approximately how many full-time employees did Microsoft have as of June 30, 2023?",
]


@st.cache_resource(show_spinner="Loading the index and models (first visit takes about a minute)...")
def load_rag() -> RAG:
    """Download the index once and load the retrieval models; shared by all visitors."""
    snapshot_download(INDEX_REPO, repo_type="dataset", revision=INDEX_REVISION, local_dir=INDEX_DIR)
    return RAG(filter_filings=True, search="hybrid", rerank=True, load_generator=False)


st.set_page_config(page_title="SEC Filings Copilot", page_icon="📄")
st.title("SEC Filings Copilot")
st.markdown(
    "Search 75 SEC 10-K and 10-Q filings from 32 companies. This hosted demo shows the **passages retrieved** "
    "for a question; the full system answers from them with citations, or refuses, using Qwen3-8B running "
    f"locally on a laptop. Code and evaluation results: [GitHub]({GITHUB})."
)

example = st.selectbox("Pick an example, or type your own below", [""] + EXAMPLES)
question = st.text_input("Question", value=example, placeholder="e.g. What was Microsoft's revenue in fiscal 2023?")

if question.strip():
    rag = load_rag()
    with st.spinner("Searching..."):
        chunks = rag.retrieve(question.strip())
    st.subheader(f"Top {TOP_K} passages")
    for n, c in enumerate(chunks, 1):
        st.markdown(f"**{n}. {c['company']}**, {c['form']} FY{c['fiscal_year']} {c['fiscal_period']}, "
                    f"Item {c['item']} {c['title']}  \n`{c['id']}` · reranker score {c['score']:.2f} · "
                    f"[filing on EDGAR]({c['url']})")
        with st.expander("Show text"):
            st.text(c["text"])
