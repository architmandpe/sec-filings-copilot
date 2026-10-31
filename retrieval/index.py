"""Split parsed 10-K sections into chunks, embed them, and save the index."""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingest.edgar import get_cik, list_filings

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = 400
OVERLAP_TOKENS = 50
INDEX_DIR = Path("data/index")


def chunk_text(text: str, tokenizer) -> list[str]:
    """Split text into windows of CHUNK_TOKENS tokens that overlap by OVERLAP_TOKENS."""
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    chunks = []
    for start in range(0, len(offsets), CHUNK_TOKENS - OVERLAP_TOKENS):
        window = offsets[start:start + CHUNK_TOKENS]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + CHUNK_TOKENS >= len(offsets):
            break
    return chunks


def build_chunks(ticker: str, tokenizer) -> list[dict]:
    """Chunk every section of the company's downloaded 10-Ks, with metadata for filtering and citation."""
    cik = get_cik(ticker)
    chunks = []
    for filing in list_filings(cik):
        fiscal_year = int(filing["report_date"][:4])  # wrong for companies that label FY by start year; see README
        sections = json.loads((Path("data/parsed") / cik / f"{filing['accession']}.json").read_text())
        for section in sections:
            for n, text in enumerate(chunk_text(section["text"], tokenizer)):
                chunks.append({
                    "id": f"{ticker}-{fiscal_year}-{section['item']}-{n:03d}",
                    "ticker": ticker,
                    "cik": cik,
                    "form": filing["form"],
                    "fiscal_year": fiscal_year,
                    "item": section["item"],
                    "title": section["title"],
                    "accession": filing["accession"],
                    "url": filing["url"],
                    "text": text,
                })
    return chunks


if __name__ == "__main__":
    model = SentenceTransformer(EMBED_MODEL)
    chunks = build_chunks("AAPL", model.tokenizer)
    embeddings = model.encode(
        [c["text"] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    with open(INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks, embeddings shape {embeddings.shape}, device {model.device}")
