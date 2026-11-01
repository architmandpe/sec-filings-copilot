"""Download the FinanceBench open-source questions and save them as JSONL for inspection."""

import json
from pathlib import Path

from datasets import load_dataset

OUT_PATH = Path("data/evals/financebench.jsonl")


if __name__ == "__main__":
    dataset = load_dataset("PatronusAI/financebench", split="train")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"{len(dataset)} questions")
    print(f"columns: {dataset.column_names}")
