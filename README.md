# SEC Filings Copilot

A fully local, open-source question-answering and analysis system over SEC 10-K and 10-Q filings.
Answers questions with cited evidence, computes financial metrics, and refuses when evidence is missing.

**Status:** Week 1, data ingestion.

## Setup

```
uv sync
cp .env.example .env   # then edit .env with your name and email
uv run python -m ingest.edgar
```
