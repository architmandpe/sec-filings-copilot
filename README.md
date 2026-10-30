# SEC Filings Copilot

A fully local, open-source question-answering and analysis system over SEC 10-K and 10-Q filings.
Answers questions with cited evidence, computes financial metrics, and refuses when evidence is missing.

**Status:** Week 1, data ingestion.

## Setup

```
uv sync
cp .env.example .env   # then edit .env with your name and email
uv run python -m ingest.edgar   # download filings into data/raw/
uv run python -m ingest.parse   # split them into Item sections in data/parsed/
```

## What failed and why

A running log of problems found, why they happened, and what was done about them.

### Parsing

| Problem | Cause | Status |
|---|---|---|
| Searching for "Item 1A" finds the table of contents, not the section | Every 10-K lists each Item twice: in the table of contents, then as the real heading | **Fixed:** keep the *last* occurrence of each Item heading, since the table of contents always comes first |
| Page footers (e.g. `Apple Inc. \| 2023 Form 10-K \| 19`) remain inside section text | Footers are ordinary text in the HTML; seen as Item 6 "[Reserved]" having 8 words instead of 0 | **Open:** footer formats differ by company, so the fix waits until more companies are ingested |
| Signatures page is attached to Item 16 | No Item heading follows the signatures to end the section | **Open:** harmless for now |

### Data limitations

| Limitation | Consequence |
|---|---|
| Items 10–14 (directors, executive compensation, ownership) are incorporated by reference from the proxy statement (DEF 14A), which is not ingested | Questions such as executive pay cannot be answered from 10-Ks; the system should refuse them |
