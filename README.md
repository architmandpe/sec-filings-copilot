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

### XBRL financial data

| Problem | Cause | Status |
|---|---|---|
| Filtering by `fiscal_year = 2023` returns three different years of revenue | In companyfacts, `fy` is the fiscal year of the *filing* that reported a value, and each 10-K repeats the two prior years for comparison | **Fixed:** the `annual` view identifies the period by its dates and keeps one row per company, concept and fiscal year, with the latest filing's value winning |
| Microsoft's FY2008 and FY2009 values were labeled FY2010 | First rule labeled a period with the first 10-K that reported it, but XBRL data starts at the FY2010 10-K, whose comparison years have no 10-K of their own in the data | **Fixed:** each 10-K defines its own fiscal year-end (latest full-year period it reports); periods without their own 10-K are dropped instead of mislabeled |
| Two rows for the same concept and year (e.g. Apple's 2017 buyback authorization) | Instant facts at dates other than the fiscal year-end, such as announcement dates, were kept | **Fixed:** only facts dated on a fiscal year-end are kept. Verified by a query that counts duplicate company/concept/year rows: 0 |

### Data limitations

| Limitation | Consequence |
|---|---|
| Items 10–14 (directors, executive compensation, ownership) are incorporated by reference from the proxy statement (DEF 14A), which is not ingested | Questions such as executive pay cannot be answered from 10-Ks; the system should refuse them |
| XBRL concept names change over time (e.g. revenue moved from `SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax` with the ASC 606 standard in 2018) | A query on one concept name silently misses older years; metrics need a mapping to several candidate concepts |
