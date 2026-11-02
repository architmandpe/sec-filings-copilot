# SEC Filings Copilot

A fully local, open-source question-answering and analysis system over SEC 10-K and 10-Q filings.
Answers questions with cited evidence, computes financial metrics, and refuses when evidence is missing.

**Status:** Week 1: thin end-to-end baseline complete (dense retrieval → cited answer from a local LLM).

## Setup

```
uv sync
cp .env.example .env   # then edit .env with your name and email
uv run python -m evals.financebench   # download FinanceBench questions into data/evals/
uv run python -m evals.manifest       # list the 72 10-K/10-Q filings they need
uv run python -m ingest.xbrl    # load XBRL financial facts into data/financials.duckdb
uv run python -m ingest.edgar   # download Apple's latest 10-Ks into data/raw/
uv run python -m ingest.corpus  # match and download the FinanceBench filings into data/raw/
uv run python -m ingest.parse   # split them into Item sections in data/parsed/
uv run python -m retrieval.index   # chunk and embed sections into data/index/
uv run python -m retrieval.search  # show the top chunks for sample questions
uv run python -m agent.answer      # retrieve and answer sample questions with citations
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
| Some 10-Ks were assigned the wrong period end (Coca-Cola FY2016 → March 2017, Boeing FY2017 → December 2018, Activision FY2014 → February 2015) | The period end was the *latest* end date among a filing's facts, but filings also tag forward-looking facts (Boeing: next year's 21% tax rate) and subsequent events (Activision: a dividend declared in February). Taking the *most common* end date instead failed differently: some 10-Ks tag more facts for the prior year than for their own | **Fixed:** the `filings` view uses the latest year-long (10-K) or quarter-long (10-Q) duration that ends on or before the filing date. Verified: the derived period end equals EDGAR's own `reportDate` for all 72 corpus filings |
| 513 duplicate rows in the `annual` view after loading 32 companies | Three causes: (1) older filings start a year on the previous year-end (2009-12-31 vs 2010-01-01), so the same value appeared under two start dates; (2) the wrong period ends above; (3) the duplicate check ignored `taxonomy`, and some concepts exist in two taxonomies | **Fixed:** `period_start` is taken from the latest filing instead of being grouped on; the check now groups by taxonomy too. Verified: 0 duplicates |
| `annual` values can differ from what a filing originally reported (J&J FY2022 revenue: $94.9B in the FY2022 10-K, $80.0B in the view) | "Latest filing wins" picks restatements: after the 2023 Kenvue separation, J&J restated 2022 without its Consumer Health segment | **Open:** questions about "the FY2022 10-K" need the as-filed value; checking FinanceBench answers against XBRL must use the value from that filing's own accession |

### Evaluation corpus (FinanceBench)

| Problem | Cause | Status |
|---|---|---|
| Half the FinanceBench filings have no CIK in their link | FinanceBench took PDFs from many sites (its own CDN, company investor-relations sites, hosting providers); only its CDN links contain `CIK-##########` | **Fixed:** a company's CIK is taken from any of its links; 15 companies with none are looked up by ticker (Activision Blizzard, delisted, is hard-coded). Verified against EDGAR's company name for all 31 |
| The accession prefix looks like a CIK but often isn't | Many filings are submitted by filing agents, whose CIK forms the accession prefix (e.g. `0001558370-19-000470` for a 3M 10-K) | **Avoided:** CIKs never come from accession numbers |
| JPMorgan's 2021–2023 10-Ks/10-Qs are missing from `filings.recent` | `recent` holds one year or 1,000 filings, and JPMorgan files ~23,000 prospectuses a year; older filings are on 71 extra pages | **Fixed:** filings are found via the XBRL catalog, then only the submissions page covering the filing date is fetched |
| "Fiscal year 2023" is ambiguous | Ulta calls its year ending January 2023 "fiscal 2022" and FinanceBench calls it 2023; Best Buy's quarter ending July 2023 is in "fiscal 2024"; J&J's year ending January 1, 2023 is "fiscal 2022" | **Fixed:** a fiscal year is named by the calendar year it ends in, minus 7 days for 52/53-week years ending in early January; a 10-Q takes the fiscal year of the next 10-K year-end. Verified: all 9 FinanceBench filings with a known accession match, and every manifest filing matches exactly one accession |
| A wrong J&J filing passed the "exactly one match" check | Uniqueness doesn't prove correctness; J&J's link had no accession to compare against | **Fixed** (rule above); found by reading the question evidence ("At January 1, 2023"). Lesson: only known answers test correctness |
| CVS 2018 10-K is 0.9 MB (CVS 2022: 6.0 MB) | Probably MD&A and financial statements are in Exhibit 13 and incorporated by reference | **Open:** check Item 8 when parsing (Step 2.2d) |
| Some FinanceBench gold answers are wrong (Pfizer Upjohn costs: 77.78 instead of ~70) or contradict their justification (American Water Works working capital) | Human labeling errors | **Open:** failures are checked against the filing during error analysis; label corrections are logged |

### Chunking and retrieval

| Problem | Cause | Status |
|---|---|---|
| Chunk `fiscal_year` is taken from the calendar year of the fiscal year-end date | Simplification for the baseline; correct for Apple (September) and Microsoft (June) | **Open:** wrong for companies that name a fiscal year by its starting year (some retailers); use the filing's own fiscal year label when they are added |
| Some chunks contain nothing but a page footer (e.g. Item 6 "[Reserved]" → `Apple Inc. \| 2024 Form 10-K \| 19`) | Footers remain in section text (see Parsing), and every non-empty section produces at least one chunk | **Open:** junk chunks can take retrieval slots; fix footers and/or drop very short chunks, and measure the effect in Week 3 |
| Retrieval scores cannot tell answerable questions from unanswerable ones (CEO pay: top score 0.750, supply chain: 0.754) | Dense search always returns the k nearest chunks, however unrelated | **By design:** refusal must come from the generator (and later a reranker score), not a similarity threshold |

### Answer generation

| Problem | Cause | Status |
|---|---|---|
| Supply-chain question answered partly with a financial-risk chunk (credit risk, counterparty failure), cited correctly but irrelevant | Dense retrieval returned macroeconomic Risk Factor chunks, and the model listed every excerpt as part of the answer | **Open:** measure answer relevance separately from citation faithfulness in Week 2; try a reranker in Week 3 |
| Answers mix fiscal years without saying so (supply-chain answer cites FY2024 and FY2025 chunks) | No fiscal-year filter in retrieval; near-duplicate passages exist in every year's 10-K | **Open:** fiscal-year metadata filter in Week 3 |
| Refusals tested only on questions where no retrieved chunk was relevant (CEO pay, 2030 revenue) | Baseline smoke test had 4 questions | **Open:** false refusals (refusing when the answer is in the excerpts) are measured with the Week 2 question set |

Verified: the baseline's answer for Apple's FY2024 R&D ($31,370 million, from the 10-K text) matches the XBRL fact in DuckDB.

### Data limitations

| Limitation | Consequence |
|---|---|
| Items 10–14 (directors, executive compensation, ownership) are incorporated by reference from the proxy statement (DEF 14A), which is not ingested | Questions such as executive pay cannot be answered from 10-Ks; the system should refuse them |
| XBRL concept names change over time (e.g. revenue moved from `SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax` with the ASC 606 standard in 2018) | A query on one concept name silently misses older years; metrics need a mapping to several candidate concepts |
