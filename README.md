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
uv run python -m ingest.corpus  # match and download the 75 corpus filings into data/raw/, list them in data/corpus.jsonl
uv run python -m ingest.parse   # split them into Item sections in data/parsed/
uv run python -m retrieval.index   # chunk and embed sections into data/index/
uv run python -m evals.questions   # convert FinanceBench into the eval question format, with audited corrections
uv run python -m evals.gold        # check that every question's evidence is found in the indexed chunks
uv run python -m retrieval.search  # show the top chunks for sample questions
uv run python -m agent.answer      # retrieve and answer sample questions with citations
```

## Evaluation data

| Set | Questions | Source of gold answers |
|---|---|---|
| FinanceBench (open-source subset, 10-K/10-Q only) | 127 of 150 (8-Ks and earnings releases excluded) | FinanceBench labels, with corrections from a 20-question audit (`evals/financebench_audit.csv`) |
| Own (`evals/questions_own.yaml`) | 100: table 20, computation 20, multi-hop 15, text 15, unanswerable 30 | Quoted from the corpus filings; every derived answer recomputed; checked with `python -m evals.own` |

FinanceBench has no unanswerable or multi-company questions, so the own set supplies them. Gold evidence is stored as text and matched to chunks at eval time (`evals/gold.py`), so labels survive re-chunking.

**Dev/test split** (`evals/split.json`, made by `python -m evals.split`, seed 2026): whole companies go to one side, so the two sides never share filing text. Companies linked by a multi-company question stay together, questions with no filing are assigned one at a time, and sides are balanced by (source, type). All tuning and error analysis use dev; test is scored once at the end. The file is committed and frozen: re-running keeps every existing assignment and only places new companies or questions, and fails if an existing question would change sides.

| Split | Companies | FinanceBench | Own | Unanswerable | Total |
|---|---|---|---|---|---|
| dev | 16 | 64 | 50 | 15 | 114 |
| test | 16 | 63 | 50 | 15 | 113 |

## Results

Every eval run is appended to `evals/results.jsonl` (commit, clean/dirty, model revisions, config, metrics by source and type); per-question details go to `data/evals/runs/`.

### Retrieval ablation (dev, 99 answerable questions)

`python -m evals.retrieval --note "..."`. recall@k: share of a question's evidence pieces with a gold chunk in the top k, averaged over questions; MRR: 1 / rank of the first gold chunk (0 if not in the top 20).

| Change | recall@1 | recall@5 | recall@10 | recall@20 | MRR | Run |
|---|---|---|---|---|---|---|
| Baseline: dense `bge-small-en-v1.5`, 400-token chunks, 50 overlap | 0.061 | 0.172 | 0.308 | 0.399 | 0.159 | `20261102-224225` |

Baseline by type (recall@5): table 0.267 (n=15, but recall@1 = 0), text 0.156 (n=32), computation 0.136 (n=44), multi-hop 0.250 (n=8). The generator sees the evidence for only ~1 in 6 facts, so retrieval caps answer accuracy.

Where the baseline's misses come from (`python -m evals.misses`, 115 evidence pieces not in the top 5): 89 (77%) had the right filing in the top 5 but not the evidence chunk; 12 (10%) got only other filings of the right company; 14 (12%) got no chunk from the right company. Across all questions, 55% of top-5 slots go to the wrong filing (38% right company, other filing; 17% other company).

## What failed and why

A running log of problems found, why they happened, and what was done about them.

### Parsing

| Problem | Cause | Status |
|---|---|---|
| Searching for "Item 1A" finds the table of contents, not the section | Every 10-K lists each Item twice: in the table of contents, then as the real heading | **Fixed:** keep the *last* occurrence of each Item heading, since the table of contents always comes first |
| Page footers (e.g. `Apple Inc. \| 2023 Form 10-K \| 19`) remain inside section text | Footers are ordinary text in the HTML; seen as Item 6 "[Reserved]" having 8 words instead of 0 | **Open:** footer formats differ by company, so the fix waits until more companies are ingested |
| Signatures page is attached to Item 16 | No Item heading follows the signatures to end the section | **Open:** harmless for now |
| 10-Qs lost 87–99% of their text (coverage 0.01–0.13) | 10-Q Items restart in Part II, so "keep the last Item 1" kept Part II's Legal Proceedings and dropped Part I (financial statements, MD&A) | **Fixed:** 10-Q sections are keyed by Part and Item (`I-1`, `I-2`, `II-1A`). Coverage now 0.92–0.99 |
| Microsoft's MD&A and financial statements were filed under the wrong Items (Item 7: 267 words) | Microsoft prints `Item 8` at the top of 41 pages; keep-last chose the last page | **Fixed:** heading lines repeated 3+ times are treated as page headers and ignored |
| Adobe 2017's Item 8 was a press-release paragraph | `Item 8.01. Other Events` (an 8-K item number quoted in the text) matched as Item 8 | **Fixed:** a dot followed by a digit is not a heading. The first attempt also rejected `Item 7—Title` (Costco) and accepted `Item 1, Business Segments, provides…` (a cross-reference); the final regex lists the allowed separators: `.` `:` dashes, space, end of line |
| CVS 2018's MD&A and financial statements are missing from the 10-K | Incorporated by reference from Exhibit 13 (the annual report) | **Fixed:** every 10-K's EDGAR index page is checked for `EX-13*` documents; CVS 2018 is the only one, parsed as section `EX-13.1` |
| Financial statements located after Item 15 are labeled Item 15/16 (Ulta, Corning, Netflix, PayPal, Activision, PepsiCo); JPMorgan's 10-K and 10-Qs put everything after a cross-reference index | Item 8 only says "see Item 15"; JPMorgan's filings are an annual report wrapped in a 10-K | **Open:** no text is lost (coverage ≥ 0.91 for every filing), only the Item label is wrong; fix when section filters are added (Week 3) |
| In some older filings a table row's label and numbers land on separate lines (Walmart 2018, Amazon 2017, General Mills 2020: `Accounts payable` then its values below) | Table cells contain their own block tags (`<p>`, `<div>`), so each cell gets a newline | **Open:** numbers are separated from their labels inside chunks; address with table-aware chunking (Week 3) |
| Cover pages were missing from the index, so questions like "which debt securities are registered on an exchange?" (3 FinanceBench questions) had no evidence anywhere | `split_sections` only kept text from the first Item heading onward | **Fixed:** text before the first heading becomes a `COVER` section (it also contains the table of contents). Found because the gold-evidence matcher found no chunk for those 3 questions |
| Pfizer's 10-Q has table-of-contents text under `II-3`/`II-4` | Those Items have no heading in the body, so their table-of-contents line is the last occurrence | **Open:** ~1,100 words of junk; harmless |

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
| CVS 2018 10-K is 0.9 MB (CVS 2022: 6.0 MB) | MD&A and financial statements are in Exhibit 13, incorporated by reference | **Fixed:** see Parsing |
| Some FinanceBench gold answers are wrong (Pfizer Upjohn costs: 77.78 instead of ~70) or contradict their justification (American Water Works working capital, Adobe operating margin) | Human labeling errors | **Measured:** a random sample of 20 in-scope questions (seed 2026, stratified by question type; `evals/audit.py`, verdicts in `evals/financebench_audit.csv`) found 18 correct, 1 wrong (Pfizer), 1 ambiguous (PepsiCo: the question asks for the income statement, the answer comes from the cash flow statement). Estimated error rate 5% (10% counting ambiguous); with only 20 questions the 95% interval is roughly 1–24%. Decision rule fixed in advance (≤ ~10% → primary eval set with corrections) is met. Failures are still checked against the filing during error analysis |
| A document-level dev/test split would leak | A company's 10-Ks repeat most of their text from year to year (risk factors, business description), so tuning on Apple 2023 and testing on Apple 2024 tests on near-duplicate text; Week 6 embedding fine-tuning on dev chunks would make it worse | **Avoided:** split by company instead (6 multi-company questions keep their companies together). Verified: 0 questions span both sides, and a second run changes nothing. Own table/text questions are slightly uneven (table 8 dev / 12 test, text 9 / 6) because companies move as units, so per-type scores are reported alongside the overall score |

### Chunking and retrieval

| Problem | Cause | Status |
|---|---|---|
| Chunk `fiscal_year` is taken from the calendar year of the fiscal year-end date | Simplification for the baseline; correct for Apple (September) and Microsoft (June) | **Open:** wrong for companies that name a fiscal year by its starting year (some retailers); use the filing's own fiscal year label when they are added |
| Some chunks contain nothing but a page footer (e.g. Item 6 "[Reserved]" → `Apple Inc. \| 2024 Form 10-K \| 19`) | Footers remain in section text (see Parsing), and every non-empty section produces at least one chunk | **Open:** junk chunks can take retrieval slots; fix footers and/or drop very short chunks, and measure the effect in Week 3 |
| Baseline recall@5 is 0.172; the assumed cause (retrieving the right company's other-year filings) was wrong | Measured with a miss breakdown: 77% of misses have the right filing in the top 5 but the wrong chunk of it. Dense embeddings match the question's concepts ("capital-intensive", "ROA") to prose that discusses them, not to the statement rows that contain the numbers | **Open (Week 3):** BM25 for exact terms, reranker for ranks 6–20, title-prefixed chunks; company/year filters for the 55% of top-5 slots taken by wrong filings (measure an oracle filter first as the upper bound). Lesson: measure the hypothesis before building the fix |
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
