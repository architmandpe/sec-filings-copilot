# SEC Filings Copilot

A fully local, open-source question-answering and analysis system over SEC 10-K and 10-Q filings.
Answers questions with cited evidence, computes financial metrics, and refuses when evidence is missing.

**Status:** Week 2: evaluation harness built and baseline measured (retrieval and answers). Week 3 (retrieval improvements) is next.

## At a glance

| | |
|---|---|
| Corpus | 75 SEC filings (10-K, 10-Q, one Exhibit 13) from 32 companies, 19,729 chunks; 894,619 XBRL facts in DuckDB |
| Eval set | 227 questions: 127 FinanceBench (audited) + 100 own (5 types, 30 unanswerable); company-level dev/test split, 114 / 113 |
| Models | Embeddings `BAAI/bge-small-en-v1.5`; generator `mlx-community/Qwen3-8B-4bit` on an M4 MacBook Air (24 GB), thinking off |
| Cost | $0: everything runs locally with open-source models |
| Baseline (dev) | retrieval recall@5 0.172, MRR 0.159; answer accuracy 0.243 (0.75 when the evidence was retrieved); refusal recall 1.0; citation validity 1.0 |

## Key findings

The main results so far, with the numbers behind them. Details are in [Results](#results) and [What failed and why](#what-failed-and-why).

1. **Retrieval, not the LLM, is the bottleneck.** For 88 of 99 answerable dev questions the evidence was not in the 5 chunks given to the LLM (recall@5 = 0.172). When it was, the LLM answered 6 of 8 scored questions correctly (0.75); when it was not, 0.18. 56 of 58 false refusals happened because retrieval missed the evidence, so the model was right to refuse.
2. **The obvious explanation for retrieval misses was wrong, and measuring showed it.** Expected cause: the right company's filing from the wrong year. A miss breakdown showed only 10% of misses are wrong-year; 77% have the right filing in the top 5 but the wrong chunk of it. Dense embeddings rank prose that *discusses* a concept ("capital-intensive") above the statement rows that *contain* the numbers. Table evidence is never ranked first (recall@1 = 0) but is often in the top 20 (0.60). This reorders the Week 3 plan: within-filing ranking (BM25, reranker, chunk titles) before year filters.
3. **Raising k does not fix it.** 60% of evidence ranks below position 20 of 19,729 chunks; k = 20 would raise recall only from 0.17 to 0.40 at 4× the prompt length.
12. **Choosing the right filing is worth at most +0.14 recall@5; ranking inside the filing is the bigger problem.** An oracle filter that searches only the evidence filings (from the gold labels: an upper bound no real system can reach) raised recall@5 from 0.172 to 0.308 (+0.136, 95% CI [+0.076, +0.202], paired bootstrap; 16 questions better). More than predicted: removing other filings' chunks also promotes right-filing chunks that ranked 6–10, so filters help even the "right filing, wrong chunk" misses. But even searching only the right filing (median 240 chunks instead of 19,729), 69% of evidence is outside the top 5 and 39% outside the top 20, and table evidence is still never ranked first (recall@1 = 0). Most of the remaining gap needs better ranking (BM25, reranker).
13. **A rule-based company + fiscal-year filter reaches the oracle.** Company names (case and punctuation ignored, 2 aliases) and years (2019, FY2019, FY19; a year without its own filing maps to the next two 10-Ks, which repeat it) inferred from the question text kept the evidence filing searchable for 99 of 99 dev questions while searching 1.3 filings on average instead of 75. recall@5 0.172 → 0.308 (+0.136, 95% CI [+0.076, +0.202]), identical to the oracle at k ≤ 5 (it trails only at k = 20: 0.556 vs 0.611, from the extra filings). The year matters: company alone gives +0.040, because one company's filings from different years repeat the same text. Caveat: the rules were designed on dev questions, so the test set will show how well they generalize.
14. **BM25 and an off-the-shelf reranker did not improve the top 5; they improved the top 20.** On top of the filter, no combination changed recall@5 beyond noise: hybrid 0.288, dense + reranker 0.303, hybrid + reranker 0.323, vs 0.308 (all 95% intervals include 0). The reranker reshuffles rather than improves: 18 questions better and 18 worse. It helps text questions (recall@5 0.281 → 0.406, not significant on 32) and recall@1 (0.091 → 0.146), and hurts computation questions, where it prefers segment and notes tables over the primary statements. Hybrid + reranker does raise recall@20 from 0.556 to 0.672 (+0.116, CI [+0.025, +0.212]), so the evidence is increasingly *near* the top, not in it. Two causes seen in the misses: question and table use different words ("capital expenditure" vs "Purchases of property, plant and equipment"), and tables are flattened one cell per line (`$ / 1,747 / $ / 119`), which neither the embedding model nor the reranker (trained on prose) reads well. Decision: keep filter + dense (fastest, no worse) and fix how tables are represented next.
4. **Generator behavior is trustworthy.** 100% of cited chunk IDs were actually retrieved (no invented citations), 98% of answers cite a source, and all 15 unanswerable dev questions were refused. Refusal recall must be rechecked once retrieval improves (today the model refuses 64% of all questions).
5. **Benchmark labels were audited before being trusted.** A stratified random sample of 20 FinanceBench answers checked against the filings: 18 correct, 1 wrong, 1 ambiguous (5–10% error, 95% interval ~1–24%). The decision rule (≤ ~10% → use as primary set, with corrections) was fixed before the audit.
6. **The dev/test split is by company, not by document.** A company's 10-Ks repeat most text year to year, so a document-level split would leak near-duplicate text from dev into test (and into Week 6 embedding fine-tuning). Verified: 0 questions span both sides; the split file is frozen in git.
7. **The gold labels are a lower bound.** 12 answers were correct although the labeled evidence chunk was not retrieved: the same number also appears in next year's 10-K (comparison column) or elsewhere in the same filing. True retrieval quality is somewhat higher than recall@k shows.
8. **SEC data has traps that silently produce wrong numbers.** XBRL `fy` is the filing's fiscal year, not the period's (each 10-K repeats 2 prior years); concept names change (revenue after ASC 606); "latest filing wins" returns restated values (J&J FY2022 revenue $94.9B as filed vs $80.0B restated); "fiscal 2023" means different calendar periods at Ulta, Best Buy and J&J. Each was fixed and verified against EDGAR (derived period end = EDGAR `reportDate` for all 72 filings; 0 duplicate facts).
9. **A valid citation is not a supporting citation.** Citation validity was 1.0 (every cited chunk was retrieved), but labeling the 41 answered baseline questions found only 16 (39%) faithful: in 25, at least one claim is not stated in its cited chunk. 8 answers were correct but unsupported by their citations. The most common cause (11 of 25) is the wrong period: a number read from the wrong column of a multi-year table, or from another year's filing. Labels by Claude following a written guide (`evals/judge_labels.csv`), verified by a human.
11. **An LLM judge was validated against labels before being used.** The judge (Gemma 4 12B, a different model family from the Qwen generator, to avoid a model favoring its own answers) graded the same 41 answers. Rule fixed in advance: trust a criterion if agreement ≥ 0.85 and Cohen's kappa ≥ 0.6. **Correctness passed** (agreement 0.951, kappa 0.902; 11 of 11 free-text answers, which the automatic scorer cannot grade at all). **Faithfulness failed** with the first prompt (agreement 0.805, kappa 0.624): all 8 disagreements were the judge calling an unsupported answer faithful, so it overstated faithfulness (59% vs 39% labeled). It checked that numbers appear in the excerpts, not that their period, scope (segment vs total) and direction match: the generator's most common errors. One allowed revision made the judge list each claim with its period and scope next to the supporting text: agreement 0.854, kappa 0.678 (borderline pass), lenient errors down from 8 to 1, now slightly strict (29% faithful vs 39% labeled). Judge faithfulness is a slightly conservative estimate.
10. **Thinking mode is off for a measured reason.** The same answer took 375 tokens with Qwen3 thinking on vs 54 off (~7× slower). On this laptop, prompt processing runs at ~750 tokens/s but generation at ~26 tokens/s, so long contexts are cheap and long answers are expensive. Peak memory 5.6 GB.

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

Evaluation:

```
uv run python -m evals.own         # check the hand-written questions (format, evidence found in the filings)
uv run python -m evals.split       # assign questions to dev/test (frozen: only places new ones)
uv run python -m evals.retrieval --note "..."   # recall@k and MRR on dev, logged to evals/results.jsonl
uv run python -m evals.misses      # where the latest retrieval run's misses came from
uv run python -m evals.answers --note "..."     # full RAG on dev (~18 min): accuracy, refusals, citations
uv run python -m evals.answers --limit 5        # quick test on 5 random questions (not logged)
uv run python -m evals.judge --run <answers run>  # LLM judge: correctness + faithfulness (~40 min)
uv run python -m evals.compare <run A> <run B>    # is B better than A? paired bootstrap 95% CI per metric
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

Every eval run is appended to `evals/results.jsonl` (commit, clean/dirty, model revisions, config, metrics by source and type, and a **fingerprint**: a hash of the scored questions, their gold answers and evidence, and the scoring settings; runs are comparable only when fingerprints match). Per-question details of logged runs are committed in `evals/runs/`, so any result can be recomputed and two runs compared question by question; quick `--limit` test runs go to the gitignored `data/evals/runs/`. Runs before 2026-11-06 have no fingerprint.

### Retrieval ablation (dev, 99 answerable questions)

`python -m evals.retrieval --note "..."`. recall@k: share of a question's evidence pieces with a gold chunk in the top k, averaged over questions; MRR: 1 / rank of the first gold chunk (0 if not in the top 20).

| Change | recall@1 | recall@5 | recall@10 | recall@20 | MRR | Run |
|---|---|---|---|---|---|---|
| Baseline: dense `bge-small-en-v1.5`, 400-token chunks, 50 overlap | 0.061 | 0.172 | 0.308 | 0.399 | 0.159 | `20261102-224225`, rerun `20261106-122253` (fingerprint `5c6cd2004fb8`) |
| *Upper bound, not a real system:* oracle filing filter (search only the evidence filings, taken from gold labels) | 0.091 | 0.308 | 0.465 | 0.611 | 0.242 | `20261106-124301`; recall@5 +0.136, 95% CI [+0.076, +0.202] |
| Company filter (companies named in the question → their filings) | 0.071 | 0.212 | 0.333 | 0.449 | 0.177 | `20261106-125639`; recall@5 +0.040 [+0.010, +0.081] |
| **Company + fiscal-year filter** (`retrieval/filters.py`) | 0.091 | **0.308** | 0.439 | 0.556 | 0.235 | `20261106-125643`; recall@5 +0.136 [+0.076, +0.202]; equals the oracle at k ≤ 5 |
| Filter + hybrid: dense + hand-written BM25, reciprocal rank fusion (k = 60) | 0.081 | 0.288 | 0.439 | 0.625 | 0.222 | `20261106-142816`; vs filter: recall@5 −0.020 [−0.091, +0.051], recall@20 +0.069 [−0.013, +0.152] |
| Filter + dense + reranker (`bge-reranker-base`, top 50) | 0.146 | 0.303 | 0.490 | 0.641 | 0.257 | `20261106-143927`; vs filter: recall@5 −0.005 [−0.096, +0.086]; 2.4 s/question |
| Filter + hybrid + reranker (top 50) | 0.146 | 0.323 | 0.475 | 0.672 | 0.263 | `20261106-143520`; vs filter: recall@5 +0.015 [−0.081, +0.116] (18 better, 18 worse), recall@20 **+0.116 [+0.025, +0.212]**; 1.9 s/question |

Search time per question: dense or hybrid ~0.01 s; with the reranker ~2 s (50 cross-encoder passes on the M4 GPU).

The retrieval eval is exactly reproducible: the rerun returned the same top-20 chunk list for 99 of 99 questions, so differences between Week 3 runs come from the change being tested, not from run-to-run noise in the system. (Noise from having only 99 questions still applies; `evals/compare.py` measures it.)

**Comparing runs** (`python -m evals.compare <A> <B>`): for each question, B's score minus A's; the mean is the improvement, and a paired bootstrap (resample the questions with replacement 10,000 times) gives its 95% confidence interval. A change counts as an improvement only if the interval is above 0. It refuses runs with different fingerprints. Checks: baseline vs its rerun gives +0.000 [0, 0] on every metric; a simulated recall@5 gain of +0.136 (17 questions better) gives [+0.076, +0.202]; a single improved question (+0.010) gives [0.000, +0.030], not significant. On 99 questions, gains smaller than about 0.03–0.05 in recall@5 will be hard to separate from noise.

Baseline by type (recall@5): table 0.267 (n=15, but recall@1 = 0), text 0.156 (n=32), computation 0.136 (n=44), multi-hop 0.250 (n=8). The generator sees the evidence for only ~1 in 6 facts, so retrieval caps answer accuracy.

Where the baseline's misses come from (`python -m evals.misses`, 115 evidence pieces not in the top 5): 89 (77%) had the right filing in the top 5 but not the evidence chunk; 12 (10%) got only other filings of the right company; 14 (12%) got no chunk from the right company. Across all questions, 55% of top-5 slots go to the wrong filing (38% right company, other filing; 17% other company).

### Answers (dev, 114 questions)

`python -m evals.answers --note "..."`. Accuracy covers number and yes/no answers (a refusal counts as wrong); free-text answers wait for an LLM judge. acc|found / acc|miss: accuracy when retrieval did / did not put all the evidence in the top 5.

| Change | Accuracy | acc\|found | acc\|miss | False refusals | Refusal recall | Refusal precision | Citation rate | Citation validity | Run |
|---|---|---|---|---|---|---|---|---|---|
| Baseline: dense top-5, `Qwen3-8B-4bit`, thinking off | 0.243 | 0.75 | 0.182 | 0.586 | 1.0 | 0.205 | 0.976 | 1.0 | `20261104-233612` |

### LLM judge validation (41 answered baseline questions)

`python -m evals.judge --run <answer run>`. Judge `mlx-community/gemma-4-12B-it-qat-4bit`, compared with the labels in `evals/judge_labels.csv`. Rule fixed in advance: use a criterion's judge scores only if agreement ≥ 0.85 and kappa ≥ 0.6; one prompt revision allowed, both reported. An unparsed verdict (no `VERDICT:` line) counts as "n".

Faithfulness v2 notes: 2 of 41 replies hit the 800-token limit before the verdict and count as "n" (a rule chosen after seeing this run, stated here for that reason). Leaving them out gives 0.872 / 0.719 (n=39); counting both as disagreements gives 0.829, a fail. With 41 answers one or two verdicts move agreement across 0.85, so the pass is borderline. v2 fixed 6 of v1's 8 lenient verdicts; its remaining errors are mostly strict: it demands a quote for a conclusion drawn from cited facts ("capital intensive" from $23.1B capex; "North America" for "United States and Canada"). The limit is now 1,200 tokens. Correctness verdicts were identical in both runs (greedy decoding). The v2 numbers were recomputed from that run's per-question verdicts with the fail-closed rule; its entry in `evals/results.jsonl` shows the earlier calculation (0.872 / 0.719, unparsed left out). The judge was not rerun with the 1,200-token limit: greedy decoding makes the 39 parsed verdicts identical, so only the 2 cut-off answers could change.

| Criterion | Prompt | Agreement | Kappa | Judge y, label n | Judge n, label y | Judge y rate (labels) | Verdict | Run |
|---|---|---|---|---|---|---|---|---|
| Correct | v1 | 0.951 | 0.902 | 1 | 1 | 0.537 (0.537) | **Trusted** | `20261105-220148` |
| Faithful | v1: single verdict | 0.805 | 0.624 | 8 | 0 | 0.585 (0.390) | Failed (too lenient) | `20261105-220148` |
| Faithful | v2: claim by claim | 0.854 | 0.678 | 1 | 5 | 0.293 (0.390) | **Trusted, borderline** | `20261105-230101` |

Of 99 answerable questions, retrieval found all the evidence for only 11 (6 correct, 2 refused, 3 free text). For the other 88, the model refused 56: correct behavior for what it was shown. So 56 of the 58 false refusals are retrieval failures; the generator follows the refusal rule and never cites a chunk it was not given.

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
| A run made right after another was logged as "dirty" although its code was committed (`20261106-125643`, code = commit `3ca5d81`) | Run details became committed files in `evals/runs/`, so the previous run's new, uncommitted details file counted as an uncommitted change | **Fixed:** the dirty check ignores eval outputs (`evals/results.jsonl` and `evals/runs/`) |

### Chunking and retrieval

| Problem | Cause | Status |
|---|---|---|
| Chunk `fiscal_year` is taken from the calendar year of the fiscal year-end date | Simplification for the baseline; correct for Apple (September) and Microsoft (June) | **Open:** wrong for companies that name a fiscal year by its starting year (some retailers); use the filing's own fiscal year label when they are added |
| Some chunks contain nothing but a page footer (e.g. Item 6 "[Reserved]" → `Apple Inc. \| 2024 Form 10-K \| 19`) | Footers remain in section text (see Parsing), and every non-empty section produces at least one chunk | **Open:** junk chunks can take retrieval slots; fix footers and/or drop very short chunks, and measure the effect in Week 3 |
| Baseline recall@5 is 0.172; the assumed cause (retrieving the right company's other-year filings) was wrong | Measured with a miss breakdown: 77% of misses have the right filing in the top 5 but the wrong chunk of it. Dense embeddings match the question's concepts ("capital-intensive", "ROA") to prose that discusses them, not to the statement rows that contain the numbers | **Open (Week 3):** BM25 for exact terms, reranker for ranks 6–20, title-prefixed chunks; company/year filters for the 55% of top-5 slots taken by wrong filings (measure an oracle filter first as the upper bound). Lesson: measure the hypothesis before building the fix |
| Retrieval scores cannot tell answerable questions from unanswerable ones (CEO pay: top score 0.750, supply chain: 0.754) | Dense search always returns the k nearest chunks, however unrelated | **By design:** refusal must come from the generator (and later a reranker score), not a similarity threshold |
| Other filings' chunks pushed the right filing's chunks out of the top 5 (55% of top-5 slots went to other filings; 38% to the same company's other years) | No metadata filtering: one company's 10-Ks repeat the same text every year, and dense search can't tell the years apart | **Fixed** with a company + fiscal-year filter inferred from the question: recall@5 0.172 → 0.308, equal to the oracle at k ≤ 5 |
| BM25 + RRF lowered the top 5 (recall@5 0.308 → 0.288) while raising the top 20 (0.556 → 0.625) | The two lists find different chunks; fusion promotes BM25-only hits and pushes some dense top-5 hits down. Keyword matching can't bridge finance synonyms ("capital expenditure" vs "Purchases of PP&E") | **Measured:** kept as an option (`--search hybrid`) for candidate generation, not the default |
| An off-the-shelf cross-encoder (`bge-reranker-base`) reshuffled the top 5 without improving it (18 questions better, 18 worse) | Trained on prose passages; our tables are flattened one cell per line, and FinanceBench questions carry long instructions ("Round your answer to two decimal places…") it treats as content. It prefers segment/notes tables over primary statements (`fb-02987`: Activision balance sheet pushed out of the top 20) | **Open:** fix table representation (one row per line, cells joined), then re-test; candidates: a larger reranker (`bge-reranker-v2-m3`), stripping answer-format instructions from the query |
| Tables reach the index one cell per line (`Net cash provided by operating activities / 6,439 / 6,240 / 6,662`) | `html_to_text` breaks the line after every block tag (`p`, `div`), and many filings wrap each cell's content in one: 99.6% of cells in 3M 2018, 86% in Activision 2019 and Coca-Cola 2017, but 5% in Apple 2024 (whose rows stay on one line) | **Open (next step):** render each table row on one line with cells separated |

### Answer generation

| Problem | Cause | Status |
|---|---|---|
| Supply-chain question answered partly with a financial-risk chunk (credit risk, counterparty failure), cited correctly but irrelevant | Dense retrieval returned macroeconomic Risk Factor chunks, and the model listed every excerpt as part of the answer | **Open:** measure answer relevance separately from citation faithfulness in Week 2; try a reranker in Week 3 |
| Answers mix fiscal years without saying so (supply-chain answer cites FY2024 and FY2025 chunks) | No fiscal-year filter in retrieval; near-duplicate passages exist in every year's 10-K | **Open:** fiscal-year metadata filter in Week 3 |
| Refusals tested only on questions where no retrieved chunk was relevant (CEO pay, 2030 revenue) | Baseline smoke test had 4 questions | **Measured:** false refusal rate 0.586 on dev, but 56 of 58 happen when retrieval missed the evidence (refusing is then correct). Refusal recall 1.0 is easy while the model refuses 64% of everything; recheck it once retrieval improves |
| 12 answers scored correct although "evidence missing" from the top 5 | Gold chunks are the chunks containing the quoted passage, but the same number appears elsewhere: next year's 10-K repeats it as a comparison column (`own-002` cited `APPLE_2025_10K`), and revenue appears in the income statement, segment tables and MD&A (`own-006`) | **Open:** recall@k and "evidence found" are lower bounds on true retrieval quality; consider accepting any chunk of the right filing that contains the answer value as gold (Week 3) |
| Free cash flow question refused although the cash flow statement was retrieved (`fb-04854`) | FCF is not printed in a 10-K; it is computed (operating cash flow − capital expenditures), and the model did not know the definition | **Open:** a generator failure; Week 4 calculator tool / metric definitions |
| Citation validity 1.0, yet only 16 of 41 answered questions are faithful to their citations | Validity only checks that a cited ID was retrieved, not that the chunk states the claim. Unsupported claims come from: the wrong period (11: wrong column of a multi-year table or another year's filing, e.g. `fb-04417`, `fb-00859`, `own-064`), a segment table read as company totals (`own-015`, `own-035`), numbers absent from the cited chunk (`own-025`, `own-036`, `own-068`) | **Measured** with labels (`evals/judge_labels.csv`); the LLM faithfulness judge failed validation with prompt v1 (see below) and is being revised. Fixes: year filter (Week 3), numbers from XBRL with explicit periods (Week 4), verifier (Week 7) |
| The automatic number/yes-no scorer disagrees with the labels on 3 of 30 answers | 1% tolerance accepts 18.00 when the question asks for 2 decimals (17.98, `fb-05915`); a "yes" conveyed without the word is missed (`fb-00790`); a correct "yes" with a wrong supporting number passes (`fb-01244`) | **Open:** 90% agreement; tighten the tolerance for questions that specify rounding |
| An answer was cut off mid-calculation and its citations were not counted (`fb-02981`) | It hit `MAX_TOKENS = 512`, and it cited sources in parentheses instead of square brackets | **Open:** check how often answers hit the token limit; accept or forbid other citation formats |
| LLM faithfulness judge (prompt v1) called 8 unsupported answers faithful, and never the reverse | It checked that the answer's numbers appear in the excerpts, not whether they belong to the stated period (`fb-01107`, `fb-00566`, `own-067`), scope (`own-015`: segment revenue as total) or direction (`fb-00283`: received vs pay; `own-071`: "did not decline" next to a 3% decrease), nor uncited inputs (`own-036`, `fb-03282`) | **Fixed (v2):** the judge lists each claim with its period and scope next to the supporting text and the period and scope of that text. 6 of 8 fixed; `fb-00566` still lenient (each number is real but the reasoning uses the wrong pair of years); v2 is slightly strict instead (5 answers judged unfaithful that were labeled faithful) |
| 2 of 41 v2 faithfulness replies had no verdict (`fb-01902`, `fb-01107`) | The claim list plus self-rechecking ran past `FAITHFUL_MAX_TOKENS = 800` | **Fixed:** unparsed counts as "n" (fail closed); limit raised to 1,200 |
| The correctness judge missed a self-contradicting answer (`own-071`) and an implicit "yes" (`fb-00790`) | Same blind spots as the automatic scorer for `fb-00790` | **Accepted:** 2 of 41 disagreements, within the pre-set threshold |
| A partial calculation followed by a refusal counts as a refusal (`fb-02608`) | `is_refusal` looks for the refusal sentence anywhere in the answer | **By design for now:** the answer is wrong either way; revisit if mixed answers become common |

Verified: the baseline's answer for Apple's FY2024 R&D ($31,370 million, from the 10-K text) matches the XBRL fact in DuckDB.

### Data limitations

| Limitation | Consequence |
|---|---|
| Items 10–14 (directors, executive compensation, ownership) are incorporated by reference from the proxy statement (DEF 14A), which is not ingested | Questions such as executive pay cannot be answered from 10-Ks; the system should refuse them |
| XBRL concept names change over time (e.g. revenue moved from `SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax` with the ASC 606 standard in 2018) | A query on one concept name silently misses older years; metrics need a mapping to several candidate concepts |
| XBRL companyfacts also contain executive pay (`ecd` taxonomy) for 8 companies, taken from proxy statements, not 10-Ks | The Week 4 SQL tool must expose only 10-K/10-Q facts, or proxy questions that should be refused would be answered from data outside the corpus |

### Model choice

| Measurement | Result | Decision |
|---|---|---|
| Qwen3-8B (4-bit, MLX) with thinking on vs off, same question | 375 vs 54 generated tokens for the same answer (~7× slower) | Thinking off for the baseline; whether it helps multi-hop questions is tested in Week 4 |
| Speed on an M4 MacBook Air, ~2.3–3k-token prompts | Prompt processing ~720–777 tokens/s (3–4 s to first token); generation ~25–27 tokens/s; peak memory 5.6 GB | Long contexts are cheap and long answers expensive: prompt processing is parallel (compute-bound), generation is one token at a time (memory-bandwidth-bound) |
| Full dev answer eval | ~9 s per question, 114 questions in ~18 minutes | Fast enough to rerun after every change |
