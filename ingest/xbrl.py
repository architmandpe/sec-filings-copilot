"""Flatten SEC companyfacts JSON into rows and load them into DuckDB."""

import json
from pathlib import Path

import duckdb
import pandas as pd

from evals.manifest import MANIFEST_PATH
from ingest.edgar import get_company_facts

DB_PATH = Path("data/financials.duckdb")
EXTRA_CIKS = ["0000320193"]  # Apple: the Week 1 baseline company, not in FinanceBench

COLUMNS = [
    "cik", "company", "taxonomy", "concept", "label", "unit",
    "period_start", "period_end", "value", "accession",
    "fiscal_year", "fiscal_period", "form", "filed", "frame",
]

FILINGS_VIEW = """
CREATE OR REPLACE VIEW filings AS
WITH reports AS (
    -- One row per 10-K/10-Q. Its period ends on the latest year-long (10-K) or quarter-long (10-Q) duration
    -- it reports, ignoring durations that end after the filing date or have other lengths: forward-looking
    -- facts (next year's tax rate) and subsequent events (a dividend declared in February) would push it too far.
    SELECT cik, accession, any_value(form) AS form, any_value(fiscal_period) AS fiscal_period,
           min(filed) AS filed, max(period_end) AS period_end
    FROM facts
    WHERE taxonomy = 'us-gaap' AND period_end <= filed
      AND ((form = '10-K' AND datediff('day', period_start, period_end) BETWEEN 350 AND 380)
        OR (form = '10-Q' AND datediff('day', period_start, period_end) BETWEEN 75 AND 120))
    GROUP BY cik, accession
)
SELECT r.*,
    -- A fiscal year is named by the calendar year it ends in (Ulta's year ending January 2023 is FY2023),
    -- except that 52/53-week years ending in the first days of January belong to the previous year
    -- (J&J's year ending January 1, 2023 is FY2022); a 10-Q belongs to the fiscal year of the next
    -- 10-K year-end on or after its quarter
    year((SELECT min(k.period_end) FROM reports k
          WHERE k.cik = r.cik AND k.form = '10-K' AND k.period_end >= r.period_end) - INTERVAL 7 DAY) AS fiscal_year
FROM reports r
"""

ANNUAL_VIEW = """
CREATE OR REPLACE VIEW annual AS
WITH annual_facts AS (
    -- Full-year values from 10-Ks: durations of about one year, or instants (balance-sheet items)
    SELECT * FROM facts
    WHERE form = '10-K'
      AND (period_start IS NULL OR datediff('day', period_start, period_end) BETWEEN 350 AND 380)
)
SELECT
    f.cik, f.company, f.taxonomy, f.concept, f.unit, p.fiscal_year,
    arg_max(f.period_start, f.filed) AS period_start,  -- some filers start a year on the previous year-end (Dec 31 vs Jan 1)
    f.period_end,
    arg_max(f.value, f.filed) AS value,  -- latest filing wins, so restatements replace originals
    arg_max(f.accession, f.filed) AS accession,
    max(f.filed) AS filed
FROM annual_facts f
JOIN filings p ON p.cik = f.cik AND p.period_end = f.period_end AND p.form = '10-K'  -- only periods that end some 10-K's own fiscal year
GROUP BY ALL
"""


def flatten_facts(facts: dict) -> list[tuple]:
    """Turn the nested companyfacts JSON into one row per reported value."""
    cik = str(facts["cik"]).zfill(10)
    rows = []
    for taxonomy, concepts in facts["facts"].items():
        for concept, details in concepts.items():
            for unit, entries in details["units"].items():
                for e in entries:
                    rows.append((
                        cik, facts["entityName"], taxonomy, concept, details.get("label"), unit,
                        e.get("start"), e["end"], e["val"], e["accn"],
                        e.get("fy"), e.get("fp"), e["form"], e["filed"], e.get("frame"),
                    ))
    return rows


def load_facts(ciks: list[str]) -> None:
    """Rebuild the facts table from the companyfacts data of the given companies."""
    rows = []
    for cik in ciks:
        company_rows = flatten_facts(get_company_facts(cik))
        print(f"  {cik}: {len(company_rows)} facts")
        rows += company_rows
    df = pd.DataFrame(rows, columns=COLUMNS)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
            CREATE OR REPLACE TABLE facts AS
            SELECT * REPLACE (
                CAST(period_start AS DATE) AS period_start,
                CAST(period_end AS DATE) AS period_end,
                CAST(filed AS DATE) AS filed,
                CAST(fiscal_year AS INTEGER) AS fiscal_year
            )
            FROM df
        """)
        con.execute(FILINGS_VIEW)
        con.execute(ANNUAL_VIEW)
        count = con.execute("SELECT count(*) FROM facts").fetchone()[0]
    print(f"Loaded {count} facts into {DB_PATH}")


def corpus_ciks() -> list[str]:
    """Every company in the FinanceBench manifest, plus Apple."""
    with open(MANIFEST_PATH) as f:
        manifest_ciks = {json.loads(line)["cik"] for line in f}
    return sorted(manifest_ciks | set(EXTRA_CIKS))


if __name__ == "__main__":
    load_facts(corpus_ciks())
