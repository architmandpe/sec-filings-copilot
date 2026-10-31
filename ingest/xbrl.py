"""Flatten SEC companyfacts JSON into rows and load them into DuckDB."""

from pathlib import Path

import duckdb
import pandas as pd

from ingest.edgar import get_cik, get_company_facts

DB_PATH = Path("data/financials.duckdb")
COLUMNS = [
    "cik", "company", "taxonomy", "concept", "label", "unit",
    "period_start", "period_end", "value", "accession",
    "fiscal_year", "fiscal_period", "form", "filed", "frame",
]

ANNUAL_VIEW = """
CREATE OR REPLACE VIEW annual AS
WITH annual_facts AS (
    -- Full-year values from 10-Ks: durations of about one year, or instants (balance-sheet items)
    SELECT * FROM facts
    WHERE form = '10-K'
      AND (period_start IS NULL OR datediff('day', period_start, period_end) BETWEEN 350 AND 380)
),
fiscal_years AS (
    -- Each 10-K's own fiscal year ends on the latest full-year period it reports
    SELECT DISTINCT cik, max(period_end) AS period_end, any_value(fiscal_year) AS fiscal_year
    FROM annual_facts
    WHERE period_start IS NOT NULL
    GROUP BY cik, accession
)

SELECT
    f.cik, f.company, f.taxonomy, f.concept, f.unit, p.fiscal_year, f.period_start, f.period_end,
    arg_max(f.value, f.filed) AS value,  -- latest filing wins, so restatements replace originals
    arg_max(f.accession, f.filed) AS accession,
    max(f.filed) AS filed
FROM annual_facts f
JOIN fiscal_years p USING (cik, period_end)
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


def load_facts(tickers: list[str]) -> None:
    """Rebuild the facts table from the companyfacts data of the given companies."""
    rows = []
    for ticker in tickers:
        rows += flatten_facts(get_company_facts(get_cik(ticker)))
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
        con.execute(ANNUAL_VIEW)
        
        count = con.execute("SELECT count(*) FROM facts").fetchone()[0]
    print(f"Loaded {count} facts into {DB_PATH}")


if __name__ == "__main__":
    load_facts(["AAPL", "MSFT"])
