"""Build the evaluation corpus: match each needed filing to its EDGAR accession, download it, and list it in data/corpus.jsonl."""

import json
from pathlib import Path

import duckdb
import pandas as pd

from evals.manifest import MANIFEST_PATH
from ingest.edgar import CACHE_DIR, download_filing, fetch_cached, find_filing, list_documents
from ingest.xbrl import DB_PATH

CORPUS_PATH = Path("data/corpus.jsonl")

# Apple's last three 10-Ks: the Week 1 baseline, not in FinanceBench; their accessions double as known answers
EXTRA_FILINGS = [
    {"doc_name": f"APPLE_{year}_10K", "company": "Apple", "cik": "0000320193", "form": "10-K",
     "fiscal_year": year, "fiscal_period": "FY", "expected_accession": accession}
    for year, accession in [(2023, "0000320193-23-000106"), (2024, "0000320193-24-000123"), (2025, "0000320193-25-000079")]
]


def load_needed() -> list[dict]:
    """Return every filing the corpus needs: the FinanceBench manifest plus the extra filings."""
    with open(MANIFEST_PATH) as f:
        return [json.loads(line) for line in f] + EXTRA_FILINGS


def match_filings(needed: list[dict]) -> list[dict]:
    """Find each needed filing's accession in the XBRL filings catalog."""
    m = pd.DataFrame(needed)
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        cursor = con.execute("""
            SELECT m.doc_name, m.company, m.cik, m.form, m.fiscal_year, m.fiscal_period, m.expected_accession,
                   f.accession, CAST(f.filed AS VARCHAR) AS filed, CAST(f.period_end AS VARCHAR) AS period_end
            FROM m
            LEFT JOIN filings f
              ON f.cik = m.cik AND f.form = m.form AND f.fiscal_period = m.fiscal_period AND f.fiscal_year = m.fiscal_year
            ORDER BY m.doc_name
        """)
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def check_matches(needed: list[dict], rows: list[dict]) -> None:
    """Stop unless every needed filing matched exactly one accession, and the known ones match."""
    names = [r["doc_name"] for r in rows]
    if len(names) != len(needed) or len(set(names)) != len(needed) or any(r["accession"] is None for r in rows):
        raise ValueError(f"{len(rows)} matches for {len(needed)} needed filings; each needs exactly one")
    for r in rows:
        if r["expected_accession"] and r["accession"] != r["expected_accession"]:
            raise ValueError(f"{r['doc_name']}: matched {r['accession']}, expected {r['expected_accession']}")


def download_exhibits(filing: dict) -> list[dict]:
    """Download a 10-K's Exhibit 13 (annual report), where some companies put MD&A and financial statements."""
    if filing["form"] != "10-K":
        return []
    exhibits = [d for d in list_documents(filing) if d["type"].startswith("EX-13")]
    for d in exhibits:
        fetch_cached(d["url"], CACHE_DIR / filing["cik"] / filing["accession"] / d["document"])
    return [{"type": d["type"], "document": d["document"]} for d in exhibits]


if __name__ == "__main__":
    needed = load_needed()
    rows = match_filings(needed)
    check_matches(needed, rows)
    with open(CORPUS_PATH, "w") as f:
        for r in rows:
            filing = find_filing(r["cik"], r["accession"], r["filed"])
            path = download_filing(filing)
            exhibits = download_exhibits(filing)
            entry = {k: r[k] for k in ("doc_name", "company", "cik", "form", "fiscal_year", "fiscal_period", "accession", "period_end")}
            f.write(json.dumps({**entry, "document": filing["document"], "url": filing["url"], "exhibits": exhibits}) + "\n")
            note = "" if filing["report_date"] == r["period_end"] else f"  report_date {filing['report_date']}"
            extra = "  + " + ", ".join(e["type"] for e in exhibits) if exhibits else ""
            print(f"{r['doc_name']:<28} {r['accession']}  {r['period_end']}  {path.stat().st_size / 1e6:5.1f} MB{extra}{note}")
