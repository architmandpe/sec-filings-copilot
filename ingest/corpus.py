"""Download every filing in the FinanceBench manifest, matched to EDGAR accessions via the XBRL filings catalog."""

import duckdb

from evals.manifest import MANIFEST_PATH
from ingest.edgar import download_filing, find_filing
from ingest.xbrl import DB_PATH


def match_manifest() -> list[dict]:
    """Find each manifest filing's accession in the XBRL filings catalog."""
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        cursor = con.execute(f"""
            SELECT m.doc_name, m.cik, m.expected_accession, f.accession,
                   CAST(f.filed AS VARCHAR) AS filed, CAST(f.period_end AS VARCHAR) AS period_end
            FROM '{MANIFEST_PATH}' m
            LEFT JOIN filings f
              ON f.cik = m.cik AND f.form = m.form AND f.fiscal_period = m.fiscal_period AND f.fiscal_year = m.fiscal_year
            ORDER BY m.doc_name
        """)
        columns = [c[0] for c in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def check_matches(rows: list[dict]) -> None:
    """Stop unless every manifest filing matched exactly one accession, and the known ones match FinanceBench's."""
    with open(MANIFEST_PATH) as f:
        expected = sum(1 for _ in f)
    names = [r["doc_name"] for r in rows]
    if len(names) != expected or len(set(names)) != expected or any(r["accession"] is None for r in rows):
        raise ValueError(f"{len(rows)} matches for {expected} manifest filings; each needs exactly one")
    for r in rows:
        if r["expected_accession"] and r["accession"] != r["expected_accession"]:
            raise ValueError(f"{r['doc_name']}: matched {r['accession']}, FinanceBench used {r['expected_accession']}")


if __name__ == "__main__":
    rows = match_manifest()
    check_matches(rows)
    for r in rows:
        filing = find_filing(r["cik"], r["accession"], r["filed"])
        path = download_filing(filing)
        note = "" if filing["report_date"] == r["period_end"] else f"  report_date {filing['report_date']}"
        print(f"{r['doc_name']:<28} {r['accession']}  {r['period_end']}  {path.stat().st_size / 1e6:5.1f} MB{note}")
