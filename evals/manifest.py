"""Build the list of filings FinanceBench needs: one row per in-scope 10-K or 10-Q."""

import json
import re
from pathlib import Path

from ingest.edgar import get_cik, get_submissions

FINANCEBENCH_PATH = Path("data/evals/financebench.jsonl")
MANIFEST_PATH = Path("data/evals/manifest.jsonl")
FORMS = {"10k": "10-K", "10q": "10-Q"}  # FinanceBench doc_type → EDGAR form; 8-Ks and earnings releases are out of scope

CIK_IN_LINK = re.compile(r"CIK-(\d{10})")
ACCESSION_IN_LINK = re.compile(r"(\d{10}-\d{2}-\d{6})")
DOC_NAME = re.compile(r"_(\d{4})(Q[1-4])?_10[KQ]$")

# Companies none of whose FinanceBench links contain a CIK
TICKERS = {
    "3M": "MMM", "AMD": "AMD", "Adobe": "ADBE", "Amcor": "AMCR", "American Express": "AXP",
    "Coca-Cola": "KO", "JPMorgan": "JPM", "Johnson & Johnson": "JNJ", "Kraft Heinz": "KHC",
    "Lockheed Martin": "LMT", "Microsoft": "MSFT", "PepsiCo": "PEP", "Ulta Beauty": "ULTA", "Verizon": "VZ",
}
DELISTED_CIKS = {"Activision Blizzard": "0000718877"}  # acquired by Microsoft in 2023, so no ticker


def load_questions() -> list[dict]:
    with open(FINANCEBENCH_PATH) as f:
        return [json.loads(line) for line in f]


def company_ciks(questions: list[dict]) -> dict[str, str]:
    """Map each company to its CIK: from any FinanceBench link that contains one, otherwise by ticker."""
    ciks = dict(DELISTED_CIKS)
    for q in questions:
        match = CIK_IN_LINK.search(q["doc_link"])
        if match:
            cik = ciks.setdefault(q["company"], match.group(1))
            if cik != match.group(1):
                raise ValueError(f"{q['company']} has two CIKs: {cik} and {match.group(1)}")
    for company, ticker in TICKERS.items():
        if company not in ciks:
            ciks[company] = get_cik(ticker)
    return ciks


def build_manifest(questions: list[dict]) -> list[dict]:
    """Return one row per in-scope filing, sorted by doc_name."""
    ciks = company_ciks(questions)
    filings = {}
    for q in questions:
        if q["doc_type"] not in FORMS:
            continue
        year, quarter = DOC_NAME.search(q["doc_name"]).groups()
        accession = ACCESSION_IN_LINK.search(q["doc_link"])
        filings[q["doc_name"]] = {
            "doc_name": q["doc_name"],
            "company": q["company"],
            "cik": ciks[q["company"]],
            "form": FORMS[q["doc_type"]],
            "fiscal_year": int(year),
            "fiscal_period": quarter or "FY",
            "expected_accession": accession.group(1) if accession else None,
        }
    return sorted(filings.values(), key=lambda f: f["doc_name"])


if __name__ == "__main__":
    manifest = build_manifest(load_questions())
    with open(MANIFEST_PATH, "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")
    companies = {row["company"]: row["cik"] for row in manifest}
    known = sum(row["expected_accession"] is not None for row in manifest)
    print(f"{len(manifest)} filings, {len(companies)} companies, {len(set(companies.values()))} distinct CIKs, "
          f"{known} with a known accession")
    for company, cik in sorted(companies.items()):
        print(f"  {cik}  {company:<22} EDGAR: {get_submissions(cik)['name']}")
