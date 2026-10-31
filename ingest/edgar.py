import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.environ["SEC_USER_AGENT"]
CACHE_DIR = Path("data/raw")
MIN_INTERVAL = 0.15  # seconds between requests (~6/s), safely under SEC's 10/s limit

_last_request = 0.0


def fetch(url: str) -> bytes:
    """GET a URL from SEC, sleeping first if the previous request was too recent."""
    global _last_request
    wait = MIN_INTERVAL - (time.monotonic() - _last_request)
    if wait > 0:
        time.sleep(wait)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    _last_request = time.monotonic()
    response.raise_for_status()
    return response.content


def fetch_cached(url: str, path: Path) -> bytes:
    """Return the file at path if it exists, otherwise download it and save it there."""
    if path.exists():
        return path.read_bytes()
    content = fetch(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def get_cik(ticker: str) -> str:
    """Look up a ticker's CIK, zero-padded to 10 digits as EDGAR URLs expect."""
    raw = fetch_cached(
        "https://www.sec.gov/files/company_tickers.json",
        CACHE_DIR / "company_tickers.json",
    )
    for company in json.loads(raw).values():
        if company["ticker"] == ticker.upper():
            return str(company["cik_str"]).zfill(10)
    raise ValueError(f"Unknown ticker: {ticker}")


def get_submissions(cik: str) -> dict:
    """Return the company's filing index from the submissions API."""
    raw = fetch_cached(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        CACHE_DIR / cik / "submissions.json",
    )
    return json.loads(raw)


def list_filings(cik: str, form: str = "10-K", limit: int = 3) -> list[dict]:
    """Return the most recent filings of one form type, newest first."""
    recent = get_submissions(cik)["filings"]["recent"]
    filings = []
    for i, form_type in enumerate(recent["form"]):
        if form_type != form:
            continue
        accession = recent["accessionNumber"][i]
        document = recent["primaryDocument"][i]
        filings.append({
            "cik": cik,
            "form": form_type,
            "accession": accession,
            "filing_date": recent["filingDate"][i],
            "report_date": recent["reportDate"][i],
            "document": document,
            "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}",
        })
        if len(filings) == limit:
            break
    return filings


def download_filing(filing: dict) -> Path:
    """Download a filing's primary document and return the local path."""
    path = CACHE_DIR / filing["cik"] / filing["accession"] / filing["document"]
    fetch_cached(filing["url"], path)
    return path

def get_company_facts(cik: str) -> dict:
    """Return every XBRL fact the company has reported, from the companyfacts API."""
    raw = fetch_cached(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        CACHE_DIR / cik / "companyfacts.json",
    )
    return json.loads(raw)

if __name__ == "__main__":
    cik = get_cik("AAPL")
    for filing in list_filings(cik):
        path = download_filing(filing)
        print(filing["report_date"], filing["accession"], path)
