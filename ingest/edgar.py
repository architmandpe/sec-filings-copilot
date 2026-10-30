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


if __name__ == "__main__":
    print(get_cik("MSFT"))