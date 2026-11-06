"""Work out which filings a question is about (company and fiscal year), so search can skip all the others.

Rules, designed on the dev questions (see README):
  company  every corpus company whose name appears in the question, ignoring case and punctuation
           ("AMCOR", "Coca Cola"), plus a few aliases ("JnJ"); a name that is also an ordinary word ("Block")
           must match with its capital letter. Several companies -> all of their filings.
  year     every year written as 2019, FY2019, FY 2019 or FY19. A year keeps that fiscal year's filings; if the
           company has none, the 10-Ks of the next two years, which repeat it as comparison columns. Years that
           match nothing (e.g. "expects to pay in 2024") are ignored.
  fallback no company found -> None (search everything); no year, or no year matched -> all the company's filings.
"""

import re

ALIASES = {"Johnson & Johnson": ["JnJ", "J&J"]}  # names beyond case and punctuation differences
CASE_SENSITIVE = {"Block"}  # company names that are also ordinary English words
YEAR = re.compile(r"(?<!\d)(20\d\d)(?!\d)|\bFY ?(\d\d)\b", re.IGNORECASE)
LATER_YEARS = 2  # a 10-K's statements also cover the two prior fiscal years


def normalize(text: str) -> str:
    """Lowercase, punctuation to spaces (keeping &), padded with spaces so matches fall on word boundaries."""
    return " " + re.sub(r"[^a-z0-9&]+", " ", text.lower()).strip() + " "


def years(question: str) -> set[int]:
    """Every year the question mentions: 2019, FY2019, FY 2019, FY19 -> 2019."""
    return {int(full) if full else 2000 + int(short) for full, short in YEAR.findall(question)}


class FilingFilter:
    """Knows the filings in the index (company and fiscal year of each) and picks the ones a question is about."""

    def __init__(self, chunks: list[dict]):
        self.filings = {c["doc_name"]: (c["company"], c["fiscal_year"]) for c in chunks}
        self.companies = sorted({company for company, _ in self.filings.values()})

    def find_companies(self, question: str) -> set[str]:
        found, text = set(), normalize(question)
        for company in self.companies:
            if company in CASE_SENSITIVE:
                if re.search(rf"\b{re.escape(company)}\b", question):
                    found.add(company)
            elif any(normalize(name) in text for name in [company] + ALIASES.get(company, [])):
                found.add(company)
        return found

    def infer(self, question: str, use_year: bool = True) -> set[str] | None:
        """The filings to search, or None if the question names no corpus company (then search everything)."""
        companies = self.find_companies(question)
        if not companies:
            return None
        by_company = {co: [(doc, fy) for doc, (c, fy) in self.filings.items() if c == co] for co in companies}
        everything = {doc for docs in by_company.values() for doc, _ in docs}
        asked = years(question) if use_year else set()
        if not asked:
            return everything
        keep = set()
        for docs in by_company.values():
            for year in asked:
                exact = [doc for doc, fy in docs if fy == year]
                later = [doc for doc, fy in docs if year < fy <= year + LATER_YEARS]
                keep.update(exact or later)
        return keep or everything
