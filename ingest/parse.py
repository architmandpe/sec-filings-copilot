"""Convert EDGAR filing HTML into clean plain text."""
import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

from ingest.corpus import CORPUS_PATH

BLOCK_TAGS = ["div", "p", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
# "Item 7.", "Item 7 Title", "Item 7—Title", but not 8-K item numbers quoted in the text ("Item 8.01. Other Events")
ITEM_HEADING = re.compile(r"^item\s+(\d{1,2}[a-c]?)(?:\.(?!\d)|[:—–-]|\s|$)\s*(.*)$", re.IGNORECASE)
PART_HEADING = re.compile(r"^part\s+(i{1,3}|iv)\b", re.IGNORECASE)
PAGE_HEADER_MIN_REPEATS = 3  # table of contents + body = 2; a line printed on many pages is a running page header


def html_to_text(html: bytes) -> str:
    """Return the visible text of a filing, one line per block element."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "ix:header"]):
        tag.decompose()
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_after("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" ")
    text = soup.get_text().replace("\xa0", " ")
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)

def split_sections(text: str, form: str) -> list[dict]:
    """Split filing text into Items, keeping the last heading of each Item to skip the table of contents.

    10-Q Items are numbered again in each Part (Part I Item 2 is MD&A, Part II Item 2 is share repurchases),
    so 10-Q Items are keyed by Part as well, e.g. "I-2".
    """
    lines = text.splitlines()
    repeats = Counter(lines)
    starts = {}
    part = "I"
    for i, line in enumerate(lines):
        part_match = PART_HEADING.match(line)
        if part_match:
            part = part_match.group(1).upper()
        match = ITEM_HEADING.match(line)
        if match and repeats[line] < PAGE_HEADER_MIN_REPEATS:
            item = match.group(1).upper()
            key = f"{part}-{item}" if form == "10-Q" else item
            starts[key] = (i, match.group(2))
    headings = sorted((start, item, title) for item, (start, title) in starts.items())
    sections = []
    for n, (start, item, title) in enumerate(headings):
        end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
        sections.append({"item": item, "title": title, "text": "\n".join(lines[start + 1:end])})
    return sections


if __name__ == "__main__":
    with open(CORPUS_PATH) as f:
        corpus = [json.loads(line) for line in f]
    for filing in corpus:
        folder = Path("data/raw") / filing["cik"] / filing["accession"]
        text = html_to_text((folder / filing["document"]).read_bytes())
        sections = split_sections(text, filing["form"])
        for exhibit in filing["exhibits"]:
            exhibit_text = html_to_text((folder / exhibit["document"]).read_bytes())
            sections.append({"item": exhibit["type"], "title": "Annual report (incorporated by reference)", "text": exhibit_text})
        out_dir = Path("data/parsed") / filing["cik"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{filing['accession']}.txt").write_text(text)
        (out_dir / f"{filing['accession']}.json").write_text(json.dumps(sections, indent=2, ensure_ascii=False))
        items = " ".join(s["item"] for s in sections)
        print(f"{filing['doc_name']:<28} {len(sections):>3} sections: {items}")
