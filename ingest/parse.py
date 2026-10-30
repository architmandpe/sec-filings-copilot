"""Convert EDGAR filing HTML into clean plain text."""
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

BLOCK_TAGS = ["div", "p", "br", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
ITEM_HEADING = re.compile(r"^item\s+(\d{1,2}[a-c]?)[.:]?\s*(.*)$", re.IGNORECASE)


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

def split_sections(text: str) -> list[dict]:
    """Split 10-K text into Items, keeping the last heading of each Item to skip the table of contents."""
    lines = text.splitlines()
    starts = {}
    for i, line in enumerate(lines):
        match = ITEM_HEADING.match(line)
        if match:
            starts[match.group(1).upper()] = (i, match.group(2))
    headings = sorted((start, item, title) for item, (start, title) in starts.items())
    sections = []
    for n, (start, item, title) in enumerate(headings):
        end = headings[n + 1][0] if n + 1 < len(headings) else len(lines)
        sections.append({"item": item, "title": title, "text": "\n".join(lines[start + 1:end])})
    return sections


if __name__ == "__main__":
    for path in sorted(Path("data/raw").glob("*/*/*.htm")):
        text = html_to_text(path.read_bytes())
        sections = split_sections(text)
        out_dir = Path("data/parsed") / path.parent.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.parent.name}.txt").write_text(text)
        (out_dir / f"{path.parent.name}.json").write_text(json.dumps(sections, indent=2, ensure_ascii=False))
        print(path.parent.name)
        for s in sections:
            print(f"  Item {s['item']:<3} {len(s['text'].split()):>6} words  {s['title'][:60]}")
