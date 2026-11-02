"""Find each question's gold chunks: the chunks of its evidence filing that contain the evidence text.

Gold is stored as evidence text, not chunk IDs, so it survives re-chunking. Both texts are reduced to
lowercase letters and digits, which removes the PDF artifacts in FinanceBench's evidence ("Costofgoodssold",
"OperatingÂ income", "Flow s"); the evidence is cut into fixed-size windows, and a chunk scores the number
of windows it contains.
"""

import json
import re
from collections import defaultdict

from evals.questions import QUESTIONS_PATH
from retrieval.index import INDEX_DIR

WINDOW = 40  # characters of normalized text, roughly 6-8 words
MIN_WINDOWS = 3  # a chunk must contain at least this many windows (or all of them, for short evidence)
RELATIVE = 0.5  # ... and at least half as many as the best-matching chunk


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def windows(text: str) -> list[str]:
    """Consecutive non-overlapping windows of the normalized text; a short quote is one window."""
    norm = normalize(text)
    if 0 < len(norm) < WINDOW:
        return [norm]
    return [norm[i:i + WINDOW] for i in range(0, len(norm) - WINDOW + 1, WINDOW)]


def match_evidence(evidence_text: str, chunks: list[dict]) -> tuple[list[str], float]:
    """Return the gold chunk IDs for one piece of evidence, and the coverage of the best chunk (0-1)."""
    ws = windows(evidence_text)
    if not ws:
        return [], 0.0
    scores = [(sum(w in c["norm"] for w in ws), c["id"]) for c in chunks]
    best = max((s for s, _ in scores), default=0)
    needed = min(MIN_WINDOWS, len(ws))
    gold = [cid for s, cid in scores if s >= needed and s >= RELATIVE * best]
    return gold, best / len(ws)


def load_chunks_by_doc() -> dict[str, list[dict]]:
    """Index chunks grouped by filing, each with its normalized text."""
    by_doc = defaultdict(list)
    with open(INDEX_DIR / "chunks.jsonl") as f:
        for chunk in map(json.loads, f):
            by_doc[chunk["doc_name"]].append({"id": chunk["id"], "norm": normalize(chunk["text"])})
    return by_doc


if __name__ == "__main__":
    by_doc = load_chunks_by_doc()
    with open(QUESTIONS_PATH) as f:
        questions = [json.loads(line) for line in f]
    rows = []
    for q in questions:
        for e in q["evidence"]:
            gold, coverage = match_evidence(e["text"], by_doc[e["doc_name"]])
            rows.append((coverage, len(gold), q["id"], e["doc_name"], len(windows(e["text"]))))
    rows.sort()
    found = sum(1 for r in rows if r[1] > 0)
    print(f"{len(rows)} evidence pieces in {len(questions)} questions; {found} with at least one gold chunk")
    for low, high in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        print(f"  best coverage {low:.1f}-{min(high, 1):.1f}: {sum(low <= r[0] < high for r in rows)}")
    print("\nLowest coverage:")
    for coverage, n_gold, qid, doc, n_windows in rows[:15]:
        print(f"  {coverage:.2f}  {n_gold:>2} gold  {qid:<10} {doc:<28} {n_windows} windows")
