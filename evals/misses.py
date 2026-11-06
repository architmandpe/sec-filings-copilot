"""Retrieval miss breakdown: when an evidence piece is not in the top k, where did the top k come from?

Reads the per-question details of a retrieval run (no new retrieval), so it explains a logged score.
"""

import argparse
import json
from collections import Counter

from evals.gold import load_chunks_by_doc, match_evidence, normalize
from evals.runlog import RUNS_DIR, run_path
from evals.split import load_questions
from retrieval.index import INDEX_DIR

K = 5  # what the generator sees
CATEGORIES = {
    "right_filing": "right filing in top k, but not the evidence chunk",
    "right_company": "right company, but only other filings (other years / 10-Qs)",
    "other_company": "no chunk from the right company",
}


def classify(gold_doc: str, company: str, top: list[dict]) -> str:
    """The closest the top k got to the evidence."""
    if any(c["doc_name"] == gold_doc for c in top):
        return "right_filing"
    if any(c["company"] == company for c in top):
        return "right_company"
    return "other_company"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run id, e.g. 20261102-224225-retrieval (default: the latest retrieval run)")
    args = parser.parse_args()
    path = run_path(args.run) if args.run else max(RUNS_DIR.glob("*-retrieval.jsonl"))
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    questions = {q["id"]: q for q in load_questions()}
    with open(INDEX_DIR / "chunks.jsonl") as f:
        chunks = {c["id"]: c for c in map(json.loads, f)}
    by_doc = load_chunks_by_doc()

    pieces, misses, slots = 0, [], Counter()
    for row in rows:
        q = questions[row["id"]]
        top = [chunks[cid] for cid in row["retrieved"][:K]]
        companies = {chunks[by_doc[d][0]["id"]]["company"] for d in q["filings"]}
        for c in top:  # where every top-k slot came from, hit or miss
            slots["right filing(s)" if c["doc_name"] in q["filings"] else
                  "right company, other filing" if c["company"] in companies else "other company"] += 1
        for e in q["evidence"]:
            gold = set(match_evidence(e["text"], by_doc[e["doc_name"]])[0])
            if not gold:
                continue
            pieces += 1
            if gold & {c["id"] for c in top}:
                continue
            company = chunks[next(iter(gold))]["company"]
            others = [{"id": c["id"], "norm": normalize(c["text"])} for c in top if c["doc_name"] != e["doc_name"]]
            misses.append({
                "id": q["id"], "source": q["source"], "type": q["type"], "question": q["question"],
                "gold_doc": e["doc_name"], "category": classify(e["doc_name"], company, top),
                "evidence_elsewhere": bool(match_evidence(e["text"], others)[0]) if others else False,
                "top_docs": [c["doc_name"] for c in top],
            })

    print(f"Run {path.stem}: {len(rows)} questions, {pieces} evidence pieces, {len(misses)} not in the top {K}\n")
    print(f"Where the top {K} slots came from (all {sum(slots.values())} slots):")
    for name, n in slots.most_common():
        print(f"  {n:>4}  {n / sum(slots.values()):>5.0%}  {name}")

    print(f"\nMissed pieces, by the closest the top {K} got:")
    print(f"  {'':<16}{'all':>6}{'financebench':>14}{'own':>6}")
    for cat, label in CATEGORIES.items():
        n = Counter(m["source"] for m in misses if m["category"] == cat)
        print(f"  {cat:<16}{n.total():>6}{n['financebench']:>14}{n['own']:>6}   {label}")
    elsewhere = sum(m["evidence_elsewhere"] for m in misses)
    print(f"\nMissed pieces whose evidence text is in a top-{K} chunk of ANOTHER filing: {elsewhere}")
    print("  (the same fact retrieved from a different year's filing, e.g. a prior-year column)")

    print("\nExamples (right company, other filings only; one line per question):")
    examples = {m["id"]: m for m in misses if m["category"] == "right_company"}  # a question can miss several pieces
    for m in list(examples.values())[:8]:
        print(f"  {m['id']:<10} {m['question'][:80]}")
        print(f"             gold {m['gold_doc']}  top: {', '.join(m['top_docs'])}")
