"""Freeze the dev/test split. Whole companies go to one side, so dev and test never share filing text."""

import json
import random
from collections import Counter
from pathlib import Path

from evals.own import load_own
from evals.questions import QUESTIONS_PATH
from ingest.corpus import CORPUS_PATH

SPLIT_PATH = Path("evals/split.json")  # committed: the split is part of the eval, not rebuildable data
SEED = 2026
SPLITS = ("dev", "test")


def load_questions() -> list[dict]:
    """FinanceBench questions followed by the hand-written ones."""
    with open(QUESTIONS_PATH) as f:
        return [json.loads(line) for line in f] + load_own()


def load_company_of() -> dict[str, str]:
    """doc_name -> company."""
    with open(CORPUS_PATH) as f:
        return {row["doc_name"]: row["company"] for row in map(json.loads, f)}


def load_split() -> dict[str, str]:
    """question id -> "dev" or "test" (what the eval harness uses)."""
    return json.loads(SPLIT_PATH.read_text())["questions"]


def company_groups(questions: list[dict], company_of: dict[str, str]) -> list[set[str]]:
    """Sets of companies that must share a split because questions link them (directly or through a chain)."""
    groups = []
    for q in questions:
        linked = {company_of[d] for d in q["filings"]}
        for g in [g for g in groups if g & linked]:
            groups.remove(g)
            linked |= g
        if linked:
            groups.append(linked)
    return groups


def choose(counts: dict[str, Counter], add: Counter, rng: random.Random) -> str:
    """The split that leaves dev and test closest in (source, type) counts after adding `add`; ties broken at random."""

    def imbalance(target: str) -> int:
        dev = counts["dev"] + add if target == "dev" else counts["dev"]
        test = counts["test"] + add if target == "test" else counts["test"]
        return sum(abs(dev[k] - test[k]) for k in dev.keys() | test.keys())

    scores = {s: imbalance(s) for s in SPLITS}
    return rng.choice([s for s in SPLITS if scores[s] == min(scores.values())])


def build_split(questions: list[dict], company_of: dict[str, str], old: dict) -> dict:
    """Assign new companies and new filing-less questions; everything already in `old` keeps its split."""
    rng = random.Random(SEED)
    companies = dict(old.get("companies", {}))
    old_questions = old.get("questions", {})
    loose = {i: s for i, s in old_questions.items()}  # filing-less questions are assigned one by one
    key = lambda q: (q["source"], q["type"])

    new_groups = []
    for g in company_groups(questions, company_of):
        assigned = {companies[c] for c in g if c in companies}
        if len(assigned) > 1:
            raise ValueError(f"a question links companies already in both splits: {sorted(g)}")
        if assigned:
            side = assigned.pop()
            for c in g:
                companies.setdefault(c, side)  # a new company joins the split of the company it is linked to
        else:
            new_groups.append(g)

    def side_of(q: dict) -> str | None:
        return companies.get(company_of[q["filings"][0]]) if q["filings"] else loose.get(q["id"])

    counts = {s: Counter() for s in SPLITS}
    for q in questions:
        if side_of(q):
            counts[side_of(q)][key(q)] += 1

    def group_counts(g: set[str]) -> Counter:
        return Counter(key(q) for q in questions if q["filings"] and company_of[q["filings"][0]] in g)

    new_groups.sort(key=sorted)  # fixed order first: set order can differ between Python runs
    rng.shuffle(new_groups)
    new_groups.sort(key=lambda g: -group_counts(g).total())  # largest first balances best; the shuffle breaks ties
    for g in new_groups:
        add = group_counts(g)
        side = choose(counts, add, rng)
        counts[side] += add
        for c in g:
            companies[c] = side

    new_loose = [q for q in questions if not q["filings"] and q["id"] not in loose]
    rng.shuffle(new_loose)
    for q in new_loose:
        side = choose(counts, Counter([key(q)]), rng)
        counts[side][key(q)] += 1
        loose[q["id"]] = side

    assignment = {q["id"]: side_of(q) for q in questions}
    for i, s in old_questions.items():
        if i in assignment and assignment[i] != s:
            raise ValueError(f"{i} moved from {s} to {assignment[i]}: its filings changed company")
    return {"seed": SEED, "companies": dict(sorted(companies.items())), "questions": dict(sorted(assignment.items()))}


if __name__ == "__main__":
    questions = load_questions()
    company_of = load_company_of()
    old = json.loads(SPLIT_PATH.read_text()) if SPLIT_PATH.exists() else {}
    split = build_split(questions, company_of, old)
    SPLIT_PATH.write_text(json.dumps(split, indent=2) + "\n")

    side = split["questions"]
    spanning = [q["id"] for q in questions if len({split["companies"][company_of[d]] for d in q["filings"]}) > 1]
    moved = [i for i, s in old.get("questions", {}).items() if side.get(i) != s]
    print(f"Questions whose filings span both splits: {len(spanning)} (expect 0)")
    print(f"Previously assigned questions that moved: {len(moved)} (expect 0)\n")

    types = ["text", "table", "computation", "multi_hop", "unanswerable"]
    rows = Counter((side[q["id"]], q["source"], q["type"]) for q in questions)
    print(f"{'':<18}" + "".join(f"{t:>13}" for t in types) + f"{'total':>7}")
    for s in SPLITS:
        for source in ("financebench", "own"):
            n = [rows[s, source, t] for t in types]
            print(f"{s + ' ' + source:<18}" + "".join(f"{x:>13}" for x in n) + f"{sum(n):>7}")
    for s in SPLITS:
        names = [c for c, x in split["companies"].items() if x == s]
        print(f"\n{s} ({len(names)} companies): {', '.join(names)}")
