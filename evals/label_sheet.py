"""Make the hand-labeling sheet that the LLM judge is validated against.

For the answered (not refused) questions of one answer run (all, or a sample stratified by question type), writes:
  labels_path(run)            one row per answer, with empty `correct` and `faithful` columns to fill in (committed);
                              evals/judge_labels.csv for the Week 2 baseline run, evals/judge_labels_<run>.csv otherwise
  data/evals/label_review_<run>.md   the material to judge from: question, gold answer, answer, and the full text
                              of every source the answer cites (passages, XBRL facts, calculator results)

Never overwrites an existing CSV, so labels can't be lost by re-running.
"""

import argparse
import csv
import json
import random
import sys
from collections import defaultdict

from evals.runlog import cited_chunks, run_path
from retrieval.index import INDEX_DIR

LABELS_PATH = "evals/judge_labels.csv"  # labels of the Week 2 baseline run (the first judge validation)
LABELED_RUNS = {"20261104-233612-answers": LABELS_PATH}
SAMPLE_SEED = 2026
MIN_PER_TYPE = 2
COLUMNS = ["id", "type", "answer_kind", "auto_correct", "correct", "faithful", "note", "question"]

GUIDE = """# Judge labels: how to label

For each answer below, fill two columns in `evals/judge_labels.csv` with `y` or `n` (and a short `note` when unsure).

**correct**: compare the answer with the gold answer only.
- `y` if it states the gold answer's key facts (numbers equal up to rounding or units, same names), and
  nothing that contradicts the gold answer. If the question asks for several things (e.g. three years), all are needed.
- `n` otherwise, including partially correct answers.
- `auto_correct` shows the automatic number/yes-no score; check it rather than copying it (that also tests the scorer).

**faithful**: compare the answer with the cited chunks only (ignore the gold answer).
- `y` if every factual claim (each number, name, date, and the company and period it is attributed to) is stated in a
  chunk cited for it. A calculation counts as supported if its inputs are cited and the arithmetic is right.
- `n` if any claim is not in its cited chunks, has no citation, or is attributed to a different period than the
  cited chunk covers (e.g. a "fiscal 2023" fact cited to a 2016 10-K).
- Ignore sentences about what the excerpts do not contain ("the filings do not mention ...").
- A calculator result (`CALC-n`) shows its expression and the sources of its inputs; it supports a computed number
  if the inputs are the right facts for the claim (company, metric, period) and the result matches.
"""


def labels_path(run_id: str) -> str:
    """The labels file for one answer run."""
    return LABELED_RUNS.get(run_id, f"evals/judge_labels_{run_id}.csv")


def stratified_sample(rows: list[dict], n: int) -> list[dict]:
    """n rows, split across question types in proportion to their counts (at least MIN_PER_TYPE each when available,
    largest remainders get the rest), chosen at random with a fixed seed; returned in run order."""
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)
    share = {t: n * len(g) / len(rows) for t, g in by_type.items()}
    take = {t: min(len(by_type[t]), max(MIN_PER_TYPE, int(share[t]))) for t in by_type}
    for t in sorted(by_type, key=lambda t: share[t] - int(share[t]), reverse=True):
        if sum(take.values()) >= n:
            break
        if take[t] < len(by_type[t]):
            take[t] += 1
    rng = random.Random(SAMPLE_SEED)
    chosen = {r["id"] for t, g in sorted(by_type.items()) for r in rng.sample(g, take[t])}
    return [r for r in rows if r["id"] in chosen]


def source_header(c: dict) -> str:
    if c["form"] == "calculation":
        return "calculator result"
    if c["form"] == "XBRL":
        return f"XBRL fact, {c['company']} fiscal {c['fiscal_year']}, {c['title']}"
    return f"{c['company']} {c['form']} FY{c['fiscal_year']} {c['fiscal_period']}, Item {c['item']}: {c['title']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="answer run id, e.g. 20261104-233612-answers")
    parser.add_argument("--sample", type=int, help="label only this many answers, stratified by question type")
    args = parser.parse_args()
    path, review_path = labels_path(args.run), f"data/evals/label_review_{args.run}.md"
    try:
        out = open(path, "x", newline="")  # "x" = create; fails if the file already exists
    except FileExistsError:
        sys.exit(f"{path} already exists; not overwriting labels. Delete it first to start over.")

    with open(run_path(args.run)) as f:
        answered = [r for r in map(json.loads, f) if not r["refused"]]
    if args.sample:
        answered = stratified_sample(answered, args.sample)
    with open(INDEX_DIR / "chunks.jsonl") as f:
        chunks = {c["id"]: c for c in map(json.loads, f)}

    with out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for r in answered:
            auto = {True: "y", False: "n", None: ""}[r["correct"]]
            writer.writerow({"id": r["id"], "type": r["type"], "answer_kind": r["answer_kind"], "auto_correct": auto,
                             "correct": "", "faithful": "", "note": "", "question": r["question"][:100]})

    with open(review_path, "w") as f:
        f.write(GUIDE + f"\nRun `{args.run}`: {len(answered)} answered questions.\n")
        for n, r in enumerate(answered, 1):
            f.write(f"\n---\n\n## {n}. {r['id']} ({r['type']})\n\n**Question:** {r['question']}\n\n"
                    f"**Gold answer:** {r['gold_answer']}\n\n**Answer:**\n\n{r['answer']}\n\n")
            if not r["citations"]:
                f.write("**Cited chunks:** none\n")
            found = {c["id"]: c for c in cited_chunks(r, chunks)}
            for cid in dict.fromkeys(r["citations"]):  # unique, in citation order
                c = found.get(cid)
                if c is None:
                    f.write(f"**[{cid}]** (not a chunk ID)\n\n")
                    continue
                f.write(f"**[{cid}]** {source_header(c)}\n\n> " + c["text"].replace("\n", "\n> ") + "\n\n")

    print(f"Wrote {len(answered)} rows to {path} and the review file {review_path}")
