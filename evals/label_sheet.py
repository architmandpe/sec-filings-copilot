"""Make the hand-labeling sheet that the LLM judge is validated against.

For every answered (not refused) question in one answer run, writes:
  evals/judge_labels.csv      one row per answer, with empty `correct` and `faithful` columns to fill in (committed)
  data/evals/label_review.md  the material to judge from: question, gold answer, answer, and the full text of
                              every chunk the answer cites

Never overwrites an existing CSV, so labels can't be lost by re-running.
"""

import argparse
import csv
import json
import sys

from evals.runlog import run_path
from retrieval.index import INDEX_DIR

LABELS_PATH = "evals/judge_labels.csv"
REVIEW_PATH = "data/evals/label_review.md"
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
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="answer run id, e.g. 20261104-233612-answers")
    args = parser.parse_args()
    try:
        out = open(LABELS_PATH, "x", newline="")  # "x" = create; fails if the file already exists
    except FileExistsError:
        sys.exit(f"{LABELS_PATH} already exists; not overwriting labels. Delete it first to start over.")

    with open(run_path(args.run)) as f:
        answered = [r for r in map(json.loads, f) if not r["refused"]]
    with open(INDEX_DIR / "chunks.jsonl") as f:
        chunks = {c["id"]: c for c in map(json.loads, f)}

    with out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        for r in answered:
            auto = {True: "y", False: "n", None: ""}[r["correct"]]
            writer.writerow({"id": r["id"], "type": r["type"], "answer_kind": r["answer_kind"], "auto_correct": auto,
                             "correct": "", "faithful": "", "note": "", "question": r["question"][:100]})

    with open(REVIEW_PATH, "w") as f:
        f.write(GUIDE + f"\nRun `{args.run}`: {len(answered)} answered questions.\n")
        for n, r in enumerate(answered, 1):
            f.write(f"\n---\n\n## {n}. {r['id']} ({r['type']})\n\n**Question:** {r['question']}\n\n"
                    f"**Gold answer:** {r['gold_answer']}\n\n**Answer:**\n\n{r['answer']}\n\n")
            if not r["citations"]:
                f.write("**Cited chunks:** none\n")
            for cid in dict.fromkeys(r["citations"]):  # unique, in citation order
                c = chunks.get(cid)
                if c is None:
                    f.write(f"**[{cid}]** (not a chunk ID)\n\n")
                    continue
                f.write(f"**[{cid}]** {c['company']} {c['form']} FY{c['fiscal_year']} {c['fiscal_period']}, "
                        f"Item {c['item']}: {c['title']}\n\n> " + c["text"].replace("\n", "\n> ") + "\n\n")

    print(f"Wrote {len(answered)} rows to {LABELS_PATH} and the review file {REVIEW_PATH}")
