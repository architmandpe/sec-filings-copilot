"""Compare two eval runs question by question: is run B really better than run A, or is the difference noise?

For every question scored in both runs, take B's score minus A's (a paired comparison: each question is compared
with itself, so easy vs hard questions don't add noise). The mean of those differences is the improvement. Its 95%
confidence interval comes from a paired bootstrap: resample the questions with replacement many times and see how
much the mean difference moves. If the interval excludes 0, the change helped (or hurt) beyond sampling noise.

Runs are compared only if they scored the same benchmark: same fingerprint (questions, gold answers, evidence,
scoring settings). Runs logged before fingerprints existed fall back to "same question IDs", with a warning.
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np

from evals.retrieval import TYPES
from evals.runlog import RESULTS_PATH, run_path

RESAMPLES = 10_000
SEED = 0  # fixed, so the same two runs always give the same interval


def per_question_retrieval(row: dict) -> dict:
    """Each retrieval metric is already a per-question number."""
    return {m: row[m] for m in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr")}


def per_question_answers(row: dict) -> dict:
    """1/0 per question; a metric is left out when it doesn't apply to the question (same rules as answers.summarize)."""
    if row["answer_kind"] == "refuse":
        return {"refusal_recall": float(row["refused"])}
    m = {"false_refusal_rate": float(row["refused"])}
    if row["correct"] is not None:
        m["accuracy"] = float(row["correct"])
    if row["evidence_retrieved"] is not None:
        m["evidence_found"] = float(row["evidence_retrieved"])
    return m


PER_QUESTION = {"retrieval": per_question_retrieval, "answers": per_question_answers}
HEADLINE = {"retrieval": "recall@5", "answers": "accuracy"}
LOWER_IS_BETTER = {"false_refusal_rate"}


def load_run(run_id: str) -> tuple[dict | None, dict[str, dict]]:
    """The run's log entry (None if it was not logged) and its per-question details by question ID."""
    entry = None
    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["run_id"] == run_id:
                entry = r
    with open(run_path(run_id)) as f:
        details = {r["id"]: r for r in map(json.loads, f)}
    return entry, details


def bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """95% interval of the mean difference: resample questions with replacement RESAMPLES times."""
    picks = rng.integers(0, len(diffs), size=(RESAMPLES, len(diffs)))
    means = diffs[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(a: dict[str, dict], b: dict[str, dict], ids: list[str], stage: str) -> dict[str, dict]:
    """For every metric: n, mean in A and B, mean difference, 95% interval, and questions better/worse/same."""
    scores_a = {i: PER_QUESTION[stage](a[i]) for i in ids}
    scores_b = {i: PER_QUESTION[stage](b[i]) for i in ids}
    metrics = list(dict.fromkeys(m for i in ids for m in scores_a[i]))  # in first-seen order
    rng = np.random.default_rng(SEED)
    out = {}
    for m in metrics:
        pairs = [(scores_a[i][m], scores_b[i][m]) for i in ids if m in scores_a[i] and m in scores_b[i]]
        if not pairs:
            continue
        va, vb = np.array(pairs).T
        diffs = vb - va
        sign = -1 if m in LOWER_IS_BETTER else 1  # "better" means lower for false refusals
        low, high = bootstrap_ci(diffs, rng)
        out[m] = {"n": len(pairs), "a": va.mean(), "b": vb.mean(), "diff": diffs.mean(), "low": low, "high": high,
                  "better": int((sign * diffs > 0).sum()), "worse": int((sign * diffs < 0).sum()),
                  "same": int((diffs == 0).sum())}
    return out


def verdict(r: dict) -> str:
    if r["low"] > 0:
        return "B higher"
    if r["high"] < 0:
        return "B lower"
    return "no clear difference"


def print_table(results: dict[str, dict], first: str) -> None:
    print(f"{first:<22}{'n':>5}{'A':>9}{'B':>9}{'B - A':>9}   {'95% CI':<18}{'better':>7}{'worse':>7}{'same':>6}   verdict")
    for name, r in results.items():
        ci = f"[{r['low']:+.3f}, {r['high']:+.3f}]"
        print(f"{name:<22}{r['n']:>5}{r['a']:>9.3f}{r['b']:>9.3f}{r['diff']:>+9.3f}   {ci:<18}"
              f"{r['better']:>7}{r['worse']:>7}{r['same']:>6}   {verdict(r)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="baseline run id, e.g. 20261106-122253-retrieval")
    parser.add_argument("b", help="run id with the change being tested")
    args = parser.parse_args()

    (entry_a, a), (entry_b, b) = load_run(args.a), load_run(args.b)
    stage_a, stage_b = args.a.rsplit("-", 1)[1], args.b.rsplit("-", 1)[1]
    if stage_a != stage_b or stage_a not in PER_QUESTION:
        sys.exit(f"can compare two retrieval runs or two answers runs, not {stage_a} vs {stage_b}")
    stage = stage_a

    fp_a, fp_b = (e.get("fingerprint") if e else None for e in (entry_a, entry_b))
    if fp_a and fp_b and fp_a != fp_b:
        sys.exit(f"different fingerprints ({fp_a} vs {fp_b}): the runs scored different questions, gold answers "
                 "or scoring settings, so their metrics are not comparable")
    if not (fp_a and fp_b):
        print(f"warning: {'A' if not fp_a else 'B'} has no fingerprint (logged before 2026-11-06 or not logged); "
              "checking only that both runs scored the same question IDs")
    if a.keys() != b.keys():
        sys.exit(f"different questions: {len(a.keys() - b.keys())} only in A, {len(b.keys() - a.keys())} only in B")
    ids = sorted(a)

    print(f"A = {args.a}" + (f"  ({entry_a['note']})" if entry_a and entry_a.get("note") else ""))
    print(f"B = {args.b}" + (f"  ({entry_b['note']})" if entry_b and entry_b.get("note") else ""))
    print(f"{len(ids)} questions, fingerprint {fp_a or fp_b or 'n/a'}, paired bootstrap with {RESAMPLES:,} resamples\n")
    print_table(compare(a, b, ids, stage), "metric (all)")

    headline = HEADLINE[stage]
    by_type = defaultdict(list)
    for i in ids:
        by_type[a[i]["type"]].append(i)
    rows = {}
    for t in TYPES:
        if by_type[t]:
            result = compare(a, b, by_type[t], stage)
            if headline in result:
                rows[f"type={t}"] = result[headline]
    if rows:
        print()
        print_table(rows, f"{headline} by type")
    if stage == "answers":
        print("\nbetter/worse count questions where B moved in the good/bad direction (for false_refusal_rate, lower is better).")
