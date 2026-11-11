"""LLM judge: grade the answers of an answer run for correctness and faithfulness.

Two separate questions, each its own model call:
  correctness   does the answer match the gold answer? (sees question, gold answer, answer)
  faithfulness  is every claim stated in the chunks the answer cites? (sees question, answer, cited chunks;
                never the gold answer, so a correct-looking answer can't sway it)
The rules are the same as the hand-labeling guide (evals/label_sheet.py). If evals/judge_labels.csv has labels for
the judged answers, the judge's agreement with them is reported: that is how we know whether to trust it.
The judge is a different model family from the generator, to avoid a model favoring its own answers.

Cost savings (a full run took ~80 min of GPU): numeric answers take their correctness from the automatic scorer
(it agreed with the judge on 49 of 52, and was right in the 3 disagreements), unless --judge-numbers; every verdict
is cached by its exact input (greedy decoding: same input, same verdict), so unchanged answers are never re-judged;
and a run can be interrupted and continued with --resume.
"""

import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path

from mlx_lm import generate, load

from agent.answer import format_context
from evals.label_sheet import labels_path
from evals.runlog import (checkpoint_path, cited_chunks, model_revision, resume_checkpoint, run_path,
                          save_checkpoint_row, save_run, start_checkpoint)
from retrieval.index import INDEX_DIR

JUDGE_MODEL = "mlx-community/gemma-4-12B-it-qat-4bit"
CACHE_PATH = Path("data/evals/judge_cache.jsonl")  # one line per verdict: input hash -> verdict and reply
CORRECT_MAX_TOKENS = 300
FAITHFUL_MAX_TOKENS = 1200  # room for one line per claim; at 800, 2 of 41 replies were cut off before the verdict
VERDICT = re.compile(r"VERDICT:\s*\**\s*([yn])\b", re.IGNORECASE)
ENDING = 'First explain your reasoning in at most 3 sentences. Then end with a final line that is exactly "VERDICT: y" or "VERDICT: n".'

CORRECT_PROMPT = """You are grading an answer to a question about a company's SEC filing.
Compare the ANSWER with the GOLD ANSWER only.

Rules:
- "y" if the answer states the gold answer's key facts (numbers equal up to rounding or units, e.g. thousands vs millions; same names) and nothing that contradicts the gold answer. If the question asks for several things (e.g. three years), all are needed.
- "n" otherwise, including partially correct answers.

QUESTION: {question}

GOLD ANSWER: {gold}

ANSWER: {answer}

""" + ENDING

# Revision 2 (claim by claim). Revision 1 asked for a single verdict and was too lenient: it checked that numbers
# appear in the excerpts, not that their period, scope and direction match (run 20261105-220148-judge).
FAITHFUL_PROMPT = """You are checking whether an answer is supported by the SEC filing excerpts it cites.
Use the CITED EXCERPTS only. Do not use outside knowledge, and do not judge whether the answer is correct.
Each excerpt starts with a header naming its company, filing and fiscal year. Tables may have several year columns: read the column headers.

Step 1. List every factual claim in the answer (each number, name, date or stated fact), one line per claim:
claim | period and scope the answer gives it | supporting text from the excerpts, or NONE | period and scope of that text
Scope means what the claim covers: the whole company or one segment, paid or received, increase or decrease.

Step 2. A claim fails if any of these is true:
- the supporting text is NONE (the claim is not in the excerpts);
- the periods differ (e.g. a number from a 2020 column reported as 2021, or a fiscal 2024 fact supported only by a fiscal 2023 filing that does not state it);
- the scope or direction differs (e.g. one segment's revenue reported as total revenue; money received reported as money to be paid);
- the answer contradicts its own excerpt (e.g. says sales did not decline while the excerpt shows a decrease).
A calculation passes if every input passes and the arithmetic is right. Unit conversions and rounding are fine.
Ignore sentences about what the excerpts do not contain.

QUESTION: {question}

ANSWER: {answer}

CITED EXCERPTS:

{excerpts}

Write the claim list, then one line naming any failed claim. End with a final line that is exactly "VERDICT: y" if every claim passes, or "VERDICT: n" if any claim fails."""


class Judge:
    """Holds the judge model in memory for repeated verdicts."""

    def __init__(self, model_name: str):
        self.model_name, self.model, self.tokenizer = model_name, None, None  # loaded on the first cache miss
        self.cache = {}
        if CACHE_PATH.exists():
            with open(CACHE_PATH) as f:
                self.cache = {row["key"]: row for row in map(json.loads, f)}
        self.hits = 0

    def verdict(self, prompt: str, max_tokens: int) -> tuple[str | None, str]:
        """Return ("y" | "n" | None if unparseable, the judge's full reply), from the cache if this exact input was
        judged before."""
        key = hashlib.sha256(json.dumps([self.model_name, max_tokens, prompt]).encode()).hexdigest()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]["verdict"], self.cache[key]["reply"]
        if self.model is None:
            self.model, self.tokenizer = load(self.model_name)
        messages = [{"role": "user", "content": prompt}]  # one user turn: Gemma's template has no system role
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        reply = generate(self.model, self.tokenizer, prompt=text, max_tokens=max_tokens)
        found = VERDICT.findall(reply)
        result = {"key": key, "verdict": found[-1].lower() if found else None, "reply": reply}
        self.cache[key] = result
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps(result) + "\n")
        return result["verdict"], reply


def agreement(pairs: list[tuple[str, str]]) -> dict:
    """How often judge and label agree, and Cohen's kappa (agreement corrected for chance), over (judge, label) pairs."""
    n = len(pairs)
    observed = sum(j == lab for j, lab in pairs) / n
    p_judge, p_label = sum(j == "y" for j, _ in pairs) / n, sum(lab == "y" for _, lab in pairs) / n
    chance = p_judge * p_label + (1 - p_judge) * (1 - p_label)
    return {
        "n": n, "agreement": round(observed, 3),
        "kappa": round((observed - chance) / (1 - chance), 3) if chance < 1 else None,
        "judge_y_label_n": sum(j == "y" and lab == "n" for j, lab in pairs),  # judge too lenient
        "judge_n_label_y": sum(j == "n" and lab == "y" for j, lab in pairs),  # judge too strict
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="answer run id to judge, e.g. 20261104-233612-answers")
    parser.add_argument("--model", default=JUDGE_MODEL, help="judge model (default: %(default)s)")
    parser.add_argument("--limit", type=int, help="judge only this many random answers (a quick test; not logged)")
    parser.add_argument("--judge-numbers", action="store_true",
                        help="also judge correctness of numeric answers (default: use the automatic scorer)")
    parser.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted judge run with its settings")
    parser.add_argument("--labeled-only", action="store_true",
                        help="judge only the answers that have labels for this run (to validate the judge first)")
    args = parser.parse_args()
    if args.resume:
        settings, git, done = resume_checkpoint(args.resume)
        args, run_id = argparse.Namespace(**settings), args.resume
        print(f"resuming {run_id}: {len(done)} answers already judged", flush=True)
    else:
        run_id, git = start_checkpoint("judge", vars(args))
        done = []

    with open(run_path(args.run)) as f:
        answered = [r for r in map(json.loads, f) if not r["refused"]]  # refusals are scored without a judge
    labels = {}  # labels grade one specific run's answers, so they are looked up by run
    if Path(labels_path(args.run)).exists():
        with open(labels_path(args.run), newline="") as f:
            labels = {row["id"]: row for row in csv.DictReader(f) if row["correct"] and row["faithful"]}
    if args.labeled_only:
        answered = [r for r in answered if r["id"] in labels]
    if args.limit:
        answered = random.Random(0).sample(answered, args.limit)
    with open(INDEX_DIR / "chunks.jsonl") as f:
        chunks = {c["id"]: c for c in map(json.loads, f)}
    judge = Judge(args.model)

    details = list(done)
    finished = {d["id"] for d in done}
    for i, r in enumerate(answered, 1):
        if r["id"] in finished:
            continue
        start = time.perf_counter()
        if r["answer_kind"] == "number" and not args.judge_numbers:
            correct, correct_reply = ("y" if r["correct"] else "n"), "automatic scorer (numeric answer)"
        else:
            correct, correct_reply = judge.verdict(
                CORRECT_PROMPT.format(question=r["question"], gold=r["gold_answer"], answer=r["answer"]),
                CORRECT_MAX_TOKENS)
        cited = cited_chunks(r, chunks)
        if cited:
            faithful, faithful_reply = judge.verdict(
                FAITHFUL_PROMPT.format(question=r["question"], answer=r["answer"], excerpts=format_context(cited)),
                FAITHFUL_MAX_TOKENS)
        else:
            faithful, faithful_reply = "n", "No valid citation, so no claim is supported."
        details.append({"id": r["id"], "type": r["type"], "answer_kind": r["answer_kind"],
                        "correct": correct, "faithful": faithful,
                        "correct_reply": correct_reply, "faithful_reply": faithful_reply,
                        "seconds": round(time.perf_counter() - start, 1)})
        save_checkpoint_row(run_id, details[-1])
        print(f"{i:>3}/{len(answered)}  {r['id']:<10} correct={correct}  faithful={faithful}  {details[-1]['seconds']:>5.1f}s",
              flush=True)

    def counted(d, name):
        """An unparsed verdict counts as "n" (fail closed: the judge did not confirm it). Details keep the raw None."""
        return d[name] or "n"

    metrics = {}
    for name in ("correct", "faithful"):
        metrics[f"{name}_rate"] = round(sum(counted(d, name) == "y" for d in details) / len(details), 3) if details else None
        metrics[f"{name}_unparsed"] = sum(d[name] is None for d in details)
        pairs = [(counted(d, name), labels[d["id"]][name]) for d in details if d["id"] in labels]
        if pairs:
            metrics[f"{name}_vs_labels"] = agreement(pairs)
    text_pairs = [(counted(d, "correct"), labels[d["id"]]["correct"]) for d in details
                  if d["id"] in labels and d["answer_kind"] == "text"]
    if text_pairs:
        metrics["correct_vs_labels_text_only"] = agreement(text_pairs)

    prompts_sha = hashlib.sha256((CORRECT_PROMPT + FAITHFUL_PROMPT).encode()).hexdigest()[:8]
    run = {
        "run_id": run_id, "stage": "judge", "source_run": args.run, "limit": args.limit, **git,
        "config": {"judge_model": args.model, "judge_revision": model_revision(args.model),
                   "numeric_correctness": "judge" if args.judge_numbers else "automatic scorer",
                   "max_tokens": {"correct": CORRECT_MAX_TOKENS, "faithful": FAITHFUL_MAX_TOKENS},
                   "prompts_sha": prompts_sha},
        "metrics": metrics,
    }
    save_run(run, details, log=not args.limit)
    checkpoint_path(run_id).unlink()

    print(f"\n{run['run_id']}  judged {args.run}  judge={args.model}  cached verdicts reused: {judge.hits}"
          f"{'  (limited run: not logged)' if args.limit else ''}")
    for name in ("correct", "faithful"):
        print(f"\n{name}: judge says y for {metrics[f'{name}_rate']}, unparsed {metrics[f'{name}_unparsed']} (counted as n)")
        for key in (f"{name}_vs_labels", "correct_vs_labels_text_only" if name == "correct" else None):
            if key in metrics:
                m = metrics[key]
                print(f"  {key}: n={m['n']}  agreement={m['agreement']}  kappa={m['kappa']}  "
                      f"too lenient (judge y, label n)={m['judge_y_label_n']}  too strict (judge n, label y)={m['judge_n_label_y']}")
