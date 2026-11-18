"""Where the agent's time goes: seconds per question and per part (model turns, each tool), p50 and p95.

    python -m evals.latency --llm-url http://127.0.0.1:8080 --note "model server, prompt cache off"

Runs a stratified sample of dev questions (the label sheets' sampler: proportional by type, fixed seed) through
Agent(llm=ServerLLM(url)) and reads agent.timings. Each answer is also compared with a reference run (default: the
agent v2 eval run) and scored with the answer eval's automatic scorer, so a faster setting of the model server
(e.g. its prompt cache) can be checked for changed or wrong answers.
The model server's own settings (e.g. --prompt-cache-size) can't be read over HTTP: write them in --note.
Logged to evals/results.jsonl (stage "latency"); per-question details in evals/runs/.

p50 = the median (half the values are lower), p95 = 95% of values are lower (the slow tail an average hides).
"""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from agent.llm import ServerLLM
from agent.loop import Agent
from evals.answers import is_refusal, judge
from evals.label_sheet import SAMPLE_SEED, stratified_sample
from evals.runlog import git_state, new_run_id, run_path, save_run
from evals.split import load_questions

REFERENCE_RUN = "20261111-163251-answers"  # agent v2, in process: the evaluated behavior
N_QUESTIONS = 20


def stats(values: list[float]) -> dict:
    """Count, p50, p95, mean and sum of a list of seconds."""
    return {"n": len(values), "p50": round(float(np.percentile(values, 50)), 2),
            "p95": round(float(np.percentile(values, 95)), 2), "mean": round(float(np.mean(values)), 2),
            "sum": round(float(np.sum(values)), 1)}


def outcome(q: dict, text: str) -> bool | None:
    """Correct? Unanswerable: refused. Number / yes-no: the automatic scorer. Free text: None (needs the judge)."""
    if q["answer_kind"] == "refuse":
        return is_refusal(text)
    return judge(q, text)


def summarize(rows: list[dict]) -> dict:
    per_part = defaultdict(list)
    for r in rows:
        for t in r["timings"]:
            per_part[t["part"]].append(t["seconds"])
    total = sum(r["seconds"] for r in rows)
    model = sum(per_part["model"])
    scored = [r for r in rows if r["correct"] is not None]
    ref_scored = [r for r in rows if r["reference_correct"] is not None]
    return {
        "question": stats([r["seconds"] for r in rows]),
        "parts": {part: stats(values) for part, values in sorted(per_part.items())},
        "model_share": round(model / total, 3),  # fraction of all seconds spent in model turns
        "model_turns_per_question": round(len(per_part["model"]) / len(rows), 2),
        "identical_answer": sum(r["identical_answer"] for r in rows),
        "identical_trace": sum(r["identical_trace"] for r in rows),
        "correct": f"{sum(r['correct'] for r in scored)}/{len(scored)}",
        "reference_correct": f"{sum(r['reference_correct'] for r in ref_scored)}/{len(ref_scored)}",
        "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm-url", default="http://127.0.0.1:8080", help="address of the mlx_lm.server")
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="questions in the sample")
    parser.add_argument("--reference", default=REFERENCE_RUN, help="run whose answers and traces to compare with")
    parser.add_argument("--note", default="", help="what is being measured, incl. the model server's settings")
    args = parser.parse_args()

    questions = {q["id"]: q for q in load_questions()}
    with open(run_path(args.reference)) as f:
        reference = [json.loads(line) for line in f]
    sample = stratified_sample(reference, args.n)

    run_id, git = new_run_id("latency"), git_state()
    agent = Agent(llm=ServerLLM(args.llm_url))
    rows = []
    for i, ref in enumerate(sample, 1):
        q = questions[ref["id"]]
        start = time.perf_counter()
        text, _ = agent.answer(q["question"])
        seconds = round(time.perf_counter() - start, 2)
        row = {"id": q["id"], "type": q["type"], "seconds": seconds, "timings": agent.timings, "answer": text,
               "identical_answer": text == ref["answer"].strip(), "identical_trace": agent.trace == ref["trace"],
               "correct": outcome(q, text), "reference_correct": outcome(q, ref["answer"]), "trace": agent.trace}
        rows.append(row)
        print(f"[{i}/{len(sample)}] {q['id']:<9} {q['type']:<12} {seconds:6.1f} s  {len(agent.timings):2} parts  "
              f"identical {row['identical_answer'] and row['identical_trace']}  correct {row['correct']}", flush=True)

    summary = summarize(rows)
    save_run({"run_id": run_id, "stage": "latency", "git": git, "note": args.note,
              "config": {"llm": agent.llm.config(), "retrieval": agent.retriever.config(), "n": len(rows),
                         "sample_seed": SAMPLE_SEED, "reference_run": args.reference},
              "summary": summary}, rows)
    print(f"\nRun {run_id}")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
