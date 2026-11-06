"""Ceiling test for the tools: run hand-written plans (get_financial lookups + calculate) for the dev FinanceBench
formula questions and score the results like the answer eval does.

A plan is what a perfect planner would do (evals/formula_plans.yaml), so failures here are tool or data problems,
not LLM mistakes, and the score bounds what the agent can reach on these questions with these tools.

    python -m evals.formulas [--note "..."]
"""

import argparse
import hashlib
import json

import yaml

from agent.calculate import calculate
from agent.financials import Financials, describe
from evals.answers import number_correct
from evals.runlog import git_state, new_run_id, save_run
from evals.split import load_questions

PLANS_PATH = "evals/formula_plans.yaml"


def run_plan(fin: Financials, plan: dict) -> dict:
    """Fetch every input, fill the expression with the values in millions, and calculate it."""
    inputs = {name: fin.get(*args) for name, args in plan["lookups"].items()}
    missing = {name: r["error"] for name, r in inputs.items() if "error" in r}
    if missing:
        return {"inputs": inputs, "error": f"inputs not found: {missing}"}
    expression = plan["expression"].format(**{name: repr(r["value"] / 1e6) for name, r in inputs.items()})
    return {"inputs": inputs, **calculate(expression)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what this run changes, for the results log")
    args = parser.parse_args()

    with open(PLANS_PATH) as f:
        plans = yaml.safe_load(f)
    questions = {q["id"]: q for q in load_questions()}
    fin = Financials()
    details = []
    for plan in plans:
        q = questions[plan["id"]]
        result = run_plan(fin, plan)
        correct = "value" in result and number_correct(f"{result['value']}", q["answer_value"])
        details.append({"id": plan["id"], "question": q["question"], "gold": q["answer_value"], "plan": plan,
                        "correct": correct, **{k: v for k, v in result.items() if k != "inputs"},
                        "inputs": {name: describe(r) for name, r in result["inputs"].items()}})
        shown = f"{result['value']:.4g}" if "value" in result else result["error"][:90]
        print(f"{plan['id']}  {'ok   ' if correct else 'WRONG'}  gold {q['answer_value']:>8}  got {shown}")

    n_correct = sum(d["correct"] for d in details)
    run = {
        "run_id": new_run_id("formulas"), "stage": "formulas", "split": "dev", "note": args.note, **git_state(),
        "n_questions": len(plans),
        "fingerprint": hashlib.sha256(json.dumps(plans, sort_keys=True).encode()).hexdigest()[:12],
        "config": {"plans": PLANS_PATH, "scorer": "evals.answers.number_correct"},
        "metrics": {"all": {"n": len(plans), "accuracy": round(n_correct / len(plans), 3), "correct": n_correct}},
    }
    save_run(run, details)
    print(f"\n{run['run_id']}  commit={run['commit']}{' (dirty)' if run['dirty'] else ''}  "
          f"correct {n_correct}/{len(plans)}")
