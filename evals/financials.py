"""Tool eval: does get_financial (agent/financials.py) return the numbers the dev gold answers state?

Cases are in evals/financials_cases.yaml. A case passes if the tool's value is within the stated rounding of the
expected number, or, for expected: null, if the tool says "not found" (returning some other number would be a
confident wrong answer, the worst outcome for this tool).

    python -m evals.financials [--note "..."]
"""

import argparse
import hashlib
import json

import yaml

from agent.financials import METRICS, Financials, describe
from evals.runlog import git_state, new_run_id, save_run

CASES_PATH = "evals/financials_cases.yaml"
DEFAULT_TOLERANCE = 0.5  # millions: half a unit of a number stated in whole millions


def check(case: dict, result: dict) -> str:
    """pass, wrong (a different number), missed (not found but a number expected), or invented (a number where
    none should exist)."""
    found = "error" not in result
    if case["expected"] is None:
        return "invented" if found else "pass"
    if not found:
        return "missed"
    ok = abs(result["value"] / 1e6 - case["expected"]) <= case.get("tolerance", DEFAULT_TOLERANCE)
    return "pass" if ok else "wrong"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", default="", help="what this run changes, for the results log")
    args = parser.parse_args()

    with open(CASES_PATH) as f:
        cases = yaml.safe_load(f)
    fin = Financials()
    details = []
    for case in cases:
        result = fin.get(case["company"], case["metric"], case["fiscal_year"])
        outcome = check(case, result)
        details.append({**case, "outcome": outcome, "result": result})
        print(f"{case['id']}  {outcome:<8} expected {case['expected']!s:>10}  {describe(result)}")

    counts = {k: sum(d["outcome"] == k for d in details) for k in ("pass", "wrong", "missed", "invented")}
    metrics = {"n": len(cases), "accuracy": round(counts["pass"] / len(cases), 3), **counts,
               "conflicting_values": sum(bool(d["result"].get("conflicting_values")) for d in details)}
    run = {
        "run_id": new_run_id("financials"), "stage": "financials", "split": "dev", "note": args.note, **git_state(),
        "n_questions": len(cases),
        "fingerprint": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()[:12],
        "config": {"metrics": {m: concepts for m, (_, concepts, _) in METRICS.items()}, "as_reported": True},
        "metrics": {"all": metrics},
    }
    save_run(run, details)
    print(f"\n{run['run_id']}  commit={run['commit']}{' (dirty)' if run['dirty'] else ''}  {metrics}")
