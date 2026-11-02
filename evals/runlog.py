"""Record every eval run in evals/results.jsonl with what is needed to reproduce it."""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from huggingface_hub import snapshot_download

RESULTS_PATH = Path("evals/results.jsonl")  # committed: one line per run, the project's history of scores
RUNS_DIR = Path("data/evals/runs")  # per-question details for error analysis (rebuildable, gitignored)


def git_state() -> dict:
    """The current commit, and whether tracked or new files differ from it (a dirty run can't be rebuilt from the commit)."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    changes = subprocess.run(["git", "status", "--porcelain", "--", ".", f":!{RESULTS_PATH}"],
                             capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "dirty": bool(changes)}


def model_revision(repo_id: str) -> str:
    """The Hugging Face commit hash of the locally cached model (the name of its snapshot folder)."""
    return Path(snapshot_download(repo_id, local_files_only=True)).name


def new_run_id(stage: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{stage}"


def save_run(run: dict, details: list[dict]) -> None:
    """Append the run summary to the results log and write the per-question details."""
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(run) + "\n")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / f"{run['run_id']}.jsonl", "w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")
