"""Record every eval run in evals/results.jsonl with what is needed to reproduce and compare it."""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from huggingface_hub import try_to_load_from_cache

RESULTS_PATH = Path("evals/results.jsonl")  # committed: one line per run, the project's history of scores
RUNS_DIR = Path("evals/runs")  # committed: per-question details of logged runs (needed to compare runs question by question)
SCRATCH_RUNS_DIR = Path("data/evals/runs")  # gitignored: details of quick --limit test runs, which are not logged

# The parts of a question that define what is being scored. Chunk IDs are left out on purpose:
# re-chunking changes them, and runs with different chunking must still be comparable.
FINGERPRINT_FIELDS = ("id", "question", "answer", "answer_kind", "answer_value", "accepted_answers", "evidence")


def git_state() -> dict:
    """The current commit, and whether tracked or new files differ from it (a dirty run can't be rebuilt from the commit)."""
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    # Eval outputs (the log and run details) don't change the code, so they don't make a run dirty.
    changes = subprocess.run(["git", "status", "--porcelain", "--", ".", f":!{RESULTS_PATH}", f":!{RUNS_DIR}"],
                             capture_output=True, text=True).stdout.strip()
    return {"commit": commit, "dirty": bool(changes)}


def model_revision(repo_id: str) -> str:
    """The Hugging Face commit hash of the locally cached model (the name of its snapshot folder).

    Looks up one file (config.json) rather than the whole snapshot, which mlx-lm downloads only partly."""
    return Path(try_to_load_from_cache(repo_id, "config.json")).parent.name


def fingerprint(questions: list[dict], settings: dict) -> str:
    """A short hash of the scored questions (with their gold answers and evidence) and the scoring settings.

    Two runs with the same fingerprint scored the same benchmark the same way, so their metrics are comparable."""
    scored = sorted(({k: q.get(k) for k in FINGERPRINT_FIELDS} for q in questions), key=lambda q: q["id"])
    blob = json.dumps({"questions": scored, "settings": settings}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def new_run_id(stage: str) -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{stage}"


def checkpoint_path(run_id: str) -> Path:
    """The progress file of an unfinished run: a header line, then one line per finished question."""
    return SCRATCH_RUNS_DIR / f"{run_id}.partial.jsonl"


def start_checkpoint(stage: str, settings: dict) -> tuple[str, dict]:
    """Start a run that can be interrupted: returns (run_id, git state at the start) and writes the header."""
    run_id, git = new_run_id(stage), git_state()
    SCRATCH_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path(run_id), "w") as f:
        f.write(json.dumps({"settings": settings, "git": git}) + "\n")
    return run_id, git


def resume_checkpoint(run_id: str) -> tuple[dict, dict, list[dict]]:
    """(settings, git state at the start, finished rows) of an interrupted run.

    Refuses to resume if the code has changed since the run started (or it started with uncommitted changes),
    because the finished and the remaining questions would then come from two different versions of the system."""
    with open(checkpoint_path(run_id)) as f:
        header, *rows = [json.loads(line) for line in f]
    now = git_state()
    if header["git"]["dirty"] or now != header["git"]:
        raise SystemExit(f"can't resume {run_id}: it started at {header['git']}, the code is now {now}")
    return header["settings"], header["git"], rows


def save_checkpoint_row(run_id: str, row: dict) -> None:
    """Append one finished question to the progress file (written immediately, so an interruption loses nothing)."""
    with open(checkpoint_path(run_id), "a") as f:
        f.write(json.dumps(row) + "\n")


def run_path(run_id: str) -> Path:
    """Where a run's per-question details are: committed runs first, then quick test runs."""
    for folder in (RUNS_DIR, SCRATCH_RUNS_DIR):
        path = folder / f"{run_id}.jsonl"
        if path.exists():
            return path
    raise FileNotFoundError(f"no details for run {run_id} in {RUNS_DIR} or {SCRATCH_RUNS_DIR}")


def cited_chunks(row: dict, index: dict[str, dict]) -> list[dict]:
    """The chunks one answer cites, unique, in citation order.

    Answer runs from 2026-11-06 on save the cited chunks with the answer ("cited_chunks"), so they stay readable
    after the index is rebuilt. Older runs fall back to the current index (id -> chunk), which shows the right text
    only if the index hasn't been re-chunked since the run."""
    source = row["cited_chunks"] if "cited_chunks" in row else index
    return [source[cid] for cid in dict.fromkeys(row["citations"]) if cid in source]


def save_run(run: dict, details: list[dict], log: bool = True) -> None:
    """Append the run summary to the results log and write the per-question details next to it.

    A quick test run (log=False) is not logged, and its details go to the gitignored scratch folder."""
    if log:
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(run) + "\n")
    folder = RUNS_DIR if log else SCRATCH_RUNS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{run['run_id']}.jsonl", "w") as f:
        for row in details:
            f.write(json.dumps(row) + "\n")
