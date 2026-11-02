"""Load and check the hand-written questions in evals/questions_own.yaml."""

import json
import re
from collections import Counter
from pathlib import Path

import yaml

from evals.gold import load_chunks_by_doc, match_evidence
from ingest.corpus import CORPUS_PATH

OWN_PATH = Path("evals/questions_own.yaml")
TYPES = {"text", "table", "computation", "multi_hop", "unanswerable"}
KINDS = {"number", "yes_no", "text", "refuse"}
TARGETS = {"unanswerable": 30, "multi_hop": 15, "table": 20, "computation": 20, "text": 15}


def load_own() -> list[dict]:
    """Return the hand-written questions with the same fields as the converted FinanceBench questions."""
    with open(OWN_PATH) as f:
        questions = yaml.safe_load(f) or []
    for q in questions:
        q.setdefault("answer_value", None)
        q.setdefault("accepted_answers", [])
        q.setdefault("notes", "")
        q["source"] = "own"
    return questions


def problems(q: dict, doc_names: set[str]) -> list[str]:
    """Format errors in one question (empty if it is valid)."""
    errors = []
    if not re.fullmatch(r"own-\d{3}", str(q.get("id"))):
        errors.append("id must look like own-001")
    for field in ("type", "question", "answer", "answer_kind", "filings", "evidence"):
        if field not in q:
            errors.append(f"missing {field}")
    if errors:
        return errors
    if q["type"] not in TYPES:
        errors.append(f"type must be one of {sorted(TYPES)}")
    if q["answer_kind"] not in KINDS:
        errors.append(f"answer_kind must be one of {sorted(KINDS)}")
    if (q["type"] == "unanswerable") != (q["answer_kind"] == "refuse"):
        errors.append("unanswerable questions, and only they, have answer_kind refuse")
    if q["answer_kind"] == "number" and not isinstance(q["answer_value"], (int, float)):
        errors.append("number answers need a numeric answer_value")
    if q["answer_kind"] == "refuse" and q["evidence"]:
        errors.append("unanswerable questions have no evidence")
    if q["answer_kind"] != "refuse" and not q["evidence"]:
        errors.append("answerable questions need at least one evidence quote")
    if q["type"] == "multi_hop" and len(q["evidence"]) < 2:
        errors.append("multi_hop questions need evidence from at least two places")
    for name in q["filings"]:
        if name not in doc_names:
            errors.append(f"unknown filing {name}")
    for e in q["evidence"]:
        if e.get("doc_name") not in q["filings"]:
            errors.append(f"evidence filing {e.get('doc_name')} is not listed in filings")
    return errors


if __name__ == "__main__":
    with open(CORPUS_PATH) as f:
        doc_names = {row["doc_name"] for row in map(json.loads, f)}
    questions = load_own()
    by_doc = load_chunks_by_doc()
    duplicates = [i for i, n in Counter(q.get("id") for q in questions).items() if n > 1]
    if duplicates:
        print(f"Duplicate ids: {duplicates}")
    failed = 0
    for q in questions:
        errors = problems(q, doc_names)
        if not errors:
            for e in q["evidence"]:
                gold, coverage = match_evidence(e["text"], by_doc[e["doc_name"]])
                if coverage < 1:
                    errors.append(f"quote not found in {e['doc_name']} (best chunk has {coverage:.0%} of it): {e['text'][:60]!r}")
        failed += bool(errors)
        status = "ok " if not errors else "FIX"
        print(f"{status} {q.get('id')}  {q.get('type', '?'):<12} {str(q.get('question'))[:70]}")
        for error in errors:
            print(f"      - {error}")
    counts = Counter(q.get("type") for q in questions)
    progress = ", ".join(f"{t} {counts[t]}/{n}" for t, n in TARGETS.items())
    print(f"\n{len(questions)} questions, {failed} to fix. Progress: {progress}")
