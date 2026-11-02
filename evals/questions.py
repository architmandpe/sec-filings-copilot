"""Convert in-scope FinanceBench questions into the eval question format, applying the audited corrections."""

import csv
import json
import re
from pathlib import Path

from evals.audit import LABELS_PATH
from evals.financebench import OUT_PATH as FINANCEBENCH_PATH

QUESTIONS_PATH = Path("data/evals/questions_financebench.jsonl")
NUMBER = re.compile(r"(-)?\$?\(?(\d[\d,]*(?:\.\d+)?)")


def parse_number(answer: str) -> float | None:
    """The first number in an answer, with its sign ("-$1561M" → -1561.0, "2.8%" → 2.8)."""
    match = NUMBER.search(answer)
    if not match:
        return None
    value = float(match.group(2).replace(",", ""))
    return -value if match.group(1) else value


def question_type(q: dict) -> str:
    """Map FinanceBench's question type and reasoning label onto our types (approximate: novel questions have no label)."""
    reasoning = q["question_reasoning"] or ""
    if q["question_type"] == "metrics-generated":
        return "table" if reasoning == "Information extraction" else "computation"
    return "computation" if "numerical" in reasoning.lower() else "text"


def answer_kind(q: dict) -> str:
    """How the answer will be scored: numeric match, yes/no match, or an LLM judge for free text."""
    if q["question_type"] == "metrics-generated":
        return "number"
    if re.match(r"(yes|no)\b", q["answer"].strip(), re.IGNORECASE):
        return "yes_no"
    return "text"


def convert(q: dict, corrections: dict) -> dict:
    """One FinanceBench row in the eval question format."""
    kind = answer_kind(q)
    question = {
        "id": "fb-" + q["financebench_id"].rsplit("_", 1)[1],
        "source": "financebench",
        "type": question_type(q),
        "question": q["question"],
        "answer": q["answer"],
        "answer_kind": kind,
        "answer_value": parse_number(q["answer"]) if kind == "number" else None,
        "accepted_answers": [],
        "filings": [q["doc_name"]],
        "evidence": [{"doc_name": e["doc_name"], "page": e["evidence_page_num"], "text": e["evidence_text"]}
                     for e in q["evidence"]],
        "financebench": {"id": q["financebench_id"], "question_type": q["question_type"],
                         "reasoning": q["question_reasoning"], "justification": q["justification"]},
        "notes": "",
    }
    audit = corrections.get(q["financebench_id"])
    if audit and audit["verdict"] == "wrong":
        question["notes"] = f"Gold answer corrected (original: {q['answer']}). {audit['note']}"
        question["answer"] = audit["corrected_answer"]
        if kind == "number":
            question["answer_value"] = parse_number(audit["corrected_answer"])
    elif audit and audit["verdict"] == "ambiguous":
        question["accepted_answers"] = [audit["corrected_answer"]]
        question["notes"] = f"Ambiguous; both answers accepted. {audit['note']}"
    return question


if __name__ == "__main__":
    with open(LABELS_PATH, newline="") as f:
        corrections = {row["financebench_id"]: row for row in csv.DictReader(f)}
    with open(FINANCEBENCH_PATH) as f:
        rows = [q for q in map(json.loads, f) if q["doc_type"] in ("10k", "10q")]
    questions = [convert(q, corrections) for q in rows]
    with open(QUESTIONS_PATH, "w") as f:
        for question in questions:
            f.write(json.dumps(question, ensure_ascii=False) + "\n")
    print(f"{len(questions)} questions → {QUESTIONS_PATH}")
    for field in ("type", "answer_kind"):
        counts = {}
        for question in questions:
            counts[question[field]] = counts.get(question[field], 0) + 1
        print(f"  {field}: {counts}")
    unparsed = [q["id"] for q in questions if q["answer_kind"] == "number" and q["answer_value"] is None]
    print(f"  numeric answers without a parsed value: {unparsed or 'none'}")
    print(f"  corrected: {sum(q['notes'].startswith('Gold answer corrected') for q in questions)}, "
          f"ambiguous: {sum(bool(q['accepted_answers']) for q in questions)}")
