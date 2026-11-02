"""Draw a reproducible random sample of in-scope FinanceBench questions and print what is needed to check each gold answer."""

import csv
import json
import random
import re
from pathlib import Path

from evals.financebench import OUT_PATH as FINANCEBENCH_PATH
from ingest.corpus import CORPUS_PATH

SAMPLE_SIZE = 20
SEED = 2026  # fixed, so the sample is reproducible and not hand-picked
REPORT_PATH = Path("data/evals/audit_sample.md")
LABELS_PATH = Path("evals/financebench_audit.csv")  # the human verdicts; committed, unlike data/
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
MAX_LINES = 8


def stratified_sample(questions: list[dict]) -> list[dict]:
    """Sample SAMPLE_SIZE questions, each question type getting a share proportional to its size."""
    rng = random.Random(SEED)
    by_type = {}
    for q in questions:
        by_type.setdefault(q["question_type"], []).append(q)
    sample = []
    for qtype, group in sorted(by_type.items()):
        share = round(SAMPLE_SIZE * len(group) / len(questions))
        sample += rng.sample(sorted(group, key=lambda q: q["financebench_id"]), share)
    return sample


def key_numbers(*texts: str) -> set[str]:
    """Numbers with 3+ digits in the gold answer and justification, as they may be printed in a filing ("1577" and "1,577")."""
    forms = set()
    for text in texts:
        for raw in NUMBER.findall(text or ""):
            digits = raw.replace(",", "")
            value = float(digits)
            if len(digits.replace(".", "").lstrip("0")) < 3 or (value.is_integer() and 1990 <= value <= 2030):
                continue  # small numbers and years match almost every line
            if value.is_integer():
                forms |= {str(int(value)), f"{int(value):,}"}
            else:
                forms.add(digits)
    return forms


def find_lines(sections: list[dict], forms: set[str]) -> list[str]:
    """Lines of the parsed filing that contain any of the key numbers, labeled with their Item."""
    if not forms:
        return []
    pattern = re.compile(r"(?<![\d,.])(" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + r")(?![\d,])")
    hits = []
    for section in sections:
        for line in section["text"].splitlines():
            if pattern.search(line):
                hits.append(f"[Item {section['item']}] {line[:220]}")
    return hits


if __name__ == "__main__":
    with open(FINANCEBENCH_PATH) as f:
        questions = [q for q in map(json.loads, f) if q["doc_type"] in ("10k", "10q")]
    with open(CORPUS_PATH) as f:
        corpus = {row["doc_name"]: row for row in map(json.loads, f)}
    sample = stratified_sample(questions)

    with open(REPORT_PATH, "w") as report:
        for n, q in enumerate(sample, 1):
            filing = corpus[q["doc_name"]]
            sections = json.loads((Path("data/parsed") / filing["cik"] / f"{filing['accession']}.json").read_text())
            forms = key_numbers(q["answer"], q["justification"])
            hits = find_lines(sections, forms)
            evidence = q["evidence"][0]
            report.write(f"## {n}. {q['financebench_id']} — {q['doc_name']} ({q['question_type']})\n\n")
            report.write(f"**Question:** {q['question']}\n\n**Gold answer:** {q['answer']}\n\n")
            report.write(f"**Justification:** {q['justification']}\n\n")
            report.write(f"**FinanceBench evidence (PDF page {evidence['evidence_page_num']}):**\n```\n{evidence['evidence_text'][:1200]}\n```\n\n")
            report.write(f"**Our filing ({filing['accession']}), lines with {sorted(forms) or 'no key numbers'}:** "
                         f"{len(hits)} found\n```\n" + "\n".join(hits[:MAX_LINES]) + "\n```\n\n")
            print(f"{n:>2}. {q['financebench_id']:<22} {q['question_type']:<18} {q['doc_name']:<28} {len(hits):>3} matching lines")

    if not LABELS_PATH.exists():  # never overwrite verdicts already entered
        with open(LABELS_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["financebench_id", "doc_name", "question_type", "verdict", "corrected_answer", "note"])
            for q in sample:
                writer.writerow([q["financebench_id"], q["doc_name"], q["question_type"], "", "", ""])
    print(f"\nReport: {REPORT_PATH}\nVerdicts to fill in: {LABELS_PATH} (verdict = correct / wrong / ambiguous)")
