"""Answer eval: run the full RAG pipeline on the eval questions and score what it says.

Scores refusals (unanswerable questions should be refused, answerable ones should not), number and yes/no
answers by matching, and citations (present, and pointing at chunks that were actually retrieved). Free-text
answers are saved but not scored here; they need an LLM judge. Each answer is also tagged with whether
retrieval found its evidence, to separate retrieval failures from generation failures.
"""

import argparse
import hashlib
import random
import re
import time
from collections import defaultdict

from agent.answer import GEN_MODEL, MAX_TOKENS, REFUSAL, RAG, SYSTEM_PROMPT, TOP_K
from evals.gold import gold_settings, load_chunks_by_doc
from evals.retrieval import evidence_gold, retrieval_config
from evals.runlog import fingerprint, git_state, model_revision, new_run_id, save_run
from evals.split import load_questions, load_split
from retrieval.search import SEARCH_MODES

TYPES = ["text", "table", "computation", "multi_hop", "unanswerable"]
REL_TOL, ABS_TOL = 0.01, 0.005  # 1% of the value, or half a unit in the 2nd decimal (for small ratios)
SCALES = (1, 1e3, 1e-3, 100, 0.01)  # million <-> billion/thousand, fraction <-> percent
CITATION = re.compile(r"\[([^\[\]]+)\]")
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def is_refusal(text: str) -> bool:
    """True if the answer contains the exact refusal sentence (ignoring case and spacing)."""
    return REFUSAL.lower().rstrip(".") in " ".join(text.lower().split())


def citations(text: str) -> list[str]:
    """Chunk IDs cited in square brackets; "[A, B]" counts as two citations."""
    return [cid.strip() for group in CITATION.findall(text) for cid in group.split(",") if cid.strip()]


def numbers(text: str) -> list[float]:
    """Every number in the answer, citations removed first (chunk IDs contain digits)."""
    return [float(n.replace(",", "")) for n in NUMBER.findall(CITATION.sub(" ", text))]


def number_correct(text: str, value: float) -> bool:
    """Any number in the answer equals the gold value within tolerance, allowing unit changes (sign ignored)."""
    tol = max(REL_TOL * abs(value), ABS_TOL)
    return any(abs(x * s - abs(value)) <= tol for x in numbers(text) for s in SCALES)


def yes_no_correct(text: str, gold: str) -> bool:
    """The answer's first yes/no word matches the gold answer's first word."""
    said = re.search(r"\b(yes|no)\b", CITATION.sub(" ", text), re.IGNORECASE)
    return bool(said) and said.group(1).lower() == gold.split()[0].strip(".,").lower()


def judge(q: dict, text: str) -> bool | None:
    """Correctness of one answer (None = not scored here: free text, or an unanswerable question)."""
    if q["answer_kind"] == "number":
        return not is_refusal(text) and number_correct(text, q["answer_value"])
    if q["answer_kind"] == "yes_no":
        return not is_refusal(text) and yes_no_correct(text, q["answer"])
    return None


def rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def summarize(rows: list[dict]) -> dict:
    """Metrics for a group of questions; None where the group has nothing to measure."""
    answerable = [r for r in rows if r["answer_kind"] != "refuse"]
    unanswerable = [r for r in rows if r["answer_kind"] == "refuse"]
    scored = [r for r in answerable if r["correct"] is not None]
    answered = [r for r in rows if not r["refused"]]
    cited = [r for r in answered if r["citations"]]
    found = [r for r in scored if r["evidence_retrieved"]]
    missing = [r for r in scored if not r["evidence_retrieved"]]
    return {
        "n": len(rows),
        "accuracy": rate(sum(r["correct"] for r in scored), len(scored)), "n_scored": len(scored),
        "accuracy_evidence_retrieved": rate(sum(r["correct"] for r in found), len(found)), "n_retrieved": len(found),
        "accuracy_evidence_missing": rate(sum(r["correct"] for r in missing), len(missing)),
        "false_refusal_rate": rate(sum(r["refused"] for r in answerable), len(answerable)),
        "refusal_recall": rate(sum(r["refused"] for r in unanswerable), len(unanswerable)),
        "refusal_precision": rate(sum(r["refused"] for r in unanswerable), sum(r["refused"] for r in rows)),
        "citation_rate": rate(len(cited), len(answered)),
        "citation_validity": rate(sum(not r["invalid_citations"] for r in cited), len(cited)),
        "seconds_per_question": round(sum(r["seconds"] for r in rows) / len(rows), 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--note", default="", help="what this run changes, for the results log")
    parser.add_argument("--limit", type=int, help="answer only this many random questions (a quick test; not logged)")
    parser.add_argument("--filter", default="none", choices=["none", "company_year"],
                        help="company_year: search only the filings the question names (default: all chunks)")
    parser.add_argument("--search", default="dense", choices=SEARCH_MODES, help="dense (baseline) or hybrid")
    parser.add_argument("--rerank", action="store_true", help="re-order the top candidates with a cross-encoder")
    args = parser.parse_args()

    split = load_split()
    questions = [q for q in load_questions() if split[q["id"]] == args.split]
    if args.limit:
        questions = random.Random(0).sample(questions, args.limit)
    by_doc = load_chunks_by_doc()
    rag = RAG(filter_filings=args.filter == "company_year", search=args.search, rerank=args.rerank)

    details = []
    for i, q in enumerate(questions, 1):
        start = time.perf_counter()
        text, chunks = rag.answer(q["question"], verbose=False)
        retrieved = {c["id"] for c in chunks}
        cited = citations(text)
        gold = evidence_gold(q, by_doc) if q["evidence"] else []
        row = {
            "id": q["id"], "source": q["source"], "type": q["type"], "answer_kind": q["answer_kind"],
            "question": q["question"], "gold_answer": q["answer"], "answer": text,
            "refused": is_refusal(text), "correct": judge(q, text),
            "citations": cited, "invalid_citations": [c for c in cited if c not in retrieved],
            "evidence_retrieved": all(piece & retrieved for piece in gold) if gold else None,
            "retrieved": [c["id"] for c in chunks], "seconds": round(time.perf_counter() - start, 1),
        }
        details.append(row)
        status = "refused" if row["refused"] else {True: "correct", False: "wrong", None: "unscored"}[row["correct"]]
        print(f"{i:>3}/{len(questions)}  {q['id']:<10} {q['type']:<12} {status:<9} {row['seconds']:>5.1f}s")

    groups = defaultdict(list)
    for row in details:
        for name in ("all", f"source={row['source']}", f"type={row['type']}"):
            groups[name].append(row)
    order = ["all", "source=financebench", "source=own"] + [f"type={t}" for t in TYPES]
    metrics = {name: summarize(groups[name]) for name in order if groups[name]}

    run = {
        "run_id": new_run_id("answers"), "stage": "answers", "split": args.split, "note": args.note,
        "limit": args.limit, **git_state(),
        "n_questions": len(questions),
        "fingerprint": fingerprint(questions, {
            "gold": gold_settings(),
            "scorer": {"rel_tol": REL_TOL, "abs_tol": ABS_TOL, "scales": SCALES, "refusal": REFUSAL}}),
        "config": retrieval_config(rag.retriever) | {"filter": args.filter} | {
            "gen_model": GEN_MODEL, "gen_revision": model_revision(GEN_MODEL), "top_k": TOP_K,
            "max_tokens": MAX_TOKENS, "thinking": False,
            "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]},
        "metrics": metrics,
    }
    save_run(run, details, log=not args.limit)

    print(f"\n{run['run_id']}  split={args.split}  filter={args.filter}  search={args.search}"
          f"{'  rerank' if args.rerank else ''}  commit={run['commit']}{' (dirty)' if run['dirty'] else ''}"
          f"{'  (limited run: not logged)' if args.limit else ''}")
    columns = {"n": "n", "accuracy": "acc", "accuracy_evidence_retrieved": "acc|found",
               "accuracy_evidence_missing": "acc|miss", "false_refusal_rate": "false_ref",
               "refusal_recall": "ref_recall", "refusal_precision": "ref_prec",
               "citation_rate": "cite_rate", "citation_validity": "cite_valid"}
    print(f"\n{'':<22}" + "".join(f"{h:>11}" for h in columns.values()))
    for name, m in metrics.items():
        cells = [f"{m[k]:>11}" if m[k] is not None else f"{'-':>11}" for k in columns]
        print(f"{name:<22}" + "".join(cells))
