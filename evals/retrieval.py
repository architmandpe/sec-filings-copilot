"""Retrieval eval: does search put each question's evidence in the top k?

Only answerable questions are scored (unanswerable ones have no evidence). A question's evidence pieces
(two or more for multi-hop) are scored separately, so finding one of two facts gives recall 0.5.
"""

import argparse
import time
from collections import defaultdict

from evals.gold import gold_settings, load_chunks_by_doc, match_evidence
from evals.runlog import fingerprint, git_state, model_revision, new_run_id, save_run
from evals.split import load_questions, load_split
from retrieval.index import CHUNK_TOKENS, EMBED_MODEL, OVERLAP_TOKENS
from retrieval.filters import FilingFilter
from retrieval.search import RERANK_CANDIDATES, SEARCH_MODES, Retriever

TYPES = ["text", "table", "computation", "multi_hop"]
KS = (1, 5, 10, 20)  # 5 = what the generator sees; 20 = the pool a reranker could reorder
METRICS = [f"recall@{k}" for k in KS] + ["mrr"]
FILTERS = ("none", "oracle", "company", "company_year")  # which filings a question may retrieve from (none = baseline)


def evidence_gold(q: dict, by_doc: dict) -> list[set[str]]:
    """One set of gold chunk IDs per evidence piece (pieces that match no chunk are dropped)."""
    pieces = [set(match_evidence(e["text"], by_doc[e["doc_name"]])[0]) for e in q["evidence"]]
    return [p for p in pieces if p]


def score(gold: list[set[str]], retrieved: list[str]) -> dict:
    """recall@k: share of evidence pieces with a gold chunk in the top k. mrr: 1 / rank of the first gold chunk (0 if none)."""
    ranks = [min((i + 1 for i, cid in enumerate(retrieved) if cid in piece), default=None) for piece in gold]
    found = [r for r in ranks if r is not None]
    row = {f"recall@{k}": sum(r <= k for r in found) / len(gold) for k in KS}
    row["mrr"] = 1 / min(found) if found else 0.0
    return row


def retrieval_config(retriever: Retriever) -> dict:
    """The retrieval settings a run depends on (shared with the answer eval)."""
    cfg = {"embed_model": EMBED_MODEL, "embed_revision": model_revision(EMBED_MODEL),
           "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS, "n_chunks": len(retriever.chunks)}
    cfg |= retriever.config()
    if cfg["rerank_model"]:
        cfg["rerank_revision"] = model_revision(cfg["rerank_model"])
    return cfg


def average(rows: list[dict]) -> dict:
    return {"n": len(rows)} | {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--note", default="", help="what this run changes, for the results log")
    parser.add_argument("--filter", default="none", choices=FILTERS,
                        help="none: search all chunks (baseline); oracle: only the question's evidence filings, "
                             "taken from the gold labels (an upper bound, not a real system); company / company_year: "
                             "filings inferred from the question text (retrieval/filters.py)")
    parser.add_argument("--search", default="dense", choices=SEARCH_MODES,
                        help="dense: embeddings only (baseline); hybrid: dense + BM25 merged by reciprocal rank fusion")
    parser.add_argument("--rerank", action="store_true", help="re-order the top candidates with a cross-encoder")
    parser.add_argument("--candidates", type=int, default=RERANK_CANDIDATES,
                        help="how many first-stage chunks the reranker re-orders (default: %(default)s)")
    args = parser.parse_args()

    split = load_split()
    questions = [q for q in load_questions() if split[q["id"]] == args.split and q["evidence"]]
    by_doc = load_chunks_by_doc()
    retriever = Retriever(mode=args.search, rerank=args.rerank, candidates=args.candidates)
    filing_filter = FilingFilter(retriever.chunks)

    def filter_docs(q: dict) -> set[str] | None:
        """The filings this question may retrieve from under --filter (None = all)."""
        if args.filter == "oracle":
            return {e["doc_name"] for e in q["evidence"]}
        if args.filter in ("company", "company_year"):
            return filing_filter.infer(q["question"], use_year=args.filter == "company_year")
        return None

    details, unmatched, scored, search_seconds = [], [], [], []
    for q in questions:
        gold = evidence_gold(q, by_doc)
        if not gold:
            unmatched.append(q["id"])
            continue
        scored.append(q)
        doc_names = filter_docs(q)
        start = time.perf_counter()
        retrieved = [c["id"] for c in retriever.search(q["question"], k=max(KS), doc_names=doc_names)]
        search_seconds.append(time.perf_counter() - start)
        evidence_docs = {e["doc_name"] for e in q["evidence"]}
        details.append({"id": q["id"], "source": q["source"], "type": q["type"], "question": q["question"],
                        "gold": [sorted(p) for p in gold], "retrieved": retrieved, **score(gold, retrieved),
                        "filter_docs": sorted(doc_names) if doc_names is not None else None,
                        "filter_kept_evidence": doc_names is None or evidence_docs <= doc_names})

    groups = defaultdict(list)
    for row in details:
        for name in ("all", f"source={row['source']}", f"type={row['type']}"):
            groups[name].append(row)
    order = ["all", "source=financebench", "source=own"] + [f"type={t}" for t in TYPES]
    metrics = {name: average(groups[name]) for name in order if groups[name]}
    filtered = [r for r in details if r["filter_docs"] is not None]
    filter_stats = {  # did the filter keep the evidence filings, and how much did it narrow the search?
        "kept_evidence": round(sum(r["filter_kept_evidence"] for r in details) / len(details), 3),
        "unfiltered": len(details) - len(filtered),  # no company found: searched everything
        "avg_filings": round(sum(len(r["filter_docs"]) for r in filtered) / len(filtered), 2) if filtered else None,
    }

    run = {
        "run_id": new_run_id("retrieval"), "stage": "retrieval", "split": args.split, "note": args.note, **git_state(),
        "n_questions": len(scored),
        "fingerprint": fingerprint(scored, {"gold": gold_settings(), "ks": KS}),
        "config": retrieval_config(retriever) | {"filter": args.filter},
        "seconds_per_question": round(sum(search_seconds) / len(search_seconds), 3),
        "metrics": metrics,
        "filter_stats": filter_stats,
    }
    save_run(run, details)

    print(f"{run['run_id']}  split={args.split}  filter={args.filter}  search={args.search}"
          f"{f'  rerank top {args.candidates}' if args.rerank else ''}  {run['seconds_per_question']} s/question  commit={run['commit']}{' (dirty)' if run['dirty'] else ''}")
    if unmatched:
        print(f"Skipped (evidence matches no chunk): {unmatched}")
    if args.filter != "none":
        lost = [r["id"] for r in details if not r["filter_kept_evidence"]]
        print(f"Filter kept the evidence filings for {filter_stats['kept_evidence']:.1%} of questions; "
              f"{filter_stats['unfiltered']} searched everything (no company found); "
              f"avg {filter_stats['avg_filings']} filings searched; lost evidence: {lost or 'none'}")
    print(f"\n{'':<24}{'n':>5}" + "".join(f"{m:>11}" for m in METRICS))
    for name, m in metrics.items():
        print(f"{name:<24}{m['n']:>5}" + "".join(f"{m[k]:>11.3f}" for k in METRICS))
