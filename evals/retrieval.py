"""Retrieval eval: does search put each question's evidence in the top k?

Only answerable questions are scored (unanswerable ones have no evidence). A question's evidence pieces
(two or more for multi-hop) are scored separately, so finding one of two facts gives recall 0.5.
"""

import argparse
from collections import defaultdict

from evals.gold import load_chunks_by_doc, match_evidence
from evals.runlog import git_state, model_revision, new_run_id, save_run
from evals.split import load_questions, load_split
from retrieval.index import CHUNK_TOKENS, EMBED_MODEL, OVERLAP_TOKENS
from retrieval.search import Retriever

TYPES = ["text", "table", "computation", "multi_hop"]
KS = (1, 5, 10, 20)  # 5 = what the generator sees; 20 = the pool a reranker could reorder
METRICS = [f"recall@{k}" for k in KS] + ["mrr"]


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


def retrieval_config(n_chunks: int) -> dict:
    """The retrieval settings a run depends on (shared with the answer eval)."""
    return {"embed_model": EMBED_MODEL, "embed_revision": model_revision(EMBED_MODEL),
            "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS, "n_chunks": n_chunks}


def average(rows: list[dict]) -> dict:
    return {"n": len(rows)} | {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--note", default="", help="what this run changes, for the results log")
    args = parser.parse_args()

    split = load_split()
    questions = [q for q in load_questions() if split[q["id"]] == args.split and q["evidence"]]
    by_doc = load_chunks_by_doc()
    retriever = Retriever()

    details, unmatched = [], []
    for q in questions:
        gold = evidence_gold(q, by_doc)
        if not gold:
            unmatched.append(q["id"])
            continue
        retrieved = [c["id"] for c in retriever.search(q["question"], k=max(KS))]
        details.append({"id": q["id"], "source": q["source"], "type": q["type"], "question": q["question"],
                        "gold": [sorted(p) for p in gold], "retrieved": retrieved, **score(gold, retrieved)})

    groups = defaultdict(list)
    for row in details:
        for name in ("all", f"source={row['source']}", f"type={row['type']}"):
            groups[name].append(row)
    order = ["all", "source=financebench", "source=own"] + [f"type={t}" for t in TYPES]
    metrics = {name: average(groups[name]) for name in order if groups[name]}

    run = {
        "run_id": new_run_id("retrieval"), "stage": "retrieval", "split": args.split, "note": args.note, **git_state(),
        "config": retrieval_config(len(retriever.chunks)),
        "metrics": metrics,
    }
    save_run(run, details)

    print(f"{run['run_id']}  split={args.split}  commit={run['commit']}{' (dirty)' if run['dirty'] else ''}")
    if unmatched:
        print(f"Skipped (evidence matches no chunk): {unmatched}")
    print(f"\n{'':<24}{'n':>5}" + "".join(f"{m:>11}" for m in METRICS))
    for name, m in metrics.items():
        print(f"{name:<24}{m['n']:>5}" + "".join(f"{m[k]:>11.3f}" for k in METRICS))
