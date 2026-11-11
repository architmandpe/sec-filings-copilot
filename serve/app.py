"""HTTP API for the agent: what curl, a UI and (later) Docker, the answer cache and Langfuse talk to.

    python -m serve.app        # loads the models once, then serves http://127.0.0.1:8000 (interactive docs at /docs)

Endpoints
    GET  /health     is the service up, and what it runs (model, retrieval settings, git commit)
    POST /search     passages for a query: the agent's search_filings tool on its own, no LLM
    POST /ask        the agent answers, streamed as server-sent events (each `data:` is JSON):
                       start   {request_id, question}
                       step    {step, tool, arguments, output}: one agent.trace entry, sent as it happens (a tool
                               call with its output cut to 300 characters, a failed check "nudge_...", or "step_limit")
                       answer  {request_id, answer, refused, citations, sources, unknown_citations, wait_seconds, seconds}
                       error   {request_id, error}
    POST /feedback   thumbs up/down and a comment on an /ask answer, by its request_id

All model work (Qwen through MLX, the embedding model, the reranker) runs on one worker thread, one job at a time: two
jobs would not make the GPU faster, the Agent keeps per-question state (agent.trace), and MLX is used from the thread
that loaded it. Requests that arrive during a question wait in the worker's queue; the web thread stays free, so
/health answers at once. Run one server process only: each process would load its own copy of the models.
A client that disconnects stops its question: skipped if it is still waiting, else ended at the agent's next step
(a model turn already running finishes first), so abandoned questions don't hold up the queue.

Every /search, /ask and /feedback call is appended to LOG_PATH, one JSON object per line.
"""

import os

# Set before anything imports Hugging Face: every model is already on disk, so start-up never waits on the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import asyncio
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, ConfigDict, Field

from agent.answer import GEN_MODEL, TOP_K
from agent.loop import Agent
from evals.answers import citations, is_refusal
from evals.runlog import git_state

HOST, PORT = "127.0.0.1", 8000  # this machine only: nothing else on the network can reach the API
LOG_PATH = Path("data/serve/requests.jsonl")
# what a client sees of each source (XBRL facts and calculator results have no url, so theirs is null)
SOURCE_FIELDS = ("id", "company", "form", "fiscal_year", "fiscal_period", "item", "title", "url", "text")

worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")


class ClientLeft(Exception):
    """Raised on the model thread to stop a question whose client has disconnected."""


class Request(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)  # strip text first, so "   " fails min_length like ""


class SearchRequest(Request):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=TOP_K, ge=1, le=20)


class AskRequest(Request):
    question: str = Field(min_length=1, max_length=2000)


class FeedbackRequest(Request):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")  # the request_id of the /ask answer being rated
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=2000)


def new_id() -> str:
    """A random 32-hex-character request ID (the same format as a Langfuse trace ID, so 5.4 can reuse it)."""
    return uuid.uuid4().hex


def elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 2)


def log(record: dict) -> None:
    """Append one request to the log. Only the web thread calls this, so lines never interleave."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record, "git": app.state.git}
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(line) + "\n")


def answered_ids(path: Path) -> set[str]:
    """Request IDs of the answers in the log, so /feedback accepts answers given before a restart."""
    if not path.exists():
        return set()
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    return {r["request_id"] for r in rows if r["endpoint"] == "/ask" and r["status"] == "ok"}


def source_view(source: dict) -> dict:
    return {field: source.get(field) for field in SOURCE_FIELDS}


def answer_view(text: str, sources: list[dict]) -> dict:
    """The answer with the sources it cites; cited IDs that no tool returned are listed separately."""
    by_id = {source["id"]: source for source in sources}
    cited = list(dict.fromkeys(citations(text)))  # unique, in order of first mention
    return {"answer": text, "refused": is_refusal(text), "citations": cited,
            "sources": [source_view(by_id[c]) for c in cited if c in by_id],
            "unknown_citations": [c for c in cited if c not in by_id]}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once: load everything before the first request is accepted, release the worker at shutdown."""
    print("Loading the index and the models...", flush=True)
    app.state.agent = await asyncio.get_running_loop().run_in_executor(worker, Agent)  # on the model thread
    app.state.git = git_state()  # the code this server runs, recorded in every log line
    app.state.answered = answered_ids(LOG_PATH)
    app.state.started = time.time()
    yield
    worker.shutdown(cancel_futures=True)


app = FastAPI(title="SEC Filings Copilot", lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    agent = app.state.agent
    return {"status": "ok", "generator": GEN_MODEL, "retrieval": agent.retriever.config(),
            "chunks": len(agent.retriever.chunks), "companies": len(agent.financials.ciks), **app.state.git,
            "uptime_seconds": round(time.time() - app.state.started)}


@app.post("/search")
async def search(body: SearchRequest) -> dict:
    request_id, start, agent = new_id(), time.perf_counter(), app.state.agent

    def job():
        doc_names = agent.filing_filter.infer(body.query)  # the same filing filter as the search_filings tool
        return doc_names, agent.retriever.search(body.query, k=body.k, doc_names=doc_names)

    record = {"request_id": request_id, "endpoint": "/search", "query": body.query, "k": body.k, "status": "error"}
    try:
        doc_names, chunks = await asyncio.get_running_loop().run_in_executor(worker, job)
        record.update(status="ok", results=[c["id"] for c in chunks])
    except Exception as error:
        record["error"] = repr(error)
        raise  # FastAPI answers 500 and prints the traceback
    finally:
        record["seconds"] = elapsed(start)
        log(record)
    return {"request_id": request_id, "query": body.query, "filings": sorted(doc_names) if doc_names else None,
            "results": [{**source_view(c), "score": round(c["score"], 4)} for c in chunks],
            "seconds": record["seconds"]}


@app.post("/ask", response_class=EventSourceResponse)
async def ask(body: AskRequest):
    request_id, start = new_id(), time.perf_counter()
    loop, events = asyncio.get_running_loop(), asyncio.Queue()

    client_left = threading.Event()  # set when the client disconnects, so the model thread stops working for it

    def send(kind: str, payload=None) -> None:
        """Called on the model thread: put an event in this request's queue, safely, on the web thread."""
        loop.call_soon_threadsafe(events.put_nowait, (kind, payload))

    def on_step(step: dict) -> None:
        if client_left.is_set():
            raise ClientLeft  # ends agent.answer at this step instead of finishing an answer nobody will read
        send("step", step)

    def job() -> None:
        if client_left.is_set():  # the client left while this question waited in the queue
            return
        send("running")
        try:
            send("answer", app.state.agent.answer(body.question, on_step=on_step))
        except ClientLeft:
            pass
        except Exception as error:  # reported to the client; the models stay loaded for the next question
            send("error", error)

    worker.submit(job)
    # "disconnected" stays if the client leaves before the answer
    record = {"request_id": request_id, "endpoint": "/ask", "question": body.question, "status": "disconnected",
              "wait_seconds": None, "trace": []}
    try:
        yield ServerSentEvent(event="start", data={"request_id": request_id, "question": body.question})
        while True:
            kind, payload = await events.get()
            if kind == "running":  # the worker picked the question up; until now it waited behind other requests
                record["wait_seconds"] = elapsed(start)
            elif kind == "step":
                record["trace"].append(payload)
                yield ServerSentEvent(event="step", data=payload)
            elif kind == "answer":
                result = answer_view(*payload)
                record.update(status="ok", **{k: result[k] for k in ("answer", "refused", "citations",
                                                                     "unknown_citations")})
                app.state.answered.add(request_id)
                yield ServerSentEvent(event="answer", data={"request_id": request_id, **result,
                                                            "wait_seconds": record["wait_seconds"],
                                                            "seconds": elapsed(start)})
                return
            else:
                record.update(status="error", error=repr(payload))
                yield ServerSentEvent(event="error", data={"request_id": request_id, "error": str(payload)})
                return
    finally:
        if record["status"] == "disconnected":
            client_left.set()
        record["seconds"] = elapsed(start)
        log(record)


@app.post("/feedback")
async def feedback(body: FeedbackRequest) -> dict:
    request_id, known = new_id(), body.request_id in app.state.answered
    log({"request_id": request_id, "endpoint": "/feedback", "answer_request_id": body.request_id,
         "rating": body.rating, "comment": body.comment, "status": "ok" if known else "unknown_request_id"})
    if not known:
        raise HTTPException(status_code=404, detail=f"no answered /ask request with id {body.request_id}")
    return {"request_id": request_id, "status": "recorded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
