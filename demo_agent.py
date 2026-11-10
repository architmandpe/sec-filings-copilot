"""Live agent demo: ask a question, watch the agent's tool calls and checks, read the cited answer.

Run from the project root, on the Mac that has the models:  python demo_agent.py   (opens http://127.0.0.1:7860)

app.py shows the RAG pipeline; this page shows the agent (agent/loop.py), the best system measured so far.
The agent is not changed: the page runs Agent.answer on a background thread and reads agent.trace, the list the
loop already fills, so the eval results describe exactly what is on screen.
"""

import os

# Set before anything imports Hugging Face or Gradio: every model is already on disk, so a bad connection during
# the demo can't slow down or break start-up.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import html
import time
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from agent.answer import GEN_MODEL
from agent.loop import MAX_NUDGES, Agent
from evals.answers import citations, is_refusal

POLL_SECONDS = 0.25  # how often the page looks for new agent steps
# Dev questions from the agent's logged run 20261107-000812: an exact number (as filed, before a later restatement),
# a computation, a two-company comparison, a text lookup, and two refusals
EXAMPLES = [
    "What were Johnson & Johnson's sales to customers in fiscal 2022?",
    "What was Apple's gross margin percentage in fiscal 2024?",
    "Which company spent more on research and development in fiscal 2023, Apple or Microsoft, and by how much?",
    "Where are Apple's outsourcing partners that manufacture its hardware products primarily located, "
    "according to its fiscal 2024 10-K?",
    "How much was Apple's CEO paid in fiscal 2024?",
    "What will Nike's revenue be in fiscal 2030?",
]
CHECKS = {  # trace entries written when code sends a would-be final answer back to the model
    "nudge_no_tool": "no tool has been used yet",
    "nudge_no_citation": "the reply cites no source",
    "nudge_unsupported_numbers": "a number in the reply is not in any tool result",
}

# One thread does all the model work: it loads the agent, then answers one question at a time. The page's own
# thread stays free to redraw while a question runs.
worker = ThreadPoolExecutor(max_workers=1)
print("Loading the models and the index...")
agent = worker.submit(Agent).result()


def fold(summary: str, text: str) -> str:
    """Text hidden under a click-to-open line, so long passages don't bury the steps."""
    return (f"<details><summary>{summary}</summary>\n\n<pre style='white-space: pre-wrap'>{html.escape(text)}</pre>"
            "\n\n</details>")


def step_markdown(number: int, step: dict) -> str:
    """One entry of agent.trace as a block on the page."""
    tool, output = step["tool"], step["output"]
    if tool in CHECKS:
        return f"**{number}. Check failed:** {CHECKS[tool]}. Sent back to the model.\n\n{fold('The reply it sent back', output)}"
    if tool == "step_limit":
        return f"**{number}. Step limit reached:** the model must answer from what it has."
    arguments = ", ".join(str(value) for value in step["arguments"].values())
    heading = f"**{number}. {tool}** `{arguments}`"
    if tool == "search_filings":
        return f"{heading}\n\n{fold('Start of the results', output)}"
    return f"{heading}\n\n{output}"


def steps_markdown(steps: list[dict], running: bool) -> str:
    """Everything the agent has done so far for this question."""
    parts = ["### What the agent did"] + [step_markdown(n, step) for n, step in enumerate(steps, 1)]
    if running:
        parts.append("*The model is working...*")
    elif sum(step["tool"] in CHECKS for step in steps) >= MAX_NUDGES:
        parts.append(f"*{MAX_NUDGES} send-backs used, so the final reply was accepted without another check.*")
    return "\n\n".join(parts)


def answer_markdown(text: str, seconds: float) -> str:
    heading = "Refused" if is_refusal(text) else "Answer"
    return f"### {heading}\n\n{text}\n\n*{seconds:.0f} s on this laptop*"


def sources_markdown(text: str, found: list[dict]) -> str:
    """Each source the answer cites: the XBRL fact with its filing, or the passage with a link to EDGAR."""
    by_id = {source["id"]: source for source in found}
    parts = []
    for cited in dict.fromkeys(citations(text)):
        source = by_id.get(cited)
        if source is None:
            parts.append(f"**`{cited}`**: not in any tool result (invented citation).")
        elif source["form"] == "XBRL":
            parts.append(f"**`{cited}`**: {source['text']}")
        else:
            parts.append(f"**`{cited}`**: {source['company']}, {source['form']} FY{source['fiscal_year']}, "
                         f"Item {source['item']} {source['title']} · [filing on EDGAR]({source['url']})\n\n"
                         f"{fold('Show text', source['text'])}")
    return "### Sources cited\n\n" + "\n\n".join(parts) if parts else ""


def ask(question: str):
    """Yield (steps, answer, sources) for the page: the steps as they happen, then the answer and its sources."""
    question = question.strip()
    if not question:
        yield "", "Type a question first.", ""
        return
    agent.trace = []  # so the page never shows the previous question's steps
    start = time.perf_counter()
    job = worker.submit(agent.answer, question)
    shown = None
    while not job.done():
        steps = list(agent.trace)
        if len(steps) != shown:
            shown = len(steps)
            yield steps_markdown(steps, running=True), "", ""
        time.sleep(POLL_SECONDS)
    try:
        text, found = job.result()
    except Exception as error:  # show it on the page; the models stay loaded for the next question
        yield steps_markdown(agent.trace, running=False), f"### Error\n\n`{error}`", ""
        return
    seconds = time.perf_counter() - start
    yield steps_markdown(agent.trace, running=False), answer_markdown(text, seconds), sources_markdown(text, found)


with gr.Blocks(title="SEC Filings Copilot: agent") as demo:
    companies = sorted(agent.financials.ciks)
    gr.Markdown(
        "# SEC Filings Copilot: agent\n"
        f"Questions about 75 10-K and 10-Q filings. A local model (`{GEN_MODEL.split('/')[-1]}`) calls tools for exact "
        "XBRL numbers, filing search and arithmetic, and code checks its answer before accepting it, or it refuses. "
        "The number tool covers fiscal years, not quarters.\n\n"
        + fold(f"The {len(companies)} companies covered", ", ".join(companies)))
    question = gr.Textbox(label="Question", placeholder=EXAMPLES[1])
    button = gr.Button("Ask", variant="primary")
    gr.Examples(EXAMPLES, inputs=question)
    with gr.Row():
        with gr.Column():
            steps = gr.Markdown()
        with gr.Column():
            answer = gr.Markdown()
            sources = gr.Markdown()
    # One event for the button and the Enter key, so a second question waits for the first to finish
    gr.on(triggers=[button.click, question.submit], fn=ask, inputs=question, outputs=[steps, answer, sources])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
