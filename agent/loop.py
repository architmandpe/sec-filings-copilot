"""The agent: Qwen decides which tools to call, a hand-written loop runs them, and the model answers from the results.

    question -> model -> tool calls? -- yes -> run them, add the results to the conversation -> model -> ...
                                     \\- no  -> final answer (cited) or refusal

Tools (Qwen3's native tool-calling format: the chat template lists them, the model replies with
<tool_call>{"name": ..., "arguments": {...}}</tool_call>):
  search_filings(query)                          passages from the filings (Week 3 retrieval), cited by chunk ID
  get_financial(company, metric, fiscal_year)    exact XBRL numbers (agent/financials.py), cited by fact ID
  calculate(expression)                          exact arithmetic (agent/calculate.py)
XBRL results are returned as pseudo-chunks (fact ID + one-line description), so citation checks and the judge
treat them like passages. No framework: the loop is the code below (decision D10).

    python -m agent.loop        # sample questions with the tool trace
"""

import json
import re

from agent.answer import GEN_MODEL, REFUSAL, TOP_K, format_context
from agent.calculate import calculate
from agent.financials import METRICS, Financials, describe
from retrieval.filters import FilingFilter
from retrieval.search import Retriever

NO_TOOL_NUDGE = ("You have not used any tool yet. Call search_filings or get_financial first, then answer from "
                 "what they return. If they don't answer the question, reply with exactly: \"{refusal}\" followed by "
                 "one sentence saying what is missing.")
FILING_NUDGE = ("The question asks about the {year} filing, but your sources are from fiscal {found}, and the {year} "
                "filing is not in the collection. Unless your sources state what the question asks about {year}, "
                "reply with exactly: \"{refusal}\" followed by one sentence saying what is missing.")
NO_CITATION_NUDGE = ("Your reply has no citation. If you still need data, call the tool now; otherwise give the "
                     "final answer citing IDs in square brackets.")
UNSUPPORTED_NUDGE = ("These numbers are not in any tool result: {numbers}. Compute them with calculate (or fetch them "
                     "with a tool), then give the final answer.")
MAX_NUDGES = 2  # send-backs per question, so a stubborn model can't loop
MAX_STEPS = 6  # model turns; a question needing more tool rounds than this is answered with what it has
STEP_TOKENS = 512  # tokens per model turn
TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# a question about one specific filing: "its 2019 10-K", "fiscal 2020 10-K", "FY2022 annual report"
NAMED_FILING = re.compile(r"(?:fiscal\s+|FY\s?)?(20\d\d)\s+(?:annual report|form 10-K|10-K|10-Q)", re.IGNORECASE)

TOOLS = [
    {"type": "function", "function": {
        "name": "get_financial",
        "description": "Exact number for one standard financial line item of one company and fiscal year, from XBRL "
                       "data as reported in that year's 10-K, in USD millions. Use it for any of these metrics: "
                       + "; ".join(f"{m} = {desc}" for m, (_, _, desc) in METRICS.items()),
        "parameters": {"type": "object", "properties": {
            "company": {"type": "string", "description": "company name, e.g. 3M, Coca-Cola"},
            "metric": {"type": "string", "enum": list(METRICS)},
            "fiscal_year": {"type": "integer", "description": "the calendar year the fiscal year ends in"}},
            "required": ["company", "metric", "fiscal_year"]}}},
    {"type": "function", "function": {
        "name": "search_filings",
        "description": f"Search the 10-K and 10-Q filings; returns the {TOP_K} most relevant passages with their IDs. "
                       "Use it for anything get_financial doesn't cover: risks, segments, strategy, auditors, "
                       "employees, explanations of changes, and line items not in the metric list. Name the "
                       "company and year in the query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "what to look for, e.g. 'Apple 2024 supply chain risks'"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression exactly. Numbers without thousands separators; "
                       "allowed: + - * / ** ( ) and avg, min, max, abs, round. Example: avg(1182, 1310) / 8975 * 100",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string"}}, "required": ["expression"]}}},
]


def system_prompt(companies: list[str]) -> str:
    return f"""You answer questions about companies using their SEC 10-K and 10-Q filings, through tools.

Companies available: {", ".join(companies)}.

Rules:
1. Use only what the tools return. Do not use outside knowledge.
2. For a standard financial line item, call get_financial. For anything else, call search_filings.
3. Do all arithmetic with calculate, never in your head. If the question defines a formula, follow it exactly; "average X between FY2018 and FY2019" is the average of the two year-end values.
4. In the final answer, cite every fact with its ID in square brackets: passage IDs from search_filings, fact IDs from get_financial. Put only IDs inside the brackets, separated by commas, e.g. [XBRL_3M_2022_revenue, XBRL_3M_2022_capex]; write formulas outside the brackets.
5. State numbers with units, rounded as the question asks.
6. If the tools don't give you the answer (forecasts, companies not listed, information not in 10-K or 10-Q filings such as executive pay), reply with exactly: "{REFUSAL}" followed by one sentence saying what is missing.
7. Keep the final answer short."""


def unsupported_numbers(answer: str, evidence: str) -> list[str]:
    """Numbers in the answer that no number in the evidence (tool outputs + question) rounds to.

    Rounding: 24.26 is supported by 24.2579 (half a unit of its last digit); a percentage may come from a fraction
    (46.2 from 0.462). Citations are removed first, because IDs contain digits."""
    from evals.answers import CITATION, NUMBER  # the answer scorer's number pattern: same definition of a number
    known = [float(n.replace(",", "")) for n in NUMBER.findall(evidence)]
    missing = []
    for raw in NUMBER.findall(CITATION.sub(" ", answer)):
        value = float(raw.replace(",", ""))
        half_unit = 0.5 * 10 ** -(len(raw.split(".")[1]) if "." in raw else 0)
        if not any(abs(k * scale - value) <= half_unit for k in known for scale in (1, 100)):
            missing.append(raw)
    return missing


def input_sources(expression: str, sources: dict[str, dict]) -> str:
    """Where the numbers in a calculation came from: lines "6489 = [XBRL_..._revenue] <fact>" for every input that
    equals a number in an XBRL fact or earlier calculation, so citing the calculation also shows its inputs' sources."""
    from evals.answers import NUMBER
    lines = []
    # expressions have no thousands separators (calculate rejects them), so "avg(100,200)" is two numbers, not 100200
    for raw in dict.fromkeys(re.findall(r"\d+(?:\.\d+)?", expression)):
        value = float(raw.replace(",", ""))
        for sid, source in sources.items():
            if source["form"] not in ("XBRL", "calculation"):
                continue
            first = source["text"].splitlines()[0]
            # only the value, not dates or accession numbers: "...: $6,489 million (2019-01-01 ..." / "... = 24.2579"
            shown = first.split(" = ")[-1] if source["form"] == "calculation" else first.split(": ", 1)[-1].split(" (")[0]
            if any(abs(float(n.replace(",", "")) - value) < 1e-9 for n in NUMBER.findall(shown)):
                lines.append(f"{raw} = [{sid}] {source['text'].splitlines()[0]}")
                break
    return "".join(f"\n  {line}" for line in lines)


def fact_id(result: dict) -> str:
    """Citable ID for an XBRL fact, e.g. XBRL_JohnsonJohnson_2022_revenue."""
    return f"XBRL_{re.sub(r'[^A-Za-z0-9]', '', result['company'])}_{result['fiscal_year']}_{result['metric']}"


class Agent:
    """Holds the retriever, the XBRL tool and the generator; answer() runs the loop for one question."""

    def __init__(self, search: str = "hybrid", rerank: bool = True):
        from mlx_lm import load
        self.retriever = Retriever(mode=search, rerank=rerank)
        self.filing_filter = FilingFilter(self.retriever.chunks)
        self.financials = Financials()
        self.system = system_prompt(sorted(self.financials.ciks))
        self.model, self.tokenizer = load(GEN_MODEL)
        self.trace = []  # tool calls of the last question, for evals and debugging

    def run_tool(self, name: str, args: dict, question: str, sources: dict) -> str:
        """Run one tool call; add what it returned to `sources` (ID -> chunk); return the text the model sees."""
        if name == "search_filings":
            query = str(args.get("query", question))
            doc_names = self.filing_filter.infer(query) or self.filing_filter.infer(question)
            chunks = self.retriever.search(query, k=TOP_K, doc_names=doc_names)
            sources.update({c["id"]: c for c in chunks})
            return format_context(chunks)
        if name == "get_financial":
            try:
                result = self.financials.get(str(args["company"]), str(args["metric"]), int(args["fiscal_year"]))
            except (KeyError, ValueError, TypeError) as e:
                return f"Error: get_financial needs company, metric and fiscal_year ({e})"
            if "error" in result:
                return describe(result)
            fid = fact_id(result)
            sources[fid] = {"id": fid, "company": result["company"], "form": "XBRL", "fiscal_year": result["fiscal_year"],
                            "fiscal_period": "FY", "item": "-", "title": result["metric"], "text": describe(result)}
            return f"[{fid}] {describe(result)}"
        if name == "calculate":
            result = calculate(str(args.get("expression", "")))
            if "error" in result:
                return f"Error: {result['error']}"
            # a citable source, like an XBRL fact, so a computed number in the answer can be checked against it
            cid = f"CALC-{sum(1 for k in sources if k.startswith('CALC-')) + 1}"
            text = f"{result['expression']} = {result['value']:.6g}" + input_sources(result["expression"], sources)
            sources[cid] = {"id": cid, "company": "", "form": "calculation", "fiscal_year": "", "fiscal_period": "",
                            "item": "-", "title": "calculator", "text": text}
            return f"[{cid}] {text}"
        return f"Error: unknown tool {name!r}; use get_financial, search_filings or calculate"

    def generate(self, messages: list[dict]) -> str:
        from mlx_lm import generate
        prompt = self.tokenizer.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True,
                                                    enable_thinking=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=STEP_TOKENS)

    def check_final(self, reply: str, question: str, messages: list[dict],
                    sources: dict[str, dict]) -> tuple[str, str] | None:
        """A reason to send a would-be final answer back (name, message to the model), or None to accept it."""
        from evals.answers import citations, is_refusal
        tool_outputs = [m["content"] for m in messages if m["role"] == "tool"]
        if not tool_outputs:
            return "nudge_no_tool", NO_TOOL_NUDGE.format(refusal=REFUSAL)
        if is_refusal(reply):
            return None
        cited = citations(reply)
        if not cited:
            return "nudge_no_citation", NO_CITATION_NUDGE
        named = NAMED_FILING.search(question)
        passages = [sources[c] for c in cited if c in sources and sources[c]["form"] in ("10-K", "10-Q")]
        if named and passages and all(str(p["fiscal_year"]) != named.group(1) for p in passages):
            found = ", ".join(sorted({str(p["fiscal_year"]) for p in passages}))
            return "nudge_filing_year", FILING_NUDGE.format(year=named.group(1), found=found, refusal=REFUSAL)
        missing = unsupported_numbers(reply, "\n".join(tool_outputs + [question]))
        if missing:
            return "nudge_unsupported_numbers", UNSUPPORTED_NUDGE.format(numbers=", ".join(missing))
        return None

    def answer(self, question: str, verbose: bool = False) -> tuple[str, list[dict]]:
        """Run the loop; return the final answer and every source the tools returned (same shape as RAG.answer)."""
        messages = [{"role": "system", "content": self.system}, {"role": "user", "content": question}]
        sources, self.trace, nudges = {}, [], 0
        for step in range(MAX_STEPS):
            reply = self.generate(messages)
            calls = TOOL_CALL.findall(reply)
            if not calls:
                nudge = self.check_final(reply, question, messages, sources) if nudges < MAX_NUDGES else None
                if nudge is None:
                    return reply.strip(), list(sources.values())
                nudges += 1
                messages += [{"role": "assistant", "content": reply.strip()}, {"role": "user", "content": nudge[1]}]
                self.trace.append({"step": step, "tool": nudge[0], "arguments": {}, "output": reply[:300]})
                if verbose:
                    print(f"  [{step}] {nudge[0]}: {nudge[1][:120]!r}")
                continue
            parsed = []
            for raw in calls:
                try:
                    call = json.loads(raw)
                    parsed.append((str(call["name"]), call.get("arguments") or {}))
                except (json.JSONDecodeError, KeyError, TypeError):
                    parsed.append(("invalid", {"raw": raw}))
            messages.append({"role": "assistant", "content": TOOL_CALL.sub("", reply).strip(),
                             "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}
                                            for n, a in parsed]})
            for name, args in parsed:
                output = (f"Error: could not parse the tool call {args['raw']!r} as JSON" if name == "invalid"
                          else self.run_tool(name, args, question, sources))
                messages.append({"role": "tool", "content": output})
                self.trace.append({"step": step, "tool": name, "arguments": args, "output": output[:300]})
                if verbose:
                    print(f"  [{step}] {name}({json.dumps(args)}) -> {output[:150]!r}")
        # step limit reached: ask for the answer from what has been gathered, without further tools
        messages.append({"role": "user", "content": "Give your final answer now from the tool results above."})
        reply = self.generate(messages)
        self.trace.append({"step": MAX_STEPS, "tool": "step_limit", "arguments": {}, "output": ""})
        return TOOL_CALL.sub("", reply).strip(), list(sources.values())


if __name__ == "__main__":
    agent = Agent()
    for question in [
        "What was Apple's gross margin percentage in fiscal 2024?",
        "What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined "
        "as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places.",
        "Where are Apple's outsourcing partners that manufacture its hardware products primarily located, "
        "according to its fiscal 2024 10-K?",
        "What will Nike's revenue be in fiscal 2030?",
    ]:
        print(f"\n### {question}")
        text, sources = agent.answer(question, verbose=True)
        print(text)
