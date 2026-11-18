"""The generator (Qwen3-8B) behind one interface, so the agent works the same with the model in this process or in a
separate model server.

    LocalLLM()        Qwen loaded in this process with mlx_lm (Apple silicon only): evals, demos, the API by default
    ServerLLM(url)    Qwen in a separate mlx_lm.server process, reached over HTTP: for the API inside Docker (step 5.5),
                      which can't use the Mac's GPU or install MLX
    llm_from_env()    ServerLLM if the LLM_URL environment variable is set, else LocalLLM

Both have chat(messages, tools, max_tokens) -> the model's raw reply (tool calls as Qwen's <tool_call> blocks), with
thinking off and greedy decoding, so agent/loop.py treats both the same.

Start the model server on the Mac, in its own terminal:
    mlx_lm.server --model mlx-community/Qwen3-8B-4bit --port 8080 --prompt-cache-size 0

Measured on 6 dev questions (step 5.2): ServerLLM gives answers and tool traces identical to the in-process eval run
(6/6) with --prompt-cache-size 0. With the server's default prompt cache (it reuses its computation for the start of
a conversation it has already seen) it was 1.6x faster, but 2 of 6 answers changed: the same arithmetic done in a
different order changes the last digits, which can flip a near-tie token.
"""

import json
import os

import requests

from agent.answer import GEN_MODEL

LLM_URL_VAR = "LLM_URL"  # environment variable with the model server's address, e.g. http://127.0.0.1:8080
TIMEOUT_SECONDS = 300  # one model turn: up to 512 tokens at ~25 tokens/s plus reading the prompt, with a wide margin


class LocalLLM:
    """Qwen in this process (the behavior of every evaluated run)."""

    def __init__(self, model: str = GEN_MODEL):
        from mlx_lm import load  # imported here, not at the top: mlx_lm exists only on Apple silicon
        self.model_name = model
        self.model, self.tokenizer = load(model)

    def chat(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        from mlx_lm import generate
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                    enable_thinking=False)
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

    def config(self) -> dict:
        return {"backend": "local", "model": self.model_name}


class ServerLLM:
    """Qwen in an mlx_lm.server process, through its OpenAI-style chat API (/v1/chat/completions).

    The chat API, not /v1/completions: for a text completion the server removes the tool-call text from the reply
    (finish_reason "tool_calls", empty text), so the agent would never see its own tool calls."""

    def __init__(self, url: str, model: str = GEN_MODEL):
        self.url, self.model_name = url.rstrip("/"), model
        try:  # fail at start-up with a clear message, not on the first question
            requests.get(f"{self.url}/health", timeout=10).raise_for_status()
        except requests.RequestException as error:
            raise RuntimeError(f"No model server at {self.url} ({error}). Start it with: "
                               f"mlx_lm.server --model {model} --port 8080 --prompt-cache-size 0") from error

    def chat(self, messages: list[dict], tools: list[dict], max_tokens: int) -> str:
        response = requests.post(f"{self.url}/v1/chat/completions", timeout=TIMEOUT_SECONDS, json={
            "model": self.model_name,  # the server switches to this model if it is running another one
            "messages": messages, "tools": tools, "max_tokens": max_tokens,
            "temperature": 0.0,  # greedy: always the most likely next token, as generate() does in process
            "seed": 0,  # a request with a seed is served on its own, not batched with others
            "chat_template_kwargs": {"enable_thinking": False}})
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        # The server splits the reply into text, reasoning and parsed tool calls; rebuild Qwen's raw format
        reply = f"<think>\n{message['reasoning']}\n</think>\n\n" if message.get("reasoning") else ""
        reply += message.get("content") or ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            reply += "\n<tool_call>\n" + json.dumps({"name": call["function"]["name"], "arguments": arguments}) \
                     + "\n</tool_call>"
        return reply

    def config(self) -> dict:
        return {"backend": "server", "model": self.model_name, "url": self.url}


def llm_from_env() -> LocalLLM | ServerLLM:
    url = os.environ.get(LLM_URL_VAR)
    return ServerLLM(url) if url else LocalLLM()
