"""OpenAI-compatible chat client with an on-disk response cache.

Works with a local Ollama server, OpenRouter, OpenAI, or any server speaking
the same API (llama.cpp, vLLM, LM Studio) through `base_url`. Every
response is stored under `cache_dir`, keyed by model and prompt, so a benchmark
can be replayed and re-graded without paying twice.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

PROVIDERS = {
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}
RETRYABLE = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    prompt_tokens: int | None  # as billed by the provider, in the model's tokens


class Client:
    def __init__(
        self,
        base_url: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        cache_dir: Path = Path("cache"),
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        retries: int = 5,
    ):
        provider = provider or os.environ.get("TOKENMIND_PROVIDER", "openrouter")
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}, choose from {PROVIDERS}")
        base_url = base_url or os.environ.get("TOKENMIND_BASE_URL", PROVIDERS[provider])
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("TOKENMIND_API_KEY", "")
        self.cache_dir = cache_dir
        self.http = httpx.Client(transport=transport, timeout=600)
        self.sleep = sleep
        self.retries = retries

    def _path(self, model: str, prompt: str, max_tokens: int) -> Path:
        key = json.dumps([model, prompt, max_tokens]).encode()
        digest = hashlib.sha256(key).hexdigest()
        # Model tags such as `llama3.2:3b` are not valid file names on Windows.
        folder = model.replace("/", "__").replace(":", "_")
        return self.cache_dir / folder / f"{digest}.json"

    def chat(self, model: str, prompt: str, max_tokens: int = 64) -> Reply:
        path = self._path(model, prompt, max_tokens)
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            return Reply(cached["answer"], cached["prompt_tokens"])
        data = self._post(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        )
        reply = Reply(
            data["choices"][0]["message"].get("content") or "",
            data.get("usage", {}).get("prompt_tokens"),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"model": model, "prompt": prompt, "answer": reply.text,
                  "prompt_tokens": reply.prompt_tokens}  # fmt: skip
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return reply

    def _post(self, body: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        for attempt in range(self.retries):
            response = self.http.post(
                f"{self.base_url}/chat/completions", json=body, headers=headers
            )
            if response.status_code in RETRYABLE:
                self.sleep(2**attempt)
                continue
            if response.is_error:
                raise LLMError(f"{response.status_code}: {response.text[:200]}")
            return response.json()
        raise LLMError(f"gave up after {self.retries} attempts")
