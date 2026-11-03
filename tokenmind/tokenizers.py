"""Count tokens with the tokenizers of several model families (H1).

Only tokenizer files are downloaded from Hugging Face, never model weights.
"""

from collections.abc import Callable
from functools import cache

import tiktoken
from tokenizers import Tokenizer

TIKTOKEN = {"gpt-4o": "o200k_base", "gpt-4": "cl100k_base"}
HUGGING_FACE = {
    "llama-3": "unsloth/Llama-3.1-8B-Instruct",
    "qwen-2.5": "Qwen/Qwen2.5-7B-Instruct",
    "mistral": "unsloth/mistral-7b-instruct-v0.3",
    "gemma-2": "unsloth/gemma-2-9b-it",
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
}
TOKENIZERS = (*TIKTOKEN, *HUGGING_FACE)


@cache
def _counter(name: str) -> Callable[[str], int]:
    if name in TIKTOKEN:
        encoding = tiktoken.get_encoding(TIKTOKEN[name])
        return lambda text: len(encoding.encode(text))
    if name not in HUGGING_FACE:
        raise ValueError(f"unknown tokenizer {name!r}, choose from {TOKENIZERS}")
    tokenizer = Tokenizer.from_pretrained(HUGGING_FACE[name])
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def count(text: str, tokenizer: str = "gpt-4o") -> int:
    return _counter(tokenizer)(text)


def counts(text: str, tokenizers=TOKENIZERS) -> dict[str, int]:
    return {name: count(text, name) for name in tokenizers}
