import json

import httpx
import pytest
from typer.testing import CliRunner

from tokenmind import cli
from tokenmind.llm import Client


def fake_client(answers_correctly: bool):
    gold = {}

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        if "Question: " not in prompt:
            text = "OK"
        else:
            question = prompt.split("Question: ")[1].split("\n")[0]
            answer = gold[question] if answers_correctly else "?"
            text = ", ".join(answer) if isinstance(answer, list) else str(answer)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    def factory(provider, cache_dir):
        return Client(provider=provider, cache_dir=cache_dir,
                      transport=httpx.MockTransport(handler))  # fmt: skip

    return gold, factory


@pytest.mark.parametrize("out_option", [[], ["--out", "custom"]])
def test_quick_benchmark_preserves_published_results(tmp_path, monkeypatch, out_option):
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "results"
    results.mkdir()
    filenames = ("items.jsonl", "tokens.csv", "answers.jsonl", "cells.csv", "best.csv")
    published = {name: f"Published {name}".encode() for name in filenames}
    for name, content in published.items():
        (results / name).write_bytes(content)
    out = tmp_path / ("custom" if out_option else "runs/quick")

    gold, factory = fake_client(answers_correctly=True)
    real_generate = cli.dataset.generate

    def generate(seed, worlds):
        items = real_generate(seed, worlds)
        gold.update({item.question: item.answer for item in items})
        return items

    monkeypatch.setattr(cli.dataset, "generate", generate)
    monkeypatch.setattr(
        cli.bench,
        "counts",
        lambda text, tokenizers: dict.fromkeys(tokenizers, len(text)),
    )
    monkeypatch.setattr(
        cli, "Client", lambda provider: factory(provider, tmp_path / "cache")
    )
    result = CliRunner().invoke(
        cli.app,
        ["benchmark", "--quick", "--models", "llama3.2:3b,qwen2.5:3b",
         *out_option],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert {path.name: path.read_bytes() for path in results.iterdir()} == published
    assert {path.name for path in out.iterdir()} == set(filenames)
    answers = (out / "answers.jsonl").read_text().splitlines()
    assert len(answers) == 25 * 6 * 2 * 2
    best = (out / "best.csv").read_text()
    assert "llama-3" in best and "qwen-2.5" in best
    # Iterates model by model: every llama answer comes before any qwen answer.
    models = [json.loads(line)["model"] for line in answers]
    assert models == sorted(models, key=["llama3.2:3b", "qwen2.5:3b"].index)


def test_benchmark_explains_an_unreachable_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def refuse(request):
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(refuse)
    monkeypatch.setattr(
        cli, "Client", lambda provider: Client(provider=provider, transport=transport)
    )
    result = CliRunner().invoke(cli.app, ["benchmark", "--quick"])
    assert result.exit_code == 1
    assert "is the server running?" in result.output
