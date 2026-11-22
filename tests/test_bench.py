import json

import httpx
import pytest

from tokenmind import bench
from tokenmind.dataset import generate
from tokenmind.llm import Client

ITEMS = generate(seed=4, worlds=2)
GOLD = {item.question: item.answer for item in ITEMS}


def answer_request(request):
    """A fake model that answers correctly from the question in the prompt."""
    prompt = json.loads(request.content)["messages"][0]["content"]
    question = prompt.split("Question: ")[1].split("\n")[0]
    answer = GOLD[question]
    text = ", ".join(answer) if isinstance(answer, list) else str(answer)
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def oracle(tmp_path):
    return Client(cache_dir=tmp_path, transport=httpx.MockTransport(answer_request))


def test_runs_every_cell_and_grades_answers(tmp_path):
    rows = list(
        bench.run(
            ITEMS,
            oracle(tmp_path),
            ["m1", "m2"],
            ["json", "symbolic"],
            [0, 4],
            tokenizers=("gpt-4o",),
        )
    )
    assert len(rows) == len(ITEMS) * 2 * 2 * 2
    assert all(row.correct for row in rows)
    cells = bench.summarize(rows)
    assert len(cells) == 2 * 5 * 2 * 2
    assert all(cell.n == 2 and cell.accuracy == 1 for cell in cells)


def test_extra_contexts_are_evaluated_too(tmp_path):
    extra = {ITEMS[0].id: {"rewrite": "anything"}}
    rows = list(
        bench.run(
            ITEMS[:1],
            oracle(tmp_path),
            ["m"],
            ["json"],
            [0],
            tokenizers=("gpt-4o",),
            extra=extra,
        )
    )
    assert [row.representation for row in rows] == ["json", "rewrite"]


def test_rows_round_trip(tmp_path):
    rows = list(
        bench.run(
            ITEMS[:2], oracle(tmp_path), ["m"], ["natural"], [0], tokenizers=("gpt-4o",)
        )
    )
    assert bench.save_rows(rows, tmp_path / "r.jsonl") == 2
    assert bench.load_rows(tmp_path / "r.jsonl") == rows


def test_saves_complete_answers_after_retrying_a_read_timeout(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text("previous answers\n", encoding="utf-8")
    sent = []
    delays = []

    def handler(request):
        assert path.read_text(encoding="utf-8") == "previous answers\n"
        sent.append(request.content)
        if len(sent) == 2:
            raise httpx.ReadTimeout("timed out", request=request)
        return answer_request(request)

    client = Client(
        cache_dir=tmp_path / "cache",
        transport=httpx.MockTransport(handler),
        sleep=delays.append,
    )
    items = ITEMS[:2]
    rows = bench.run(items, client, ["m"], ["natural"], [0], tokenizers=())

    assert bench.save_rows(rows, path) == 2
    assert len(sent) == 3
    assert sent[0] != sent[1] == sent[2]
    assert delays == [1]
    saved = bench.load_rows(path)
    assert [row.item for row in saved] == [item.id for item in items]
    assert all(row.correct for row in saved)


def test_partial_write_preserves_previous_answers(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text("previous answers\n", encoding="utf-8")

    def rows():
        yield bench.Row("m", "item", "task", "natural", 0, {}, None, "Lyon", True)
        raise RuntimeError("run failed")

    with pytest.raises(RuntimeError, match="run failed"):
        bench.save_rows(rows(), path)
    assert path.read_text(encoding="utf-8") == "previous answers\n"
    assert list(tmp_path.iterdir()) == [path]


def test_token_costs_are_measured_per_world():
    costs = bench.token_costs(ITEMS, ["natural", "symbolic"], ("gpt-4o",))
    assert len(costs) == 2 * 5
    assert costs["symbolic", 4]["gpt-4o"] < costs["natural", 0]["gpt-4o"]


def test_wilson_interval():
    low, high = bench.wilson(90, 100)
    assert 0.82 < low < 0.84 and 0.94 < high < 0.95
    assert bench.wilson(0, 0) == (0.0, 1.0)
