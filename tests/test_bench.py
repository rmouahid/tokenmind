import json

import httpx

from tokenmind import bench
from tokenmind.dataset import generate
from tokenmind.llm import Client

ITEMS = generate(seed=4, worlds=2)
GOLD = {item.question: item.answer for item in ITEMS}


def oracle(tmp_path):
    """A fake model that answers correctly from the question in the prompt."""

    def handler(request):
        prompt = json.loads(request.content)["messages"][0]["content"]
        question = prompt.split("Question: ")[1].split("\n")[0]
        answer = GOLD[question]
        text = ", ".join(answer) if isinstance(answer, list) else str(answer)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})

    return Client(cache_dir=tmp_path, transport=httpx.MockTransport(handler))


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


def test_token_costs_are_measured_per_world():
    costs = bench.token_costs(ITEMS, ["natural", "symbolic"], ("gpt-4o",))
    assert len(costs) == 2 * 5
    assert costs["symbolic", 4]["gpt-4o"] < costs["natural", 0]["gpt-4o"]


def test_wilson_interval():
    low, high = bench.wilson(90, 100)
    assert 0.82 < low < 0.84 and 0.94 < high < 0.95
    assert bench.wilson(0, 0) == (0.0, 1.0)
