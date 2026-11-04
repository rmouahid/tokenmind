import random

from tokenmind import bench, report
from tokenmind.bench import Cell
from tokenmind.dataset import TASKS
from tokenmind.renderers import LEVELS, RENDERERS

TOKENIZERS = ("gpt-4o", "llama-3", "qwen-2.5")


def fake_results(path, with_answers=True):
    rng = random.Random(0)
    path.mkdir()
    lines = ["representation,level," + ",".join(TOKENIZERS)]
    for name in RENDERERS:
        for level in range(len(LEVELS)):
            base = {"natural": 360, "json": 700, "symbolic": 250}.get(
                name, 300
            ) - 20 * level
            lines.append(f"{name},{level}," + ",".join(str(base) for _ in TOKENIZERS))
    (path / "tokens.csv").write_text("\n".join(lines) + "\n")
    if not with_answers:
        return
    cells = [
        Cell(model, task, name, level, 20, max(0.0, 0.95 - 0.1 * level * rng.random()),
             0.8, 1.0, {t: 300.0 - 20 * level for t in TOKENIZERS})
        for model in ("llama3.2:3b", "qwen2.5:3b")
        for task in TASKS
        for name in RENDERERS
        for level in range(len(LEVELS))
    ]  # fmt: skip
    bench.save_cells(cells, path / "cells.csv")
    (path / "best.csv").write_text(
        "model,task,representation,level,accuracy,natural_accuracy,tokenizer,"
        "token_change\nllama3.2:3b,lookup,symbolic,4,0.950,0.950,llama-3,-0.390\n"
    )


def test_report_with_token_costs_only(tmp_path):
    fake_results(tmp_path / "results", with_answers=False)
    written = report.write(tmp_path / "results", tmp_path / "docs")
    assert [p.name for p in written] == ["tokens.png", "results.md"]
    text = (tmp_path / "docs" / "results.md").read_text()
    assert "| gpt-4o | Symbolic L4 |" in text


def test_full_report(tmp_path):
    fake_results(tmp_path / "results")
    written = report.write(tmp_path / "results", tmp_path / "docs")
    assert {p.name for p in written} == {
        "tokens.png",
        "degradation.png",
        "pareto.png",
        "results.md",
    }
    text = (tmp_path / "docs" / "results.md").read_text()
    assert "| llama3.2:3b | lookup | Symbolic L4 | 95% | 95% | -39% (llama-3) |" in text
    assert all(p.stat().st_size > 10_000 for p in written if p.suffix == ".png")


def test_cells_round_trip_through_csv(tmp_path):
    fake_results(tmp_path / "results")
    cells = report.read_cells(tmp_path / "results" / "cells.csv")
    assert len(cells) == 2 * len(TASKS) * len(RENDERERS) * len(LEVELS)
    assert cells[0].tokens.keys() == set(TOKENIZERS)
