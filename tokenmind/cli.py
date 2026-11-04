"""Command line: generate data, measure token costs, run the benchmark."""

from pathlib import Path

import typer

from tokenmind import bench, dataset
from tokenmind.llm import Client
from tokenmind.renderers import RENDERERS
from tokenmind.tokenizers import TOKENIZERS

app = typer.Typer(no_args_is_help=True, help=__doc__)


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


@app.command()
def generate(
    out: Path = Path("data/items.jsonl"), seed: int = 0, worlds: int = 100
) -> None:
    """Write one question per task family for each seeded world."""
    out.parent.mkdir(parents=True, exist_ok=True)
    items = dataset.generate(seed, worlds)
    dataset.save(items, out)
    typer.echo(f"{len(items)} items -> {out}")


@app.command()
def tokens(
    items: Path = Path("data/items.jsonl"),
    tokenizers: str = ",".join(TOKENIZERS),
    out: Path = Path("results/tokens.csv"),
) -> None:
    """Mean context tokens per representation, level and tokenizer."""
    names = _split(tokenizers)
    costs = bench.token_costs(dataset.load(items), tokenizers=names)
    baseline = costs["natural", 0]
    lines = ["representation,level," + ",".join(names)]
    for (name, level), found in costs.items():
        lines.append(f"{name},{level}," + ",".join(f"{found[t]:.1f}" for t in names))
        saved = 1 - found[names[0]] / baseline[names[0]]
        typer.echo(f"{name:10} L{level}  {found[names[0]]:7.1f}  {saved:+.0%}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"-> {out}")


@app.command()
def run(
    models: str,
    items: Path = Path("data/items.jsonl"),
    representations: str = ",".join(RENDERERS),
    levels: str = "0,1,2,3,4",
    limit: int = 0,
    out: Path = Path("runs/results.jsonl"),
) -> None:
    """Ask each model every question in every representation (cached)."""
    loaded = dataset.load(items)
    rows = bench.run(
        loaded[:limit] if limit else loaded,
        Client(),
        _split(models),
        _split(representations),
        [int(level) for level in _split(levels)],
    )
    typer.echo(f"{bench.save_rows(rows, out)} answers -> {out}")
    for cell in bench.summarize(bench.load_rows(out)):
        typer.echo(
            f"{cell.model:32} {cell.task:11} {cell.representation:10} "
            f"L{cell.level} {cell.accuracy:6.1%} [{cell.low:.0%}-{cell.high:.0%}]"
        )
