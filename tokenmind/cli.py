"""Command line: generate data, measure token costs, run the benchmark."""

import time
from pathlib import Path

import httpx
import typer

from tokenmind import bench, dataset, optimizer, rewriter
from tokenmind.llm import PROVIDERS, Client, LLMError
from tokenmind.renderers import RENDERERS
from tokenmind.tokenizers import TOKENIZERS, tokenizer_for

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
        change = found[names[0]] / baseline[names[0]] - 1
        typer.echo(f"{name:10} L{level}  {found[names[0]]:7.1f}  {change:+.0%}")
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
    rewrites: Path | None = None,
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
        extra=rewriter.load(rewrites) if rewrites else None,
    )
    typer.echo(f"{bench.save_rows(rows, out)} answers -> {out}")
    for cell in bench.summarize(bench.load_rows(out)):
        typer.echo(
            f"{cell.model:32} {cell.task:11} {cell.representation:10} "
            f"L{cell.level} {cell.accuracy:6.1%} [{cell.low:.0%}-{cell.high:.0%}]"
        )


@app.command()
def rewrite(
    model: str,
    items: Path = Path("data/items.jsonl"),
    out: Path = Path("runs/rewrites.json"),
) -> None:
    """Ask `model` for a denser encoding of each world (kept if lossless)."""
    loaded = dataset.load(items)
    found = rewriter.rewrites(loaded, Client(), model)
    rewriter.save(found, out)
    typer.echo(f"{len(found)}/{len(loaded)} items with a lossless rewrite -> {out}")


def _winners(cells, tokenizer: str, fidelity: float) -> list[list]:
    """Best representation per model and task, tokens in `tokenizer` or the
    model's own tokenizer when it is `auto`."""
    found = []
    for model, task in sorted({(cell.model, cell.task) for cell in cells}):
        name = tokenizer_for(model) if tokenizer == "auto" else tokenizer
        winner = optimizer.best(cells, model, task, name, fidelity)
        natural = optimizer.baseline(cells, model, task)
        change = winner.tokens[name] / natural.tokens[name] - 1
        found.append([model, task, winner.representation, winner.level,
                      f"{winner.accuracy:.3f}", f"{natural.accuracy:.3f}", name,
                      f"{change:+.3f}"])  # fmt: skip
    return found


def _echo_winners(found: list[list]) -> None:
    for model, task, rep, level, accuracy, natural, name, change in found:
        typer.echo(
            f"{model:24} {task:11} {rep:9} L{level}  accuracy {float(accuracy):6.1%}"
            f" (natural {float(natural):6.1%})  tokens {float(change):+.0%} ({name})"
        )


@app.command()
def best(
    results: Path = Path("runs/results.jsonl"),
    tokenizer: str = "auto",
    fidelity: float = 0.98,
) -> None:
    """Cheapest representation per model and task within the fidelity target."""
    cells = bench.summarize(bench.load_rows(results))
    _echo_winners(_winners(cells, tokenizer, fidelity))


def _progress(rows, total: int, every: int = 25):
    start, model = time.monotonic(), None
    for done, row in enumerate(rows, 1):
        if row.model != model:
            model = row.model
            typer.echo(f"\n== {model}")
        if done % every == 0 or done == total:
            rate = done / (time.monotonic() - start)
            eta = (total - done) / rate / 60
            typer.echo(f"  {done}/{total}  {rate:.1f} answers/s  ~{eta:.0f} min left")
        yield row


@app.command()
def benchmark(
    models: str = "llama3.2:3b,qwen2.5:3b,gemma2:2b,mistral:7b",
    provider: str = "ollama",
    worlds: int = 100,
    seed: int = 0,
    levels: str = "0,1,2,3,4",
    quick: bool = typer.Option(False, help="5 worlds, levels 0 and 4: a trial run"),
    out: Path = Path("results"),
) -> None:
    """Everything in one go: dataset, token costs, answers, summary, winners.

    Answers are cached in ./cache, so an interrupted run resumes where it
    stopped when launched again with the same options.
    """
    if quick:
        worlds, levels = 5, "0,4"
    if provider not in PROVIDERS:
        raise typer.BadParameter(f"choose from {', '.join(PROVIDERS)}")
    names, client = _split(models), Client(provider=provider)
    out.mkdir(parents=True, exist_ok=True)

    typer.echo("Checking that every model answers...")
    for model in names:
        try:
            client.chat(model, "Reply with OK.", max_tokens=5)
        except httpx.TransportError as exc:
            typer.echo(f"Cannot reach {client.base_url}: is the server running?")
            raise typer.Exit(1) from exc
        except LLMError as exc:
            hint = f" (try: ollama pull {model})" if provider == "ollama" else ""
            typer.echo(f"{model} failed: {exc}{hint}")
            raise typer.Exit(1) from exc

    items = dataset.generate(seed, worlds)
    dataset.save(items, out / "items.jsonl")
    level_list = [int(level) for level in _split(levels)]
    typer.echo(f"{len(items)} questions from {worlds} worlds (seed {seed})")
    tokens(items=out / "items.jsonl", out=out / "tokens.csv")

    planned = len(items) * len(RENDERERS) * len(level_list)
    total = planned * len(names)
    typer.echo(f"\n{total} answers to collect ({planned} per model)")
    rows = bench.run(items, client, names, levels=level_list)
    bench.save_rows(_progress(rows, total), out / "answers.jsonl")

    cells = bench.summarize(bench.load_rows(out / "answers.jsonl"))
    bench.save_cells(cells, out / "cells.csv")
    found = _winners(cells, "auto", 0.98)
    header = "model,task,representation,level,accuracy,natural_accuracy,"
    lines = [header + "tokenizer,token_change"] + [",".join(map(str, r)) for r in found]
    (out / "best.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo("\nCheapest representation within 98 % of natural-text accuracy:")
    _echo_winners(found)
    typer.echo(f"\nResults in {out}/: answers.jsonl, cells.csv, best.csv, tokens.csv")


@app.command()
def report(results: Path = Path("results"), docs: Path = Path("docs")) -> None:
    """Charts and docs/results.md from the files in results/."""
    from tokenmind import report as reporting  # matplotlib is slow to import

    for path in reporting.write(results, docs):
        typer.echo(f"-> {path}")
