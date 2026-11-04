"""Turn the files in results/ into charts and a Markdown report."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from tokenmind.bench import Cell  # noqa: E402
from tokenmind.optimizer import pareto  # noqa: E402
from tokenmind.tokenizers import tokenizer_for  # noqa: E402

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SAVES, COSTS = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("cost", [SAVES, "#f0efec", COSTS])
NAMES = {
    "natural": "Natural text",
    "condensed": "Condensed",
    "json": "JSON",
    "triples": "Triples",
    "symbolic": "Symbolic",
    "hybrid": "Hybrid table",
}
LEVELS = ["as is", "compact", "short keys", "short values", "no legend"]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
    }
)


def read_tokens(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return {
        (row.pop("representation"), int(row.pop("level"))): {
            name: float(value) for name, value in row.items()
        }
        for row in rows
    }


def read_cells(path: Path) -> list[Cell]:
    with path.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    return [
        Cell(
            r["model"],
            r["task"],
            r["representation"],
            int(r["level"]),
            int(r["n"]),
            float(r["accuracy"]),
            float(r["low"]),
            float(r["high"]),
            {k[7:]: float(v) for k, v in r.items() if k.startswith("tokens_")},
        )
        for r in rows
    ]


def token_heatmap(costs, out: Path) -> None:
    """H1: token cost against natural text, per representation and tokenizer."""
    tokenizers = list(costs["natural", 0])
    reps = [name for name in NAMES if (name, 0) in costs]
    rows = [(name, level) for name in reps for level in (0, 4)]
    values = [
        [costs[r][t] / costs["natural", 0][t] - 1 for t in tokenizers] for r in rows
    ]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    norm = TwoSlopeNorm(0, -0.6, 1.0)
    ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="auto")
    for i, line in enumerate(values):
        for j, change in enumerate(line):
            change = round(change, 2) or 0.0  # no "-0%"
            ax.text(
                j,
                i,
                f"{change:+.0%}",
                ha="center",
                va="center",
                color=INK,
                fontsize=9.5,
            )
    ax.set_xticks(range(len(tokenizers)), tokenizers)
    ax.set_yticks(
        range(len(rows)),
        [NAMES[n] if lv == 0 else "   max compression" for n, lv in rows],
    )
    for boundary in range(2, len(rows), 2):
        ax.axhline(boundary - 0.5, color=SURFACE, lw=4)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Context tokens vs natural text (negative = cheaper)",
        loc="left",
        pad=34,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _pooled(cells: list[Cell], by_model: bool) -> dict[tuple, tuple[float, dict]]:
    """Accuracy (weighted by n) and tokens per (model?, representation, level)."""
    groups: dict[tuple, list[Cell]] = defaultdict(list)
    for cell in cells:
        key = (cell.representation, cell.level)
        groups[(cell.model, *key) if by_model else key].append(cell)
    pooled = {}
    for key, found in groups.items():
        n = sum(c.n for c in found)
        accuracy = sum(c.accuracy * c.n for c in found) / n
        tokens = {
            t: sum(c.tokens[t] for c in found) / len(found) for t in found[0].tokens
        }
        pooled[key] = (accuracy, tokens)
    return pooled


def degradation(cells: list[Cell], out: Path) -> None:
    """Accuracy at each compression level, all models and tasks pooled."""
    pooled = _pooled(cells, by_model=False)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for color, name in zip(SERIES, NAMES, strict=False):
        levels = sorted(lv for rep, lv in pooled if rep == name)
        if not levels:
            continue
        accuracy = [pooled[name, lv][0] for lv in levels]
        ax.plot(
            levels,
            accuracy,
            color=color,
            lw=2,
            marker="o",
            ms=7,
            mec=SURFACE,
            mew=2,
            label=NAMES[name],
        )
    ax.set_xticks(range(len(LEVELS)), LEVELS)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", color=GRID)
    ax.set_ylabel("Correct answers")
    ax.set_xlabel("Compression level")
    ax.legend(frameon=False, loc="lower left", ncols=3)
    ax.set_title(
        "Where comprehension breaks down", loc="left", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def pareto_chart(cells: list[Cell], out: Path) -> None:
    """Tokens vs accuracy per model, in the model's own tokenizer."""
    pooled = _pooled(cells, by_model=True)
    models = sorted({key[0] for key in pooled})
    fig, axes = plt.subplots(
        1, len(models), figsize=(4.2 * len(models), 4.4), sharey=True, squeeze=False
    )
    for ax, model in zip(axes[0], models, strict=True):
        tokenizer = tokenizer_for(model)
        points = [
            Cell(model, "all", rep, lv, 1, acc, 0, 1, tokens)
            for (m, rep, lv), (acc, tokens) in pooled.items()
            if m == model
        ]
        ax.scatter(
            [p.tokens[tokenizer] for p in points],
            [p.accuracy for p in points],
            color=GRID,
            s=28,
            zorder=2,
        )
        front = pareto(points, tokenizer)
        ax.plot(
            [p.tokens[tokenizer] for p in front],
            [p.accuracy for p in front],
            color=SAVES,
            lw=2,
            marker="o",
            ms=6,
            zorder=3,
        )
        for p in front:
            ax.annotate(
                f"{NAMES[p.representation]} L{p.level}",
                (p.tokens[tokenizer], p.accuracy),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=8,
                color=INK_2,
            )
        base = next(p for p in points if (p.representation, p.level) == ("natural", 0))
        ax.scatter(
            [base.tokens[tokenizer]],
            [base.accuracy],
            color=SERIES[1],
            s=60,
            zorder=4,
            label="natural text",
        )
        ax.set_title(model, loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Context tokens ({tokenizer})")
        ax.grid(color=GRID)
    axes[0][0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylabel("Correct answers")
    axes[0][0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def write(results: Path, docs: Path) -> list[Path]:
    """Write the figures and docs/results.md from whatever results/ holds."""
    figures = docs / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    costs = read_tokens(results / "tokens.csv")
    token_heatmap(costs, figures / "tokens.png")
    written = [figures / "tokens.png"]
    tokenizers = list(costs["natural", 0])
    cheapest = {t: min(costs, key=lambda key: costs[key][t]) for t in tokenizers}
    rows = [
        [
            t,
            f"{NAMES[cheapest[t][0]]} L{cheapest[t][1]}",
            f"{costs[cheapest[t]][t] / costs['natural', 0][t] - 1:+.0%}",
        ]
        for t in tokenizers
    ]
    parts = [
        "# Results\n",
        "Generated by `tokenmind report` from `results/`.\n",
        "## H1 — token cost depends on the tokenizer\n",
        "![Token cost per representation and tokenizer](figures/tokens.png)\n",
        _table(["Tokenizer", "Cheapest representation", "vs natural text"], rows),
    ]
    if (results / "cells.csv").exists():
        cells = read_cells(results / "cells.csv")
        degradation(cells, figures / "degradation.png")
        pareto_chart(cells, figures / "pareto.png")
        written += [figures / "degradation.png", figures / "pareto.png"]
        parts += [
            "\n## Accuracy against tokens, per model\n",
            "![Pareto frontier per model](figures/pareto.png)\n",
            "## Where comprehension breaks down\n",
            "![Accuracy per compression level](figures/degradation.png)\n",
        ]
    if (results / "best.csv").exists():
        with (results / "best.csv").open(encoding="utf-8") as source:
            best = list(csv.DictReader(source))
        parts += [
            "## H2 — the best representation per model and task\n",
            "Cheapest representation within 98 % of natural-text accuracy.\n",
            _table(
                ["Model", "Task", "Best", "Accuracy", "Natural text", "Tokens"],
                [
                    [
                        b["model"],
                        b["task"],
                        f"{NAMES[b['representation']]} " f"L{b['level']}",
                        f"{float(b['accuracy']):.0%}",
                        f"{float(b['natural_accuracy']):.0%}",
                        f"{float(b['token_change']):+.0%} ({b['tokenizer']})",
                    ]
                    for b in best
                ],
            ),
        ]
    report = docs / "results.md"
    report.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return [*written, report]
