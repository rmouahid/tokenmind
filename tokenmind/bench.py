"""Run the matrix model × task × representation × level and summarise it."""

import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from tokenmind.dataset import Item, World
from tokenmind.grading import grade
from tokenmind.llm import Client
from tokenmind.renderers import LEVELS, RENDERERS, render
from tokenmind.tokenizers import TOKENIZERS, counts

ALL_LEVELS = tuple(range(len(LEVELS)))
PROMPT = (
    "Answer the question using only the facts below. Reply with the answer "
    "only: a name, a value or a number. For a list, separate items with "
    "commas.\n\nFacts:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class Row:
    model: str
    item: str
    task: str
    representation: str
    level: int
    tokens: dict[str, int]  # context tokens per tokenizer, legend included
    prompt_tokens: int | None
    response: str
    correct: bool


@dataclass
class Cell:
    model: str
    task: str
    representation: str
    level: int
    n: int
    accuracy: float
    low: float  # 95 % Wilson interval
    high: float
    tokens: dict[str, float]


def variants(world: World, representations=RENDERERS, levels=ALL_LEVELS):
    for name in representations:
        for level in levels:
            yield name, level, render(world, name, level)


def token_costs(items: list[Item], representations=RENDERERS, tokenizers=TOKENIZERS):
    """Mean context tokens per (representation, level, tokenizer); no model."""
    worlds = {item.id.split("-")[0]: item.world for item in items}
    totals: dict[tuple[str, int], list[dict[str, int]]] = defaultdict(list)
    for world in worlds.values():
        for name, level, text in variants(world, representations):
            totals[name, level].append(counts(text, tokenizers))
    return {
        key: {t: mean(c[t] for c in found) for t in tokenizers}
        for key, found in totals.items()
    }


def run(
    items: Iterable[Item],
    client: Client,
    models: list[str],
    representations=RENDERERS,
    levels=ALL_LEVELS,
    tokenizers=TOKENIZERS,
    extra=None,
) -> Iterable[Row]:
    """`extra` maps an item id to {representation: context} (e.g. rewrites)."""
    for item in items:
        contexts = list(variants(item.world, representations, levels))
        for name, context in ((extra or {}).get(item.id) or {}).items():
            contexts.append((name, 0, context))
        for name, level, context in contexts:
            tokens = counts(context, tokenizers)
            prompt = PROMPT.format(context=context, question=item.question)
            for model in models:
                reply = client.chat(model, prompt)
                yield Row(model, item.id, item.task, name, level, tokens,
                          reply.prompt_tokens, reply.text,
                          grade(item.answer, reply.text))  # fmt: skip


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


def summarize(rows: Iterable[Row]) -> list[Cell]:
    groups: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        groups[row.model, row.task, row.representation, row.level].append(row)
    cells = []
    for key, found in sorted(groups.items()):
        correct = sum(row.correct for row in found)
        tokens = {t: mean(row.tokens[t] for row in found) for t in found[0].tokens}
        cells.append(Cell(*key, len(found), correct / len(found),
                          *wilson(correct, len(found)), tokens))  # fmt: skip
    return cells


def save_rows(rows: Iterable[Row], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            written += 1
    return written


def load_rows(path: Path) -> list[Row]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Row(**json.loads(line)) for line in lines]
