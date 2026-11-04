"""Pick the cheapest representation that keeps answers correct."""

from tokenmind.bench import Cell


def baseline(cells: list[Cell], model: str, task: str) -> Cell:
    """Natural text at level 0, the reference every representation is held to."""
    for cell in cells:
        if (cell.model, cell.task, cell.representation, cell.level) == (
            model, task, "natural", 0,
        ):  # fmt: skip
            return cell
    raise ValueError(f"no natural-text baseline for {model} / {task}")


def best(
    cells: list[Cell], model: str, task: str, tokenizer="gpt-4o", fidelity=0.98
) -> Cell:
    """Fewest tokens with accuracy >= fidelity x accuracy of natural text."""
    pool = [cell for cell in cells if cell.model == model and cell.task == task]
    target = fidelity * baseline(cells, model, task).accuracy
    good = [cell for cell in pool if cell.accuracy >= target]
    return min(good, key=lambda cell: (cell.tokens[tokenizer], -cell.accuracy))


def pareto(cells: list[Cell], tokenizer="gpt-4o") -> list[Cell]:
    """Cells that no cheaper cell matches or beats in accuracy."""
    frontier: list[Cell] = []
    for cell in sorted(cells, key=lambda c: (c.tokens[tokenizer], -c.accuracy)):
        if not frontier or cell.accuracy > frontier[-1].accuracy:
            frontier.append(cell)
    return frontier
