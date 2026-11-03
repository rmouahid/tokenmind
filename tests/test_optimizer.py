import pytest

from tokenmind.bench import Cell
from tokenmind.optimizer import best, pareto


def cell(representation, level, accuracy, tokens, task="lookup", model="m"):
    return Cell(model, task, representation, level, 100, accuracy, 0, 1,
                {"gpt-4o": tokens})  # fmt: skip


CELLS = [
    cell("natural", 0, 0.95, 360),
    cell("json", 1, 0.96, 410),
    cell("hybrid", 0, 0.94, 260),
    cell("symbolic", 4, 0.70, 220),
    cell("natural", 0, 0.50, 360, task="aggregation"),
]


def test_best_is_the_cheapest_within_the_fidelity_target():
    assert best(CELLS, "m", "lookup").representation == "hybrid"
    assert best(CELLS, "m", "lookup", fidelity=1.0).representation == "natural"
    assert best(CELLS, "m", "lookup", fidelity=0.5).representation == "symbolic"


def test_best_needs_a_natural_baseline():
    with pytest.raises(ValueError, match="baseline"):
        best(CELLS, "other", "lookup")


def test_pareto_frontier():
    lookup = [c for c in CELLS if c.task == "lookup"]
    found = [(c.representation, c.accuracy) for c in pareto(lookup)]
    assert found == [("symbolic", 0.70), ("hybrid", 0.94), ("natural", 0.95),
                     ("json", 0.96)]  # fmt: skip
