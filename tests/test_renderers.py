import json
import random

import pytest
import tiktoken

from tokenmind.dataset import make_world
from tokenmind.renderers import LEVELS, RENDERERS, Style, missing_facts, render, value

WORLD = make_world(random.Random(11))
ENCODING = tiktoken.get_encoding("o200k_base")


@pytest.mark.parametrize("name", RENDERERS)
@pytest.mark.parametrize("level", range(len(LEVELS)))
def test_every_representation_keeps_every_fact(name, level):
    text = render(WORLD, name, level)
    assert missing_facts(text, WORLD, LEVELS[level]) == []


@pytest.mark.parametrize("name", RENDERERS)
def test_the_last_level_is_the_cheapest(name):
    # Short keys need a legend, which can cost more than it saves on a small
    # world: levels are not monotonic, only the last one must be the cheapest.
    counts = [len(ENCODING.encode(render(WORLD, name, lv))) for lv in range(5)]
    assert counts[1] <= counts[0]
    assert counts[-1] == min(counts) < counts[0]


def test_json_stays_valid_without_legend():
    data = json.loads(render(WORLD, "json", 4))
    assert set(data) == {"C", "P", "D"}


def test_short_keys_come_with_a_legend_until_the_last_level():
    assert render(WORLD, "triples", 3).startswith("Keys: ")
    assert not render(WORLD, "triples", 4).startswith("Keys: ")
    assert "P=name/age/role/salary/employer" in render(WORLD, "symbolic", 0)


def test_short_values():
    short = Style(short_values=True)
    assert value("salary", 52_000, short) == "52k"
    assert value("launched", 2021, short) == "'21"
    assert value("salary", 52_000, Style()) == "52000"
