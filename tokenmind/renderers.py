"""Render the same world as natural text, condensed text, JSON, triples, a
symbolic notation or a hybrid table, at increasing compression levels.

Any legend a representation needs to be decoded is part of its output, so its
token cost is counted fairly.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass

from tokenmind.dataset import SCHEMA, Entity, World

GROUPS = {"company": "companies", "person": "people", "product": "products"}
ALIASES = {
    "companies": "C", "people": "P", "products": "D", "name": "n",
    "city": "c", "sector": "s", "founded": "f", "age": "a", "role": "r",
    "salary": "w", "employer": "e", "maker": "m", "launched": "l", "price": "p",
}  # fmt: skip
YEARS = ("founded", "launched")


@dataclass(frozen=True)
class Style:
    compact: bool = False  # minimal whitespace
    short_keys: bool = False  # one-letter field names, decoded by a legend
    short_values: bool = False  # 52000 -> 52k, 2021 -> '21
    legend: bool = True  # drop it at the last level: the model must guess


LEVELS = (
    Style(),
    Style(compact=True),
    Style(compact=True, short_keys=True),
    Style(compact=True, short_keys=True, short_values=True),
    Style(compact=True, short_keys=True, short_values=True, legend=False),
)


def value(field: str, raw: str | int, style: Style) -> str:
    if style.short_values and field == "salary":
        return f"{raw // 1000}k"
    if style.short_values and field in YEARS:
        return f"'{raw % 100:02d}"
    return str(raw)


def _key(name: str, style: Style) -> str:
    return ALIASES[name] if style.short_keys else name


def _legend(style: Style, text: str, names=tuple(ALIASES)) -> str:
    """Prefix the aliases actually used, so short keys stay decodable."""
    if not (style.short_keys and style.legend):
        return text
    pairs = " ".join(f"{ALIASES[name]}={name}" for name in names)
    return f"Keys: {pairs}\n{text}"


def _vals(entity: Entity, style: Style) -> dict[str, str]:
    return {f: value(f, v, style) for f, v in entity.attrs.items()}


def natural(world: World, style: Style) -> str:
    templates = {
        "company": "{name} is a {sector} company based in {city}, "
        "founded in {founded}.",
        "person": "{name} is {age} years old, works as a {role} at {employer} "
        "and earns {salary} per year.",
        "product": "{name} is made by {maker}, was launched in {launched} "
        "and costs {price}.",
    }
    sentences = [
        templates[e.kind].format(name=e.name, **_vals(e, style)) for e in world.entities
    ]
    return (" " if style.compact else "\n").join(sentences)


def condensed(world: World, style: Style) -> str:
    templates = {
        "company": "{name}: {sector}, {city}, est. {founded}",
        "person": "{name}: {age}y, {role} @ {employer}, {salary}/yr",
        "product": "{name}: by {maker}, {launched}, price {price}",
    }
    lines = [
        templates[e.kind].format(name=e.name, **_vals(e, style)) for e in world.entities
    ]
    return ("; " if style.compact else "\n").join(lines)


def _as_number(text: str) -> str | int:
    return int(text) if text.isdigit() else text


def as_json(world: World, style: Style) -> str:
    data: dict[str, list] = {}
    for kind, group in GROUPS.items():
        data[_key(group, style)] = [
            {_key("name", style): e.name}
            | {_key(f, style): _as_number(v) for f, v in _vals(e, style).items()}
            for e in world.of(kind)
        ]
    if style.compact:
        text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    else:
        text = json.dumps(data, indent=2, ensure_ascii=False)
    return _legend(style, text)


def triples(world: World, style: Style) -> str:
    sep = "|" if style.compact else " | "
    lines = [
        sep.join((e.name, _key(f, style), v))
        for e in world.entities
        for f, v in _vals(e, style).items()
    ]
    fields = [f for f in ALIASES if f not in ("name", *GROUPS.values())]
    return _legend(style, "\n".join(lines), fields)


def symbolic(world: World, style: Style) -> str:
    """Positional records `P Lena Moreau/34/...` decoded by a schema line."""
    tag = {kind: ALIASES[group] for kind, group in GROUPS.items()}
    records = [
        f"{tag[e.kind]} {e.name}/" + "/".join(_vals(e, style).values())
        for e in world.entities
    ]
    body = (";" if style.compact else "\n").join(records)
    if not style.legend:
        return body
    schema = " ".join(
        f"{tag[kind]}=" + "/".join(("name", *fields)) for kind, fields in SCHEMA.items()
    )
    return f"{schema}\n{body}"


def hybrid(world: World, style: Style) -> str:
    """One header per entity type, then comma-separated rows."""
    blocks = []
    for kind, group in GROUPS.items():
        header = ",".join(_key(f, style) for f in ("name", *SCHEMA[kind]))
        rows = [",".join((e.name, *_vals(e, style).values())) for e in world.of(kind)]
        blocks.append("\n".join([f"{_key(group, style)}: {header}", *rows]))
    return _legend(style, ("\n" if style.compact else "\n\n").join(blocks))


RENDERERS: dict[str, Callable[[World, Style], str]] = {
    "natural": natural,
    "condensed": condensed,
    "json": as_json,
    "triples": triples,
    "symbolic": symbolic,
    "hybrid": hybrid,
}


def render(world: World, representation: str, level: int = 0) -> str:
    return RENDERERS[representation](world, LEVELS[level])


def missing_facts(text: str, world: World, style: Style) -> list[str]:
    """Entity names and attribute values that do not appear in `text`."""
    expected = [e.name for e in world.entities] + [
        v for e in world.entities for v in _vals(e, style).values()
    ]
    return [fact for fact in expected if fact not in text]
