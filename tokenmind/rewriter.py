"""Ask a model for denser encodings than the templates; keep only the
rewrites that still contain every name and value."""

import json
from pathlib import Path

from tokenmind.dataset import Item, World
from tokenmind.llm import Client
from tokenmind.renderers import Style, missing_facts, render

PROMPT = (
    "Rewrite the facts below as compactly as possible, for another language "
    "model to read. Keep every name and every value exactly as written. You may "
    "change the structure, drop words and invent notation, but the text must "
    "stay decodable on its own. Reply with the rewritten facts only.\n\n{context}"
)


def _unfence(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:-1] if lines[-1].startswith("```") else lines[1:]
    return "\n".join(lines).strip()


def rewrite(world: World, client: Client, model: str, source="json") -> str | None:
    """A rewrite of the compact `source` rendering, or None if it lost facts."""
    prompt = PROMPT.format(context=render(world, source, level=1))
    text = _unfence(client.chat(model, prompt, max_tokens=4096).text)
    return None if missing_facts(text, world, Style()) else text


def rewrites(items: list[Item], client: Client, model: str) -> dict:
    """{item id: {"rewrite": text}} for the worlds whose rewrite kept every fact."""
    by_world: dict[str, str | None] = {}
    found = {}
    for item in items:
        key = item.id.split("-")[0]
        if key not in by_world:
            by_world[key] = rewrite(item.world, client, model)
        if by_world[key] is not None:
            found[item.id] = {"rewrite": by_world[key]}
    return found


def save(found: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
