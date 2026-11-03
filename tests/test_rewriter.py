import httpx

from tokenmind import rewriter
from tokenmind.dataset import generate
from tokenmind.llm import Client
from tokenmind.renderers import render

ITEMS = generate(seed=9, worlds=2)


def model_replying(tmp_path, replies):
    replies = iter(replies)

    def handler(request):
        content = next(replies)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    return Client(cache_dir=tmp_path, transport=httpx.MockTransport(handler))


def test_keeps_a_rewrite_that_preserves_every_fact(tmp_path):
    world = ITEMS[0].world
    dense = render(world, "symbolic", 1)
    client = model_replying(tmp_path, [f"```\n{dense}\n```"])
    assert rewriter.rewrite(world, client, "m") == dense


def test_rejects_a_rewrite_that_drops_facts(tmp_path):
    client = model_replying(tmp_path, ["Three companies and some people."])
    assert rewriter.rewrite(ITEMS[0].world, client, "m") is None


def test_rewrites_once_per_world_and_maps_every_item(tmp_path):
    first = ITEMS[0].world
    client = model_replying(tmp_path, [render(first, "hybrid", 1), "lost"])
    found = rewriter.rewrites(ITEMS, client, "m")
    assert sorted(found) == sorted(item.id for item in ITEMS[:5])
    assert found["0-lookup"]["rewrite"] == render(first, "hybrid", 1)
    rewriter.save(found, tmp_path / "rw.json")
    assert rewriter.load(tmp_path / "rw.json") == found
