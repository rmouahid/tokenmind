import random
from collections import Counter

from tokenmind.dataset import SCHEMA, TASKS, generate, load, make_world, save


def test_same_seed_gives_the_same_items():
    assert generate(seed=7, worlds=3) == generate(seed=7, worlds=3)
    assert generate(seed=7, worlds=3) != generate(seed=8, worlds=3)


def test_one_item_per_task_and_world():
    items = generate(seed=1, worlds=4)
    assert Counter(item.task for item in items) == {task: 4 for task in TASKS}
    assert len({item.id for item in items}) == len(items)


def test_entities_follow_the_schema_and_have_unique_names():
    world = make_world(random.Random(3))
    names = [entity.name for entity in world.entities]
    assert len(set(names)) == len(names)
    for entity in world.entities:
        assert tuple(entity.attrs) == SCHEMA[entity.kind]


def test_gold_answers_are_computed_from_the_facts():
    for item in generate(seed=2, worlds=20):
        world = item.world
        if item.task == "comparison":
            a, b = (
                item.question.removeprefix("Who is older, ").rstrip("?").split(" or ")
            )
            ages = {name: world.get(name).attrs["age"] for name in (a, b)}
            assert item.answer == max(ages, key=ages.get)
        if item.task == "filtering":
            year = int(item.question.rstrip("?").split()[-1])
            launched = [
                p.name for p in world.of("product") if p.attrs["launched"] > year
            ]
            assert item.answer == sorted(launched) and launched
        if item.task == "aggregation":
            assert 0 <= item.answer <= len(world.of("person"))


def test_items_round_trip_through_jsonl(tmp_path):
    items = generate(seed=5, worlds=2)
    save(items, tmp_path / "items.jsonl")
    assert load(tmp_path / "items.jsonl") == items
