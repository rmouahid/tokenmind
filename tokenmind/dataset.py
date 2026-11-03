"""Seeded synthetic worlds and questions with exact gold answers.

Worlds are made of invented companies, people and products, so a model cannot
answer from memorised knowledge and every gold answer comes from the facts.
"""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

FIRST = "Lena Tomas Ines Marek Yara Oskar Nadia Elio Sana Ruben Mila Joris Alba".split()
LAST = (
    "Moreau Varga Lindqvist Okafor Brandt Castell Ivanova Duarte Kowal Haddad".split()
)
SYLLABLES = "ka lo ven tri mar zu sel dor phi nex ra quo bel tam vi gor".split()
CITIES = "Lyon Porto Gdansk Tampere Utrecht Graz Bilbao Turin".split()
SECTORS = "fintech biotech logistics gaming energy robotics".split()
ROLES = (
    "data engineer|designer|sales manager|accountant|product manager"
    "|backend developer|recruiter|researcher"
).split("|")

SCHEMA = {
    "company": ("city", "sector", "founded"),
    "person": ("age", "role", "salary", "employer"),
    "product": ("maker", "launched", "price"),
}
TASKS = ("lookup", "multihop", "aggregation", "comparison", "filtering")

Answer = str | int | list[str]


@dataclass
class Entity:
    kind: str
    name: str
    attrs: dict[str, str | int]


@dataclass
class World:
    entities: list[Entity]

    def of(self, kind: str) -> list[Entity]:
        return [entity for entity in self.entities if entity.kind == kind]

    def get(self, name: str) -> Entity:
        return next(entity for entity in self.entities if entity.name == name)


@dataclass
class Item:
    id: str
    task: str
    question: str
    answer: Answer
    world: World


def _coined(rng: random.Random, taken: set[str]) -> str:
    while True:
        name = "".join(rng.sample(SYLLABLES, rng.randint(2, 3))).capitalize()
        if name not in taken:
            taken.add(name)
            return name


def make_world(rng: random.Random, companies=3, people=8, products=5) -> World:
    taken: set[str] = set()
    firms = [
        Entity(
            "company",
            _coined(rng, taken),
            {
                "city": rng.choice(CITIES),
                "sector": rng.choice(SECTORS),
                "founded": rng.randint(1990, 2022),
            },
        )
        for _ in range(companies)
    ]
    staff = []
    for first, last in rng.sample([(f, la) for f in FIRST for la in LAST], people):
        staff.append(
            Entity(
                "person",
                f"{first} {last}",
                {
                    "age": rng.randint(22, 64),
                    "role": rng.choice(ROLES),
                    "salary": rng.randrange(30_000, 120_001, 1_000),
                    "employer": rng.choice(firms).name,
                },
            )
        )
    goods = [
        Entity(
            "product",
            _coined(rng, taken),
            {
                "maker": rng.choice(firms).name,
                "launched": rng.randint(2012, 2025),
                "price": rng.randrange(20, 2_001, 5),
            },
        )
        for _ in range(products)
    ]
    return World(firms + staff + goods)


def _lookup(rng, world):
    person = rng.choice(world.of("person"))
    attr, template = rng.choice(
        [
            ("role", "What is the role of {}?"),
            ("age", "How old is {}?"),
            ("salary", "What is the annual salary of {}?"),
        ]
    )
    return template.format(person.name), person.attrs[attr]


def _multihop(rng, world):
    if rng.random() < 0.5:
        person = rng.choice(world.of("person"))
        employer = world.get(person.attrs["employer"])
        question = f"In which city is the employer of {person.name} based?"
        return question, employer.attrs["city"]
    product = rng.choice(world.of("product"))
    maker = world.get(product.attrs["maker"])
    return f"In which sector is the maker of {product.name} active?", maker.attrs[
        "sector"
    ]


def _aggregation(rng, world):
    company = rng.choice(world.of("company"))
    threshold = rng.randrange(40_000, 100_001, 5_000)
    count = sum(
        1
        for person in world.of("person")
        if person.attrs["employer"] == company.name
        and person.attrs["salary"] > threshold
    )
    question = f"How many employees of {company.name} earn more than {threshold}?"
    return question, count


def _comparison(rng, world):
    while True:
        a, b = rng.sample(world.of("person"), 2)
        if a.attrs["age"] != b.attrs["age"]:
            older = a if a.attrs["age"] > b.attrs["age"] else b
            return f"Who is older, {a.name} or {b.name}?", older.name


def _filtering(rng, world):
    products = world.of("product")
    years = sorted({product.attrs["launched"] for product in products})
    if len(years) < 2:
        return None
    year = rng.choice(years[:-1])
    names = sorted(p.name for p in products if p.attrs["launched"] > year)
    return f"Which products were launched after {year}?", names


QUESTIONS = {
    "lookup": _lookup,
    "multihop": _multihop,
    "aggregation": _aggregation,
    "comparison": _comparison,
    "filtering": _filtering,
}


def generate(seed: int, worlds: int) -> list[Item]:
    """One question per task family for each of `worlds` seeded worlds."""
    rng = random.Random(seed)
    items = []
    for index in range(worlds):
        world = make_world(rng)
        while (filtering := _filtering(rng, world)) is None:
            world = make_world(rng)
        for task in TASKS:
            question, answer = (
                filtering if task == "filtering" else QUESTIONS[task](rng, world)
            )
            items.append(Item(f"{index}-{task}", task, question, answer, world))
    return items


def save(items: list[Item], path: Path) -> None:
    lines = (json.dumps(asdict(item), ensure_ascii=False) for item in items)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load(path: Path) -> list[Item]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        world = World([Entity(**entity) for entity in raw["world"]["entities"]])
        items.append(Item(**{**raw, "world": world}))
    return items
