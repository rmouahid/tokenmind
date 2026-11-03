"""Deterministic grading against gold answers: no LLM judge."""

import re

from tokenmind.dataset import Answer

NUMBER = re.compile(r"(\d[\d,  ]*(?:\.\d+)?)\s*(k\b)?", re.IGNORECASE)


def normalize(text: str) -> str:
    text = text.strip().lower().strip(" .'\"*`")
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return re.sub(r"\s+", " ", text)


def parse_number(text: str) -> int | None:
    """First number in `text`; `52,000`, `52 000` and `52k` all give 52000."""
    match = NUMBER.search(text)
    if not match:
        return None
    number = float(re.sub(r"[,  ]", "", match.group(1)))
    return round(number * 1000 if match.group(2) else number)


def grade(answer: Answer, response: str) -> bool:
    lines = [line for line in response.strip().splitlines() if line.strip()]
    if isinstance(answer, list):
        parts = re.split(r",|;|\band\b|\n", response)
        found = {normalize(part) for part in parts if normalize(part)}
        return found == {normalize(name) for name in answer}
    if not lines:
        return False
    if isinstance(answer, int):
        return parse_number(lines[0]) == answer
    first, gold = normalize(lines[0]), normalize(answer)
    # A short sentence that names the answer ("Lena Moreau is older") counts.
    in_short_sentence = len(first.split()) <= len(gold.split()) + 3 and (
        re.search(rf"\b{re.escape(gold)}\b", first) is not None
    )
    return first == gold or in_short_sentence
