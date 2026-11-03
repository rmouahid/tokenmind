import pytest

from tokenmind.grading import grade, parse_number


@pytest.mark.parametrize(
    "text, number",
    [("52000", 52000), ("52,000 per year", 52000), ("52k", 52000), ("34 years", 34),
     ("none", None)],
)  # fmt: skip
def test_parse_number(text, number):
    assert parse_number(text) == number


@pytest.mark.parametrize(
    "answer, response, correct",
    [
        ("Lyon", "Lyon", True),
        ("Lyon", "lyon.", True),
        ("Lyon", "Porto", False),
        ("data engineer", "A data engineer", True),
        ("Lena Moreau", "Lena Moreau is older", True),
        ("Lena Moreau", "Lena Moreau is older than Tomas Varga", False),
        (52000, "52k", True),
        (3, "There are 3", True),
        (3, "", False),
        (["Kalo", "Venzu"], "Venzu, Kalo", True),
        (["Kalo", "Venzu"], "Kalo and Venzu", True),
        (["Kalo", "Venzu"], "Kalo", False),
    ],
)
def test_grade(answer, response, correct):
    assert grade(answer, response) is correct
