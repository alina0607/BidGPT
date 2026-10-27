"""Matching what the player typed to a drawable class (sketch/drawer.py); no model needed."""

from sellanything_ml.sketch.data import load_categories
from sellanything_ml.sketch.drawer import match

CATS = load_categories()


def test_exact_case_articles_and_plurals():
    assert match("Dog", CATS) == ("dog", [])
    assert match("  the  dog! ", CATS) == ("dog", [])
    assert match("guitars", CATS) == ("guitar", [])
    assert match("buses", CATS) == ("bus", [])
    assert match("eiffel tower", CATS) == ("The Eiffel Tower", [])


def test_close_typos_are_accepted_far_ones_are_offered():
    assert match("bicyle", CATS) == ("bicycle", [])
    label, suggestions = match("unicorn", CATS)
    assert label is None and 1 <= len(suggestions) <= 3
    assert all(s in CATS for s in suggestions)


def test_nothing_typed_matches_nothing():
    assert match("", CATS)[0] is None
