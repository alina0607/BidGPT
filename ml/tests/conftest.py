"""Shared pytest setup."""

from pathlib import Path

import pytest

# These tests read the small-words prompt folder, which is local working data and not in the
# repository, so they run only on a machine that has it (not in CI).
SMALL_WORDS_FOLDER = (
    Path(__file__).resolve().parents[2] / "data" / "written_sales" / "prompts_3000_3049"
)
NEEDS_SMALL_WORDS_FOLDER = {
    "test_small_words.py::test_the_prompt_folder_is_up_to_date_and_names_no_held_out_word_outside_its_ban",
    "test_small_words.py::test_check_cues_passes_a_sale_that_follows_its_plan_and_flags_each_fault",
    "test_small_words.py::test_check_cues_lists_a_turned_buyer_moving_the_wrong_way_for_reading",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if SMALL_WORDS_FOLDER.exists():
        return
    skip = pytest.mark.skip(reason="needs data/written_sales/prompts_3000_3049; not in the repo")
    for item in items:
        if f"{item.path.name}::{item.name}" in NEEDS_SMALL_WORDS_FOLDER:
            item.add_marker(skip)
