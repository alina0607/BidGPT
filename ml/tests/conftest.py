"""Shared pytest setup."""

from pathlib import Path

import pytest

# These tests read a prompt folder of data/written_sales, which is local working data and not in
# the repository, so they run only on a machine that has the folder (not in CI).
WRITTEN_SALES = Path(__file__).resolve().parents[2] / "data" / "written_sales"
NEEDS_FOLDER = {
    "test_small_words.py::test_the_prompt_folder_is_up_to_date_and_names_no_held_out_word_outside_its_ban":
        "prompts_3000_3049",
    "test_small_words.py::test_check_cues_passes_a_sale_that_follows_its_plan_and_flags_each_fault":
        "prompts_3000_3049",
    "test_small_words.py::test_check_cues_lists_a_turned_buyer_moving_the_wrong_way_for_reading":
        "prompts_3000_3049",
    "test_heard_3050.py::test_the_prompt_folder_is_up_to_date_and_names_who_asks": "prompts_3050_3149",
    "test_heard_3050.py::test_check_cues_passes_a_sale_that_follows_its_plan_and_flags_who_asked":
        "prompts_3050_3149",
    "test_heard_3050.py::test_check_cues_reads_a_target_inside_another_answer_by_its_own_sentence":
        "prompts_3050_3149",
    "test_sayings_3150.py::test_the_prompts_and_cue_plan_on_disk_are_current": "prompts_3150_3234",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        folder = NEEDS_FOLDER.get(f"{item.path.name}::{item.name}")
        if folder and not (WRITTEN_SALES / folder).exists():
            item.add_marker(pytest.mark.skip(reason=f"needs data/written_sales/{folder}; not in the repo"))
