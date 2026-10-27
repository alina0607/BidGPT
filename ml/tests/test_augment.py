"""Noun swaps for buyer data v2 (buyers/augment.py)."""

import random

from sellanything_ml.buyers.augment import augment, made_up_noun, names_for, swap_noun


def test_names_include_short_forms_longest_first():
    assert names_for("alarm clock", "alarm clock") == ["alarm clock", "clock"]
    assert names_for("hedgehog", "pet hedgehog") == ["pet hedgehog", "hedgehog"]
    assert "saw" not in names_for("see saw", "seesaw")          # too short to rename safely


def test_swap_is_whole_word_keeps_capitals_and_fixes_the_article():
    names = names_for("apple", "apple")
    text = "An apple? I'd buy an apple, not pineapple. Apples too."
    assert swap_noun(text, names, "zorvel") == "A zorvel? I'd buy a zorvel, not pineapple. Apples too."
    assert swap_noun("Apple first.", names, "zorvel") == "Zorvel first."


def test_made_up_nouns_start_with_a_consonant():
    rng = random.Random(0)
    assert all(made_up_noun(rng)[0] not in "aeiou" for _ in range(200))


def test_augment_renames_prompt_and_target_together():
    row = {"item": "alarm clock", "task": "ask",
           "prompt": "<|item|>alarm clock<|end|><|ask|>type<|end|>",
           "target": "Andre: Is it a wind-up clock?<|end|>"}
    (out,), swapped = augment([row], {"alarm clock": "alarm clock"}, 1.0, random.Random(3))
    new = out["swapped_to"]
    assert swapped == 1 and f"<|item|>{new}<|end|>" in out["prompt"] and f"wind-up {new}?" in out["target"]
    assert row["prompt"].startswith("<|item|>alarm clock")      # the input row is not modified
