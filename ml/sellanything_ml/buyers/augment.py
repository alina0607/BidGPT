"""Noun swaps: teach the buyer model to copy the item's name from the prompt (buyer data v2).

    python -m sellanything_ml.buyers.augment            # buyers_v5_raw → buyers_v5

Trained on v1, the model writes the right item names for items it has seen, but on held-out items
it falls back to a seen one ("How old is the dog?" for a hedgehog): it memorized names instead of
reading them from `<|item|>`. Here a share of the training examples get their item renamed to a
made-up word, consistently in the prompt and the target, so the only way to get the name right is
to copy it. Made-up words rather than other real items, so no real item is paired with someone
else's attributes. Validation and test files are copied unchanged, so v1 and v2 compare directly.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from .datagen import BUYERS_DIR, RAW_DIR, load_split
from .format import item_names as names_for
from .schema import load_items

SWAP_P = 0.3
ONSETS = ["b", "br", "d", "dr", "f", "g", "gr", "k", "kl", "m", "n", "p", "pl", "r", "s", "st", "t", "tr", "v", "z"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "oo"]
CODAS = ["", "", "l", "n", "r", "s", "t", "x", "m"]


def made_up_noun(rng: random.Random) -> str:
    """A pronounceable non-word that starts with a consonant (so "a" stays the right article)."""
    return "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS) for _ in range(rng.choice((2, 3))))


def swap_noun(text: str, names: list[str], new: str) -> str:
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)

    def replace(m: re.Match) -> str:
        return new.capitalize() if m.group(0)[0].isupper() else new

    text = pattern.sub(replace, text)
    # "an apple" → "an zorvel": the made-up word starts with a consonant.
    return re.sub(r"\b([Aa])n (" + re.escape(new) + r")\b", r"\1 \2", text, flags=re.IGNORECASE)


def augment(rows: list[dict], nouns: dict[str, str], p: float, rng: random.Random) -> tuple[list[dict], int]:
    out, swapped = [], 0
    for row in rows:
        if rng.random() < p:
            new = made_up_noun(rng)
            names = names_for(row["item"], nouns.get(row["item"], row["item"]))
            row = {**row, "prompt": swap_noun(row["prompt"], names, new),
                   "target": swap_noun(row["target"], names, new), "swapped_to": new}
            swapped += 1
        out.append(row)
    return out, swapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=BUYERS_DIR)
    parser.add_argument("--p", type=float, default=SWAP_P, help="share of training examples renamed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    nouns = {key: item.noun for key, item in load_items().items()}
    rows, swapped = augment(load_split("train", args.src), nouns, args.p, random.Random(args.seed))
    with open(args.out / "train.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for path in [*args.src.glob("*.jsonl"), *args.src.glob("*.json")]:
        if path.name != "train.jsonl":
            shutil.copy(path, args.out / path.name)
    print(f"{swapped:,} of {len(rows):,} training examples renamed → {args.out}")


if __name__ == "__main__":
    main()
