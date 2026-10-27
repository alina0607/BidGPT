"""Baselines for the react task, so the model's Δ accuracy has something to be compared with.

    python -m sellanything_ml.llm.baselines

- majority: always predict the most common Δ in the training set.
- answer-only: for each (item, attribute, answer) seen in training, predict the Δ most often
  given to it, ignoring who the buyer is. Beating this means the model uses the persona.
  (Unseen combinations, including every held-out item, fall back to the majority Δ.)
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from ..buyers.datagen import load_split


def load(split: str) -> list[dict]:
    return [r for r in load_split(split) if r["task"] == "react"]


def main() -> None:
    train = load("train")
    majority = Counter(r["delta"] for r in train).most_common(1)[0][0]
    by_answer: dict[tuple, Counter] = defaultdict(Counter)
    for r in train:
        by_answer[(r["item"], r["attr"], r["option"])][r["delta"]] += 1

    def sign(x: int) -> int:
        return (x > 0) - (x < 0)

    report = {"majority_delta": majority}
    for split in ("test_seen", "test_heldout"):
        rows = load(split)
        for name, predict in (
            ("majority", lambda r: majority),
            ("answer_only", lambda r: (by_answer[(r["item"], r["attr"], r["option"])].most_common(1)[0][0]
                                       if (r["item"], r["attr"], r["option"]) in by_answer else majority)),
        ):
            preds = [predict(r) for r in rows]
            report[f"{split}/{name}"] = {
                "n": len(rows),
                "delta_exact": round(sum(p == r["delta"] for p, r in zip(preds, rows)) / len(rows), 4),
                "delta_sign": round(sum(sign(p) == sign(r["delta"]) for p, r in zip(preds, rows)) / len(rows), 4),
                "delta_mae": round(sum(abs(p - r["delta"]) for p, r in zip(preds, rows)) / len(rows), 4),
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
