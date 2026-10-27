"""The player-style answer test (NEXT_SESSION_PROMPT 4.1 D, owner 2026-10-13): does the buyer model still read a seller
answer when it is typed the way a player types (lower case, short forms, typos, "but/though", numbers in digits)?

    PYTHONPATH=. python tools/player_style_test.py build          # writes the two example files below
    PYTHONPATH=. python tools/eval_buyer_b1.py runs/sft_b4/best.pt --data runs/eval_sets_b2/player/original.jsonl \
        --device cpu --out … --outputs runs/eval_sets_b2/player/b4_original_outputs.jsonl      (and the same for player)
    PYTHONPATH=. python tools/player_style_test.py compare runs/eval_sets_b2/player/b4_original_outputs.jsonl \
        runs/eval_sets_b2/player/b4_player_outputs.jsonl

`picked.json` holds 62 seller answers from the held-out sales of `val_new_format5.jsonl` (in each sale the 1st, 4th,
7th and 10th answer) with a rewrite that keeps every fact, negation and number and only changes how it is typed;
7 answers that were already typed that way are kept as they are and left out. Both files hold the reactions to the
other 55 answers, in the same order, so the evaluation samples each pair with the same seed and the only difference
is the wording of the seller's answer. The sale's own reaction is the target in both.

compare reports, over the pairs where both lines are readable: how often the model moves its offer the same way on
the original and the rewritten answer, the direction against the sale's buyer on each, the mean offer error on each,
the rules passed on each, and the pairs whose direction changed (for reading).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[1] / "runs" / "eval_sets_b2"
PLAYER = HERE / "player"


def build() -> None:
    rows = [json.loads(line) for line in (HERE / "val_new_format5.jsonl").read_text().splitlines()]
    picked = json.loads((PLAYER / "picked.json").read_text())
    original, player = [], []
    for p in picked:
        if p["unchanged"]:
            continue
        for i in p["examples"]:
            ex = rows[i]
            old = f"Seller: {p['original']}<|end|>"
            assert ex["prompt"].count(old) == 1 and ex["prompt"].endswith(old + "<|react|>"), (i, p["original"])
            original.append(ex)
            player.append(ex | {"prompt": ex["prompt"].replace(old, f"Seller: {p['player']}<|end|>")})
    for name, rows_out in (("original", original), ("player", player)):
        (PLAYER / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows_out))
    print(f"{sum(not p['unchanged'] for p in picked)} answers, {len(original)} reactions in each file")


def compare(a_path: Path, b_path: Path) -> None:
    from sellanything_ml.buyers.cat_market import parse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_buyer_b1 import context
    a = [json.loads(line) for line in a_path.read_text().splitlines()]
    b = [json.loads(line) for line in b_path.read_text().splitlines()]
    exs = [json.loads(line) for line in (PLAYER / "original.jsonl").read_text().splitlines()]

    def move(ex: dict[str, Any], line: str) -> str | None:
        row = parse("react", line)
        if row is None:
            return None
        held = context(ex)["offer"]
        return "walk" if row["offer"] == 0 else "up" if row["offer"] > held else "down" if row["offer"] < held else "hold"

    pairs = []
    for ex, ra, rb in zip(exs, a, b, strict=True):
        ma, mb = move(ex, ra["output"]), move(ex, rb["output"])
        if ma and mb:
            pairs.append((ex, ra, rb, ma, mb, move(ex, ex["target"].replace("<|end|>", ""))))
    n = len(pairs)

    def rate(vals: list[bool]) -> float:
        return round(sum(vals) / max(1, len(vals)), 3)

    report = {
        "pairs": n, "of": len(exs),
        "same_move_original_vs_player": rate([ma == mb for _, _, _, ma, mb, _ in pairs]),
        "same_direction_as_sale": {"original": rate([ma == t for *_, ma, _, t in pairs]),
                                   "player": rate([mb == t for *_, mb, t in pairs])},
        "rules_ok": {"original": rate([ra["checks"].get("rules_ok", False) for _, ra, *_ in pairs]),
                     "player": rate([rb["checks"].get("rules_ok", False) for _, _, rb, *_ in pairs])},
        "mean_offer_error": {"original": round(sum(ra["checks"]["offer_error"] for _, ra, *_ in pairs) / max(1, n), 2),
                             "player": round(sum(rb["checks"]["offer_error"] for _, _, rb, *_ in pairs) / max(1, n), 2)},
    }
    print(json.dumps(report, indent=1))
    out = b_path.with_name(b_path.stem.replace("_player_outputs", "") + "_player_compare")
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    lines = []
    for ex, ra, rb, ma, mb, t in pairs:
        if ma != mb:
            answer = ex["prompt"].rsplit("Seller: ", 1)[1].replace("<|end|><|react|>", "")
            lines += [f"**{ex['name']}** (sale {ex['seed']}), sale's move {t}", f"- answer: {answer}",
                      f"- original ({ma}): {ra['output']}", f"- player ({mb}): {rb['output']}", ""]
    out.with_suffix(".md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("compare")
    c.add_argument("original", type=Path)
    c.add_argument("player", type=Path)
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    else:
        compare(a.original, a.player)


if __name__ == "__main__":
    main()
