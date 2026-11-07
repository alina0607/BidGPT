"""The seller's prompts (prompt format 8: answer, decide) and the buyers' replies to counter-offers, for the same
held-out sales as runs/eval_sets_b2 (the old, the new and the b3 validation set).

    cd ml && PYTHONPATH=. python3 tools/make_eval_sets_seller.py        # → runs/eval_sets_b5/val_{old,new,b3}_seller.jsonl

Taken from data/processed/buyers_b5/val.jsonl (sales_sft --prompt-format 8 --sources b5). The buyer prompts of
those sales are the format-5 sets themselves (format 8 writes the same buyer prompts), so a format-8 model is scored
on the buyers' tasks with runs/eval_sets_b2 and on the seller's with these. Every seed of a set must be in the B5
validation split, so none of these prompts was trained on.
"""

from __future__ import annotations

import json
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
SELLER_TASKS = ("answer", "decide", "reply")


def main() -> None:
    """Write the three seller sets, checking every seed is in the B5 validation split."""
    rows = [json.loads(line) for line in (ML / "data/processed/buyers_b5/val.jsonl").read_text().splitlines()]
    out_dir = ML / "runs/eval_sets_b5"
    out_dir.mkdir(parents=True, exist_ok=True)
    for which in ("old", "new", "b3"):
        f5 = [json.loads(line) for line in (ML / f"runs/eval_sets_b2/val_{which}_format5.jsonl").read_text().splitlines()]
        seeds = {e["seed"] for e in f5}
        missing = seeds - {r["seed"] for r in rows}
        assert not missing, (which, sorted(missing))       # a seed outside B5's validation split was trained on
        picked = [r for r in rows if r["seed"] in seeds and r["task"] in SELLER_TASKS]
        (out_dir / f"val_{which}_seller.jsonl").write_text("".join(json.dumps(r) + "\n" for r in picked))
        counts = {t: sum(r["task"] == t for r in picked) for t in SELLER_TASKS}
        print(f"{which}: {len(seeds)} sales, {counts}")


if __name__ == "__main__":
    main()
