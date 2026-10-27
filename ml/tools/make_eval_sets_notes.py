"""The held-out prompts of runs/eval_sets_b2 (the old and the new validation set) in prompt format 7, with notes.

    cd ml && python tools/make_eval_sets_notes.py           # → runs/eval_sets_b3n/val_{old,new}_format7.jsonl

Taken from data/processed/buyers_b3_notes/val.jsonl (sales_sft --prompt-format 7 --sources b3): the same sales, the
same prompts in the same order, plus the note prompts. Every non-note prompt is checked against its format-5 twin:
same buyer and answer, and the same text once the notes are taken out, so a model scored on these answers exactly the
questions B1, B2, B2f6 and B3 answered, with the older answers' notes added.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
NOTES = re.compile(r" -> [^/\n]*?(?= / |\n|<\|end\|>|$)")          # a note after a question in "Asked earlier"


def without_notes(prompt: str) -> str:
    """The prompt with the notes taken out of its "Asked earlier" line (reaction lines also hold " -> ")."""
    return "\n".join(NOTES.sub("", line) if line.startswith("Asked earlier: ") else line
                     for line in prompt.split("\n"))


def main() -> None:
    """Write both sets and check them against the format-5 sets."""
    rows = [json.loads(line) for line in (ML / "data/processed/buyers_b3_notes/val.jsonl").read_text().splitlines()]
    out_dir = ML / "runs/eval_sets_b3n"
    out_dir.mkdir(parents=True, exist_ok=True)
    for which in ("old", "new"):
        f5 = [json.loads(line) for line in (ML / f"runs/eval_sets_b2/val_{which}_format5.jsonl").read_text().splitlines()]
        seeds = {e["seed"] for e in f5}
        f7 = [r for r in rows if r["seed"] in seeds]
        plain = [r for r in f7 if r["task"] != "note"]
        assert len(plain) == len(f5), (which, len(plain), len(f5))
        for a, b in zip(f5, plain, strict=True):
            assert (a["seed"], a["task"], a["name"], a["target"]) == (b["seed"], b["task"], b["name"], b["target"])
            assert without_notes(b["prompt"]) == a["prompt"], (which, a["seed"], a["task"], a["name"])
        (out_dir / f"val_{which}_format7.jsonl").write_text("".join(json.dumps(r) + "\n" for r in f7))
        print(f"{which}: {len(f7)} prompts ({len(f7) - len(plain)} notes), every other one matches format 5")


if __name__ == "__main__":
    main()
