"""Does a buyer model read its own list of things when it writes a scene? A teacher-forced test on the written scenes of
the two B2 validation sets, on the CPU, nothing sampled.

    cd ml && python tools/probe_scene_list.py runs/sft_b2/best.pt --out runs/sft_b2/scene_list/b2.jsonl

A written scene is "a, b, c | line". For every thing whose name appears in the line, the model is given the prompt, the
list and the line up to that word, and we take the log-probability it gives to the word. Then the same thing is
swapped in the list for another drawable thing (one not in the scene) and the same word is scored again. If the model
writes the line from its list, the word gets much less likely once it is no longer on the list; if the model ignores
the list, the two numbers are about the same. `drop` = log p(word | thing listed) - log p(word | thing swapped out).
In prompt format 6 the list is three thing tokens ("<|thing:a|><|thing:b|><|thing:c|> | line"), and the swap
replaces one token: `--sets runs/eval_sets_b2f6` reads the same prompts in format 6.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML), str(ML / "tools")]
SETS = ("old", "new")
SWAPS = ("umbrella", "basket", "lantern", "bucket", "toaster", "teapot")   # common enough drawable things to swap in


def split_scene(target: str, thing_token: re.Pattern[str]) -> tuple[str, str, list[str]]:
    """A written scene as (list with the space before the bar, line with the space after it, the things)."""
    cut = target.index(" | ")
    listed, line = target[:cut + 1], target[cut + 2:]
    tokens = thing_token.findall(listed)
    return listed, line, tokens or [t.strip() for t in listed.split(",")]


def swap_cases(line: str, things: list[str], thing_tokens: dict[str, str] | None) -> Iterator[tuple[int, str, str, str, int, str]]:
    """For each listed thing whose word is in the line: (place, thing, swap-in, the word with its space, where the word
    starts in the line, the list with the thing swapped out). `thing_tokens` writes the list as tokens (format 6)."""
    for k, thing in enumerate(things):
        word = thing.split()[-1]
        m = re.search(rf"\s\b{re.escape(word)}", line, re.IGNORECASE)   # the word with the space before it
        swap = next((s for s in SWAPS if s not in things and not re.search(rf"\b{s}\b", line, re.I)), None)
        if m is None or swap is None:
            continue
        listing = [swap if j == k else t for j, t in enumerate(things)]
        swapped = "".join(thing_tokens[t] for t in listing) if thing_tokens else ", ".join(listing)
        yield k, thing, swap, line[m.start():m.end()], m.start(), swapped


def main() -> None:
    """Score every named thing of every written scene with the thing listed and swapped out; one row per thing."""
    import torch

    from sellanything_ml.buyers.format import END, THING_TOKENS
    from sellanything_ml.buyers.stories import DRAWABLE_THINGS
    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--sets", type=Path, default=ML / "runs/eval_sets_b2", help="folder of val_{old,new}_format*.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    assert all(s in DRAWABLE_THINGS for s in SWAPS)
    torch.set_num_threads(4)
    model, tok = load_model(a.checkpoint, torch.device("cpu"))

    @torch.no_grad()
    def logp(prefix: str, word: str) -> float | None:
        """log p(word | prefix), summed over the word's tokens; None if the word does not tokenize on its own."""
        head, full = tok.encode(prefix), tok.encode(prefix + word)
        if full[:len(head)] != head or len(full) == len(head):
            return None
        logits, _ = model(torch.tensor([full]))
        lp = torch.log_softmax(logits[0, len(head) - 1:len(full) - 1].float(), -1)
        return float(lp.gather(1, torch.tensor(full[len(head):]).unsqueeze(1)).sum())

    thing_token = re.compile(r"<\|thing:([^|]+)\|>")
    rows = []
    for which in SETS:
        [path] = a.sets.glob(f"val_{which}_format*.jsonl")
        for i, ex in enumerate(json.loads(line) for line in path.read_text().splitlines()):
            if ex["task"] != "imagine":
                continue
            listed, line, things = split_scene(ex["target"].replace(END, ""), thing_token)
            tokens = THING_TOKENS if "<|thing:" in listed else None
            for k, thing, swap, spoken, start, swapped in swap_cases(line, things, tokens):
                with_it = logp(ex["prompt"] + listed + "|" + line[:start], spoken)
                without = logp(ex["prompt"] + swapped + " |" + line[:start], spoken)
                if with_it is None or without is None:
                    continue
                rows.append({"set": which, "index": i, "thing": thing, "place": k, "swap": swap,
                             "logp_listed": with_it, "logp_swapped": without, "drop": with_it - without})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{a.out}: {len(rows)} things, mean drop {sum(r['drop'] for r in rows) / len(rows):.2f} nats")


if __name__ == "__main__":
    main()
