"""Does a buyer model choose its scene's things from the buyer's persona? A teacher-forced test on the written scenes of
the two B2 validation sets, on the CPU, nothing sampled.

    cd ml && python tools/probe_persona_swap.py runs/sft_b2/best.pt --out runs/sft_b2f6/persona_swap/b2.jsonl
    cd ml && python tools/probe_persona_swap.py runs/sft_b2f6/best.pt --sets runs/eval_sets_b2f6 --out …/b2f6.jsonl

For every written scene we take the log-probability the model gives to the written list of three things (everything
before " |"). Then only the persona on the buyer's card is changed to another one ("who is a student on a tight
budget" → "who breeds show cats" in format 5, the <|buyer:…|> token in format 6) and the same list is scored again.
The rest of the card and the story stay, so they can still point to the same things; the drop is what the persona
itself adds. `drop` = log p(list | own persona) - log p(list | other persona), per thing in the list.

`--swap` chooses what is changed, always to what a buyer of that next persona has in the same validation set:
`persona` (above), `cares` (only the card's "cares about" list), `card` (both), `card+opening` (both, and the
buyer's own opening line at the top of the story). Later lines of the story (the buyer's own reactions) stay.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ML), str(ML / "tools")]
SETS = ("old", "new")
SWAPS = ("persona", "cares", "card", "card+opening")
CARES = re.compile(r"cares about: [^|<]*?(?= \||<\|end\|>)")
OPENING = re.compile(r"Your opening \(\$\d+\): [^\n<]*")


def swap_persona(prompt: str, own: str, other: str, buyer_tokens: dict[str, str]) -> str:
    """The prompt with the buyer's persona replaced: the <|buyer:…|> token (format 6) or "who … |" (formats 4, 5)."""
    if buyer_tokens[own] in prompt:
        return prompt.replace(buyer_tokens[own], buyer_tokens[other], 1)
    return prompt.replace(f"who {own} |", f"who {other} |", 1)


def donors(examples: list[dict], personas: list[str], buyer_tokens: dict[str, str]) -> dict[str, dict[str, str]]:
    """For each persona, the "cares about" list and the opening line of the first buyer in the file who has it."""
    out: dict[str, dict[str, str]] = {}
    for ex in examples:
        opening = OPENING.search(ex["prompt"])
        if opening is None:
            continue
        persona = next(p for p in personas if buyer_tokens[p] in ex["prompt"] or f"who {p} |" in ex["prompt"])
        out.setdefault(persona, {"cares": CARES.search(ex["prompt"])[0], "opening": opening[0]})
    return out


def swapped_prompt(prompt: str, own: str, other: str, buyer_tokens: dict[str, str], swap: str,
                   donor: dict[str, str]) -> str:
    """The prompt with the parts named by `swap` changed to those of a buyer with the other persona."""
    if swap in ("persona", "card", "card+opening"):
        prompt = swap_persona(prompt, own, other, buyer_tokens)
    if swap in ("cares", "card", "card+opening"):
        prompt = CARES.sub(lambda _: donor["cares"], prompt, count=1)
    if swap == "card+opening" and OPENING.search(prompt):
        prompt = OPENING.sub(lambda _: donor["opening"], prompt, count=1)
    return prompt


def main() -> None:
    """Score every written list with the buyer's own persona and with the next one, and write one row per scene."""
    import torch

    from sellanything_ml.buyers.format import BUYER_TOKENS
    from sellanything_ml.buyers.stories import PERSONAS
    from sellanything_ml.llm.evaluate import load_model
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--sets", type=Path, default=ML / "runs/eval_sets_b2", help="folder of val_{old,new}_format*.jsonl")
    ap.add_argument("--swap", choices=SWAPS, default="persona", help="what to change to another buyer's")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model, tok = load_model(a.checkpoint, torch.device("cpu"))
    personas = list(PERSONAS)

    @torch.no_grad()
    def logp(prompt: str, listed: str) -> float:
        head, full = tok.encode(prompt), tok.encode(prompt + listed)
        assert full[:len(head)] == head
        logits, _ = model(torch.tensor([full]))
        lp = torch.log_softmax(logits[0, len(head) - 1:len(full) - 1].float(), -1)
        return float(lp.gather(1, torch.tensor(full[len(head):]).unsqueeze(1)).sum())

    rows = []
    for which in SETS:
        [path] = a.sets.glob(f"val_{which}_format*.jsonl")
        examples = [json.loads(line) for line in path.read_text().splitlines()]
        donor = donors(examples, personas, BUYER_TOKENS)
        for i, ex in enumerate(examples):
            if ex["task"] != "imagine":
                continue
            own = next(p for p in personas if BUYER_TOKENS[p] in ex["prompt"] or f"who {p} |" in ex["prompt"])
            other = next(personas[(personas.index(own) + k) % len(personas)] for k in range(1, len(personas))
                         if a.swap == "persona" or personas[(personas.index(own) + k) % len(personas)] in donor)
            swapped = swapped_prompt(ex["prompt"], own, other, BUYER_TOKENS, a.swap, donor.get(other, {}))
            listed = ex["target"][:ex["target"].index(" | ")]
            with_own, with_other = logp(ex["prompt"], listed), logp(swapped, listed)
            rows.append({"set": which, "index": i, "persona": own, "other": other, "list": listed,
                         "logp_own": with_own, "logp_other": with_other, "drop_per_thing": (with_own - with_other) / 3})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for which in SETS:
        got = sorted(r["drop_per_thing"] for r in rows if r["set"] == which)
        print(f"{which}: {len(got)} scenes, drop per thing mean {sum(got) / len(got):.2f}, median {got[len(got) // 2]:.2f}")


if __name__ == "__main__":
    main()
