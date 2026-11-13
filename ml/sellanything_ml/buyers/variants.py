"""One cat sale where one buyer model writes every line in several ways of picking words at once, to compare them
line by line (owner, 2026-10-28: B5 always taking the likeliest word, top-p 0.9, the game's top-k 50, and top-k 50 on
another seed, which shows how much chance alone moves a line; notebook 10 section 2 did the same on held-out sales).

Unlike paired.PairedSale, the ways share one sale: every buyer's story follows the lead way (the game's top-k 50), so
in every turn all the ways answer the very same prompt, and any difference between them comes from how the words were
picked. The other ways' openings, questions and reactions are written down next to the lead's but never enter a story.
Nothing is picked or replaced: every way's line is shown as the model wrote it.

The sale is saved like a paired one, to `logs/compare/<started>_seed<seed>_<model>_ways.json` and `.md`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from . import paired
from .cat_market import BuyerGPT, CatMarket, _typed, parse
from .paired import PairedSale, _short


@dataclass(frozen=True)
class Way:
    """How words are picked: temperature 0 always takes the likeliest; top_k/top_p None means no cut."""
    label: str
    temperature: float
    top_k: int | None = None
    top_p: float | None = None
    seed_offset: int = 0


WAYS = {
    "greedy": Way("always the likeliest", 0.0),
    "topp": Way("top-p 0.9", 0.7, top_p=0.9),
    "topk": Way("top-k 50 (the game)", 0.7, top_k=50),
    "topk2": Way("top-k 50, other seed", 0.7, top_k=50, seed_offset=1),
}
LEAD = "topk"                                   # the game's way: the stories follow it
PERSONA = re.compile(r"<\|persona\|>([^,|]+),")


class _Every:
    """The model as the sale sees it: each line is written in every way on the same prompt and seed (plus the way's
    seed offset); the lead's line is the one the sale uses, and every way's line is kept in `calls`."""

    def __init__(self, gpt: Any, ways: dict[str, Way], lead: str) -> None:
        self.gpt, self.ways, self.lead = gpt, ways, lead
        self.name, self.prompt_format = gpt.name, gpt.prompt_format
        self.calls: list[dict[str, Any]] = []

    def say(self, prompt: str, task: str, seed: int) -> str:
        lines = {key: self.gpt.say(prompt, task, seed + w.seed_offset, temperature=w.temperature, top_k=w.top_k,
                                   top_p=w.top_p) for key, w in self.ways.items()}
        who = PERSONA.search(prompt)
        self.calls.append({"task": task, "name": who.group(1) if who else None, "lines": lines})
        return lines[self.lead]


class WaysSale(PairedSale):
    """One model's sale in every way of picking words; `state` has the shape of a paired sale's (one column per way,
    named `<model>:<way>`), so the comparison screen and the notebooks read it the same way."""

    def __init__(self, market: CatMarket, every: _Every, keys: dict[str, str], seed: int, out_dir: Path | None) -> None:
        super().__init__({}, seed, out_dir)
        self.market, self.every, self.keys = market, every, keys          # keys: way -> column name ("b5:greedy")
        self.lead = keys[every.lead]
        self.stats = {k: {"lines": 0, "unreadable": 0} for k in keys.values()}
        self.offers: dict[str, dict[str, Any]] = {k: {} for k in keys.values()}                       # each way's last offer per buyer
        self.openings = {k: {} for k in keys.values()}
        for call in every.calls:
            if call["task"] == "open":
                for way, row in self._parsed(call).items():
                    readable = row is not None and not row.get("unreadable")
                    got = cast(dict[str, Any], row)       # read only when readable (so not None)
                    self.openings[way][call["name"]] = {"says": got["says"] if readable else None,
                                                        "offer": got.get("offer") or None if readable else None,
                                                        "written": call["lines"][self._way(way)]}
                    self.offers[way][call["name"]] = (row or {}).get("offer") or None

    @classmethod
    def start(cls, gpt: Any, ways: list[str], n_buyers: int = 4, seed: int | None = None,  # type: ignore[override]
              out_dir: Path | None = None, cards: dict[str, list[dict[str, Any]]] | None = None,
              save: bool = True) -> WaysSale:
        """The buyers open and the first one asks, every line written in each of `ways` (names in WAYS, the lead
        among them); the stories follow the lead. Saved in `out_dir` (default logs/compare) unless `save` is False."""
        if LEAD not in ways or len(set(ways)) != len(ways) or not set(ways) <= set(WAYS):
            raise ValueError(f"the ways must be different names from {sorted(WAYS)}, with {LEAD!r} among them")
        seed = seed if isinstance(seed, int) else random.randrange(2**31)
        every = _Every(gpt, {w: WAYS[w] for w in ways}, LEAD)
        market = CatMarket.start(cast(BuyerGPT, every), n_buyers, seed=seed, cards=cards, ask=False)     # _Every says like one
        model = _short(gpt.name)
        keys = {w: f"{model}:{w}" for w in ways}
        sale = cls(market, every, keys, seed, (out_dir or paired.COMPARE_DIR) if save else None)
        sale.log += [f"[{sale.lead}] {line}" for line in market.log]
        sale._ask()
        sale.save()
        return sale

    def _way(self, key: str) -> str:
        """b5:greedy -> greedy."""
        return next(w for w, k in self.keys.items() if k == key)

    def _parsed(self, call: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
        """Every way's line of one call, read like the sale reads the lead's (None: unreadable), counted per way."""
        out = {}
        for way, text in call["lines"].items():
            key = self.keys[way]
            row = parse(call["task"], text, self.every.prompt_format)
            self.stats[key]["lines"] += 1
            if row is None:
                self.stats[key]["unreadable"] += 1
                row = None if not text else {"says": text, "unreadable": True}
            out[key] = row
        return out

    def _ask(self) -> None:
        """The next buyer's question in the lead's words (the one the player answers), with what the other ways would
        have asked in the same place. None left: the sale is over."""
        at = len(self.every.calls)
        self.market._next_question()
        self.question = None
        if self.market.question is None:
            return
        asker, text = self.market.question
        call = next(c for c in reversed(self.every.calls[at:]) if c["task"] == "ask" and c["name"] == asker)
        asked = {k: (row or {}).get("says") for k, row in self._parsed(call).items()}
        self.question = {"asker": asker, "text": text, "by": self.lead, "ways": asked}
        self.log.append(f"{asker} asked (written by {self.lead}): {text}")

    def answer_text(self, text: str) -> tuple[list[str], dict[str, Any]]:
        """The player's answer: every buyer still here reacts once in each way on the same prompt; the lead's
        reaction goes into the stories; then the next question."""
        if self.question is None:
            raise ValueError("no question is waiting")
        text = _typed(text, "answer")
        start = len(self.log)
        self.log.append(f"You: {text}")
        before, at = dict(self.market.offers), len(self.every.calls)
        self.market._answered(text, None, "player")
        turn: dict[str, Any] = {**self.question, "answer": text, "reactions": {k: {} for k in self.keys.values()}}
        for call in self.every.calls[at:]:
            if call["task"] != "react":
                continue
            name = call["name"]
            for key, row in self._parsed(call).items():
                if row is None or row.get("unreadable"):
                    turn["reactions"][key][name] = {"says": (row or {}).get("says", ""), "readable": False,
                                                    "old": before[name], "offer": None}
                else:
                    turn["reactions"][key][name] = {"says": row["says"], "readable": True, "old": before[name],
                                                    "offer": row["offer"]}
                    self.offers[key][name] = row["offer"]
                self.log.append(f"[{key}] {name}: {turn['reactions'][key][name]['says']}")
        for name in self.market.cards:
            if before[name] <= 0:
                for key in self.keys.values():
                    turn["reactions"][key][name] = {"left_before": True}
        self.turns.append(turn)
        self._ask()
        self.save()
        return self.log[start:], {"option": None, "answer": text, "rule": "model"}

    def state(self) -> dict[str, Any]:
        """The comparison screen's state: one column per way, the lead first among equals in `lead`."""
        buyers = [{"name": n, "persona": f"{n}, who {c['persona']}.", "traits": c["cares_about"],
                   "offers": {k: self.offers[k].get(n) for k in self.keys.values()}}
                  for n, c in self.market.cards.items()]
        return {"item": "cat", "label": "buy", "noun": "cat", "compare": list(self.keys.values()), "seed": self.seed,
                "lead": self.lead, "labels": {k: WAYS[w].label for w, k in self.keys.items()},
                "intro": (f"One model, {len(self.keys)} ways to pick each word. Every buyer's story follows "
                          f"{self.lead} (the game's way), so in every turn each way answers the very same prompt."),
                "finished": self.question is None, "buyers": buyers, "openings": self.openings, "turns": self.turns,
                "question": self.question, "sold": None, "log": self.log, "model_stats": self.stats,
                "saved": str(self._path(".md"))}

    def _path(self, suffix: str) -> Path:
        stamp = self.started.strftime("%Y-%m-%d_%H%M%S")
        model = self.lead.partition(":")[0]
        return (self.out_dir or paired.COMPARE_DIR) / f"{stamp}_seed{self.seed}_{model}_ways{suffix}"
