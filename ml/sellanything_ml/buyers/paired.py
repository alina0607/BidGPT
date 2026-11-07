"""One cat sale played by two buyer models side by side, to compare them line by line (owner, 2026-10-28: "the same
buyer must have both a B4f9 and a B5 answer at the same time").

Each model gets its own CatMarket on the same seed, so the buyers are the same people with the same cards; each model
writes its own buyers' openings, reactions and offers, which stay apart. What is shared is what the player compares
against: every question is put to both sales (the models take turns writing it, in the asking buyer's name), and the
player's one typed answer goes to both. Nothing is picked or replaced: each model's line is shown as it wrote it.

After the openings and after every answer the sale is written to `logs/compare/<started>_seed<seed>.json` and `.md`
(the same sale, buyer by buyer and turn by turn, both models next to each other), to study after playing.
"""

from __future__ import annotations

import datetime
import json
import random
from pathlib import Path
from typing import Any

from .cat_market import CatMarket, _typed

COMPARE_DIR = Path(__file__).resolve().parents[2] / "logs" / "compare"


def _short(gpt_name: str) -> str:
    """buyer_gpt_b4f9 -> b4f9."""
    return gpt_name.removeprefix("buyer_gpt_")


class PairedSale:
    """Two models' sales of one cat in lockstep; `state` is what the game's comparison screen draws."""

    def __init__(self, sales: dict[str, CatMarket], seed: int, out_dir: Path | None) -> None:
        self.sales = sales
        self.seed = seed
        self.turns: list[dict[str, Any]] = []
        self.question: dict[str, str] | None = None
        self.started = datetime.datetime.now()
        self.out_dir = out_dir
        self.log: list[str] = []            # every line of both sales, "[b5] Name: ...", for the server's play log
        # each model's opening per buyer (what they said and their first offer; None: the opening was unreadable)
        self.openings = {m: {n: {"says": s.stories[n].opening.partition("): ")[2] or None, "offer": s.offers[n] or None}
                             for n in s.cards} for m, s in sales.items()}

    @classmethod
    def start(cls, gpts: list[Any], n_buyers: int = 4, seed: int | None = None,
              out_dir: Path | None = None, cards: dict[str, list[dict[str, Any]]] | None = None,
              save: bool = True) -> PairedSale:
        """Both models open on the same buyers; the first question is written by the first model. The sale is saved
        in `out_dir` (default logs/compare) unless `save` is False."""
        if len(gpts) < 2 or len({g.name for g in gpts}) != len(gpts):
            raise ValueError("comparing needs two different models")
        seed = seed if isinstance(seed, int) else random.randrange(2**31)
        sales = {_short(g.name): CatMarket.start(g, n_buyers, seed=seed, cards=cards, ask=False) for g in gpts}
        first = next(iter(sales.values()))
        assert all(s.cards == first.cards for s in sales.values())      # the same people with the same cards
        sale = cls(sales, seed, (out_dir or COMPARE_DIR) if save else None)
        for model, s in sales.items():
            sale.log += [f"[{model}] {line}" for line in s.log]
        sale._ask()
        sale.save()
        return sale

    # --- one turn --------------------------------------------------------------------------------

    def _ask(self) -> None:
        """The next question, written by the models in turn (the other one first if it has nobody left to ask), and
        put to every sale in the asking buyer's name. None left in any sale: the sale is over."""
        self.question = None
        models = list(self.sales)
        start = len(self.turns) % len(models)
        for model in models[start:] + models[:start]:
            writer = self.sales[model]
            writer._next_question()
            if writer.question is None:
                continue
            asker, text = writer.question
            for other, s in self.sales.items():
                if other != model:
                    s.put_question(asker, text, source=f"model_{model}")
            self.question = {"asker": asker, "text": text, "by": model}
            self.log.append(f"{asker} asked (written by {model}): {text}")
            return

    def sell(self, buyer: str) -> dict[str, Any]:
        """A comparison has two sets of offers: nothing is sold; the next sale starts a new comparison."""
        raise ValueError("a comparison sale is not sold; start the next one")

    def scene(self, buyer: str) -> list[dict[str, Any]]:
        raise ValueError("scenes are not compared yet")

    def answer(self, option: str) -> list[str]:
        """These buyers ask in their own words: the player types."""
        raise ValueError("this buyer asks in their own words; type your answer")

    def answer_text(self, text: str) -> tuple[list[str], dict[str, Any]]:
        """The player's answer goes to both sales; every buyer still there reacts in each; then the next question."""
        if self.question is None:
            raise ValueError("no question is waiting")
        text = _typed(text, "answer")
        start = len(self.log)
        self.log.append(f"You: {text}")
        turn = {**self.question, "answer": text, "reactions": {}}
        for model, s in self.sales.items():
            turn["reactions"][model] = self._reactions(model, s, text)
        self.turns.append(turn)
        self._ask()
        self.save()
        return self.log[start:], {"option": None, "answer": text, "rule": "model"}

    def _reactions(self, model: str, s: CatMarket, text: str) -> dict[str, dict[str, Any]]:
        """One sale's reactions to the answer, per buyer: what they said and the offer before and after (a buyer who
        had left is not asked; an unreadable line is kept as written, with no offer)."""
        before, at = dict(s.offers), len(s.log)
        s._answered(text, None, "player")
        said = {}
        for line, source in zip(s.log[at:], s.sources[at:], strict=True):
            name, _, says = line.partition(": ")
            said[name] = {"says": says, "readable": source == "model"}
            self.log.append(f"[{model}] {line}")
        out = {}
        for name in s.cards:
            if before[name] <= 0:
                out[name] = {"left_before": True}
            elif name in said:
                readable = said[name]["readable"]
                out[name] = {"says": said[name]["says"], "readable": readable, "old": before[name],
                             "offer": s.offers[name] if readable else None}
        return out

    # --- what the game draws and what is kept --------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The comparison screen's state: the buyers' cards, each model's openings and current offers, every turn."""
        first = next(iter(self.sales.values()))
        buyers = [{"name": n, "persona": f"{n}, who {c['persona']}.", "traits": c["cares_about"],
                   "offers": {m: (s.offers[n] or None) for m, s in self.sales.items()}}
                  for n, c in first.cards.items()]
        return {"item": "cat", "label": "buy", "noun": "cat", "compare": list(self.sales), "seed": self.seed,
                "finished": self.question is None, "buyers": buyers, "openings": self.openings, "turns": self.turns,
                "question": self.question, "sold": None, "log": self.log,
                "model_stats": {m: dict(s.stats) for m, s in self.sales.items()}, "saved": str(self._path(".md"))}

    def _path(self, suffix: str) -> Path:
        stamp = self.started.strftime("%Y-%m-%d_%H%M%S")
        return (self.out_dir or COMPARE_DIR) / f"{stamp}_seed{self.seed}{suffix}"

    def save(self) -> None:
        """The sale so far, as JSON (everything) and Markdown (to read): rewritten after every turn."""
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        state = self.state()
        self._path(".json").write_text(json.dumps(state, ensure_ascii=False, indent=1))
        self._path(".md").write_text(report(state))


def report(state: dict[str, Any]) -> str:
    """A paired sale to read: who the buyers are, then the openings and every turn, each buyer's line from each model
    next to the other's, with a blank column for which one is more logical."""
    models = state["compare"]
    intro = state.get("intro") or ("The same buyers (same seed and cards) hear the same question and the same "
                                   "answer; each model writes its own buyers' lines and keeps its own offers.")
    out = [f"# Comparing {' vs '.join(models)}: cat sale, seed {state['seed']}", "", intro, "", "## Buyers", ""]
    for b in state["buyers"]:
        out.append(f"- **{b['name']}**: {b['persona'].partition(', who ')[2]} Cares about: {', '.join(b['traits'])}.")
    header = "| Buyer | " + " | ".join(models) + " | More logical |"
    rule = "|---|" + "---|" * len(models) + "---|"

    def cell(r: dict[str, Any] | None) -> str:
        if not r:
            return "(no line)"
        if r.get("left_before"):
            return "(had left)"
        if not r.get("readable", True):
            return f"unreadable: {r['says']}"
        move = r["offer"] - r["old"] if "old" in r else 0
        price = f"${r['offer']}" + (f" ({move:+d})" if "old" in r else "")
        return f"{price} {r['says']}".replace("|", "/")

    out += ["", "## Openings", "", header, rule]
    for b in state["buyers"]:
        n = b["name"]
        out.append(f"| {n} | " + " | ".join(cell({"offer": state['openings'][m][n]['offer'] or 0, "says":
                                                  state['openings'][m][n]['says'] or "(left)"}) for m in models) + " | |")
    for i, t in enumerate(state["turns"], 1):
        out += ["", f"## Turn {i}: {t['asker']} asks (written by {t['by']})", "", f"> {t['text']}", ""]
        out += _other_questions(t)
        out += [f"**Your answer:** {t['answer']}", "", header, rule]
        for b in state["buyers"]:
            n = b["name"]
            out.append(f"| {n} | " + " | ".join(cell(t["reactions"][m].get(n)) for m in models) + " | |")
    if state["question"]:
        q = state["question"]
        out += ["", f"## Waiting: {q['asker']} asks (written by {q['by']})", "", f"> {q['text']}", ""]
        out += _other_questions(q)
    out += ["", "## Offers now", "", "| Buyer | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for b in state["buyers"]:
        out.append(f"| {b['name']} | " + " | ".join(f"${b['offers'][m]}" if b["offers"][m] else "left" for m in models)
                   + " |")
    return "\n".join(out) + "\n"


def _other_questions(q: dict[str, Any]) -> list[str]:
    """What the other ways of picking words would have asked in the same place (variants.WaysSale), as a list."""
    other = {k: v for k, v in (q.get("ways") or {}).items() if k != q["by"]}
    if not other:
        return []
    return ["The other ways would have asked:", ""] + [f"- {k}: {v or '(unreadable)'}" for k, v in other.items()] + [""]
