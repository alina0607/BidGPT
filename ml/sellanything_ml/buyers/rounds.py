"""A selling round as the game sees it: start, answer questions, sell (docs/protocol.md).

The engine decides who asks what and how offers move; this module adds the words (v1 phrase
pools; the buyer model replaces them in v2) and packages everything into one state dict the game
can draw directly.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from . import answers, phrases
from . import format as fmt
from .datagen import ask_reason_tag, reason_tag, reason_topic
from .engine import Round, make_buyers, refuses_risky, round_price
from .schema import Item, Traits, load_items, load_labels, load_traits

if TYPE_CHECKING:
    from .voice import ModelVoice


# Why a buyer liked or disliked an answer, in a few words for their card: (tag, +1/-1) is the
# trait-driven cause the engine counted, so the player sees a reason, not a random number.
REASONS = {
    ("space", 1): "fits my home", ("space", -1): "too big",
    ("condition", 1): "good condition", ("condition", -1): "too worn",
    ("low_maintenance", 1): "easy upkeep", ("low_maintenance", -1): "too much work",
    ("family", 1): "good for kids", ("family", -1): "not for kids",
    ("status", 1): "prestige", ("status", -1): "no prestige",
    ("performance", 1): "works well", ("performance", -1): "won't do the job",
    ("style", 1): "looks good", ("style", -1): "looks dull",
    ("active", 1): "good outdoors", ("active", -1): "not outdoorsy",
}


def _reaction(buyer, reaction, option, outbid: bool) -> tuple[str, bool] | None:
    """What this buyer personally made of the answer, for their card: (words, liked). Only when the
    answer moved their interest, they raised their bid, or they left; an offer can still move for
    everyone because the item is worth more or less, which the amount next to the offer shows."""
    if reaction is not None and reaction.left:
        return "walked away", False
    if outbid:
        return "outbid the others", True
    if reaction is None or not reaction.delta:
        return None
    # The strongest of this buyer's tags pulling the same way as the change in interest.
    sign = 1 if reaction.delta > 0 else -1
    scored = [(abs(buyer.weights.get(t, 0) * s), t) for t, s in option.signals.items()
              if buyer.weights.get(t, 0) * s * sign > 0]
    if not scored:
        return None
    return REASONS[(max(scored)[1], sign)], sign > 0


def _reasons(buyer, option) -> set[tuple[str, int]]:
    """The reasons this answer can honestly mean to this buyer: every tag both they and the answer
    have, with the direction it pushes their interest."""
    return {(t, 1 if buyer.weights[t] * s > 0 else -1) for t, s in option.signals.items() if buyer.weights.get(t, 0) * s}


def _mood(reaction) -> str:
    """The sprite a reaction shows: walking off, or how the answer landed."""
    if reaction.left:
        return "left"
    return "happy" if reaction.delta > 0 else "unhappy" if reaction.delta < 0 else "idle"


@dataclass
class RoundSession:
    item_key: str
    label: str
    traits: Traits
    rng: random.Random
    round: Round | None = None
    names: list[str] = field(default_factory=list)
    personas: dict[str, str] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    # Who wrote each log line: "player", "model" (the buyer GPT), "phrases" (the pools, or the
    # fallback when the guardrail rejected a model line) or "engine" (bids, walking away).
    sources: list[str] = field(default_factory=list)
    # The log as the buyer model hears it: a typed answer becomes the option it matched, because the
    # model was trained on option texts; asking again ("Sorry, which is it?") is left out.
    heard: list[str] = field(default_factory=list)
    question: tuple | None = None          # (asker, attribute)
    sold: dict | None = None
    voice: ModelVoice | None = None        # None: phrase pools only
    # What each buyer's sprite should play (docs/art.md): idle, talk, happy, unhappy, left, won.
    moods: dict[str, str] = field(default_factory=dict)
    # The last answer, per buyer: how their offer moved, and what they made of it personally
    # ({"amount": 2, "reaction": "too much work", "liked": False}; no reaction if it didn't move them).
    changes: dict[str, dict] = field(default_factory=dict)

    # --- lifecycle ---------------------------------------------------------------------------

    @classmethod
    def start(cls, item_key: str, n_buyers: int = 4, seed: int | None = None,
              items: dict[str, Item] | None = None, labels: dict[str, str] | None = None,
              traits: Traits | None = None, voice=None) -> RoundSession:
        labels = labels or load_labels()
        if item_key not in labels:
            raise ValueError(f"unknown item {item_key!r}")
        items = items or load_items()
        traits = traits or load_traits()
        rng = random.Random(seed)
        s = cls(item_key, labels[item_key], traits, rng, voice=voice)
        if s.label == "refuse":
            s.names = rng.sample(traits.names, n_buyers)
            s.moods = {name: "unhappy" for name in s.names}
            for name, line in zip(s.names, phrases.refuse_lines(item_key, n_buyers, rng)):
                said = None
                if voice is not None:
                    chosen = rng.sample(list(traits.traits), rng.choice((1, 2)))
                    s.personas[name] = f"{name}, who " + " and ".join(traits.traits[t].persona for t in chosen) + "."
                    said = voice.refuse(item_key, s.personas[name], name)
                s._say(name, said or line, "model" if said else "phrases")
            return s
        if item_key not in items:
            raise ValueError(f"{item_key!r} can be sold but has no schema")
        item = items[item_key]
        s.round = Round(item, make_buyers(item, traits, n_buyers, rng))
        s.names = [b.name for b in s.round.buyers]
        s.personas = {b.name: b.persona(traits) for b in s.round.buyers}
        s.moods = {name: "idle" for name in s.names}
        if s.label == "risky":
            s._turn_away_the_wary()
        s._next_question()
        return s

    @property
    def _live(self) -> Round:
        """The round being negotiated; start() sets it for every item that is not refused."""
        return cast(Round, self.round)

    def _turn_away_the_wary(self) -> None:
        """A dangerous or black-market item: buyers with a trait that won't touch it say why and leave
        before the first question; the rest negotiate as usual (maybe nobody does)."""
        for buyer in self._live.buyers:
            trait = refuses_risky(buyer, self.traits)
            if trait is None:
                continue
            self._live.dismiss(buyer)
            self.moods[buyer.name] = "left"
            self.changes[buyer.name] = {"amount": 0.0, "reaction": "too dangerous", "liked": False}
            self._say(buyer.name, phrases.risky_refusal(trait, self.rng), "phrases")

    def answer_text(self, text: str) -> tuple[list[str], dict[str, str] | None]:
        """A typed answer: apply the option it matches, or have the asker ask again if it matches none.
        Returns the new log lines and what it was taken as ({"option", "answer", "rule"}), or None
        when the buyer asked again."""
        if self.question is None:
            raise ValueError("no question is waiting for an answer")
        typed = answers.clean(text)
        if not typed:
            raise ValueError("the answer is empty")
        asker, attr = self.question
        if getattr(self.voice, "understands", False):     # a buyer model trained to read answers
            option, rule = cast("ModelVoice", self.voice).parse(self._live.item.noun, attr, typed), "model"
        else:                                             # no such model: the rule matcher
            found = answers.match(typed, attr)
            option, rule = found.option, found.rule
        if option is not None:
            heard = {"option": option, "answer": attr.option(option).answer, "rule": rule}
            return self.answer(option, typed=typed), heard
        start = len(self.log)
        self.log.append(f"You: {typed}")
        self.sources.append("player")
        self.log.append(f"{asker.name}: {phrases.ask_again(attr.question, self.rng)}")
        self.sources.append("phrases")
        return self.log[start:], None

    def answer(self, option_id: str, typed: str | None = None) -> list[str]:
        """Apply the player's answer (an option, or text that matched it); returns the lines this
        answer added to the log."""
        if self.question is None:
            raise ValueError("no question is waiting for an answer")
        asker, attr = self.question
        option = attr.option(option_id)
        start = len(self.log)
        self._say("You", typed or option.answer, "player", heard=option.answer)
        snapshot = list(self.heard)               # what every buyer has heard when they react
        before = self._live.offers()
        reactions, bids = self._live.answer(asker, attr, option_id)
        by_name = {b.name: b for b in self._live.buyers}
        self._record_changes(before, reactions, bids, option)
        self.moods.update({r.buyer: _mood(r) for r in reactions})
        noun = self._live.item.noun
        for r in reactions:
            if r.left:
                self._say(r.buyer, phrases.leave(noun, self.rng), "engine")
            elif r.speaks:
                line = None
                if self.voice is not None:
                    buyer = by_name[r.buyer]
                    view = fmt.BuyerView(noun, self._attrs(), self.personas[r.buyer], snapshot, self._names(),
                                         tuple(buyer.weights))
                    line = self.voice.react(view, r.buyer, r.delta, _reasons(buyer, option))
                source = "model" if line else "phrases"
                if line is None:
                    tag = reason_tag(by_name[r.buyer].weights, option.signals)
                    line = phrases.react_with_reason(r.delta, noun, option.factor, tag, self.rng,
                                                     reason_topic(self._live.item, attr))
                self._say(r.buyer, line, source)
        for bid in bids:
            self._say(bid.buyer, phrases.raise_bid(round_price(bid.amount), self.rng), "engine")
        self._next_question()
        return self.log[start:]

    def scene(self, buyer: str) -> list[dict]:
        """How this buyer pictures the item at their place (design-v2 §5.4): the item and up to three
        props with boxes and reasons, using what has been answered so far. Nothing is drawn yet."""
        from . import scenes
        if self.round is None:
            raise ValueError("nobody pictures an item they won't buy")
        found = next((b for b in self.round.buyers if b.name == buyer), None)
        if found is None:
            raise ValueError(f"no buyer named {buyer}")
        props, groups = scenes.tables()
        return scenes.plan(self.round.item, groups[self.item_key], found, self.traits, props,
                           answered=self.round.answered)

    def sell(self, buyer: str) -> dict:
        if self.round is None or self.question is not None:
            raise ValueError("the round isn't finished")
        offers = self.round.offers()
        if buyer not in offers:
            raise ValueError(f"{buyer!r} has no offer")
        self.sold = {"buyer": buyer, "price": offers[buyer],
                     "typical": self.round.item.typical}
        self.moods[buyer] = "won"
        return self.sold

    # --- helpers -----------------------------------------------------------------------------

    def _attrs(self) -> list[str]:
        return list(self._live.item.attributes)

    def _names(self) -> tuple[str, ...]:
        return tuple(fmt.item_names(self.item_key, self._live.item.noun))

    def _next_question(self) -> None:
        self.question = self._live.next_question()
        if self.question:
            asker, attr = self.question
            if self.moods.get(asker.name) == "idle":     # a fresh reaction shows first
                self.moods[asker.name] = "talk"
            line = None
            if self.voice is not None:
                view = fmt.BuyerView(self._live.item.noun, self._attrs(), self.personas[asker.name], self.heard,
                                     self._names(), tuple(t for t, w in asker.weights.items() if w > 0))
                line = self.voice.ask(view, asker.name, attr.name, attr.question)
            source = "model" if line else "phrases"
            if line is None:
                line = phrases.ask_with_reason(attr.question, ask_reason_tag(asker, attr), self.rng,
                                               reason_topic(self._live.item, attr))
            self._say(asker.name, line, source)

    def _record_changes(self, before: dict[str, float], reactions, bids, option) -> None:
        after = self._live.offers()
        by_reaction = {r.buyer: r for r in reactions}
        outbid = {bid.buyer for bid in bids}
        by_name = {b.name: b for b in self._live.buyers}
        # Buyers who already left keep the reason they left; everyone else gets this answer's change.
        self.changes = {name: change for name, change in self.changes.items() if name not in before}
        for name, old in before.items():
            change: dict = {"amount": round(after.get(name, 0.0) - old, 2)}
            if (said := _reaction(by_name[name], by_reaction.get(name), option, name in outbid)) is not None:
                change["reaction"], change["liked"] = said
            self.changes[name] = change

    def _say(self, speaker: str, line: str, source: str, heard: str | None = None) -> None:
        self.log.append(f"{speaker}: {line}")
        self.sources.append(source)
        self.heard.append(f"{speaker}: {heard or line}")

    def state(self) -> dict:
        """Everything the game draws: buyers, log, the open question and its answer options."""
        if self.round is None:
            return {"item": self.item_key, "label": self.label, "finished": True,
                    "buyers": [{"name": n, "persona": "", "offer": None, "present": False, "mood": self.moods[n]}
                               for n in self.names],
                    "log": self.log, "sources": self.sources, "question": None}
        q = None
        if self.question:
            asker, attr = self.question
            q = {"asker": asker.name, "attribute": attr.name,
                 "options": [{"id": o.id, "answer": o.answer} for o in attr.options]}
        offers = self.round.offers()
        return {
            "item": self.item_key, "label": self.label, "noun": self.round.item.noun,
            "finished": self.question is None,
            "buyers": [{"name": b.name, "persona": self.personas[b.name], "offer": offers.get(b.name),
                        "present": b.present, "mood": self.moods[b.name],
                        "traits": [self.traits.traits[t].label for t in b.traits],
                        "change": self.changes.get(b.name)} for b in self.round.buyers],
            "log": self.log, "sources": self.sources, "question": q, "sold": self.sold,
            "voice": "model" if self.voice is not None else "phrases",
        }
