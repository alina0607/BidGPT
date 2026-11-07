"""A cat sale in the game, run by the buyer GPT (B plan): the player is the seller and types every answer.

Every buyer is the same model with their own prompt, built by the same code as the training data
(buyers/sales_sft.py), so the model sees at play time exactly what it was trained on. The game only decides whose
turn it is to ask; the model writes each opening and offer, each question, each reaction with its reason and new
offer, and the three things each buyer pictures at home. Buyers have no spending limit (owner, 2026-10-12): the
engine keeps every offer the model writes, counts every line it could not read, and never rewrites or replaces a
line or samples twice to pick a better one (the project's only standard is the model itself). A format-4 model
(B1) was trained with a budget on its card, so its prompts still show one; nothing holds its offers to it.

The game speaks the same protocol as for other items (docs/protocol.md): `round_state` with a question that has no
options, typed answers (`heard` echoes what was typed), `round_sell`, and `scene`.

A prompt-format-8 model (B5) also learned the seller's side, so the seller can be the model too (owner, 2026-10-28):
it answers each question from the cat's sheet (its facts, a way of talking and an asking price, `seller_sheet`) and
what has been said, and at the end decides on each final offer (accept, counter or refuse); a buyer it counters
answers once (accept or walk). With the seller played by the model the player can be one of the buyers (`player`), or
nobody, and the model plays both sides (tools/self_play.py). The seller sees what the training sales gave it: the last
twelve public lines (openings, questions, answers and reactions), never a buyer's card or story.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import stories
from .asking import asking
from .format import ANSWER_FORMATS, END, LINE_FIRST_FORMATS, SELLER_FORMATS
from .sales_sft import CARDS_FILE, TASK_TOKENS, Story, animal, buyer_prompt, card, money, seller_card, seller_prompt
from .schema import DATA_DIR

ML_DIR = DATA_DIR.parent
# newest first; B4f9 is retired (owner, 2026-10-28); the game's buyers are B6 (owner, 2026-11-06)
CHECKPOINTS = (ML_DIR / "runs" / "sft_b6" / "best.pt", ML_DIR / "runs" / "sft_b5" / "best.pt",
               ML_DIR / "runs" / "sft_b3_resume" / "best.pt", ML_DIR / "runs" / "sft_b2" / "best.pt",
               ML_DIR / "runs" / "sft_b1" / "best.pt")
SELLER_CHECKPOINTS = (ML_DIR / "runs" / "sft_b6" / "best.pt",        # prompt formats 8 and 10: the seller's side too
                      ML_DIR / "runs" / "sft_b5" / "best.pt")
TYPICAL = 75                               # a typical cat's price in the sales the model learned from
# note: prompt format 7; answer, decide and reply: format 8 (the longest written ones are 117, 64 and 51 tokens).
# open was 48 until B5: 1,923 of B5's 8,030 written openings are longer (the b3 sales', up to 83 tokens) and were cut.
MAX_NEW = {"open": 96, "ask": 48, "react": 80, "imagine": 64, "note": 32, "answer": 120, "decide": 72, "reply": 56}
HEARD = 12                                 # public lines the seller sees, as in the training sales (_Walk.heard)
SELLER_ACTIONS = ("accept", "counter", "refuse")
# A line-first scene (format 9) ends with its list, so a cut would lose the list: room for the longest written scene
# (85 tokens in B4's sales; 126 of 3,740 were longer than 64). Older formats keep 64, so their evaluations do not move.
LINE_FIRST_IMAGINE = 96
DOLLARS = re.compile(r"\$\s?(\d+)")
THING = re.compile(r"<\|thing:([^|]+)\|>")          # a format-6 scene thing token


def max_new(task: str, prompt_format: int) -> int:
    """The most tokens the model may write for a task in a prompt format."""
    return LINE_FIRST_IMAGINE if task == "imagine" and prompt_format in LINE_FIRST_FORMATS else MAX_NEW[task]


class BuyerGPT:
    """The fine-tuned model and its tokenizer; `say` completes one buyer's prompt."""

    def __init__(self, path: Path, device: str = "cpu", temperature: float = 0.7, top_k: int = 50):
        import torch
        from ..llm.evaluate import load_model
        self.torch = torch
        self.model, self.tok = load_model(path, torch.device(device))
        self.prompt_format = self.model.prompt_format          # 4: B1, budget on the card; 5: no budget
        self.name = "buyer_gpt_" + path.parent.name.removeprefix("sft_")     # runs/sft_b2/best.pt -> buyer_gpt_b2
        self.device, self.temperature, self.top_k = device, temperature, top_k

    def say(self, prompt: str, task: str, seed: int, temperature: float | None = None, top_k: int | None = None,
            top_p: float | None = None) -> str:
        """The model's line for this prompt, sampled with the given seed, without the prompt and <|end|>. Without
        `temperature` it picks words the game's way (the model's temperature and top-k); with it, `top_k` and `top_p`
        are taken as given (None: no cut), for comparing ways to pick (variants.py)."""
        torch = self.torch
        torch.manual_seed(seed)          # same seed, same line: a whole sale can be replayed exactly
        ids = torch.tensor([self.tok.encode(prompt)], device=self.device)
        if temperature is None:
            temperature, top_k = self.temperature, self.top_k
        with torch.no_grad():
            out = self.model.generate(ids, max_new(task, self.prompt_format), temperature=temperature,
                                      top_k=top_k, stop_id=self.tok.eos_id, top_p=top_p)
        # keep only the new tokens (the prompt is echoed back by generate)
        return self.tok.decode(out[0, ids.size(1):].tolist()).replace(END, "").strip()


def load_default(device: str = "cpu") -> BuyerGPT | None:
    """The newest buyer GPT that has a checkpoint on disk, or None (the game then uses the phrase pools)."""
    path = next((p for p in CHECKPOINTS if p.exists()), None)
    return BuyerGPT(path, device) if path else None


def load_named(name: str, device: str = "cpu") -> BuyerGPT | None:
    """runs/sft_<name>/best.pt ("b5"), or None when there is no such checkpoint."""
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError("a model name is letters, digits and _ (runs/sft_<name>/best.pt)")
    path = ML_DIR / "runs" / f"sft_{name}" / "best.pt"
    return BuyerGPT(path, device) if path.exists() else None


def load_seller(device: str = "cpu") -> BuyerGPT | None:
    """The newest model that learned the seller's side too (prompt format 8 or 10), or None."""
    path = next((p for p in SELLER_CHECKPOINTS if p.exists()), None)
    return BuyerGPT(path, device) if path else None


def seller_sheet(seed: int) -> dict[str, Any]:
    """The cat for a model seller: the same sheet a written sale on this seed has (stories.sale_for_seed), with the
    asking price set from the cat's facts alone (asking.asking), as for the written sales' decisions."""
    sale = stories.sale_for_seed(seed)
    return {**sale, "asking": asking(sale)}


def load_cards(path: Path = CARDS_FILE) -> dict[str, list[dict[str, Any]]]:
    """Persona → the buyer cards for that persona (what they care about; a budget, shown to format-4 models only).
    A card is the buyer's setup, like the persona itself; the model writes everything the buyer says and offers."""
    return json.loads(path.read_text())


def _amount(text: str) -> int | None:
    """"$55" → 55, "out" → 0 (withdrawn), anything else → None."""
    if text == "out":
        return 0
    m = DOLLARS.fullmatch(text)
    return int(m[1]) if m else None


def _parse_open(parts: list[str]) -> dict[str, Any] | None:
    """"greeting | $offer" with an offer above 0."""
    offer = _amount(parts[-1]) if len(parts) >= 2 else None
    return {"says": " | ".join(parts[:-1]), "offer": offer} if offer else None


def _parse_react(parts: list[str]) -> dict[str, Any] | None:
    """"sign | what they say | $offer" (or "out")."""
    if len(parts) < 3 or parts[0] not in stories.SIGNS:
        return None
    offer = _amount(parts[-1])
    return None if offer is None else {"sign": parts[0], "says": " | ".join(parts[1:-1]), "offer": offer}


def _parse_imagine(parts: list[str], line_first: bool = False) -> dict[str, Any] | None:
    """"thing, thing, thing | what they say", or from format 6 "<|thing:a|><|thing:b|><|thing:c|> | what they say",
    or in format 9 (line_first) "what they say | thing, thing, thing"."""
    if len(parts) < 2:
        return None
    listed, says = (parts[-1], " | ".join(parts[:-1])) if line_first else (parts[0], " | ".join(parts[1:]))
    things = THING.findall(listed) or [t.strip() for t in listed.split(",") if t.strip()]
    return {"things": things, "says": says} if things else None


def _parse_decide(parts: list[str]) -> dict[str, Any] | None:
    """"accept | - | what the seller says", "counter | $price | ..." or "refuse | - | ..." (prompt format 8)."""
    if len(parts) < 3 or parts[0] not in SELLER_ACTIONS:
        return None
    price = None if parts[1] == "-" else _amount(parts[1])
    if (parts[0] == "counter") != bool(price) or (price is None and parts[1] != "-"):
        return None             # only a counter names a price, and it must name one
    return {"action": parts[0], "price": price, "says": " | ".join(parts[2:])}


def _parse_reply(parts: list[str]) -> dict[str, Any] | None:
    """A buyer's answer to a counter-offer: "accept | what they say" or "walk | ..." (prompt format 8)."""
    if len(parts) < 2 or parts[0] not in ("accept", "walk"):
        return None
    return {"action": parts[0], "says": " | ".join(parts[1:])}


def parse(task: str, text: str, prompt_format: int = 5) -> dict[str, Any] | None:
    """A model line split into its fields, or None when it does not have the trained shape."""
    if task == "imagine":
        return _parse_imagine([p.strip() for p in text.split(" | ")], prompt_format in LINE_FIRST_FORMATS)
    if task == "ask":
        return {"says": text} if text else None
    if task in ("note", "answer"):              # format 7: the note of an answer; format 8: the seller's answer
        return {"says": text} if text and "|" not in text else None
    parts = [p.strip() for p in text.split(" | ")]
    return {"open": _parse_open, "react": _parse_react, "decide": _parse_decide, "reply": _parse_reply}[task](parts)


def decide_problem(row: dict[str, Any], offer: int, ask: int) -> str | None:
    """The written decisions' rules (data/written_sales/seller_decide/check_decide.py) for one model decision: an
    accept is at most $10 under the asking price, a counter is a multiple of $5 above the offer and at most the asking
    price. Counted and shown, never enforced: the engine keeps what the model decided."""
    if row["action"] == "accept" and offer < ask - 10:
        return f"accepts ${offer}, more than $10 under the asking ${ask}"
    if row["action"] == "counter" and (row["price"] % 5 or not offer < row["price"] <= ask):
        return f"counter ${row['price']} must be a multiple of $5 above ${offer} and at most ${ask}"
    return None


def _typed(text: str, what: str) -> str:
    """What the player typed, on one line and cut to the game's text box (120 characters)."""
    text = " ".join(str(text).split())[:120]
    if not text:
        raise ValueError(f"empty {what}")
    return text


@dataclass
class CatMarket:
    """One cat sale in the game: the buyers' cards, stories and offers, the log the game shows, and the round state.
    Same interface as rounds.RoundSession, so server.py drives both the same way."""
    gpt: BuyerGPT
    rng: random.Random
    cards: dict[str, dict[str, Any]] = field(default_factory=dict)
    stories: dict[str, Story] = field(default_factory=dict)
    offers: dict[str, int] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    moods: dict[str, str] = field(default_factory=dict)
    changes: dict[str, dict[str, int]] = field(default_factory=dict)    # this round's offer change per buyer
    question: tuple[str, str] | None = None     # (asker, question) waiting for the player's answer
    sold: dict[str, Any] | None = None
    pictured: dict[str, list[dict[str, Any]]] = field(default_factory=dict)    # each buyer's scene, written once
    stats: dict[str, int] = field(default_factory=lambda: {"lines": 0, "unreadable": 0})
    item_key: str = "cat"
    label: str = "buy"
    # what the seller has heard, in the training sales' words: openings, questions, answers, reactions (_Walk.heard)
    public: list[str] = field(default_factory=list)
    seller: dict[str, Any] | None = None        # the cat's sheet when the model is the seller (seller_sheet)
    player: str | None = None                   # the player's name when the player is one of the buyers
    waiting: str | None = None                  # what the player-buyer must do next: open, ask, react or reply
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)   # the model seller's decision per buyer

    # --- lifecycle ---------------------------------------------------------------------------

    @classmethod
    def start(cls, gpt: BuyerGPT, n_buyers: int = 4, seed: int | None = None,
              cards: dict[str, list[dict[str, Any]]] | None = None, seller: dict[str, Any] | None = None,
              player: str | None = None, ask: bool = True) -> CatMarket:
        """n buyers with different names and personas, each with a card from the pool; every buyer opens, then the
        first one asks. The same seed gives the same sale. With `seller` (a seller_sheet) the model is the seller and
        the sale runs until the player is needed; `player` names the player when they join as the last buyer.
        `ask=False` stops after the openings (paired.PairedSale puts one question to two models' sales)."""
        if (seller is not None or player) and gpt.prompt_format not in SELLER_FORMATS:
            raise ValueError(f"a prompt-format-{gpt.prompt_format} model never learned the seller's side")
        if player and seller is None:
            raise ValueError("a player who buys needs a model seller")
        rng = random.Random(seed)
        # cards are the setup (who the buyer is), like a level layout; everything they say comes from the model
        pool = cards or load_cards()
        if gpt.prompt_format == 4:      # B1 learned with a budget on every card: only the cards that have one
            pool = {p: [c for c in cs if "budget" in c] for p, cs in pool.items()}
        m = cls(gpt, rng, seller=seller, player=player)
        personas = [p for p in stories.PERSONAS if pool.get(p)]
        names = [n for n in stories.NAMES if n != player]
        for name, persona in zip(rng.sample(names, n_buyers), rng.sample(personas, n_buyers), strict=True):
            m.cards[name] = {"persona": persona, **rng.choice(pool[persona])}
            m.stories[name] = m._story(name)
            m.moods[name] = "idle"
        for name in list(m.cards):
            m._open(name)
        if player:
            m._seat_player()
        elif ask:
            m._next_question()
            m._advance()
        return m

    def _seat_player(self) -> None:
        """The player-buyer sits last, after the model buyers opened: no card, nothing the model writes for them."""
        self.cards[self.player] = {"persona": "is you", "cares_about": []}
        self.stories[self.player] = self._story(self.player)
        self.offers[self.player], self.moods[self.player], self.waiting = 0, "idle", "open"

    def _story(self, name: str) -> Story:
        """A buyer's story kept the way this model's training sales kept it (format 10: older answers too)."""
        return Story(name, answers=self.gpt.prompt_format in ANSWER_FORMATS)

    def _say(self, name: str, task: str, offer: int | None, story: str) -> dict[str, Any] | None:
        """One line from the model for this buyer and task, parsed; None (logged and counted) if it cannot be read."""
        prompt = buyer_prompt(card(name, self.cards[name], offer, self.gpt.prompt_format), story, TASK_TOKENS[task],
                              animal(self.gpt.prompt_format))
        text = self.gpt.say(prompt, task, self.rng.randrange(2**31))
        self.stats["lines"] += 1
        row = parse(task, text, self.gpt.prompt_format)
        if row is None:
            # no retry and no fallback line: an unreadable line is shown as it is and counted, so a bad model looks bad
            self.stats["unreadable"] += 1
            self._log(name, text or "...", "model_unreadable")
            return None
        return row

    def _log(self, name: str, text: str, source: str) -> None:
        """A line for the game's log, with where it came from (model, model_unreadable, player)."""
        self.log.append(f"{name}: {text}")
        self.sources.append(source)

    def _open(self, name: str) -> None:
        """The buyer's greeting and first offer. One the engine cannot read leaves the buyer out of the sale (counted
        in `stats`): the engine never makes up an offer for them."""
        row = self._say(name, "open", None, "(nothing yet)")
        self.offers[name] = row["offer"] if row else 0
        if row is None or row["offer"] == 0:
            self.moods[name] = "left"
            return
        self._log(name, row["says"], "model")
        self.stories[name].opening = f"Your opening ({money(row['offer'])}): {row['says']}"
        self.public.append(f"{name} ({money(row['offer'])}): {row['says']}")

    def present(self) -> list[str]:
        """Buyers still bidding (offer above 0), in seating order."""
        return [n for n in self.cards if self.offers[n] > 0]

    def _next_question(self) -> None:
        """The next buyer who has not asked yet writes their question; none left: the questions are over."""
        self.question = None
        # the only thing the engine decides: whose turn it is (the next buyer still here who hasn't asked yet)
        for name in self.present():
            if name in self.asked:
                continue
            self.asked.append(name)
            if name == self.player:     # the player-buyer types their own question (player_ask)
                self.waiting = "ask"
                return
            row = self._say(name, "ask", self.offers[name], self.stories[name].text())
            if row:
                self.put_question(name, row["says"])
                return

    def put_question(self, name: str, text: str, source: str = "model") -> None:
        """`name` asks `text`: the question that waits for the player (from this sale's model, or `source` names
        the other model of a paired sale, which wrote it in this buyer's name)."""
        if name not in self.asked:
            self.asked.append(name)
        self._log(name, text, source)
        self.public.append(f"{name} asked: {text}")
        self.question = (name, text)
        self.moods[name] = "talk"

    # --- the player ----------------------------------------------------------------------------

    def answer(self, option: str) -> list[str]:
        """Picking an option does not apply here: these buyers ask in their own words, so the player types."""
        raise ValueError("this buyer asks in their own words; type your answer")

    def answer_text(self, text: str) -> tuple[list[str], dict[str, Any]]:
        """The player's typed answer: every buyer still here reacts, then the next buyer asks."""
        if self.seller is not None:
            raise ValueError("the model is the seller in this sale")
        if self.question is None:
            raise ValueError("no question is waiting")
        text = _typed(text, "answer")
        start = len(self.log)
        self._answered(text, "You", "player")
        self._next_question()
        return self.log[start:], {"option": None, "answer": text, "rule": "model"}

    def _answered(self, text: str, who: str | None, source: str) -> None:
        """The seller's answer to the waiting question, heard by everyone; every model buyer still here reacts.
        `who` is the name the game's log shows it under (None: it is in the log already)."""
        asker, q = self.question
        if who:
            self.log.append(f"{who}: {text}")
            self.sources.append(source)
        self.public.append(f"You: {text}")          # the seller's own view, as in the training sales
        # the seller's words go into every buyer's story exactly like the seller lines in the training sales
        for s in self.stories.values():
            s.asked(asker, q)
            s.answered(text)
        self.changes = {}
        for name in self.present():
            if name != self.player:
                self._react(name)

    def _react(self, name: str) -> None:
        """One buyer reacts to the answer just typed; their offer and mood follow what the model wrote."""
        old = self.offers[name]
        row = self._say(name, "react", old, self.stories[name].text(live=True))
        if row is None:
            self.moods[name] = "idle"
            return
        self._log(name, row["says"], "model")
        self.stories[name].rounds[-1].append(f"You ({money(old)} -> {money(row['offer'])}): {row['says']}")
        self.public.append(f"{name} ({money(old)} -> {money(row['offer'])}): {row['says']}")
        self.offers[name] = row["offer"]
        self.changes[name] = {"amount": row["offer"] - old}
        self.moods[name] = ("left" if row["offer"] == 0 else "happy" if row["offer"] > old
                            else "unhappy" if row["offer"] < old else "idle")

    def sell(self, buyer: str) -> dict[str, Any]:
        """Sell to a buyer still bidding, once the questions are over, at their current offer."""
        if self.seller is not None:
            raise ValueError("the model is the seller in this sale; it decides on every offer itself")
        if self.question is not None:
            raise ValueError("the round isn't finished")
        if buyer not in self.present():
            raise ValueError(f"{buyer!r} has no offer")
        self.sold = {"buyer": buyer, "price": self.offers[buyer], "typical": TYPICAL}
        self.moods[buyer] = "won"
        return self.sold

    def scene(self, buyer: str) -> list[dict[str, Any]]:
        """The cat and the three things this buyer pictures with it (written by the model), in the scene layout.
        Where a thing goes is display: its zone in props.json, or beside the cat."""
        from . import scenes
        if buyer not in self.present():
            raise ValueError(f"no buyer named {buyer} is still here")
        if buyer in self.pictured:
            return self.pictured[buyer]
        row = self._say(buyer, "imagine", self.offers[buyer], self.stories[buyer].text())
        props, _ = scenes.tables()
        objects = [{"label": "cat", "role": "item", "zone": "item", "box": list(scenes.ITEM_BOX), "share": 1.0,
                    "reason": ""}]
        used: set[str] = set()
        for thing in (row["things"] if row else [])[:scenes.MAX_PROPS]:
            prop = props.get(thing)
            zone = prop.zone if prop and prop.zone not in used else next(
                z for z in scenes.ZONES if z not in used)
            used.add(zone)
            objects.append({"label": thing, "role": "prop", "zone": zone, "box": list(scenes.ZONES[zone]),
                            "share": scenes.SIZES[prop.size if prop else "medium"], "reason": row["says"]})
        self.pictured[buyer] = objects
        return objects

    # --- the model as the seller (prompt format 8) ---------------------------------------------

    def heard(self) -> str:
        """The seller's view: the last HEARD public lines of the sale."""
        return "\n".join(self.public[-HEARD:]) or "(nothing yet)"

    def _seller_say(self, task: str, story: str) -> tuple[dict[str, Any] | None, str]:
        """One seller line from the model, parsed (None when unreadable: shown as it is and counted), and the text."""
        prompt = seller_prompt(seller_card(self.seller), story, TASK_TOKENS[task], animal(self.gpt.prompt_format))
        text = self.gpt.say(prompt, task, self.rng.randrange(2**31))
        self.stats["lines"] += 1
        row = parse(task, text, self.gpt.prompt_format)
        if row is None:
            self.stats["unreadable"] += 1
            self._log("Seller", text or "...", "model_unreadable")
        return row, text

    def _advance(self) -> None:
        """With a model seller: answer the waiting question, let every buyer react, and go on to the next question and
        at the end to the decisions, until the player-buyer has to act or the sale is over."""
        if self.seller is None:
            return
        while self.waiting is None and self.sold is None:
            if self.question is not None:
                self._seller_answers()
            elif not self.decisions and self.present():
                self._decide_all()
            else:
                self._close()

    def _seller_answers(self) -> None:
        """The model seller answers the waiting question and every model buyer reacts; then the player-buyer reacts
        (waiting) or the next buyer asks."""
        row, text = self._seller_say("answer", self.heard())
        # an unreadable answer is still what the seller said (already shown and counted): the buyers hear it
        self._answered(row["says"] if row else (text or "..."), "Seller" if row else None, "model")
        if self.player in self.present():
            self.waiting = "react"
        else:
            self._next_question()

    def _decide_all(self) -> None:
        """The seller's decision on every final offer (accept, counter or refuse), each from the same view of the sale
        with that buyer's final offer, as in the training sales; a model buyer answers a counter-offer at once."""
        heard, ask = self.heard(), self.seller["asking"]
        for name in self.present():
            offer = self.offers[name]
            row, _ = self._seller_say("decide", heard + f"\n{name}'s final offer: {money(offer)}")
            if row is None:
                self.decisions[name] = {"offer": offer, "action": None}      # unreadable: no deal with this buyer
                continue
            self.decisions[name] = {"offer": offer, **row, "problem": decide_problem(row, offer, ask)}
            self._log("Seller", f"({name}, {row['action']}"
                      + (f" at {money(row['price'])}" if row["price"] else "") + f") {row['says']}", "model")
            if row["action"] == "counter" and name != self.player:
                story = self.stories[name].text() + f"\nSeller ({money(row['price'])}): {row['says']}"
                reply = self._say(name, "reply", offer, story)
                self.decisions[name]["reply"] = reply
                if reply:
                    self._log(name, reply["says"], "model")
        if self.player in self.decisions and self.decisions[self.player]["action"] == "counter":
            self.waiting = "reply"

    def _close(self) -> None:
        """The deals the decisions made (an accepted offer, or a counter-offer the buyer accepted); the cat goes to the
        highest price, the earlier seat on a tie. No deal: the cat stays (sold names no buyer)."""
        deals = {}
        for name, d in self.decisions.items():
            if d["action"] == "accept":
                deals[name] = d["offer"]
            elif d["action"] == "counter" and (d.get("reply") or {}).get("action") == "accept":
                deals[name] = d["price"]
        buyer = max(deals, key=lambda n: deals[n], default=None)     # max keeps the first of equal prices
        self.sold = {"buyer": buyer, "price": deals.get(buyer), "typical": TYPICAL, "asking": self.seller["asking"],
                     "deals": deals}
        if buyer:
            self.moods[buyer] = "won"

    # --- the player as one of the buyers ---------------------------------------------------------

    def _expect(self, step: str) -> None:
        if self.waiting != step:
            raise ValueError(f"it is not the time to {step}" + (f"; waiting for: {self.waiting}" if self.waiting else ""))

    def player_open(self, says: str, offer: int) -> list[str]:
        """The player-buyer's greeting and first offer; then the questions start."""
        self._expect("open")
        says = _typed(says, "greeting")
        if not isinstance(offer, int) or offer <= 0:
            raise ValueError("the first offer must be a whole number of dollars above 0")
        start = len(self.log)
        self._log(self.player, f"{says} ({money(offer)})", "player")
        self.public.append(f"{self.player} ({money(offer)}): {says}")
        self.offers[self.player], self.waiting = offer, None
        self._next_question()
        self._advance()
        return self.log[start:]

    def player_ask(self, text: str) -> list[str]:
        """The player-buyer's question; the seller answers and every buyer reacts."""
        self._expect("ask")
        text = _typed(text, "question")
        start = len(self.log)
        self._log(self.player, text, "player")
        self.public.append(f"{self.player} asked: {text}")
        self.question, self.waiting = (self.player, text), None
        self._advance()
        return self.log[start:]

    def player_react(self, says: str, offer: int) -> list[str]:
        """The player-buyer's reaction to the answer: what they say and their new offer (0: they walk away)."""
        self._expect("react")
        says = _typed(says, "reaction")
        if not isinstance(offer, int) or offer < 0:
            raise ValueError("the offer must be a whole number of dollars (0 to walk away)")
        start, old = len(self.log), self.offers[self.player]
        self._log(self.player, f"{says} ({money(old)} -> {money(offer)})", "player")
        self.public.append(f"{self.player} ({money(old)} -> {money(offer)}): {says}")
        self.offers[self.player], self.waiting = offer, None
        self.changes[self.player] = {"amount": offer - old}
        self._next_question()
        self._advance()
        return self.log[start:]

    def player_reply(self, accept: bool, says: str) -> list[str]:
        """The player-buyer's answer to the seller's counter-offer; then the sale closes."""
        self._expect("reply")
        says = _typed(says, "reply")
        start = len(self.log)
        self._log(self.player, says, "player")
        self.decisions[self.player]["reply"] = {"action": "accept" if accept else "walk", "says": says}
        self.waiting = None
        self._advance()
        return self.log[start:]

    # --- what the game draws -------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """The round_state the game draws (docs/protocol.md); `voice` names the model that wrote the lines, and each
        buyer's `model` the one that plays them."""
        q = None if self.question is None else {"asker": self.question[0], "attribute": "", "options": []}
        state = {
            "item": "cat", "label": "buy", "noun": "cat",
            # with a model seller the sale ends with its decisions, not with the last question
            "finished": self.sold is not None if self.seller is not None else self.question is None,
            "buyers": [{"name": n, "persona": f"{n}, who {c['persona']}.",
                        "offer": self.offers[n] if self.offers[n] > 0 else None, "present": self.offers[n] > 0,
                        "mood": self.moods[n], "traits": c["cares_about"], "change": self.changes.get(n),
                        "model": None if n == self.player else self.gpt.name}
                       for n, c in self.cards.items()],
            "log": self.log, "sources": self.sources, "question": q, "sold": self.sold, "voice": self.gpt.name,
            "model_stats": dict(self.stats)}
        if self.seller is not None:     # the price tag on the stall, the player's turn, and the seller's decisions
            state |= {"asking": self.seller["asking"], "player": self.player, "waiting": self.waiting,
                      "decisions": self.decisions}
        return state
