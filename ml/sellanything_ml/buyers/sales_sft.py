"""Buyer GPT, B plan: the official cat sales as supervised fine-tuning examples, one per decision a buyer makes
(devlog 2026-10-11 §3; owner decisions in HANDOFF.md).

    python -m sellanything_ml.buyers.sales_sft            # → data/processed/buyers_b2/{train,val}.jsonl, meta.json

Every example is what ONE buyer knows at that moment and what that buyer did next:

    open     their card (no offer yet)                           → greeting | $offer
    ask      card + their own story so far                       → the question they ask
    react    card + story + this round's question and answer     → sign | what they say | $offer (or "out")
    imagine  card + story after the last round                   → three drawable things | what they say
                                                                   (format 9: what they say | the three things)

The story is the buyer's own opening, the earlier questions (only the questions, so they do not ask one again;
owner's idea, 2026-10-11) and the last K rounds in full: who asked what, the seller's answer and their own
reaction. Buyers never see other buyers' reactions (owner, 2026-10-11). The card carries the offer the buyer holds,
so facts from older rounds reach the prompt through it.

Nothing in a prompt gives the answer away: the engine only decides whose turn it is to ask, and the model writes
the question, the reaction with its reason, the new offer and the scene (memory: buyer-model-decides).

Sales are split by seed: about one in ten (by a hash of the seed, so adding sales never moves one) is validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .format import (ANIMAL_TOKENS, ANSWER, ASK, BUYER_TOKENS, DECIDE, END, IMAGINE, ITEM, LINE_FIRST_FORMATS, LOG,
                     NOTE, NOTE_FORMATS, OPEN, PERSONA, REACT, REPLY, SELLER, SELLER_FORMATS, THING_TOKENS,
                     TOKEN_FORMATS)
from .schema import DATA_DIR
from .stories import mention_order

PROMPT_FORMAT = 5          # 4 (buyer GPT B1): the card shows a budget; 5: no budget (owner, 2026-10-12: buyers have
                           # no spending limit, their raises and cuts only have to be reasonable); 6: as 5, with the
                           # animal, the persona and the scene's three things as tokens (format.py); 7: as 5, and
                           # older rounds keep a short note of their answer, which the buyer writes as a task;
                           # 9: as 5, the scene written line first (format.LINE_FIRST_FORMATS)
K = 2                                      # full rounds in the story (fits 768 tokens: devlog 2026-10-11 §3)
WRITTEN_SALES = DATA_DIR.parents[1] / "data" / "written_sales"     # sales written to the prompts, not by Qwen
SOURCES_B1 = (DATA_DIR / "processed" / "buyer_sales_v9_official.jsonl",    # Qwen v9, every line reviewed and corrected
              WRITTEN_SALES / "sales.jsonl")                              # seeds 2048-2099
# B2 adds the sales written to the format-5 prompts with the quality gates (seeds 2200-, no budget, a note per answer).
# sales_b2.jsonl keeps growing; B2 and its format-6 twin were trained on its first 150 sales (seeds 2200-2349), kept
# as they were in sales_b2_150.jsonl so both models' data can be rebuilt bit for bit.
SOURCES_B2 = SOURCES_B1 + (WRITTEN_SALES / "sales_b2_150.jsonl",)
# The next data version: every format-5 sale written so far, plus the 100 sales Muse wrote to the same prompts (seeds
# 2100-2199), rewritten to the same rules and passing the same checks (owner, 2026-10-13).
EXTERNAL_SALES = WRITTEN_SALES.parent / "external_sales"
SOURCES_B3 = SOURCES_B1 + (WRITTEN_SALES / "sales_b2.jsonl", EXTERNAL_SALES / "muse" / "sales_v3.jsonl")
# B4 (2026-10-13): B3 plus the sales written to the scene quotas (2500-2599) and the first human-seller trial
SOURCES_B4 = SOURCES_B3 + (WRITTEN_SALES / "sales_b3.jsonl",)
# B5 (2026-10-14): the same sales rebuilt with the seller's decisions and asking price (write_sales.py), for prompt
# format 8's seller tasks; Qwen v9 has no decisions, so its buyers train as before and it adds no seller examples
SOURCES_B5 = (SOURCES_B1[0], WRITTEN_SALES / "sales_decide.jsonl", WRITTEN_SALES / "sales_b2_decide.jsonl",
              EXTERNAL_SALES / "muse" / "sales_v3_decide.jsonl", WRITTEN_SALES / "sales_b3_decide.jsonl")
SOURCES = SOURCES_B2
OUT_DIR = DATA_DIR / "processed" / "buyers_b2"
CARDS_FILE = DATA_DIR / "cat_buyer_cards.json"   # persona → the cards written for it (the game's buyer setup)
VAL_EVERY = 10                             # one sale in VAL_EVERY (by seed hash) is validation
Example = dict[str, Any]


def is_val(seed: int) -> bool:
    """Whether the sale on this seed is in the validation split (about one in VAL_EVERY)."""
    # hash the seed instead of shuffling: a sale lands on the same side every time, even after more sales are added,
    # so the validation set I compare versions on never quietly changes
    return int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) % VAL_EVERY == 0


def money(offer: int) -> str:
    """An offer as the model reads and writes it: "$70", or "out" for 0."""
    # an offer of 0 means the buyer walked away, and "out" reads better than "$0" for the model and for me
    return f"${offer}" if offer else "out"


def card(name: str, c: dict[str, Any], offer: int | None, prompt_format: int = PROMPT_FORMAT) -> str:
    """Who the buyer is; with the offer they hold, except before their opening. Format 4 also shows the budget;
    format 6 gives the persona as its token."""
    budget = f" | budget ${c['budget']}" if prompt_format == 4 else ""
    now = f" | offer now ${offer}" if offer is not None else ""
    who = f", {BUYER_TOKENS[c['persona']]}" if prompt_format in TOKEN_FORMATS else f", who {c['persona']}"
    return f"{name}{who} | cares about: {', '.join(c['cares_about'])}{budget}{now}"


def animal(prompt_format: int = PROMPT_FORMAT, kind: str = "cat") -> str:
    """What is for sale, as the prompt names it: the word, or from format 6 the animal's token."""
    return ANIMAL_TOKENS[kind] if prompt_format in TOKEN_FORMATS else kind


def scene_target(e: dict[str, Any], prompt_format: int = PROMPT_FORMAT) -> str:
    """A scene as the model writes it: "things | line", or in format 9 "line | things" with the things in the order
    the line names them."""
    if prompt_format in LINE_FIRST_FORMATS:
        return f"{e['says']} | {', '.join(mention_order(e['things'], e['says']))}"
    return f"{scene_things(e['things'], prompt_format)} | {e['says']}"


def scene_things(things: list[str], prompt_format: int = PROMPT_FORMAT) -> str:
    """A scene's three things as the model writes them: "a, b, c", or from format 6 their three tokens."""
    return "".join(THING_TOKENS[t] for t in things) if prompt_format in TOKEN_FORMATS else ", ".join(things)


class Story:
    """One buyer's own view of a sale as it goes: their opening, every round's question and answer, and their own
    reactions."""

    def __init__(self, name: str, notes: bool = False):
        self.name = name
        self.opening = ""
        self.rounds: list[list[str]] = []          # per round: question line, answer line, own reaction line
        self.questions: list[str] = []
        self.notes: list[str] = []                 # per round: the note of its answer ("" when there is none)
        self.with_notes = notes                    # prompt format 7: older rounds show their notes

    def asked(self, asker: str, question: str) -> None:
        """A new round starts with this question ("You asked" when it is this buyer's own)."""
        who = "You" if asker == self.name else asker
        self.rounds.append([f"{who} asked: {question}"])
        self.questions.append(question)
        self.notes.append("")

    def text(self, live: bool = False, k: int = K) -> str:
        """The story so far: opening, the questions of older rounds, the last k finished rounds and, if `live`,
        the round under way (its question and answer, before this buyer reacts)."""
        done = self.rounds[:-1] if live else self.rounds
        lines = [self.opening] if self.opening else []
        # rounds older than k only keep their questions: the buyer still knows what was asked (so it won't ask it
        # again), but the answers are gone. This is the "forgets after two rounds" problem B1 has (devlog 10-12 §6)
        older = self.questions[:max(0, len(done) - k)]
        if older and self.with_notes:             # format 7: the answers come back as the buyer's notes
            lines.append("Asked earlier: " + " / ".join(f"{q} -> {n}" if n else q
                                                         for q, n in zip(older, self.notes, strict=False)))
        elif older:
            lines.append("Asked earlier: " + " / ".join(older))
        for r in done[len(done) - min(k, len(done)):] + (self.rounds[-1:] if live else []):
            lines += r
        return "\n".join(lines) or "(nothing yet)"


def buyer_prompt(card_line: str, story: str, task_token: str, item: str = "cat") -> str:
    """What one buyer sees: the item, their card, their story so far, and the token of the task to do next."""
    return f"{ITEM}{item}{END}{PERSONA}{card_line}{END}{LOG}{story}{END}{task_token}"


def seller_card(sale: dict[str, Any]) -> str:
    """Who the seller is (prompt format 8): the cat's own facts, their way of talking and their asking price."""
    cat = sale["cat"]
    facts = [cat["age"], cat["breed"], cat["home"], cat["fixed"], cat["health"], cat["virus"], *cat["company"],
             cat["temperament"], *cat["quirks"], *cat.get("extra", []),
             *cat.get("life", [])]
    asking = f" | asking {money(sale['asking'])}" if sale.get("asking") else ""
    return f"facts: {'; '.join(facts)} | style: {sale.get('seller_style', '')}{asking}"


def seller_prompt(card_line: str, story: str, task_token: str, item: str = "cat") -> str:
    """What the seller sees: the item, their card, what has been said in the sale, and the task token."""
    return f"{ITEM}{item}{END}{SELLER}{card_line}{END}{LOG}{story}{END}{task_token}"


class _Walk:
    """Goes through one sale's events, keeping each buyer's story and offer, and collects the examples."""

    def __init__(self, sale: dict[str, Any], prompt_format: int = PROMPT_FORMAT):
        self.cards, self.seed, self.prompt_format = sale["cards"], sale["seed"], prompt_format
        self.stories = {n: Story(n, prompt_format in NOTE_FORMATS) for n in self.cards}
        self.offers: dict[str, int] = {}
        self.out: list[Example] = []
        self.current: tuple[str, str] | None = None          # (asker, question) of the round under way
        self.meta = {"seed": self.seed, "author": sale.get("author", "qwen_v9_edited")}
        self.seller = prompt_format in SELLER_FORMATS                  # format 8: the seller's examples too
        self.seller_card = seller_card(sale["sale"]) if self.seller else ""
        self.seller_log: list[str] = []      # everything said in the open: openings, questions, answers, reactions

    def add_seller(self, task: str, story: str, target: str, name: str = "seller") -> None:
        """One seller example (format 8): the seller's prompt for the task and what the seller did."""
        prompt = seller_prompt(self.seller_card, story, TASK_TOKENS[task], animal(self.prompt_format))
        self.out.append({"task": task, "name": name, "prompt": prompt, "target": target + END, **self.meta})

    def heard(self, k: int = 12) -> str:
        """The seller's view: the last k public lines of the sale."""
        return "\n".join(self.seller_log[-k:]) or "(nothing yet)"

    def add(self, task: str, name: str, offer: int | None, story: str, target: str) -> None:
        """One example: this buyer's prompt for the task and what they did (ending <|end|>)."""
        prompt = buyer_prompt(card(name, self.cards[name], offer, self.prompt_format), story, TASK_TOKENS[task],
                              animal(self.prompt_format))
        self.out.append({"task": task, "name": name, "prompt": prompt, "target": target + END, **self.meta})

    def open(self, e: dict[str, Any]) -> None:
        """The greeting and first offer, written from the card alone."""
        name = e["name"]
        self.add("open", name, None, "(nothing yet)", f"{e['says']} | {money(e['offer'])}")
        self.offers[name] = e["offer"]
        self.stories[name].opening = f"Your opening ({money(e['offer'])}): {e['says']}"
        self.seller_log.append(f"{name} ({money(e['offer'])}): {e['says']}")

    def ask(self, e: dict[str, Any]) -> None:
        """The question this buyer asks, from their story so far; it starts the round under way."""
        # the offer comes from the reactions: v9's ask events kept the offer from before the hand corrections
        # (293 of 451 differ; the reaction chain itself has no break)
        name = e["name"]
        self.add("ask", name, self.offers[name], self.stories[name].text(), e["says"])
        self.current = (name, e["says"])
        self.seller_log.append(f"{name} asked: {e['says']}")

    def answer(self, e: dict[str, Any]) -> None:
        """The seller's answer goes into every buyer's story with the question (no example: the player is the
        seller)."""
        # everyone hears the question and the answer, but each buyer only ever sees their own reactions
        if self.current is None or self.current[0] != e["to"]:
            raise ValueError(f"sale {self.seed}: an answer to {e['to']} with no question from them")
        if self.seller:                     # format 8: the seller answers from their sheet and what was said
            self.add_seller("answer", self.heard(), e["says"])
        self.seller_log.append(f"You: {e['says']}")
        for s in self.stories.values():
            s.asked(*self.current)
            s.rounds[-1].append(f"Seller: {e['says']}")
        if self.prompt_format in NOTE_FORMATS and e.get("note"):
            # the asker writes the note (everyone heard the same answer, so one note serves every buyer's story)
            asker = self.current[0]
            self.add("note", asker, self.offers[asker], self.stories[asker].text(live=True), e["note"])
            for s in self.stories.values():
                s.notes[-1] = e["note"]

    def react(self, e: dict[str, Any]) -> None:
        """One buyer's reaction to the answer: sign, what they say and the new offer."""
        name, s = e["name"], self.stories[e["name"]]
        # sanity check: if the offer chain does not add up, the sale file is broken and I'd rather crash than train
        # on it
        if e["old"] != self.offers[name]:
            raise ValueError(f"sale {self.seed}: {name} reacts from ${e['old']}, the record says ${self.offers[name]}")
        # the target puts the sign first, then the spoken reason, then the number, so the reason is written before
        # the model commits to an amount
        self.add("react", name, e["old"], s.text(live=True), f"{e['sign']} | {e['says']} | {money(e['offer'])}")
        s.rounds[-1].append(f"You ({money(e['old'])} -> {money(e['offer'])}): {e['says']}")
        self.seller_log.append(f"{name} ({money(e['old'])} -> {money(e['offer'])}): {e['says']}")
        self.offers[name] = e["offer"]

    def imagine(self, e: dict[str, Any]) -> None:
        """The three things this buyer pictures with the cat at home, and what they say about them."""
        name = e["name"]
        self.add("imagine", name, self.offers[name], self.stories[name].text(), scene_target(e, self.prompt_format))

    def decide(self, e: dict[str, Any]) -> None:
        """Format 8: the seller's decision on this buyer's final offer, and the buyer's reply to a counter-offer."""
        if not self.seller:
            return
        name = e["name"]
        price = money(e["price"]) if e.get("price") else "-"
        story = self.heard() + f"\n{name}'s final offer: {money(e['offer'])}"
        self.add_seller("decide", story, f"{e['action']} | {price} | {e['says']}", name)
        if e.get("reply"):
            buyer_story = self.stories[name].text() + f"\nSeller ({price}): {e['says']}"
            self.add("reply", name, e["offer"], buyer_story, f"{e['reply'][0]} | {e['reply'][1]}")

    def sold(self, e: dict[str, Any]) -> None:
        """The sale's end: the player picks the buyer, so there is nothing for a buyer to learn."""


TASK_TOKENS = {"open": OPEN, "ask": ASK, "react": REACT, "imagine": IMAGINE, "note": NOTE, "answer": ANSWER,
               "decide": DECIDE, "reply": REPLY}
STEPS = ("open", "ask", "answer", "react", "imagine", "decide", "sold")


def examples(sale: dict[str, Any], prompt_format: int = PROMPT_FORMAT) -> list[Example]:
    """Every decision of every buyer in one sale, in order, checked against the sale's own bookkeeping."""
    walk = _Walk(sale, prompt_format)
    for e in sale["events"]:
        if e["kind"] not in STEPS:      # getattr below would happily call the wrong thing, so check the name first
            raise ValueError(f"sale {walk.seed}: unknown event kind {e['kind']!r}")
        getattr(walk, e["kind"])(e)
    return walk.out


def write_cards(sales: list[dict[str, Any]], path: Path = CARDS_FILE) -> None:
    """Every buyer card in the sales, by persona, for setting up buyers in the game (buyers/cat_market.py)."""
    cards: dict[str, list[dict[str, Any]]] = {}
    for sale in sales:
        for c in sale["cards"].values():
            entry = {"cares_about": c["cares_about"]}
            if "budget" in c:
                entry["budget"] = c["budget"]                                   # format-4 models only
            if entry not in cards.setdefault(c["persona"], []):
                cards[c["persona"]].append(entry)
    path.write_text(json.dumps(dict(sorted(cards.items())), indent=1))


def build(sources: tuple[Path, ...] = SOURCES, out_dir: Path = OUT_DIR, tokenizer: Path | None = None,
          prompt_format: int = PROMPT_FORMAT) -> dict[str, Any]:
    """Write train.jsonl and val.jsonl (and meta.json) from the sales; report counts and, with a tokenizer, the
    prompt + target lengths in tokens."""
    sales = [json.loads(line) for path in sources for line in path.read_text().splitlines() if line.strip()]
    seeds = [s["seed"] for s in sales]
    if len(seeds) != len(set(seeds)):          # two sales on one seed = the same cat and buyers twice
        raise ValueError("a seed appears in two sales")
    splits: dict[str, list[Example]] = {"train": [], "val": []}
    for sale in sales:
        splits["val" if is_val(sale["seed"]) else "train"].extend(examples(sale, prompt_format))
    out_dir.mkdir(parents=True, exist_ok=True)
    if prompt_format == PROMPT_FORMAT and sources == SOURCES:   # the game's cards follow the sales of the model it plays
        write_cards(sales)
    for split, rows in splits.items():
        (out_dir / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report: dict[str, Any] = {
        "sales": {split: len({r["seed"] for r in rows}) for split, rows in splits.items()},
        "examples": {split: dict(Counter(r["task"] for r in rows)) for split, rows in splits.items()},
        "authors": dict(Counter(s.get("author", "qwen_v9_edited") for s in sales))}
    if tokenizer:
        from ..llm.tokenizer import Tokenizer
        tok = Tokenizer(tokenizer)
        lengths: dict[str, list[int]] = {}
        for r in splits["train"] + splits["val"]:
            lengths.setdefault(r["task"], []).append(len(tok.encode(r["prompt"] + r["target"])))
        report["tokens"] = {task: {"median": sorted(v)[len(v) // 2], "p95": sorted(v)[int(len(v) * 0.95)],
                                   "max": max(v), "over_512": sum(x > 512 for x in v),
                                   "over_768": sum(x > 768 for x in v)} for task, v in lengths.items()}
    tasks = ["open", "ask", "react", "imagine"] + (["note"] if prompt_format in NOTE_FORMATS else []) + \
        (["answer", "decide", "reply"] if prompt_format in SELLER_FORMATS else [])
    meta = {"prompt_format": prompt_format, "tasks": tasks, "k": K,
            "sources": [str(p) for p in sources], **report}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return report


def main() -> None:
    """Build the current format's data (or, with --prompt-format 4, B1's) and print the report."""
    from ..llm.tokenizer import TOKENIZER_DIR
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR / "sp8192.model")
    ap.add_argument("--prompt-format", type=int, default=PROMPT_FORMAT, choices=[4, 5, 6, 7, 8, 9],
                    help="4 rebuilds buyer GPT B1's data (budget on the card); 6 uses the persona, animal and thing "
                         "tokens; 9 writes the scene line first")
    ap.add_argument("--sources", choices=["b2", "b3", "b4", "b5"], default="b2",
                    help="b2: the sales of B2 (default); b3: all format-5 sales so far plus Muse's 100; "
                         "b4: b3 plus sales_b3.jsonl (the scene-quota sales)")
    a = ap.parse_args()
    sources = (SOURCES_B1 if a.prompt_format == 4 else SOURCES_B3 if a.sources == "b3"
               else SOURCES_B4 if a.sources == "b4" else SOURCES_B5 if a.sources == "b5" else SOURCES)   # B1: its two
    print(json.dumps(build(sources, a.out, a.tokenizer, a.prompt_format), indent=1))


if __name__ == "__main__":
    main()
