"""Buy a cat in the terminal: the model (prompt format 8, B5) is the seller and the other buyers, you are one buyer.

    cd ml && PYTHONPATH=. python -m sellanything_ml.cat_buy --name Alina
    cd ml && PYTHONPATH=. python -m sellanything_ml.cat_buy --checkpoint runs/sft_b6/best.pt --seed 7 --buyers 3

The sale is the game's own (buyers/cat_market.py with a model seller): the other buyers open, you open, and in seat
order everyone asks one question; the seller answers each and everyone reacts with a new offer (you type yours). At
the end the seller decides on every final offer: accept, a counter-offer you accept or walk away from, or refuse.
The stall shows the asking price, like a price tag; the cat's facts you learn only from the seller's answers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .buyers.cat_market import BuyerGPT, CatMarket, load_seller, seller_sheet


def show(lines: list[str]) -> None:
    """Print the sale's new lines."""
    for line in lines:
        print(line)


def ask(prompt: str) -> str:
    """A line the player types (asked again while empty)."""
    while not (text := input(prompt).strip()):
        pass
    return text


def ask_offer(prompt: str, allow_zero: bool) -> int:
    """A whole number of dollars the player types (0 only when walking away is allowed)."""
    while True:
        raw = input(prompt).strip().lstrip("$")
        if raw.isdigit() and (int(raw) > 0 or allow_zero):
            return int(raw)
        print("   a whole number of dollars" + (" (0 to walk away)" if allow_zero else ""))


def main() -> None:
    """One sale at the stall, with the player as the last buyer."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path,
                    help="a model that learned the seller's side, prompt format 8 or 10 (default: runs/sft_b6)")
    ap.add_argument("--name", default="Alina", help="your name at the stall")
    ap.add_argument("--buyers", type=int, default=3, help="model buyers besides you")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    gpt = BuyerGPT(a.checkpoint, a.device) if a.checkpoint else load_seller(a.device)
    if gpt is None:
        raise SystemExit("no model that learned the seller's side on disk yet (runs/sft_b6/best.pt)")
    seed = a.seed if a.seed is not None else int.from_bytes(os.urandom(3), "big")
    m = CatMarket.start(gpt, a.buyers, seed=seed, seller=seller_sheet(seed), player=a.name)
    print(f"A cat for sale, asking ${m.seller['asking']} (sale {seed}). Other buyers: "
          + ", ".join(f"{n} (who {c['persona']})" for n, c in m.cards.items() if n != a.name) + "\n")
    show(m.log)
    shown = len(m.log)
    while m.sold is None:
        step = m.waiting
        if step == "open":
            m.player_open(ask("\nYour greeting: "), ask_offer("Your first offer: $", allow_zero=False))
        elif step == "ask":
            m.player_ask(ask("\nYour question to the seller: "))
        elif step == "react":
            m.player_react(ask(f"\nWhat you say (offer now ${m.offers[a.name]}): "),
                           ask_offer("Your offer now: $", allow_zero=True))
        elif step == "reply":
            d = m.decisions[a.name]
            yes = ask(f"\nThe seller counters at ${d['price']}. Accept? (y/n) ").lower().startswith("y")
            m.player_reply(yes, ask("What you say: "))
        show(m.log[shown:])
        shown = len(m.log)
    sold = m.sold
    print("\n" + (f"Sold to {sold['buyer']} at ${sold['price']}." if sold["buyer"] else "The cat stays with the seller.")
          + (" That's you!" if sold["buyer"] == a.name else ""))
    print("The cat was: " + "; ".join([m.seller["cat"]["age"], m.seller["cat"]["breed"], m.seller["cat"]["health"]]))


if __name__ == "__main__":
    main()
