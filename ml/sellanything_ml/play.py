"""Play one selling round in the terminal, using the price engine and v1 phrases (no model).

    python -m sellanything_ml.play bicycle
    python -m sellanything_ml.play tiger            # a refuse item
    python -m sellanything_ml.play dog --auto       # answers picked at random (for testing)
"""

from __future__ import annotations

import argparse
import random

from .buyers import Round, load_items, load_labels, load_traits, make_buyers, round_price
from .buyers import phrases


def money(x: float) -> str:
    return f"${round_price(x):,.2f}".replace(".00", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("item", help="Quick, Draw! class, e.g. bicycle or 'cell phone'")
    parser.add_argument("--buyers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--auto", action="store_true", help="pick answers at random")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labels, items, traits = load_labels(), load_items(), load_traits()
    if args.item not in labels:
        raise SystemExit(f"Unknown item {args.item!r}")

    if labels[args.item] == "refuse":
        print(f"You try to sell a {args.item}.\n")
        names = rng.sample(traits.names, args.buyers)
        for name, line in zip(names, phrases.refuse_lines(args.item, args.buyers, rng)):
            print(f"{name}: {line}")
        print("\nNobody buys it.")
        return
    if args.item not in items:
        raise SystemExit(f"{args.item!r} can be sold, but has no schema yet")

    item = items[args.item]
    rnd = Round(item, make_buyers(item, traits, args.buyers, rng))
    print(f"You're selling a {item.noun}. Buyers:\n")
    for b in rnd.buyers:
        print(f"  {b.persona(traits)}  (opens at {money(b.offer)})")
    print()

    while (q := rnd.next_question()) is not None:
        asker, attr = q
        print(f"{asker.name}: {phrases.ask(attr.question, rng)}")
        for i, option in enumerate(attr.options, 1):
            print(f"   {i}. {option.answer}")
        if args.auto:
            choice = rng.randrange(len(attr.options))
            print(f"   > {choice + 1}")
        else:
            choice = -1
            while not 0 <= choice < len(attr.options):
                raw = input("   > ").strip()
                choice = int(raw) - 1 if raw.isdigit() else -1
        option = attr.options[choice]
        print(f"You: {option.answer}")

        reactions, bids = rnd.answer(asker, attr, option.id)
        for r in reactions:
            if r.left:
                print(f"{r.buyer}: {phrases.leave(item.noun, rng)}  [leaves]")
            elif r.speaks:
                line = phrases.react(r.delta, item.noun, rng, option.factor)
                print(f"{r.buyer}: {line}  [interest {r.delta:+d}]")
        for bid in bids:
            print(f"{bid.buyer}: {phrases.raise_bid(round_price(bid.amount), rng)}")
        offers = "  ".join(f"{name} {money(v)}" for name, v in rnd.offers().items())
        print(f"   offers: {offers}\n")

    final = rnd.offers()
    if not final:
        print("Everyone walked away.")
        return
    best = max(final, key=final.__getitem__)
    print("Final offers: " + ", ".join(f"{n} {money(v)}" for n, v in final.items()))
    print(f"Best offer: {best} at {money(final[best])} "
          f"(typical price for a {item.noun}: {money(item.typical)})")


if __name__ == "__main__":
    main()
