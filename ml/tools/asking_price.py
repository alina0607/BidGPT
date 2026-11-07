"""The seller's asking price for a cat sale (owner, 2026-10-13): part of the seller's card, like a buyer's card.

    cd ml && python3 tools/asking_price.py ../data/written_sales/sales_b3.jsonl ... --out ../data/written_sales/asking_prices.json

It is set from the cat's facts alone, never from what any buyer offered, so the seller's later decision (accept,
refuse or one counter-offer) is a real judgement against a price fixed before the sale. Start from the typical price
($75), add for what makes a cat worth more (a kitten, a named breed, a clean health record) and take off for what
costs a buyer (old age, illness, FIV positive, a missing leg), then round to $5.

Checked against real adoptions (tools/petfinder_market_check.py, 2026-10-14; the game only needs a reasonable market,
not PetFinder's): the age curve points the same way (cats of 1 to 2 years had 10.4 times a kitten's odds of waiting
100 days unadopted, 3 years and over 11.0, comparing cats of the same rescuer), and the long-haired breeds here
(Persian, Maine Coon, Ragdoll) are already priced up as named breeds (long fur 0.31). The health, vaccine and FIV
steps rest on what they cost a buyer, not on that data (they do not hold within rescuer). One simplification is kept:
a 7-month cat gets the kitten step like a 2-month one; changing it would rewrite 249 decision lines in 27 written
sales for $5, so it stays.
"""

from __future__ import annotations

import sys

from sellanything_ml.buyers.asking import STATED, asking


def main() -> None:
    """Asking prices for every sale in the given sales files (one JSON object per line, as write_sales.py builds them)."""
    import json
    from pathlib import Path
    args = sys.argv[1:]
    out = Path(args[args.index("--out") + 1]) if "--out" in args else None
    files = [a for a in args if a.endswith(".jsonl")]
    prices = {}
    for path in files:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            prices[str(row["seed"])] = STATED.get(str(row["seed"]), asking(row["sale"]))
    spread = sorted(prices.values())
    print(f"{len(spread)} sales: min ${spread[0]}, quartiles ${spread[len(spread) // 4]} / ${spread[len(spread) // 2]} / "
          f"${spread[3 * len(spread) // 4]}, max ${spread[-1]}")
    if out:
        out.write_text(json.dumps(dict(sorted(prices.items())), indent=1) + "\n")


if __name__ == "__main__":
    main()
