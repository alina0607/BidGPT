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
from typing import Any

PEDIGREE = ("Maine Coon", "British Shorthair", "Scottish Fold", "Sphynx", "Bengal", "Siamese", "Persian", "Ragdoll")
HEALTH = {"microchipped and healthy": 5, "fully vaccinated, vet check last month": 5,
          "bloodwork last month was normal": 5, "no vaccines yet": -5, "dewormed but not vaccinated": -5,
          "no vaccination records": -5, "vaccinated but no vet visit in two years": -5, "no bloodwork in years": -5,
          "recovering from a cold": -5, "on medication for an overactive thyroid": -10, "has some stiff joints": -10,
          "bloodwork shows early kidney disease, on a kidney diet": -15}
VIRUS = {"tested negative for FIV and FeLV": 5, "tested FIV positive": -15}
# Old sales (B1, seeds 2048-2099) where the seller named a price during the sale: that price is the asking price,
# so the seller's decisions never go back on their own word (found 2026-10-13 while adding the decisions).
STATED = {"2050": 50, "2052": 30, "2055": 50, "2057": 60, "2058": 50, "2061": 70, "2068": 40}
EXTRA = {"heart": -20, "one eye": -5, "thyroid": -10, "three legs": -10, "deaf": -5, "teeth": -5}


def years(cat: dict[str, Any]) -> float:
    """The cat's age in years (an extra "very old" fact such as "19 years old" replaces the sheet's age)."""
    for kind, fact in zip(cat.get("extra_kinds", []), cat.get("extra", []), strict=False):
        if kind == "very old":
            return float(fact.split()[0])
    return cat["months"] / 12


def asking(sale: dict[str, Any]) -> int:
    """The asking price in whole dollars, a multiple of 5 between $20 and $150."""
    cat = sale["cat"]
    price = float(sale.get("typical_price", 75))
    age = years(cat)
    price += 15 if age < 1 else 5 if age < 4 else 0 if age < 9 else -10 if age < 12 else -15 if age < 15 else -20
    kinds = cat.get("extra_kinds", [])
    if any(b in cat["breed"] for b in PEDIGREE) or "no papers" in kinds:
        price += 5 if "no papers" in kinds else 20
    price += 5 if cat["fixed"] == "spayed/neutered" else 0
    price += HEALTH.get(cat["health"], 0) + VIRUS.get(cat["virus"], 0)
    price += sum(EXTRA.get(k, 0) for k in kinds)
    return int(min(150, max(20, 5 * round(price / 5))))


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
