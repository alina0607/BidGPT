"""Cat sales written by Claude from scratch (owner, 2026-10-11: Claude writes the sales instead of Qwen), checked
with the same rules as the teacher's and laid out exactly like them, so they train the buyer GPT the same way.

    python tools/write_sales.py <specs_dir> <out.jsonl> [author]

One spec per sale, `<specs_dir>/<seed>.json`; the cat, the buyers and the seller's style come from
stories.seed_sale(seed), so they are the same as a teacher sale on that seed:

    {"cards": {"Chloe": {"cares_about": [...], "budget": 90}, ...},
     "open": {"Chloe": [65, "Hi! ..."], ...},
     "rounds": [{"name": "Andre", "q": "...", "a": "...", "react": {"Chloe": ["0", 65, "..."], ...}}, ...],
     "imagine": {"Chloe": [["teddy-bear", "crayon", "swing set"], "..."], ...}}

Every buyer still bidding must react in every round and picture the cat at the end; a buyer who withdraws (offer 0)
does neither afterwards. Each reaction and scene must pass stories.reaction_problem / imagine_problem, no two scenes
may share two things, and the sale goes to the best final offer. Sales that break a rule are reported, not written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sellanything_ml.buyers import stories

ASKING_FILE = Path(__file__).resolve().parents[2] / "data" / "written_sales" / "asking_prices.json"
ASKING: dict[str, int] = json.loads(ASKING_FILE.read_text()) if ASKING_FILE.exists() else {}   # tools/asking_price.py


def _rounds(m: stories.Market, spec: dict[str, Any], problems: list[str]) -> None:
    for k, rnd in enumerate(spec["rounds"], 1):
        if m.offers.get(rnd["name"], 0) <= 0:
            problems.append(f"round {k}: {rnd['name']} has withdrawn and cannot ask")
        m.events.append({"kind": "ask", "round": k, "name": rnd["name"], "offer": m.offers.get(rnd["name"], 0),
                         "says": rnd["q"]})
        m.events.append({"kind": "answer", "round": k, "to": rnd["name"], "q": rnd["q"], "says": rnd["a"],
                         **({"note": rnd["note"]} if rnd.get("note") else {})})     # prompt format 7 reads it
        missing = set(m.present()) - set(rnd["react"])
        extra = set(rnd["react"]) - set(m.present())
        if missing or extra:
            problems.append(f"round {k}: missing {sorted(missing)}, not bidding {sorted(extra)}")
        for name in [n for n in m.present() if n in rnd["react"]]:
            sign, offer, says = rnd["react"][name]
            row = {"sign": sign, "offer": offer, "says": says}
            problem = stories.reaction_problem(row, m.cards[name], m.offers[name], name, rnd["a"])
            if problem:
                problems.append(f"round {k} {name}: {problem}")
            m.events.append({"kind": "react", "name": name, "old": m.offers[name], "offer": offer, "sign": sign,
                             "says": says})
            m.offers[name] = offer


def _scenes(m: stories.Market, spec: dict[str, Any], problems: list[str]) -> None:
    if set(spec["imagine"]) != set(m.present()):
        problems.append(f"scenes for {sorted(spec['imagine'])}, buyers still bidding {sorted(m.present())}")
    taken: list[set[str]] = []
    for name in [n for n in m.present() if n in spec["imagine"]]:
        things, says = spec["imagine"][name]
        problem = stories._scene_problem({"things": things, "says": says}, taken)
        if problem:
            problems.append(f"scene {name}: {problem}")
        taken.append(set(things))
        m.events.append({"kind": "imagine", "name": name, "things": things, "says": says})


def build(seed: int, spec: dict[str, Any], author: str = "Alina Chuang") -> tuple[dict[str, Any], list[str]]:
    """The sale laid out like a teacher sale, and every rule it breaks."""
    s = stories.sale_for_seed(seed)
    m = stories.Market(s)
    problems: list[str] = []
    for b in s["buyers"]:
        n = b["name"]
        m.cards[n] = {"persona": b["persona"], **spec["cards"][n]}
        offer, says = spec["open"][n]
        if "budget" in m.cards[n] and offer > m.cards[n]["budget"]:      # sales from 2026-10-12 have no budget
            problems.append(f"{n} opens above the budget")
        m.offers[n] = offer
        m.events.append({"kind": "open", "name": n, "offer": offer, "says": says})
    _rounds(m, spec, problems)
    _scenes(m, spec, problems)
    # the seller's decision on every buyer still bidding (seller_decide/, owner 2026-10-13): accept, refuse, or one
    # counter-offer with the buyer's reply; checked by data/written_sales/seller_decide/check_decide.py
    for name, (action, price, says, reply) in spec.get("decide", {}).items():
        m.events.append({"kind": "decide", "name": name, "offer": m.offers.get(name), "action": action,
                         "price": price, "says": says, "reply": reply})
    best = max(m.present(), key=m.offers.get)
    m.events.append({"kind": "sold", "name": best, "price": m.offers[best]})
    problems += [f"a line uses a dash: {e['says'][:40]}" for e in m.events
                 if any(d in e.get("says", "") for d in stories.DASHES)]
    asking = ASKING.get(str(seed))
    if asking is not None:
        s = {**s, "asking": asking}                   # the seller's asking price, part of the seller's card
    row = {"seed": seed, "sale": s, "cards": m.cards, "events": m.events, "author": author,
           "check": stories.check(m), "text": stories.render(m)}
    return row, problems


def main() -> None:
    """Build every spec in a folder and write the sales that break no rule."""
    specs, out = Path(sys.argv[1]), Path(sys.argv[2])
    author = sys.argv[3] if len(sys.argv) > 3 else "Alina Chuang"          # who wrote the sales, kept in each row
    written, failed = [], {}
    for f in sorted(specs.glob("*.json")):
        row, problems = build(int(f.stem), json.loads(f.read_text()), author)
        if problems:
            failed[f.stem] = problems
        else:
            written.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in written))
    print(json.dumps({"written": len(written), "failed": failed}, indent=1))


if __name__ == "__main__":
    main()
