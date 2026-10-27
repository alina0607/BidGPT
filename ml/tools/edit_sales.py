"""Hand-correct teacher sales to the owner's standard, sentence by sentence (owner, 2026-10-11).

    python tools/edit_sales.py sheet <sales.jsonl> <seed>                 # what to read: each round, every reaction
    python tools/edit_sales.py apply <sales.jsonl> <edits_dir> <out.jsonl>

An edit file `<edits_dir>/<seed>.json` changes events by their index:

    {"events": {"57": {"says": "...", "sign": "-", "offer": 60}, "190": {"things": [...], "says": "..."}},
     "insert": {"23": [{"kind": "react", "name": "Ines", "sign": "0", "offer": 70, "says": "..."}]},
     "add_imagine": [{"name": "Ines", "things": [...], "says": "..."}]}

`insert` adds events after an event index (e.g. the reactions of a round the teacher left without any);
`{"drop": true}` removes an event (e.g. a buyer's reaction the teacher wrote twice).

`apply` rebuilds the offers in order: a reaction keeps its own change of offer (new − old) unless the edit gives a new
offer, a withdrawal stays a withdrawal, an offer never passes the budget; the sale goes to the best final offer. Every
edited line keeps the teacher's version in "original" and is marked "edited_by": "claude". Every reaction and scene
is then checked with stories.reaction_problem / imagine_problem; a sale with a line that still breaks a rule is
reported and not written.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sellanything_ml.buyers import stories


def _sales(path: Path) -> dict[int, dict[str, Any]]:
    return {r["seed"]: r for r in map(json.loads, path.read_text().splitlines())}


def sheet(path: Path, seed: int) -> str:
    """The sale as it is read for editing: cards, then each round's question, answer and every reaction with its
    event index and what the program checks say."""
    r = _sales(path)[seed]
    out = [f"seed {seed}"] + [f"CARD {n}: {c['persona']} | cares: {', '.join(c['cares_about'])} | budget ${c['budget']}"
                              for n, c in r["cards"].items()]
    answer = ""
    for i, e in enumerate(r["events"]):
        k = e["kind"]
        if k == "open":
            out.append(f"[{i}] OPEN {e['name']} ${e['offer']}: {e['says']}")
        elif k == "ask":
            out.append(f"\n[{i}] ROUND {e['round']} {e['name']} asks: {e['says']}")
        elif k == "answer":
            answer = e["says"]
            out.append(f"[{i}] SELLER: {answer}")
        elif k == "react":
            p = stories.reaction_problem(e, r["cards"][e["name"]], e["old"], e["name"], answer)
            out.append(f"[{i}] {e['name']} ({e['sign']}) {e['old']}->{e['offer']}: {e['says']}"
                       + (f"   !! {p}" if p else ""))
        elif k == "imagine":
            p = stories.imagine_problem(e)
            out.append(f"[{i}] IMAGINE {e['name']} {e['things']}: {e['says']}" + (f"   !! {p}" if p else ""))
        elif k == "sold":
            out.append(f"[{i}] SOLD {e['name']} ${e['price']}")
    return "\n".join(out)


def apply_edits(r: dict[str, Any], edits: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The sale with the edits applied and its offers rebuilt; and every rule a line still breaks."""
    cards, changes, inserts = r["cards"], edits.get("events", {}), edits.get("insert", {})
    staged: list[tuple[int | None, dict, dict]] = []          # (teacher's index or None if added, event, its edit)
    for i, e in enumerate(r["events"]):
        e, edit = dict(e), changes.get(str(i), {})
        if edit:
            e["original"] = {k: e[k] for k in edit if k in e}
            e.update(edit)
            e["edited_by"] = "claude"
        staged.append((i, e, edit))
        staged += [(None, {**x, "edited_by": "claude", "original": None}, x) for x in inserts.get(str(i), [])]
    offers: dict[str, int] = {}
    events, problems, answer = [], [], ""
    for i, e, edit in staged:
        if edit.get("drop"):
            continue                                           # e.g. the teacher wrote one buyer's reaction twice
        if e["kind"] == "open":
            offers[e["name"]] = e["offer"]
        elif e["kind"] == "answer":
            answer = e["says"]
        elif e["kind"] == "react":
            old = offers[e["name"]]
            if old == 0:
                continue                                       # withdrawn earlier (after an edit): no more reactions
            if "offer" not in edit:
                orig_old, orig_new = r["events"][i]["old"], r["events"][i]["offer"]     # a teacher's event
                e["offer"] = 0 if orig_new == 0 else min(max(1, old + orig_new - orig_old), cards[e["name"]]["budget"])
            e["old"] = old
            offers[e["name"]] = e["offer"]
            p = stories.reaction_problem(e, cards[e["name"]], old, e["name"], answer)
            if p:
                problems.append(f"[{i}] {e['name']}: {p}")
        elif e["kind"] == "imagine":
            if offers.get(e["name"], 0) == 0:
                continue
            p = stories.imagine_problem(e)
            if p:
                problems.append(f"[{i}] {e['name']}: {p}")
        elif e["kind"] == "sold":
            for extra in edits.get("add_imagine", []):
                scene = {"kind": "imagine", **extra, "edited_by": "claude", "original": None}
                p = stories.imagine_problem(scene)
                if p:
                    problems.append(f"[added] {extra['name']}: {p}")
                events.append(scene)
            present = {n: o for n, o in offers.items() if o > 0}
            if not present:
                continue
            best = max(present, key=present.get)
            e = {"kind": "sold", "name": best, "price": present[best]}
        events.append(e)
    taken: list[set[str]] = []
    for e in events:
        if e["kind"] == "imagine":
            things = set(e["things"])
            if any(len(t & things) >= 2 for t in taken):
                problems.append(f"{e['name']}: shares two things with another buyer's scene")
            taken.append(things)
    out = dict(r)
    out["events"], out["edited"] = events, sum(e.get("edited_by") == "claude" for e in events)
    m = stories.Market(r["sale"])
    m.cards, m.events = cards, events
    out["check"] = stories.check(m)
    out["text"] = stories.render(m)
    return out, problems


def main() -> None:
    """`sheet` prints one sale for editing; `apply` writes every sale whose edits pass the rules."""
    cmd = sys.argv[1]
    if cmd == "sheet":
        print(sheet(Path(sys.argv[2]), int(sys.argv[3])))
        return
    sales, edits_dir, out = _sales(Path(sys.argv[2])), Path(sys.argv[3]), Path(sys.argv[4])
    written, failed = [], {}
    for f in sorted(edits_dir.glob("*.json")):
        fixed, problems = apply_edits(sales[int(f.stem)], json.loads(f.read_text()))
        if problems:
            failed[f.stem] = problems
        else:
            written.append(fixed)
    out.write_text("".join(json.dumps(r) + "\n" for r in written))
    print(json.dumps({"written": len(written), "edited_lines": sum(r["edited"] for r in written), "failed": failed},
                     indent=1))


if __name__ == "__main__":
    main()
